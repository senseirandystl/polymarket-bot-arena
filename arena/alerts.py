"""Multi-channel production alerts (Telegram, Discord, email).

Fires on drawdowns / risk pauses, regime shifts, evolution cycles, health
degradation, and kill-switch events. Secrets live in the encrypted credentials
store; channel toggles + event filters live in arena_state ``alerts_config``.

Every send is:
  * debounced per (event_type, key) for ALERT_DEBOUNCE_SEC
  * best-effort (never raises into the trading hot path)
  * logged to arena_state ``alerts_log`` (ring buffer) for the dashboard
"""

from __future__ import annotations

import json
import logging
import smtplib
import threading
import time
import urllib.error
import urllib.request
from email.mime.text import MIMEText
from typing import Any, Optional

import config
import db

logger = logging.getLogger("arena.alerts")

STATE_KEY = "alerts_config"
LOG_KEY = "alerts_log"
DEBOUNCE_KEY = "alerts_debounce"

EVENT_TYPES = (
    "drawdown",
    "risk_pause",
    "kill_switch",
    "regime_shift",
    "evolution",
    "error",
    "health",
    "test",
)

CHANNELS = ("telegram", "discord", "email")

# credential keys (encrypted store)
CRED_TELEGRAM_TOKEN = "alert_telegram_bot_token"
CRED_TELEGRAM_CHAT = "alert_telegram_chat_id"
CRED_DISCORD_WEBHOOK = "alert_discord_webhook"
CRED_SMTP_HOST = "alert_smtp_host"
CRED_SMTP_PORT = "alert_smtp_port"
CRED_SMTP_USER = "alert_smtp_user"
CRED_SMTP_PASS = "alert_smtp_password"
CRED_SMTP_FROM = "alert_smtp_from"
CRED_SMTP_TO = "alert_smtp_to"

_lock = threading.Lock()
_debounce: dict[str, float] = {}


def _default_config() -> dict[str, Any]:
    return {
        "enabled": bool(getattr(config, "ALERTS_ENABLED", False)),
        "channels": {
            "telegram": False,
            "discord": False,
            "email": False,
        },
        "events": {e: True for e in EVENT_TYPES if e != "test"},
        "min_level": "info",  # info | warn | critical
        "debounce_sec": float(getattr(config, "ALERT_DEBOUNCE_SEC", 300)),
    }


def load_config() -> dict[str, Any]:
    base = _default_config()
    raw = db.get_arena_state(STATE_KEY)
    if not raw:
        return base
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return base
    if not isinstance(data, dict):
        return base
    base["enabled"] = bool(data.get("enabled", base["enabled"]))
    if isinstance(data.get("channels"), dict):
        for ch in CHANNELS:
            if ch in data["channels"]:
                base["channels"][ch] = bool(data["channels"][ch])
    if isinstance(data.get("events"), dict):
        for ev in EVENT_TYPES:
            if ev in data["events"]:
                base["events"][ev] = bool(data["events"][ev])
    if data.get("min_level") in ("info", "warn", "critical"):
        base["min_level"] = data["min_level"]
    try:
        base["debounce_sec"] = float(data.get("debounce_sec") or base["debounce_sec"])
    except (TypeError, ValueError):
        pass
    return base


def save_config(cfg: dict[str, Any]) -> dict[str, Any]:
    merged = _default_config()
    if isinstance(cfg, dict):
        if "enabled" in cfg:
            merged["enabled"] = bool(cfg["enabled"])
        if isinstance(cfg.get("channels"), dict):
            for ch in CHANNELS:
                if ch in cfg["channels"]:
                    merged["channels"][ch] = bool(cfg["channels"][ch])
        if isinstance(cfg.get("events"), dict):
            for ev in EVENT_TYPES:
                if ev in cfg["events"]:
                    merged["events"][ev] = bool(cfg["events"][ev])
        if cfg.get("min_level") in ("info", "warn", "critical"):
            merged["min_level"] = cfg["min_level"]
        if cfg.get("debounce_sec") is not None:
            merged["debounce_sec"] = max(30.0, float(cfg["debounce_sec"]))
    db.set_arena_state(STATE_KEY, json.dumps(merged))
    return merged


def _cred(key: str) -> Optional[str]:
    try:
        import credentials_store
        v = credentials_store.get_credential(key)
        return str(v).strip() if v else None
    except Exception:
        return None


def channel_status() -> dict[str, dict]:
    """Whether each channel has credentials + is toggled on."""
    cfg = load_config()
    return {
        "telegram": {
            "enabled": bool(cfg["channels"].get("telegram")),
            "configured": bool(_cred(CRED_TELEGRAM_TOKEN) and _cred(CRED_TELEGRAM_CHAT)),
        },
        "discord": {
            "enabled": bool(cfg["channels"].get("discord")),
            "configured": bool(_cred(CRED_DISCORD_WEBHOOK)),
        },
        "email": {
            "enabled": bool(cfg["channels"].get("email")),
            "configured": bool(
                _cred(CRED_SMTP_HOST) and _cred(CRED_SMTP_TO)
                and (_cred(CRED_SMTP_FROM) or _cred(CRED_SMTP_USER))
            ),
        },
    }


def _level_rank(level: str) -> int:
    return {"info": 0, "warn": 1, "warning": 1, "critical": 2, "error": 2}.get(
        (level or "info").lower(), 0)


def _should_send(cfg: dict, event_type: str, level: str, debounce_key: str) -> bool:
    if not cfg.get("enabled"):
        return False
    if event_type != "test" and not cfg.get("events", {}).get(event_type, True):
        return False
    if _level_rank(level) < _level_rank(cfg.get("min_level") or "info"):
        return False
    if event_type == "test":
        return True
    deb = float(cfg.get("debounce_sec") or 300)
    now = time.time()
    with _lock:
        last = _debounce.get(debounce_key, 0.0)
        if now - last < deb:
            return False
        _debounce[debounce_key] = now
    return True


def _append_log(entry: dict) -> None:
    try:
        raw = db.get_arena_state(LOG_KEY)
        log = json.loads(raw) if raw else []
        if not isinstance(log, list):
            log = []
    except Exception:
        log = []
    log.insert(0, entry)
    log = log[:80]
    try:
        db.set_arena_state(LOG_KEY, json.dumps(log, default=str))
    except Exception:
        pass


def get_alert_log(limit: int = 30) -> list:
    try:
        raw = db.get_arena_state(LOG_KEY)
        log = json.loads(raw) if raw else []
        return log[:limit] if isinstance(log, list) else []
    except Exception:
        return []


def _http_json(url: str, payload: dict, timeout: float = 8.0) -> tuple[bool, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "User-Agent": "pba-alerts/1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")[:300]
            return 200 <= resp.status < 300, body
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:
        return False, str(e)


def _send_telegram(title: str, body: str, level: str) -> tuple[bool, str]:
    token = _cred(CRED_TELEGRAM_TOKEN)
    chat = _cred(CRED_TELEGRAM_CHAT)
    if not token or not chat:
        return False, "telegram credentials missing"
    icon = {"critical": "🔴", "warn": "🟠", "warning": "🟠", "info": "🟢"}.get(
        level, "⚪")
    text = f"{icon} *{title}*" + (f"\n{body}" if body else "")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    return _http_json(url, {
        "chat_id": chat,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    })


def _send_discord(title: str, body: str, level: str) -> tuple[bool, str]:
    webhook = _cred(CRED_DISCORD_WEBHOOK)
    if not webhook:
        return False, "discord webhook missing"
    color = {"critical": 0xF85149, "warn": 0xF0883E, "warning": 0xF0883E,
             "info": 0x3FB950}.get(level, 0x8B949E)
    return _http_json(webhook, {
        "username": "PBA Risk",
        "embeds": [{
            "title": title[:256],
            "description": body[:2000],
            "color": color,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }],
    })


def _send_email(title: str, body: str, level: str) -> tuple[bool, str]:
    host = _cred(CRED_SMTP_HOST)
    to_addr = _cred(CRED_SMTP_TO)
    if not host or not to_addr:
        return False, "smtp host/to missing"
    port = int(_cred(CRED_SMTP_PORT) or "587")
    user = _cred(CRED_SMTP_USER)
    password = _cred(CRED_SMTP_PASS)
    from_addr = _cred(CRED_SMTP_FROM) or user or "arena@localhost"
    msg = MIMEText(f"[{level.upper()}] {title}\n\n{body}\n", "plain", "utf-8")
    msg["Subject"] = f"[PBA {level.upper()}] {title}"
    msg["From"] = from_addr
    msg["To"] = to_addr
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=10) as s:
                if user and password:
                    s.login(user, password)
                s.sendmail(from_addr, [a.strip() for a in to_addr.split(",")],
                           msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=10) as s:
                s.ehlo()
                try:
                    s.starttls()
                except smtplib.SMTPException:
                    pass
                if user and password:
                    s.login(user, password)
                s.sendmail(from_addr, [a.strip() for a in to_addr.split(",")],
                           msg.as_string())
        return True, "sent"
    except Exception as e:
        return False, str(e)


def notify(
    event_type: str,
    title: str,
    body: str = "",
    *,
    level: str = "info",
    key: str = "",
    detail: Optional[dict] = None,
) -> dict[str, Any]:
    """Send an alert on all enabled+configured channels. Never raises."""
    result: dict[str, Any] = {
        "sent": False, "skipped": True, "channels": {}, "event_type": event_type,
    }
    try:
        cfg = load_config()
        event_type = event_type if event_type in EVENT_TYPES else "error"
        level = (level or "info").lower()
        if level == "error":
            level = "critical"
        deb_key = f"{event_type}:{key or title}"
        if not _should_send(cfg, event_type, level, deb_key):
            result["reason"] = "filtered_or_debounced"
            return result
        result["skipped"] = False
        # The human-facing message is title + body only. The structured
        # ``detail`` dict is kept in the arena_state log for the dashboard /
        # debugging, but is NOT dumped as raw JSON into the notification —
        # that produced messages like `{"from": "normal", "confidence": 0.82...}`.
        full_body = body
        channels_cfg = cfg.get("channels") or {}
        any_ok = False
        for ch, sender in (
            ("telegram", _send_telegram),
            ("discord", _send_discord),
            ("email", _send_email),
        ):
            if not channels_cfg.get(ch):
                continue
            ok, msg = sender(title, full_body, level)
            result["channels"][ch] = {"ok": ok, "detail": msg}
            any_ok = any_ok or ok
            if not ok:
                logger.warning("alert %s failed: %s", ch, msg)
        result["sent"] = any_ok
        _append_log({
            "ts": time.time(),
            "event_type": event_type,
            "level": level,
            "title": title,
            "body": body[:300],
            "detail": detail or None,
            "sent": any_ok,
            "channels": result["channels"],
        })
    except Exception as e:
        logger.warning("notify failed: %s", e)
        result["error"] = str(e)
    return result


def send_test(channel: Optional[str] = None) -> dict[str, Any]:
    """Force a test alert (ignores event filters; still needs channel on)."""
    cfg = load_config()
    # Temporarily enable for test path via direct senders
    results = {}
    title = "PBA alert test"
    body = f"Test message at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
    targets = [channel] if channel in CHANNELS else list(CHANNELS)
    for ch in targets:
        if channel is None and not (cfg.get("channels") or {}).get(ch):
            results[ch] = {"ok": False, "detail": "channel disabled"}
            continue
        sender = {"telegram": _send_telegram, "discord": _send_discord,
                  "email": _send_email}[ch]
        ok, msg = sender(title, body, "info")
        results[ch] = {"ok": ok, "detail": msg}
    _append_log({
        "ts": time.time(), "event_type": "test", "level": "info",
        "title": title, "body": body, "sent": any(r.get("ok") for r in results.values()),
        "channels": results,
    })
    return {"success": True, "channels": results}


# ---------------------------------------------------------------------------
# Convenience emitters used by arena / risk / regime
# ---------------------------------------------------------------------------

# Human-readable regime names for notifications (raw ids stay in `detail`).
_REGIME_LABELS = {
    "high_vol_trend": "High-vol trend",
    "low_vol_range": "Low-vol range",
    "high_vol_chop": "High-vol chop",
    "low_vol_trend": "Low-vol trend",
    "normal": "Normal",
    "unknown": "Unknown",
}


def _regime_label(regime: str) -> str:
    if not regime:
        return "Unknown"
    return _REGIME_LABELS.get(regime, regime.replace("_", " ").title())


def alert_regime_shift(from_regime: str, to_regime: str, confidence: float = 0.0):
    notify(
        "regime_shift",
        f"Regime shift Detected: {_regime_label(from_regime)} → "
        f"{_regime_label(to_regime)} · {(confidence or 0) * 100:.0f}% confidence",
        "",
        level="info",
        key=f"{from_regime}->{to_regime}",
        detail={"from": from_regime, "to": to_regime, "confidence": confidence},
    )


def alert_evolution(cycle: int, trigger: str, summary: str = ""):
    notify(
        "evolution",
        f"Evolution cycle #{cycle}",
        f"trigger={trigger}\n{summary}"[:800],
        level="info",
        key=f"evo:{cycle}",
        detail={"cycle": cycle, "trigger": trigger},
    )


def alert_risk(action: str, reason: str, bot: Optional[str] = None,
               level: str = "warn"):
    event = "kill_switch" if action in ("kill", "unkill") else (
        "drawdown" if "drawdown" in (reason or "") else "risk_pause"
    )
    if action in ("kill",):
        level = "critical"
        event = "kill_switch"
    title = f"Risk {action}" + (f" · {bot}" if bot else "")
    notify(event, title, reason or action, level=level,
           key=f"{action}:{bot or 'portfolio'}:{reason[:40]}")


def alert_health(title: str, body: str, level: str = "warn"):
    notify("health", title, body, level=level, key=title)


def alert_error(where: str, message: str):
    notify("error", f"Error: {where}", message[:800], level="critical",
           key=f"err:{where}:{message[:60]}")

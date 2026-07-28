"""Multi-channel production alerts (Telegram, Discord, email).

Fires on drawdowns / risk pauses, regime shifts, evolution cycles, hourly
performance digests, health degradation, and kill-switch events. Secrets live
in the encrypted credentials store; channel toggles + per-event-type filters
live in arena_state ``alerts_config``.

Timing: all scheduled digests and day boundaries use **America/New_York (ET)**,
matching the rest of the arena (session skips, dashboard today/week rolls).
Timestamps in the DB remain UTC; EOD/hourly labels and "today" stats convert.

Defaults (when no saved ``alerts_config``):
  * master switch ON if any channel has credentials configured
  * each configured channel toggled ON; unconfigured channels stay OFF
  * every event type ON (dashboard can mute individually, e.g. regime_shift)

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
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Any, Optional

import config
import db

logger = logging.getLogger("arena.alerts")

STATE_KEY = "alerts_config"
LOG_KEY = "alerts_log"
DEBOUNCE_KEY = "alerts_debounce"
HOURLY_STATE_KEY = "alerts_last_hourly"
DAILY_STATE_KEY = "alerts_last_daily_date"
SKIP_STORM_STATE_KEY = "alerts_skip_storm_snap"
BANKROLL_STATE_KEY = "alerts_low_bankroll_level"

_ET_ZONE = "America/New_York"

EVENT_TYPES = (
    "drawdown",
    "risk_pause",
    "kill_switch",
    "regime_shift",
    "evolution",
    "hourly_report",
    "daily_report",
    "low_bankroll",
    "feed_stale",
    "feed_restored",
    "live_fill",
    "lane_change",
    "core_lane_tune",
    "skip_storm",
    "resolver_stuck",
    "portfolio_rebalance",
    "startup",
    "error",
    "health",
    "soak_report",
    "test",
)

# Operator-facing labels for dashboard toggles (event id → short name).
EVENT_LABELS: dict[str, str] = {
    "drawdown": "Drawdown",
    "risk_pause": "Risk pause",
    "kill_switch": "Kill switch",
    "regime_shift": "Regime shift",
    "evolution": "Evolution",
    "hourly_report": "Hourly report",
    "daily_report": "Daily EOD",
    "low_bankroll": "Low bankroll",
    "feed_stale": "Feed stale",
    "feed_restored": "Feed restored",
    "live_fill": "Live fill",
    "lane_change": "Lane change",
    "core_lane_tune": "Core-lane tune",
    "skip_storm": "Skip storm",
    "resolver_stuck": "Resolver stuck",
    "portfolio_rebalance": "Portfolio rebalance",
    "startup": "Startup",
    "error": "Error",
    "health": "Health",
    "soak_report": "Soak report",
    "test": "Test",
}

# Short operator-facing strategy blurbs for evolution spawn digests.
_STRATEGY_BLURBS = {
    "momentum": "BTC short-term trend (mom lane + trend thesis)",
    "phantom": "EMA crossover / breakout swing (analyze-dominant)",
    "mean_reversion": "Drift-gated z-score fade when market lags",
    "mean_reversion_sl": "Drift-gated mean reversion (legacy SL variant)",
    "mean_reversion_tp": "Drift-gated mean reversion with take-profit",
    "sentiment": "In-market flow reader (pm/cvd via analyze)",
    "hybrid": "Balanced ensemble of sub-strategies",
    "sniper": "Late/cheap/strong price zones + drift gate",
    "arbitrage": "Market-neutral two-legged book arb",
    "late_window_maker": "Late-window maker quotes (taker fills today)",
    "fee_zone_maker": "Fee-zone maker quotes (taker fills today)",
    "copy": "Whale copy-trade follower",
}

CHANNELS = ("telegram", "discord", "email")


def _et_now() -> datetime:
    """Current time as an aware America/New_York datetime (DST-correct)."""
    try:
        return db._et_now()
    except Exception:
        pass
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(timezone.utc).astimezone(ZoneInfo(_ET_ZONE))
    except Exception:
        return datetime.now(timezone.utc) - timedelta(hours=5)


def _et_day_bounds_utc(day: str) -> tuple[str, str]:
    """Return ``[start, end)`` UTC strings for an ET calendar day ``YYYY-MM-DD``."""
    try:
        from zoneinfo import ZoneInfo
        et = ZoneInfo(_ET_ZONE)
        start_et = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=et)
        end_et = start_et + timedelta(days=1)
        return (
            start_et.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            end_et.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception:
        # Fallback: treat the calendar day as if it were already UTC-shaped
        return f"{day} 00:00:00", (
            datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d %H:%M:%S")


def _format_et_now() -> str:
    """Human timestamp for test/ops messages, always labeled ET."""
    return _et_now().strftime("%Y-%m-%d %H:%M:%S ET")


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


def _channel_configured() -> dict[str, bool]:
    """Whether each channel has enough credentials to send."""
    return {
        "telegram": bool(_cred(CRED_TELEGRAM_TOKEN) and _cred(CRED_TELEGRAM_CHAT)),
        "discord": bool(_cred(CRED_DISCORD_WEBHOOK)),
        "email": bool(
            _cred(CRED_SMTP_HOST) and _cred(CRED_SMTP_TO)
            and (_cred(CRED_SMTP_FROM) or _cred(CRED_SMTP_USER))
        ),
    }


def _default_config() -> dict[str, Any]:
    """Fresh-install defaults: master + configured channels ON when creds exist."""
    configured = _channel_configured()
    any_configured = any(configured.values())
    # Explicit config.ALERTS_ENABLED still forces ON even without creds
    # (useful for tests / ops that wire channels later).
    enabled = bool(getattr(config, "ALERTS_ENABLED", False)) or any_configured
    return {
        "enabled": enabled,
        "channels": {
            "telegram": bool(configured["telegram"]),
            "discord": bool(configured["discord"]),
            "email": bool(configured["email"]),
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
    configured = _channel_configured()
    return {
        ch: {
            "enabled": bool(cfg["channels"].get(ch)),
            "configured": bool(configured.get(ch)),
        }
        for ch in CHANNELS
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


def _telegram_escape_md(text: str) -> str:
    """Escape Telegram legacy Markdown specials so bot/snake_case text is safe.

    Underscores in names like ``meanrev-v1`` / ``low_vol_range`` and unpaired
    ``*`` in free-form bodies cause HTTP 400 "can't parse entities" while
    simpler messages still deliver — intermittent failures that look fine
    from the operator's POV when most digests succeed.
    """
    if not text:
        return ""
    # Order matters: escape backslash first so we don't double-escape.
    out = str(text).replace("\\", "\\\\")
    for ch in ("_", "*", "`", "["):
        out = out.replace(ch, "\\" + ch)
    return out


def _send_telegram(title: str, body: str, level: str) -> tuple[bool, str]:
    token = _cred(CRED_TELEGRAM_TOKEN)
    chat = _cred(CRED_TELEGRAM_CHAT)
    if not token or not chat:
        return False, "telegram credentials missing"
    icon = {"critical": "🔴", "warn": "🟠", "warning": "🟠", "info": "🟢"}.get(
        level, "⚪")
    # Bold only the title (after escaping its contents); body is escaped plain.
    safe_title = _telegram_escape_md(title)
    safe_body = _telegram_escape_md(body) if body else ""
    text = f"{icon} *{safe_title}*" + (f"\n{safe_body}" if safe_body else "")
    # Telegram hard limit ~4096; truncate cleanly rather than 400 on length.
    if len(text) > 4000:
        text = text[:3990] + "…"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    ok, detail = _http_json(url, {
        "chat_id": chat,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    })
    # Fallback: if Markdown still fails (edge entities), resend plain text so
    # the operator never loses a critical alert to parse mode.
    if not ok and "parse" in (detail or "").lower():
        plain = f"{icon} {title}" + (f"\n{body}" if body else "")
        if len(plain) > 4000:
            plain = plain[:3990] + "…"
        ok2, detail2 = _http_json(url, {
            "chat_id": chat,
            "text": plain,
            "disable_web_page_preview": True,
        })
        if ok2:
            return True, "sent_plain_fallback"
        return False, f"md={detail}; plain={detail2}"
    return ok, detail


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
    body = f"Test message at {_format_et_now()}"
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


def _format_signal_weights(strategy_type: str) -> str:
    """Live non-zero lane weights for a strategy type (drift/mom/strat …)."""
    try:
        from bots.base_bot import BaseBot
        profile = BaseBot.STRATEGY_SIGNAL_PROFILE.get(
            strategy_type, BaseBot.DEFAULT_SIGNAL_PROFILE
        )
    except Exception:
        return "n/a"
    if not isinstance(profile, dict):
        return "n/a"
    parts = [
        f"{k}={float(v):.2f}"
        for k, v in profile.items()
        if v is not None and float(v) > 0
    ]
    return ", ".join(parts) if parts else "all lanes 0"


def _format_key_params(params: Optional[dict], limit: int = 6) -> str:
    """Compact numeric/string param summary for a newly spawned bot."""
    if not isinstance(params, dict) or not params:
        return ""
    # Prefer short, operator-useful keys; skip huge nested blobs.
    prefer = (
        "min_edge", "min_drift", "lookback", "z_entry", "z_exit",
        "ema_fast", "ema_slow", "breakout", "tp_pct", "window_frac",
        "cheap_max", "strong_min", "size_mult", "kelly_fraction",
    )
    items: list[tuple[str, Any]] = []
    for k in prefer:
        if k in params:
            items.append((k, params[k]))
    if len(items) < 3:
        for k, v in params.items():
            if k in {p[0] for p in items}:
                continue
            if isinstance(v, (int, float, str, bool)) and len(str(v)) < 40:
                items.append((k, v))
            if len(items) >= limit:
                break
    items = items[:limit]
    if not items:
        return ""
    bits = []
    for k, v in items:
        if isinstance(v, float):
            bits.append(f"{k}={v:.3g}")
        else:
            bits.append(f"{k}={v}")
    return ", ".join(bits)


def format_evolution_summary(report: Optional[dict], trigger: str = "") -> str:
    """Human-readable evolution digest: culled / survived / introduced."""
    lines: list[str] = []
    if trigger:
        lines.append(f"Trigger: {trigger}")
    if not isinstance(report, dict):
        return "\n".join(lines) if lines else "No report."

    if report.get("skipped"):
        reason = report.get("reason") or "skipped"
        lines.append(f"No roster change ({reason}).")

    individuals = report.get("individuals") or []
    elites = set(report.get("elites") or [])
    replaced = list(report.get("replaced") or [])
    spawned = list(report.get("spawned") or [])

    survivors: list[str] = []
    for ind in individuals:
        name = ind.get("name")
        if not name or name in replaced:
            continue
        tag = "elite" if (ind.get("elite") or name in elites) else (
            ind.get("status") or "survivor"
        )
        pnl = ind.get("pnl")
        wr = ind.get("win_rate")
        n = ind.get("trades")
        fit = ind.get("fitness")
        extra = []
        if fit is not None:
            extra.append(f"fit={float(fit):.2f}")
        if pnl is not None:
            extra.append(f"pnl=${float(pnl):+.2f}")
        if wr is not None and n is not None:
            extra.append(f"WR={float(wr) * 100:.0f}% n={int(n)}")
        survivors.append(
            f"  · {name} [{tag}]" + (f" ({', '.join(extra)})" if extra else "")
        )

    # Culled: prefer individuals with metrics; fall back to name list
    culled_lines: list[str] = []
    by_name = {ind.get("name"): ind for ind in individuals if ind.get("name")}
    for name in replaced:
        ind = by_name.get(name) or {}
        extra = []
        if ind.get("pnl") is not None:
            extra.append(f"pnl=${float(ind['pnl']):+.2f}")
        if ind.get("win_rate") is not None:
            extra.append(f"WR={float(ind['win_rate']) * 100:.0f}%")
        if ind.get("trades") is not None:
            extra.append(f"n={int(ind['trades'])}")
        culled_lines.append(
            f"  · {name}" + (f" ({', '.join(extra)})" if extra else "")
        )

    lines.append(f"Survived ({len(survivors)}):")
    lines.extend(survivors if survivors else ["  · (none)"])
    lines.append(f"Culled ({len(culled_lines)}):")
    lines.extend(culled_lines if culled_lines else ["  · (none)"])
    lines.append(f"Introduced ({len(spawned)}):")
    if not spawned:
        lines.append("  · (none)")
    else:
        for sp in spawned:
            name = sp.get("name") or "?"
            st = sp.get("strategy_type") or "?"
            blurb = _STRATEGY_BLURBS.get(st, st.replace("_", " "))
            parents = sp.get("parents") or []
            parent_s = " × ".join(parents) if parents else "?"
            replaced_name = sp.get("replaced") or "?"
            weights = _format_signal_weights(st)
            params_s = _format_key_params(sp.get("params"))
            lines.append(f"  · {name} ({st})")
            lines.append(f"      {blurb}")
            lines.append(f"      parents: {parent_s} · replaces: {replaced_name}")
            lines.append(f"      signal weights: {weights}")
            if params_s:
                lines.append(f"      params: {params_s}")

    body = "\n".join(lines)
    # Telegram/Discord-friendly cap; full detail still in arena_state log
    return body[:1800]


def alert_evolution(
    cycle: int,
    trigger: str,
    summary: str = "",
    report: Optional[dict] = None,
):
    """Emit an evolution-cycle digest.

    Prefer a structured ``report`` from ``run_ga_cycle`` (culled / survived /
    spawned with strategy + signal weights). ``summary`` is a legacy free-text
    fallback when no report is available.
    """
    if report is not None:
        body = format_evolution_summary(report, trigger=trigger)
    else:
        body = f"trigger={trigger}\n{summary}".strip()[:800]
    n_culled = len((report or {}).get("replaced") or [])
    n_spawned = len((report or {}).get("spawned") or [])
    title = f"Evolution cycle #{cycle}"
    if report is not None:
        if report.get("skipped") and not n_culled:
            title = f"Evolution cycle #{cycle} — no roster change"
        else:
            title = (
                f"Evolution cycle #{cycle} — "
                f"culled {n_culled}, introduced {n_spawned}"
            )
    notify(
        "evolution",
        title,
        body,
        level="info",
        key=f"evo:{cycle}",
        detail={
            "cycle": cycle,
            "trigger": trigger,
            "replaced": (report or {}).get("replaced"),
            "spawned": [
                s.get("name") for s in ((report or {}).get("spawned") or [])
            ],
            "elites": (report or {}).get("elites"),
        },
    )


def _window_trade_stats(hours: float = 1.0) -> dict[str, Any]:
    """Resolved + open trade stats over the last ``hours``."""
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=float(hours))
    ).strftime("%Y-%m-%d %H:%M:%S")
    with db.get_conn() as conn:
        resolved = conn.execute(
            """SELECT bot_name,
                      COUNT(*) AS n,
                      SUM(CASE WHEN outcome IN ('win','exit_tp') THEN 1 ELSE 0 END) AS wins,
                      SUM(CASE WHEN outcome IN ('loss','exit_sl') THEN 1 ELSE 0 END) AS losses,
                      COALESCE(SUM(pnl), 0) AS pnl,
                      AVG(entry_price) AS avg_entry
               FROM trades
               WHERE outcome IN ('win','loss','exit_tp','exit_sl')
                 AND created_at >= ?
               GROUP BY bot_name
               ORDER BY pnl DESC""",
            (cutoff,),
        ).fetchall()
        open_n = conn.execute(
            """SELECT COUNT(*) AS n FROM trades
               WHERE outcome IS NULL AND created_at >= ?""",
            (cutoff,),
        ).fetchone()
        # "Today" = current ET calendar day (DB stores UTC timestamps).
        try:
            day_start = db.et_day_start_utc(0)
        except Exception:
            day_start = datetime.now(timezone.utc).strftime("%Y-%m-%d 00:00:00")
        day = conn.execute(
            """SELECT COUNT(*) AS n,
                      COALESCE(SUM(pnl), 0) AS pnl,
                      SUM(CASE WHEN outcome IN ('win','exit_tp') THEN 1 ELSE 0 END) AS wins,
                      SUM(CASE WHEN outcome IN ('loss','exit_sl') THEN 1 ELSE 0 END) AS losses
               FROM trades
               WHERE outcome IN ('win','loss','exit_tp','exit_sl')
                 AND created_at >= ?""",
            (day_start,),
        ).fetchone()
    bots = []
    hour_pnl = 0.0
    hour_n = 0
    hour_wins = 0
    hour_losses = 0
    for r in resolved:
        pnl = float(r["pnl"] or 0)
        n = int(r["n"] or 0)
        w = int(r["wins"] or 0)
        l = int(r["losses"] or 0)
        hour_pnl += pnl
        hour_n += n
        hour_wins += w
        hour_losses += l
        bots.append({
            "bot": r["bot_name"],
            "n": n,
            "wins": w,
            "losses": l,
            "pnl": round(pnl, 2),
            "wr": (w / n) if n else 0.0,
        })
    day_n = int(day["n"] or 0) if day else 0
    day_pnl = float(day["pnl"] or 0) if day else 0.0
    day_wins = int(day["wins"] or 0) if day else 0
    day_losses = int(day["losses"] or 0) if day else 0
    return {
        "hours": hours,
        "hour_pnl": round(hour_pnl, 2),
        "hour_n": hour_n,
        "hour_wins": hour_wins,
        "hour_losses": hour_losses,
        "hour_wr": (hour_wins / hour_n) if hour_n else 0.0,
        "open": int(open_n["n"] or 0) if open_n else 0,
        "day_pnl": round(day_pnl, 2),
        "day_n": day_n,
        "day_wins": day_wins,
        "day_losses": day_losses,
        "day_wr": (day_wins / (day_wins + day_losses))
        if (day_wins + day_losses) else 0.0,
        "bots": bots,
    }


def arena_trading_mode() -> str:
    """Live if any active bot is live; else paper (config fallback)."""
    try:
        for row in db.get_active_bots() or []:
            name = row["bot_name"] if hasattr(row, "keys") else row.get("bot_name")
            if not name:
                continue
            mode = db.get_bot_mode(name)
            if (mode or "").lower() == "live":
                return "Live"
        return "Paper"
    except Exception:
        try:
            m = config.get_current_mode()
            return "Live" if m == "live" else "Paper"
        except Exception:
            return "Paper"


def _risk_note() -> str:
    try:
        from arena import risk_engine
        st = risk_engine.load_state()
        if st.get("kill_switch"):
            return f"KILL SWITCH ({st.get('kill_reason') or 'armed'})"
        port = st.get("portfolio") or {}
        paused = [
            n for n, b in (st.get("bots") or {}).items()
            if (b or {}).get("status") == "paused"
        ]
        note = f"portfolio {port.get('status') or '—'}"
        if paused:
            note += f" · paused: {', '.join(paused[:5])}"
        return note
    except Exception:
        return ""


def format_hourly_report(
    stats: dict[str, Any],
    pool: Optional[float] = None,
    risk_note: str = "",
    mode: str = "Paper",
) -> tuple[str, str]:
    """Return (title, body) for an hourly performance digest (times in ET)."""
    now = _et_now()
    end = now.strftime("%H:%M")
    start = (now - timedelta(hours=1)).strftime("%H:%M")
    mode_label = (mode or "Paper").strip().title()
    if mode_label.lower() not in ("live", "paper"):
        mode_label = "Paper"
    title = f"Hourly report · {mode_label} · {start}–{end} ET"

    hp = float(stats.get("hour_pnl") or 0)
    hn = int(stats.get("hour_n") or 0)
    hw = int(stats.get("hour_wins") or 0)
    hl = int(stats.get("hour_losses") or 0)
    hwr = float(stats.get("hour_wr") or 0) * 100
    open_n = int(stats.get("open") or 0)
    dp = float(stats.get("day_pnl") or 0)
    dn = int(stats.get("day_n") or 0)
    dwr = float(stats.get("day_wr") or 0) * 100

    lines = [f"Mode: {mode_label}"]
    if pool is not None:
        lines.append(f"Pool: ${pool:,.2f}")
    lines.append(
        f"Last hour: {hp:+.2f} · {hn} resolved ({hw}W/{hl}L"
        + (f", {hwr:.0f}% WR" if hn else "")
        + f") · {open_n} open"
    )
    lines.append(
        f"Today: {dp:+.2f} · {dn} resolved"
        + (f" · {dwr:.0f}% WR" if dn else "")
    )
    bots = list(stats.get("bots") or [])
    if bots:
        top = bots[:3]
        bot_bits = [
            f"{b['bot']} {float(b['pnl']):+.2f} ({int(b['n'])}t)"
            for b in top
        ]
        lines.append("Top: " + " · ".join(bot_bits))
        losers = [b for b in bots if float(b.get("pnl") or 0) < 0]
        if losers:
            worst = sorted(losers, key=lambda b: float(b["pnl"]))[:2]
            lines.append(
                "Weak: "
                + " · ".join(
                    f"{b['bot']} {float(b['pnl']):+.2f}" for b in worst
                )
            )
    else:
        lines.append("No resolved trades in the last hour.")
    if risk_note:
        lines.append(f"Risk: {risk_note}")
    return title, "\n".join(lines)


def alert_hourly_report() -> dict[str, Any]:
    """Build and send one hourly performance digest (always attempts notify)."""
    stats = _window_trade_stats(hours=1.0)
    pool = None
    try:
        pool = float(db.get_paper_available())
    except Exception:
        pass
    mode = arena_trading_mode()
    title, body = format_hourly_report(
        stats, pool=pool, risk_note=_risk_note(), mode=mode,
    )
    hour_key = _et_now().strftime("%Y%m%d%H")
    return notify(
        "hourly_report",
        title,
        body,
        level="info",
        key=f"hourly:{hour_key}",
        detail={**stats, "mode": mode},
    )


def maybe_send_hourly_report() -> Optional[dict[str, Any]]:
    """Send hourly digest when interval elapsed. Safe to call every loop tick."""
    interval = float(getattr(config, "ALERT_HOURLY_REPORT_SEC", 3600))
    try:
        last = float(db.get_arena_state(HOURLY_STATE_KEY) or 0)
    except (TypeError, ValueError):
        last = 0.0
    now = time.time()
    if last > 0 and (now - last) < interval:
        return None
    # Avoid a stampede of "empty first hour" on brand-new installs: wait until
    # at least one full interval after arena start (last==0 → set baseline).
    if last <= 0:
        try:
            db.set_arena_state(HOURLY_STATE_KEY, str(now))
        except Exception:
            pass
        return None
    result = alert_hourly_report()
    try:
        db.set_arena_state(HOURLY_STATE_KEY, str(now))
    except Exception:
        pass
    return result


def _day_trade_stats(day: str) -> dict[str, Any]:
    """Resolved trade stats for an ET calendar day ``YYYY-MM-DD``.

    ``created_at`` is stored UTC; we convert the ET day to a half-open UTC
    range so overnight (ET) trades land on the correct EOD report.
    """
    start_utc, end_utc = _et_day_bounds_utc(day)
    with db.get_conn() as conn:
        resolved = conn.execute(
            """SELECT bot_name,
                      COUNT(*) AS n,
                      SUM(CASE WHEN outcome IN ('win','exit_tp') THEN 1 ELSE 0 END) AS wins,
                      SUM(CASE WHEN outcome IN ('loss','exit_sl') THEN 1 ELSE 0 END) AS losses,
                      COALESCE(SUM(pnl), 0) AS pnl,
                      AVG(entry_price) AS avg_entry
               FROM trades
               WHERE outcome IN ('win','loss','exit_tp','exit_sl')
                 AND created_at >= ? AND created_at < ?
               GROUP BY bot_name
               ORDER BY pnl DESC""",
            (start_utc, end_utc),
        ).fetchall()
        open_n = conn.execute(
            """SELECT COUNT(*) AS n FROM trades WHERE outcome IS NULL""",
        ).fetchone()
    bots = []
    total_pnl = 0.0
    total_n = 0
    total_w = 0
    total_l = 0
    entry_sum = 0.0
    entry_n = 0
    for r in resolved:
        n = int(r["n"] or 0)
        w = int(r["wins"] or 0)
        l = int(r["losses"] or 0)
        pnl = float(r["pnl"] or 0)
        total_pnl += pnl
        total_n += n
        total_w += w
        total_l += l
        if r["avg_entry"] is not None and n:
            entry_sum += float(r["avg_entry"]) * n
            entry_n += n
        bots.append({
            "bot": r["bot_name"], "n": n, "wins": w, "losses": l,
            "pnl": round(pnl, 2), "wr": (w / n) if n else 0.0,
        })
    wr = (total_w / total_n) if total_n else 0.0
    avg_entry = (entry_sum / entry_n) if entry_n else None
    be_gap = (wr - avg_entry) if avg_entry is not None else None
    return {
        "day": day,
        "pnl": round(total_pnl, 2),
        "n": total_n,
        "wins": total_w,
        "losses": total_l,
        "wr": wr,
        "avg_entry": avg_entry,
        "be_gap": be_gap,
        "open": int(open_n["n"] or 0) if open_n else 0,
        "bots": bots,
    }


def format_daily_report(
    stats: dict[str, Any],
    pool: Optional[float] = None,
    mode: str = "Paper",
    risk_note: str = "",
) -> tuple[str, str]:
    day = stats.get("day") or "?"
    mode_label = (mode or "Paper").strip().title()
    title = f"Daily EOD · {mode_label} · {day} ET"
    pnl = float(stats.get("pnl") or 0)
    n = int(stats.get("n") or 0)
    w = int(stats.get("wins") or 0)
    l = int(stats.get("losses") or 0)
    wr = float(stats.get("wr") or 0) * 100
    be = stats.get("be_gap")
    lines = [f"Mode: {mode_label}"]
    if pool is not None:
        lines.append(f"Pool: ${pool:,.2f}")
    lines.append(
        f"Day P&L: {pnl:+.2f} · {n} resolved ({w}W/{l}L"
        + (f", {wr:.0f}% WR" if n else "")
        + ")"
    )
    if be is not None:
        lines.append(
            f"Break-even gap: {float(be) * 100:+.1f}¢ "
            f"(avg entry {float(stats.get('avg_entry') or 0) * 100:.0f}¢)"
        )
    lines.append(f"Open positions: {int(stats.get('open') or 0)}")
    bots = list(stats.get("bots") or [])
    if bots:
        lines.append(
            "Bots: "
            + " · ".join(
                f"{b['bot']} {float(b['pnl']):+.2f} ({int(b['n'])}t)"
                for b in bots[:6]
            )
        )
    else:
        lines.append("No resolved trades.")
    if risk_note:
        lines.append(f"Risk: {risk_note}")
    return title, "\n".join(lines)


def alert_daily_report(day: Optional[str] = None) -> dict[str, Any]:
    """Send end-of-day summary for an ET calendar day (default: yesterday ET)."""
    if not day:
        day = (_et_now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
    stats = _day_trade_stats(day)
    pool = None
    try:
        pool = float(db.get_paper_available())
    except Exception:
        pass
    mode = arena_trading_mode()
    title, body = format_daily_report(
        stats, pool=pool, mode=mode, risk_note=_risk_note(),
    )
    return notify(
        "daily_report", title, body, level="info",
        key=f"daily:{day}", detail={**stats, "mode": mode},
    )


def maybe_send_daily_report() -> Optional[dict[str, Any]]:
    """Once per ET day after ALERT_DAILY_REPORT_HOUR_ET, send yesterday's EOD."""
    now = _et_now()
    hour = int(getattr(
        config, "ALERT_DAILY_REPORT_HOUR_ET",
        getattr(config, "ALERT_DAILY_REPORT_HOUR_UTC", 0),
    ))
    grace = int(getattr(config, "ALERT_DAILY_REPORT_GRACE_MIN", 5))
    if now.hour < hour:
        return None
    if now.hour == hour and now.minute < grace:
        return None
    report_day = (now.date() - timedelta(days=1)).isoformat()
    last = db.get_arena_state(DAILY_STATE_KEY)
    if last == report_day:
        return None
    result = alert_daily_report(report_day)
    try:
        db.set_arena_state(DAILY_STATE_KEY, report_day)
    except Exception:
        pass
    return result


def alert_low_bankroll(available: float, threshold: float, bankroll: float):
    level = "critical" if available < threshold * 0.5 else "warn"
    notify(
        "low_bankroll",
        f"Low bankroll · ${available:.2f} available",
        f"Available ${available:.2f} is below threshold ${threshold:.2f} "
        f"(bankroll base ${bankroll:.2f}). Top up paper balance or reduce size.",
        level=level,
        key=f"low_bankroll:{level}",
        detail={"available": available, "threshold": threshold,
                "bankroll": bankroll},
    )


def maybe_alert_low_bankroll() -> Optional[dict]:
    try:
        avail = float(db.get_paper_available())
        bankroll = float(db.get_paper_bankroll())
    except Exception:
        return None
    floor_usd = float(getattr(config, "ALERT_LOW_BANKROLL_USD", 25.0))
    frac = float(getattr(config, "ALERT_LOW_BANKROLL_FRAC", 0.50))
    base = max(
        bankroll,
        float(getattr(config, "PAPER_BANKROLL_DEFAULT", 200.0) or 200.0),
    )
    threshold = max(floor_usd, frac * base)
    if avail >= threshold:
        try:
            db.set_arena_state(BANKROLL_STATE_KEY, "ok")
        except Exception:
            pass
        return None
    # Hysteresis levels so we don't spam: warn once, critical once per level
    level = "critical" if avail < threshold * 0.5 else "warn"
    last = db.get_arena_state(BANKROLL_STATE_KEY)
    if last == level:
        return None
    alert_low_bankroll(avail, threshold, bankroll)
    try:
        db.set_arena_state(BANKROLL_STATE_KEY, level)
    except Exception:
        pass
    return {"available": avail, "threshold": threshold, "level": level}


def publish_price_feed_status() -> dict[str, Any]:
    """Write Binance feed heartbeat into arena_state for health + alerts."""
    status: dict[str, Any] = {"ts": time.time(), "stale": False, "symbols": {}}
    try:
        from signals.price_feed import get_feed
        feed = get_feed()
        stale_sec = float(getattr(config, "ALERT_FEED_STALE_SEC", 90.0))
        any_stale = False
        for sym in ("btc", "eth"):
            sig = feed.get_signals(sym) if feed else {}
            latest = float(sig.get("latest") or 0)
            # price_feed marks stale at 60s; we use config threshold via last update
            is_stale = bool(sig.get("stale"))
            last_up = 0.0
            try:
                last_up = float(feed._last_update.get(sym, 0) or 0)  # noqa: SLF001
            except Exception:
                last_up = 0.0
            age = (time.time() - last_up) if last_up else None
            if age is not None and age > stale_sec:
                is_stale = True
            if latest <= 0 and last_up <= 0:
                is_stale = True
            status["symbols"][sym] = {
                "latest": latest, "stale": is_stale,
                "age_sec": round(age, 1) if age is not None else None,
            }
            any_stale = any_stale or is_stale
        status["stale"] = any_stale
    except Exception as e:
        status["error"] = str(e)
        status["stale"] = True
    try:
        db.set_arena_state("price_feed_status", json.dumps(status))
    except Exception:
        pass
    return status


_FEED_STALE_FLAG_KEY = "price_feed_was_stale"


def maybe_alert_feed_stale() -> Optional[dict]:
    """Alert when the Binance price feed goes stale — and again when it recovers.

    Recovery fires only on a stale→healthy edge (tracked in arena_state) so
    operators get a clear "restored" notification after an outage, not a
    continuous healthy spam.
    """
    status = publish_price_feed_status()
    was_stale = False
    try:
        was_stale = db.get_arena_state(_FEED_STALE_FLAG_KEY) in ("1", "true", "on")
    except Exception:
        was_stale = False

    if status.get("stale"):
        try:
            db.set_arena_state(_FEED_STALE_FLAG_KEY, "1")
        except Exception:
            pass
        syms = status.get("symbols") or {}
        parts = []
        for s, info in syms.items():
            if info.get("stale"):
                age = info.get("age_sec")
                parts.append(
                    f"{s.upper()} age={age:.0f}s" if age is not None
                    else f"{s.upper()} unavailable"
                )
        body = ", ".join(parts) if parts else status.get("error") or "feed stale"
        notify(
            "feed_stale",
            "Price feed stale / unavailable",
            body + "\nRestart arena to reconnect Binance WebSocket if this persists.",
            level="warn",
            key="feed_stale",
            detail=status,
        )
        return status

    # Healthy — clear sticky flag and notify once on recovery
    if was_stale:
        try:
            db.set_arena_state(_FEED_STALE_FLAG_KEY, "0")
        except Exception:
            pass
        syms = status.get("symbols") or {}
        parts = []
        for s, info in syms.items():
            latest = info.get("latest")
            age = info.get("age_sec")
            if latest:
                parts.append(
                    f"{s.upper()} ${float(latest):,.0f}"
                    + (f" age={age:.0f}s" if age is not None else "")
                )
            else:
                parts.append(s.upper())
        body = ", ".join(parts) if parts else "Binance WebSocket healthy again"
        notify(
            "feed_restored",
            "Price feed restored",
            body,
            level="info",
            key="feed_restored",
            detail=status,
        )
        return {"restored": True, **status}

    try:
        db.set_arena_state(_FEED_STALE_FLAG_KEY, "0")
    except Exception:
        pass
    return None


def alert_live_fill(
    bot: str,
    reason: str,
    *,
    side: str = "",
    market_id: str = "",
    detail: Optional[dict] = None,
):
    """Live-mode fill failure / anomaly (slippage, order error, naked arb leg)."""
    reason = (reason or "unknown")[:200]
    title = f"Live fill anomaly · {bot}"
    body = f"reason={reason}"
    if side:
        body += f" · side={side}"
    if market_id:
        body += f" · market={str(market_id)[:16]}"
    level = "critical" if any(
        x in reason.lower()
        for x in ("naked", "order failed", "error", "missing_token", "reject")
    ) else "warn"
    notify(
        "live_fill", title, body, level=level,
        key=f"live_fill:{bot}:{reason[:40]}",
        detail={"bot": bot, "reason": reason, "side": side,
                "market_id": market_id, **(detail or {})},
    )


def alert_lane_change(
    action: str,
    lane: str,
    *,
    accuracy: Optional[float] = None,
    n: Optional[int] = None,
    detail: Optional[dict] = None,
):
    """Lane auto-approve or auto-demote."""
    action = (action or "change").lower()
    if action in ("demote", "disable", "disabled", "auto-disabled"):
        title = f"Lane auto-demoted · {lane}"
        level = "warn"
        event_action = "demote"
    else:
        title = f"Lane auto-approved · {lane}"
        level = "info"
        event_action = "approve"
    bits = []
    if accuracy is not None:
        bits.append(f"accuracy={float(accuracy) * 100:.1f}%")
    if n is not None:
        bits.append(f"n={int(n)}")
    body = " · ".join(bits) if bits else event_action
    notify(
        "lane_change", title, body, level=level,
        key=f"lane:{event_action}:{lane}",
        detail={"action": event_action, "lane": lane,
                "accuracy": accuracy, "n": n, **(detail or {})},
    )


def alert_core_lane_tune(changes: list[dict]):
    """Notify when core-lane tuner applies material weight shifts."""
    if not changes:
        return
    min_shift = float(getattr(config, "ALERT_CORE_LANE_MIN_SHIFT", 0.05))
    material = [
        c for c in changes
        if abs(float(c.get("to", 0) or 0) - float(c.get("from", 0) or 0))
        >= min_shift - 1e-9
    ]
    if not material:
        return
    lines = [
        f"  · {c.get('strategy')}.{c.get('lane')}: "
        f"{float(c.get('from', 0)):.2f}→{float(c.get('to', 0)):.2f}"
        + (f" (acc {float(c['accuracy']) * 100:.0f}%)"
           if c.get("accuracy") is not None else "")
        for c in material[:12]
    ]
    notify(
        "core_lane_tune",
        f"Core-lane tuner · {len(material)} weight shift(s)",
        "\n".join(lines),
        level="info",
        key=f"core_tune:{int(time.time() // 300)}",
        detail={"changes": material},
    )


def maybe_alert_skip_storm() -> Optional[dict]:
    """High skip volume with almost no fills over a rolling window."""
    try:
        raw = db.get_arena_state("skip_counts")
        counts = json.loads(raw) if raw else {}
        if not isinstance(counts, dict):
            counts = {}
    except Exception:
        counts = {}
    total_skips = int(sum(int(v or 0) for v in counts.values()))
    now = time.time()
    window = float(getattr(config, "ALERT_SKIP_STORM_WINDOW_SEC", 600))
    min_skips = int(getattr(config, "ALERT_SKIP_STORM_MIN_SKIPS", 200))
    max_trades = int(getattr(config, "ALERT_SKIP_STORM_MAX_TRADES", 2))

    prev_raw = db.get_arena_state(SKIP_STORM_STATE_KEY)
    try:
        prev = json.loads(prev_raw) if prev_raw else {}
    except Exception:
        prev = {}
    prev_total = int(prev.get("total") or 0)
    prev_ts = float(prev.get("ts") or 0)
    # Always refresh snapshot
    snap = {"total": total_skips, "ts": now, "counts": counts}
    try:
        db.set_arena_state(SKIP_STORM_STATE_KEY, json.dumps(snap))
    except Exception:
        pass
    if prev_ts <= 0 or (now - prev_ts) < window * 0.5:
        return None  # need a prior baseline
    if (now - prev_ts) > window * 2:
        return None  # gap too large (restart)
    delta = total_skips - prev_total
    if delta < min_skips:
        return None
    # Trades placed in the same window
    cutoff = (
        datetime.now(timezone.utc) - timedelta(seconds=now - prev_ts)
    ).strftime("%Y-%m-%d %H:%M:%S")
    try:
        with db.get_conn() as conn:
            n_trades = conn.execute(
                "SELECT COUNT(*) AS n FROM trades WHERE created_at >= ?",
                (cutoff,),
            ).fetchone()["n"]
    except Exception:
        n_trades = 0
    if int(n_trades or 0) > max_trades:
        return None
    top = sorted(counts.items(), key=lambda kv: -int(kv[1] or 0))[:5]
    top_s = ", ".join(f"{k}={v}" for k, v in top)
    notify(
        "skip_storm",
        f"Skip storm · +{delta} skips, {int(n_trades or 0)} trades",
        f"Window ~{(now - prev_ts) / 60:.0f}m. Top reasons: {top_s or '—'}\n"
        "Often session skip / dead-zone / no_edge stacking — check Overview → Skips.",
        level="warn",
        key=f"skip_storm:{int(now // window)}",
        detail={"delta_skips": delta, "trades": n_trades, "counts": counts},
    )
    return {"delta_skips": delta, "trades": n_trades}


def maybe_alert_resolver_stuck() -> Optional[dict]:
    """Pending trades that never resolved past ALERT_RESOLVER_STUCK_AGE_MIN."""
    age_min = float(getattr(config, "ALERT_RESOLVER_STUCK_AGE_MIN", 15.0))
    min_count = int(getattr(config, "ALERT_RESOLVER_STUCK_MIN_COUNT", 2))
    cutoff = (
        datetime.now(timezone.utc) - timedelta(minutes=age_min)
    ).strftime("%Y-%m-%d %H:%M:%S")
    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                """SELECT id, bot_name, market_id, created_at, side
                   FROM trades
                   WHERE outcome IS NULL AND created_at <= ?
                   ORDER BY created_at ASC LIMIT 20""",
                (cutoff,),
            ).fetchall()
    except Exception:
        return None
    if len(rows) < min_count:
        return None
    lines = [
        f"  · {r['bot_name']} {r['side']} m={str(r['market_id'] or '')[:12]} "
        f"since {r['created_at']}"
        for r in rows[:8]
    ]
    notify(
        "resolver_stuck",
        f"Resolver stuck · {len(rows)} open trade(s) >{age_min:.0f}m",
        "Pending trades past expected resolution window:\n" + "\n".join(lines),
        level="warn",
        key="resolver_stuck",
        detail={"count": len(rows), "age_min": age_min,
                "ids": [r["id"] for r in rows]},
    )
    return {"count": len(rows)}


def alert_portfolio_rebalance(
    reason: str,
    weights: dict[str, float],
    prev_weights: Optional[dict[str, float]] = None,
    *,
    method: str = "",
):
    """Digest after a portfolio rebalance when weights moved materially."""
    prev = prev_weights or {}
    min_shift = float(getattr(config, "ALERT_PORTFOLIO_REBALANCE_MIN_SHIFT", 0.08))
    shifts = []
    names = set(weights) | set(prev)
    for n in names:
        a = float(prev.get(n) or 0)
        b = float(weights.get(n) or 0)
        if abs(b - a) >= min_shift - 1e-9:
            shifts.append((n, a, b, b - a))
    # Always notify on force/manual/regime; timer only if material shift
    force_reasons = ("manual", "window_change", "regime")
    is_notable = bool(shifts) or any(
        str(reason).startswith(p) for p in force_reasons
    ) or str(reason) in force_reasons
    if not is_notable and not shifts:
        return
    if not shifts and str(reason) == "timer":
        return
    shifts.sort(key=lambda t: -abs(t[3]))
    lines = [f"Reason: {reason}" + (f" · method={method}" if method else "")]
    if shifts:
        lines.append("Weight moves:")
        for n, a, b, d in shifts[:8]:
            lines.append(f"  · {n}: {a * 100:.0f}%→{b * 100:.0f}% ({d * 100:+.0f}pp)")
    else:
        top = sorted(weights.items(), key=lambda kv: -kv[1])[:6]
        lines.append(
            "Weights: "
            + " · ".join(f"{k} {v * 100:.0f}%" for k, v in top)
        )
    notify(
        "portfolio_rebalance",
        f"Portfolio rebalance · {reason}",
        "\n".join(lines),
        level="info",
        key=f"portfolio:{reason}:{int(time.time() // 60)}",
        detail={"reason": reason, "weights": weights, "shifts": shifts},
    )


def alert_startup(
    bots: Optional[list] = None,
    *,
    extra: str = "",
):
    """Arena process start / restart notice."""
    mode = arena_trading_mode()
    names = []
    if bots:
        try:
            names = [getattr(b, "name", str(b)) for b in bots]
        except Exception:
            names = []
    if not names:
        try:
            names = [r["bot_name"] for r in (db.get_active_bots() or [])]
        except Exception:
            names = []
    pool = None
    try:
        pool = float(db.get_paper_available())
    except Exception:
        pass
    lines = [
        f"Mode: {mode}",
        f"Bots ({len(names)}): {', '.join(names[:12]) or '—'}",
    ]
    if pool is not None:
        lines.append(f"Paper pool: ${pool:,.2f}")
    if extra:
        lines.append(extra)
    notify(
        "startup",
        f"Arena started · {mode}",
        "\n".join(lines),
        level="info",
        key=f"startup:{int(time.time() // 30)}",
        detail={"mode": mode, "bots": names, "pool": pool},
    )


def run_periodic_alerts() -> dict[str, Any]:
    """Evolution-loop host: hourly/daily digests + ops threshold alerts."""
    out: dict[str, Any] = {}
    for name, fn in (
        ("hourly", maybe_send_hourly_report),
        ("daily", maybe_send_daily_report),
        ("low_bankroll", maybe_alert_low_bankroll),
        ("feed_stale", maybe_alert_feed_stale),
        ("skip_storm", maybe_alert_skip_storm),
        ("resolver_stuck", maybe_alert_resolver_stuck),
    ):
        try:
            r = fn()
            if r is not None:
                out[name] = r if not isinstance(r, dict) else {
                    k: r[k] for k in list(r)[:8]
                    if k not in ("channels",)
                }
        except Exception as e:
            logger.debug("periodic alert %s failed: %s", name, e)
            out[name] = {"error": str(e)}
    return out


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

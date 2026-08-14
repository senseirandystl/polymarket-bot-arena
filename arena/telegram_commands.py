"""Inbound Telegram command bot — query the arena and steer it from a chat.

``arena/alerts.py`` is outbound-only (the arena pushes digests at you). This
module is the other direction: a long-poll loop on Telegram ``getUpdates``
that accepts ``/hour``, ``/status``, ``/kill`` … from the operator's phone.

**Why long-poll and not a webhook.** A webhook needs a public HTTPS endpoint;
the arena runs on a laptop / VPS behind launchd with no inbound exposure.
``getUpdates`` reaches out, so nothing new is listening on the network.

**Where it runs.** Inside the DASHBOARD process, not the arena. The dashboard
is the one that stays up when the arena dies — which is exactly when you want
to be able to ask it what happened. It reads the same SQLite DB (WAL) and
writes control intents through the same paths the dashboard HTTP endpoints
use (``risk_engine.set_kill_switch`` / ``pause_bot``, ``db.retire_bot``, the
``pending_bot_deploys`` queue the arena coordinator drains every ~30s).

**Security.** The bot token is a bearer credential — anyone who discovers the
bot's username can message it. The ONLY thing gating control commands is the
chat-id allowlist: an update whose ``chat.id`` is not the configured
``alert_telegram_chat_id`` is dropped silently (no reply — a reply would
confirm to a stranger that the bot is live). Control commands are separately
gated by ``config.TELEGRAM_COMMANDS_CONTROL_ENABLED``, and ``/retire``
requires an explicit ``confirm`` word because retiring a running bot is not
undoable from chat.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

import config
import db
from arena.alerts import (
    CRED_TELEGRAM_CHAT,
    CRED_TELEGRAM_TOKEN,
    _cred,
    _telegram_escape_md,
    _window_trade_stats,
    arena_trading_mode,
)

logger = logging.getLogger(__name__)

OFFSET_KEY = "telegram_cmd_offset"
MAX_REPLY_CHARS = 4000  # Telegram's own limit is 4096; leave room for the icon

# Per-command timestamps for the rate limiter (command name -> last handled).
_last_command_at: dict[str, float] = {}
# Destructive commands awaiting their `confirm` word (key -> expiry ts).
_pending_confirm: dict[str, float] = {}
_CONFIRM_TTL_SEC = 120.0
# Highest acked update id seen in THIS process — see _load_offset.
_offset_memo: list[int] = [0]
# Unauthorized-chat log throttle (chat id -> last logged ts).
_unauth_logged: dict[str, float] = {}
_UNAUTH_LOG_EVERY_SEC = 300.0

_poller_thread: Optional[threading.Thread] = None
_poller_lock = threading.Lock()
_stop = threading.Event()


# ---------------------------------------------------------------------------
# Credentials / transport
# ---------------------------------------------------------------------------

def _token() -> Optional[str]:
    return _cred(CRED_TELEGRAM_TOKEN)


def _chat_id() -> Optional[str]:
    return _cred(CRED_TELEGRAM_CHAT)


def _clip(text: str) -> str:
    text = str(text or "")
    return text if len(text) <= MAX_REPLY_CHARS else text[:MAX_REPLY_CHARS - 1] + "…"


def _send(chat_id: str, text: str) -> bool:
    """Reply into ``chat_id``. Plain text — command output is full of
    ``snake_case`` bot names and ``*`` that trip Telegram's Markdown parser
    (the same class of intermittent 400 that ``_telegram_escape_md`` exists
    to fix on the outbound side)."""
    token = _token()
    if not token:
        return False
    payload = json.dumps({
        "chat_id": chat_id,
        "text": _clip(text),
        "disable_web_page_preview": True,
    }).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json",
                 "User-Agent": "pba-commands/1"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception as e:
        # Type only: urllib/HTTP exceptions embed the request URL, and
        # sendMessage's URL carries the bot token.
        logger.warning("telegram reply failed: %s", type(e).__name__)
        return False


def _get_updates(offset: int, timeout: int,
                 limit: int = 100) -> Optional[list[dict]]:
    """One long-poll.

    Returns the (possibly empty) update list on success and **None on
    failure** — the distinction is load-bearing. Collapsing an error into
    ``[]`` would make the startup backlog drain think there was nothing to
    ack (so yesterday's ``/kill`` executes on the next poll) and would turn a
    401/409/429 into a hot loop hammering both Telegram and SQLite.
    """
    token = _token()
    if not token:
        return None
    q = urllib.parse.urlencode({
        "offset": offset,
        "timeout": int(timeout),
        "limit": int(limit),
        # Commands only — never pull down media/edits we would ignore anyway.
        "allowed_updates": json.dumps(["message"]),
    })
    url = f"https://api.telegram.org/bot{token}/getUpdates?{q}"
    try:
        with urllib.request.urlopen(url, timeout=timeout + 10) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        # Log the TYPE only: several urllib/http exceptions embed the failing
        # URL in their message, and the URL carries the bot token.
        logger.debug("telegram getUpdates failed: %s", type(e).__name__)
        return None
    if not isinstance(data, dict) or not data.get("ok"):
        # ok:false is 401 (revoked token) / 409 (a webhook is set) / 429.
        logger.warning("telegram getUpdates rejected: error_code=%s",
                       (data or {}).get("error_code")
                       if isinstance(data, dict) else "?")
        return None
    return [u for u in (data.get("result") or []) if isinstance(u, dict)]


def _load_offset() -> int:
    """Next un-acked update id.

    In-memory value wins when it is ahead of the DB: a `database is locked`
    on the persist path (routine — the arena writes this same WAL DB every
    second) must not make an already-executed batch look un-acked and replay.
    """
    stored = 0
    try:
        stored = int(db.get_arena_state(OFFSET_KEY) or 0)
    except (TypeError, ValueError):
        stored = 0
    return max(stored, _offset_memo[0])


def _save_offset(value: int) -> None:
    value = int(value)
    _offset_memo[0] = max(_offset_memo[0], value)
    try:
        db.set_arena_state(OFFSET_KEY, str(value))
    except Exception as e:
        logger.warning("telegram offset persist failed (%s) — holding offset "
                       "in memory for this process", type(e).__name__)


# ---------------------------------------------------------------------------
# Lazily-imported collaborators (kept as functions so tests can patch them)
# ---------------------------------------------------------------------------

def _risk_engine():
    from arena import risk_engine
    return risk_engine


def _strategy_catalog() -> list[dict]:
    from arena.startup import strategy_catalog
    return strategy_catalog()


def _confirm_gate(key: str, args: list[str], prompt: str) -> Optional[str]:
    """Two-message handshake for destructive commands.

    Returns a reply string while the command is NOT yet confirmed (prompt or
    expiry notice), or None when the caller may proceed. A lone
    ``/retire x confirm`` with no prior prompt is refused: otherwise a single
    replayed or spoofed message is enough to retire a running bot.
    """
    now = time.time()
    for k, exp in list(_pending_confirm.items()):   # sweep, keeps dict bounded
        if exp < now:
            _pending_confirm.pop(k, None)
    said_confirm = any(a.lower() == "confirm" for a in args)
    pending = _pending_confirm.get(key)
    if said_confirm and pending and pending >= now:
        _pending_confirm.pop(key, None)
        return None
    if said_confirm:
        # Deliberately does NOT arm the gate: if a bare `confirm` both refused
        # AND opened the window, the same message delivered twice would
        # complete the handshake without the operator ever seeing the prompt.
        return ("Confirmation expired or never requested — send the command "
                f"WITHOUT 'confirm' first.\n{prompt}")
    _pending_confirm[key] = now + _CONFIRM_TTL_SEC
    return prompt


def _active_bot_names() -> list[str]:
    return [
        str(b.get("bot_name")) for b in (db.get_active_bots() or [])
        if b.get("bot_name")
    ]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_command(text: str) -> Optional[tuple[str, list[str]]]:
    """``"/deploy phantom hybrid"`` → ``("deploy", ["phantom", "hybrid"])``.

    Accepts a bare word without the slash (phones autocorrect it away), and
    strips the ``@botname`` suffix Telegram appends in group chats. Returns
    None when the message is not a command we know how to shape.
    """
    if not text:
        return None
    parts = str(text).strip().split()
    if not parts:
        return None
    head = parts[0]
    name = head.lstrip("/").split("@", 1)[0].lower()
    if not name or not name.replace("_", "").isalnum():
        return None
    cmd = _REGISTRY.get(name)
    if not head.startswith("/"):
        # Bare words: reports only. The alert chat is also a conversation
        # ("Kill the losses" / "pause all") and must never mutate trading state.
        if cmd is None or cmd.control:
            return None
    return name, parts[1:]


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Command:
    name: str
    handler: Callable[[list[str]], str]
    help: str
    control: bool


_REGISTRY: dict[str, _Command] = {}


def _register(name: str, help_text: str, *, control: bool = False):
    def deco(fn):
        _REGISTRY[name] = _Command(name, fn, help_text, control)
        return fn
    return deco


# ---------------------------------------------------------------------------
# Reporting commands
# ---------------------------------------------------------------------------

def _format_window(hours: float, stats: dict) -> str:
    label = (
        "last hour" if abs(hours - 1.0) < 1e-9 else
        "last 24h" if abs(hours - 24.0) < 1e-9 else
        "last 7d" if abs(hours - 168.0) < 1e-9 else
        f"last {hours:g}h"
    )
    n = int(stats.get("hour_n") or 0)
    pnl = float(stats.get("hour_pnl") or 0)
    wins = int(stats.get("hour_wins") or 0)
    losses = int(stats.get("hour_losses") or 0)
    wr = float(stats.get("hour_wr") or 0) * 100
    lines = [f"Performance · {label}"]
    if not n:
        lines.append(f"No resolved trades. {int(stats.get('open') or 0)} open.")
        return "\n".join(lines)
    lines.append(
        f"PnL {pnl:+.2f} · {n} resolved ({wins}W/{losses}L, {wr:.0f}% WR) "
        f"· {int(stats.get('open') or 0)} open"
    )
    lines.append("Bots:")
    for b in (stats.get("bots") or [])[:12]:
        bn = int(b.get("n") or 0)
        lines.append(
            f"  {b.get('bot')}: {float(b.get('pnl') or 0):+.2f} "
            f"({bn}t, {float(b.get('wr') or 0) * 100:.0f}% WR)"
        )
    return "\n".join(lines)


def _window_hours(args: list[str], default: float) -> float:
    """Optional numeric override: ``/hour 6`` → a 6-hour window."""
    if args:
        try:
            h = float(args[0])
            if 0 < h <= 24 * 90:
                return h
        except ValueError:
            pass
    return default


@_register("hour", "/hour [h] — performance over the last hour (or h hours)")
def cmd_hour(args: list[str]) -> str:
    hours = _window_hours(args, 1.0)
    return _format_window(hours, _window_trade_stats(hours=hours))


@_register("day", "/day — performance over the last 24h")
def cmd_day(args: list[str]) -> str:
    return _format_window(24.0, _window_trade_stats(hours=24.0))


@_register("week", "/week — performance over the last 7 days")
def cmd_week(args: list[str]) -> str:
    return _format_window(168.0, _window_trade_stats(hours=168.0))


@_register("status", "/status — pool, mode, kill switch, paused bots, health")
def cmd_status(args: list[str]) -> str:
    lines = [f"Mode: {arena_trading_mode()}"]
    try:
        lines.append(f"Pool: ${float(db.get_paper_available()):,.2f}")
    except Exception:
        pass
    try:
        st = _risk_engine().load_state()
        killed = bool(st.get("kill_switch"))
        lines.append(
            f"Kill switch: {'ARMED — ' + str(st.get('kill_reason') or '') if killed else 'clear'}"
        )
        paused = [n for n, b in (st.get("bots") or {}).items()
                  if (b or {}).get("status") == "paused"]
        lines.append("Paused: " + (", ".join(paused) if paused else "none"))
        port = st.get("portfolio") or {}
        if port.get("status"):
            lines.append(f"Portfolio: {port.get('status')}")
    except Exception as e:
        lines.append(f"Risk state unavailable: {e}")
    try:
        log_age = time.time() - (config.LOG_DIR / "arena.log").stat().st_mtime
        stale = log_age > float(getattr(config, "ARENA_LOG_STALE_SEC", 300))
        lines.append(
            f"Arena log: {log_age:.0f}s old{' — STALE' if stale else ''}"
        )
    except OSError:
        lines.append("Arena log: missing")
    try:
        active = _active_bot_names()
        lines.append(f"Active bots ({len(active)}): {', '.join(active) or 'none'}")
    except Exception:
        pass
    return "\n".join(lines)


@_register("bots", "/bots — all-time WR / PnL / break-even gap per bot")
def cmd_bots(args: list[str]) -> str:
    with db.get_conn() as conn:
        rows = conn.execute("""
            SELECT bot_name, COUNT(*) n,
                ROUND(100.0*SUM(CASE WHEN outcome IN ('win','exit_tp')
                    THEN 1 ELSE 0 END)/COUNT(*), 1) wr_pct,
                ROUND(SUM(pnl), 2) pnl,
                ROUND(AVG(entry_price), 3) avg_entry,
                ROUND(1.0*SUM(CASE WHEN outcome IN ('win','exit_tp')
                    THEN 1 ELSE 0 END)/COUNT(*) - AVG(entry_price), 3) be_gap
            FROM trades
            WHERE outcome IN ('win','loss','exit_tp','exit_sl')
            GROUP BY bot_name ORDER BY pnl DESC
        """).fetchall()
    if not rows:
        return "No resolved trades yet."
    lines = ["Bots (all-time):"]
    for r in rows:
        d = dict(r)
        lines.append(
            f"  {d['bot_name']}: {d['pnl']:+.2f} · n={d['n']} "
            f"WR={d['wr_pct']}% entry={d['avg_entry']} BE={d['be_gap']}"
        )
    return "\n".join(lines)


@_register("lanes", "/lanes — lane overrides + live monitor accuracy")
def cmd_lanes(args: list[str]) -> str:
    def _state(key):
        raw = db.get_arena_state(key)
        if not raw:
            return {}
        try:
            return json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            return {}

    overrides = _state("lane_overrides")
    monitor = _state("lane_monitor")
    lines = ["Lane overrides:"]
    if overrides:
        for k, v in overrides.items():
            v = v or {}
            lines.append(
                f"  {k}: {'ON' if v.get('enabled') else 'off'}"
                + (" (core)" if v.get("core") else "")
            )
    else:
        lines.append("  none (all candidates kill-switched)")
    if monitor:
        lines.append("Monitor:")
        for k, v in monitor.items():
            v = v or {}
            acc = v.get("accuracy")
            acc_s = f"{100 * acc:.0f}%" if acc is not None else "n/a"
            lines.append(f"  {k}: {acc_s} over n={v.get('n')} [{v.get('verdict')}]")
    return "\n".join(lines)


@_register("soak", "/soak — full soak report (bots, lanes, regimes, skips)")
def cmd_soak(args: list[str]) -> str:
    from tools.soak_report import build_report, format_text
    return format_text(build_report())


@_register("help", "/help — this list")
def cmd_help(args: list[str]) -> str:
    control_on = bool(getattr(config, "TELEGRAM_COMMANDS_CONTROL_ENABLED", True))
    read = [c for c in _REGISTRY.values() if not c.control and c.help]
    ctl = [c for c in _REGISTRY.values() if c.control and c.help]
    lines = ["Arena commands", "", "Reports:"]
    lines += [f"  {c.help}" for c in sorted(read, key=lambda c: c.name)]
    lines += ["", "Control:" + ("" if control_on else " (DISABLED in config)")]
    lines += [f"  {c.help}" for c in sorted(ctl, key=lambda c: c.name)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Control commands
# ---------------------------------------------------------------------------

@_register("kill", "/kill [reason] — arm the global kill switch (halts trading)",
           control=True)
def cmd_kill(args: list[str]) -> str:
    reason = " ".join(args) or "telegram kill command"
    _risk_engine().set_kill_switch(True, reason=reason, source="telegram")
    return (f"KILL SWITCH ARMED — {reason}\n"
            "All trading halted. /unkill to clear.")


@_register("unkill",
           "/unkill confirm — clear the kill switch and resume trading",
           control=True)
def cmd_unkill(args: list[str]) -> str:
    # The risk engine arms this switch AUTOMATICALLY on a daily-loss or
    # drawdown breach. Clearing that from a phone, in one word, with no
    # sight of the numbers, is how a bad day becomes a worse one.
    source = ""
    try:
        source = str((_risk_engine().load_state() or {}).get("kill_source") or "")
    except Exception:
        pass
    warn = ""
    if source and source not in ("telegram", "dashboard", "api"):
        warn = (f"\nNOTE: this halt was armed automatically by '{source}' — "
                "check /status and /day before overriding it.")
    gate = _confirm_gate(
        "unkill", args,
        "Clearing the kill switch resumes ALL trading immediately."
        + warn + "\nSend: /unkill confirm")
    if gate:
        return gate
    _risk_engine().set_kill_switch(False, reason="telegram unkill",
                                   source="telegram")
    return "Kill switch cleared. Trading resumes on the next tick."


@_register("pause", "/pause <bot|all> — pause a bot (size 0, keeps config)",
           control=True)
def cmd_pause(args: list[str]) -> str:
    if not args:
        return "Usage: /pause <bot_name|all>\nActive: " + (
            ", ".join(_active_bot_names()) or "none")
    target = args[0]
    active = _active_bot_names()
    names = active if target.lower() == "all" else [target]
    if target.lower() != "all" and target not in active:
        return f"'{target}' is not active. Active: {', '.join(active) or 'none'}"
    re = _risk_engine()
    for n in names:
        re.pause_bot(n, reason="telegram manual pause")
    return f"Paused {len(names)}: {', '.join(names)}\n/resume to undo."


@_register("resume", "/resume <bot|all> confirm — clear a pause "
                     "(force-allows past automatic risk limits)", control=True)
def cmd_resume(args: list[str]) -> str:
    if not args:
        return "Usage: /resume <bot_name|all> confirm"
    target = args[0]
    active = _active_bot_names()
    names = active if target.lower() == "all" else [target]
    if target.lower() != "all" and target not in active:
        return f"'{target}' is not active. Active: {', '.join(active) or 'none'}"
    # resume_bot sets manual_resume, which force-allows trading PAST the
    # automatic risk limits (see risk_engine.resume_bot docstring).
    who = "all" if target.lower() == "all" else target
    gate = _confirm_gate(
        f"resume:{who}", args[1:],
        f"'/resume {who}' force-allows {len(names)} bot(s) to trade past "
        "automatic risk limits (incl. drawdown pauses).\n"
        f"Send: /resume {who} confirm")
    if gate:
        return gate
    re = _risk_engine()
    for n in names:
        re.resume_bot(n)
    return f"Resumed {len(names)}: {', '.join(names)}"


@_register("retire", "/retire <bot> confirm — permanently retire a bot",
           control=True)
def cmd_retire(args: list[str]) -> str:
    if not args:
        return "Usage: /retire <bot_name> confirm\nActive: " + (
            ", ".join(_active_bot_names()) or "none")
    name = args[0]
    active = _active_bot_names()
    if name not in active:
        return f"'{name}' is not active. Active: {', '.join(active) or 'none'}"
    gate = _confirm_gate(
        f"retire:{name}", args[1:],
        f"Retiring '{name}' is permanent — its open positions still resolve "
        f"but it stops trading and leaves the slate.\n"
        f"Send: /retire {name} confirm")
    if gate:
        return gate
    # Pause first so the bot stops sizing immediately. retire_bot only flips
    # active=0; the in-memory trader slate is rebuilt on the next coordinator
    # cycle, not within one tick.
    try:
        _risk_engine().pause_bot(name, reason="telegram retire")
    except Exception as e:
        logger.warning("telegram retire pause failed: %s", type(e).__name__)
    db.retire_bot(name)
    return (f"Retired '{name}' — paused now; it leaves the live slate on "
            "the next coordinator cycle. Use /pause if you only needed a halt.")


@_register("deploy", "/deploy <strategy...> — deploy strategies mid-run",
           control=True)
def cmd_deploy(args: list[str]) -> str:
    catalog = {e["strategy_type"] for e in _strategy_catalog()}
    active_types = {
        b.get("strategy_type") for b in (db.get_active_bots() or [])
    }
    if not args:
        avail = sorted(catalog - active_types)
        return ("Usage: /deploy <strategy> [strategy...]\n"
                "Available: " + (", ".join(avail) or "none (all active)"))

    # Merge with anything still queued so a second /deploy doesn't clobber the
    # first before the arena coordinator drains the queue (~30s).
    pending: list[str] = []
    try:
        raw = db.get_arena_state("pending_bot_deploys")
        if raw:
            prev = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(prev, dict):
                pending = list(prev.get("strategies") or [])
    except (TypeError, ValueError):
        pending = []

    queued, skipped = [], []
    for st in args:
        st = st.strip()
        if st not in catalog:
            skipped.append(f"{st} (unknown)")
        elif st in active_types:
            skipped.append(f"{st} (already active)")
        elif st in pending:
            skipped.append(f"{st} (already queued)")
        else:
            pending.append(st)
            queued.append(st)

    lines = []
    if queued:
        db.set_arena_state("pending_bot_deploys", json.dumps({
            "strategies": pending,
            "queued_at": time.time(),
            "source": "telegram",
        }))
        lines.append(f"Queued {len(queued)}: {', '.join(queued)} "
                     "— arena applies within ~30s")
    if skipped:
        lines.append("Skipped: " + ", ".join(skipped))
    return "\n".join(lines) or "Nothing to deploy."


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _permitted_senders(chat_id: str) -> set[str]:
    """User ids allowed to command. Defaults to the chat itself — true
    exactly for a private chat, so a GROUP target needs an explicit
    allowlist rather than silently trusting every member."""
    raw = getattr(config, "TELEGRAM_COMMANDS_ALLOWED_USER_IDS", ()) or ()
    if isinstance(raw, (str, int)):        # tolerate a bare scalar in config
        raw = (raw,)
    allowed = {str(u).strip() for u in raw if str(u).strip()}
    return allowed or {chat_id}


def _update_id(update: dict) -> Optional[int]:
    """One parser for both the drain and the dispatcher — divergent id
    filtering there would let an update be skipped without ever being acked,
    which stalls the offset and spins the poll loop."""
    try:
        return int(update.get("update_id"))
    except (AttributeError, TypeError, ValueError):
        return None


def _message_age_sec(msg: dict) -> Optional[float]:
    """Seconds since the message was SENT, or None if it has no usable date."""
    try:
        return time.time() - float(msg["date"])
    except (KeyError, TypeError, ValueError):
        return None


def _log_unauthorized(who: str) -> None:
    """Throttled — a stranger who finds the bot could otherwise fill the
    dashboard log at message rate."""
    now = time.time()
    if now - _unauth_logged.get(who, 0.0) < _UNAUTH_LOG_EVERY_SEC:
        return
    _unauth_logged[who] = now
    if len(_unauth_logged) > 200:            # keep the throttle map bounded
        for k, ts in sorted(_unauth_logged.items(), key=lambda kv: kv[1])[:100]:
            _unauth_logged.pop(k, None)
    logger.warning("telegram command from unauthorized sender %s ignored", who)


def _rate_limited(name: str) -> bool:
    window = float(getattr(config, "TELEGRAM_COMMANDS_RATE_LIMIT_SEC", 3.0))
    if window <= 0:
        return False
    now = time.time()
    last = _last_command_at.get(name, 0.0)
    if now - last < window:
        return True
    _last_command_at[name] = now
    return False


def handle_update(update: dict) -> bool:
    """Authorize, dispatch and reply to one Telegram update.

    Returns True when the update was ours to handle (authorized command),
    False when it was dropped — foreign chat, non-message, or plain chatter.
    """
    if not isinstance(update, dict):
        return False
    msg = update.get("message")
    if not isinstance(msg, dict):
        return False

    # --- auth boundary: allowlisted chat AND sender; no reply to anyone else -
    chat = msg.get("chat") or {}
    incoming = str(chat.get("id") or "")
    allowed = str(_chat_id() or "")
    if not allowed or incoming != allowed:
        if incoming:
            _log_unauthorized(incoming)
        return False
    # Chat scope alone is not enough: if the alert target is a GROUP, every
    # member of it would inherit /kill and /retire. Default policy is "sender
    # must be the chat itself", which is true exactly for a private chat.
    if msg.get("sender_chat"):
        # Channel post / anonymous group admin: no user identity to allowlist.
        _log_unauthorized(f"{incoming}/sender_chat")
        return False
    sender = str((msg.get("from") or {}).get("id") or "")
    permitted = _permitted_senders(allowed)
    # Unconditional: `from` is optional in the Bot API, and an auth check that
    # skips itself when its field is absent is not an auth check.
    if sender not in permitted:
        _log_unauthorized(f"{incoming}/user:{sender or 'absent'}")
        return False

    parsed = parse_command(msg.get("text") or "")
    if not parsed:
        return False
    name, args = parsed

    cmd = _REGISTRY.get(name)
    if cmd is None:
        _send(allowed, f"Unknown command /{name}. Try /help.")
        return True

    # Staleness gate. The startup drain only covers a process RESTART, but the
    # common outage here is a sleeping Mac with the dashboard still alive — the
    # backlog then lands mid-loop and would execute. Judging the message's own
    # age closes that whole class: a control command is an instruction about
    # NOW, so an old one is never safe to run. Reports are side-effect free and
    # still answer.
    if cmd.control:
        max_age = float(getattr(config, "TELEGRAM_COMMANDS_MAX_AGE_SEC", 300))
        age = _message_age_sec(msg)
        if max_age > 0 and (age is None or age > max_age):
            _send(allowed,
                  f"/{name} ignored — message is too old "
                  f"({'unknown age' if age is None else f'{age / 60:.0f} min'}; "
                  f"limit {max_age / 60:.0f} min). Re-send it if you still "
                  "want it to run.")
            return True

    if cmd.control and not getattr(
            config, "TELEGRAM_COMMANDS_CONTROL_ENABLED", True):
        _send(allowed, f"/{name} is a control command and control commands "
                       "are disabled (TELEGRAM_COMMANDS_CONTROL_ENABLED).")
        return True

    # A `confirm` follow-up is the second half of a handshake the bot itself
    # asked for — rate-limiting it would strand the operator mid-command.
    is_confirm = any(a.lower() == "confirm" for a in args)
    if not is_confirm and _rate_limited(name):
        _send(allowed, f"Too fast — /{name} is rate limited, try again shortly.")
        return True

    try:
        reply = cmd.handler(args)
    except Exception as e:
        logger.exception("telegram command /%s failed", name)
        reply = f"Error running /{name}: {e}"
    _send(allowed, reply or "(no output)")
    if cmd.control:
        logger.warning("telegram control command executed: /%s %s", name, args)
    return True


def process_updates(updates: list[dict]) -> int:
    """Handle a batch, skipping already-acked ids and advancing the offset.

    **Ack-then-execute, per update.** The offset moves past an update BEFORE
    its handler runs, so a handler that raises (or a crash mid-batch) can
    never leave a control command re-deliverable — Telegram would otherwise
    hand us the same ``/kill`` on every subsequent poll. The trade is
    at-most-once execution, which is the right side to err on here.
    """
    handled = 0
    for u in updates:
        uid = _update_id(u)
        if uid is None:
            continue
        if uid < _load_offset():
            continue  # Telegram redelivers until acked — don't re-run it
        _save_offset(uid + 1)
        if handle_update(u):
            handled += 1
    return handled


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------

def _drain_backlog() -> bool:
    """Ack everything queued while we were down WITHOUT executing it.

    Telegram holds undelivered updates for 24h. Replaying a ``/kill`` the
    operator sent yesterday — or a stale ``/deploy`` — on process start would
    be a nasty surprise, so startup only advances the offset. Pages until the
    queue is empty: ``getUpdates`` caps at 100 per call, and acking only the
    first page would drop update 101 straight into the executing loop.

    Returns False if the drain could not be completed, in which case the
    caller must NOT start dispatching — an un-drained backlog would run.
    """
    total = 0
    for _ in range(50):                      # 5k updates in one pass
        updates = _get_updates(_load_offset(), timeout=0)
        if updates is None:
            logger.warning("telegram backlog drain failed — not dispatching "
                           "until it succeeds (stale commands could replay)")
            return False
        if not updates:
            if total:
                logger.info("telegram: skipped %d backlogged update(s) "
                            "on startup", total)
            return True
        ids = [i for i in (_update_id(u) for u in updates) if i is not None]
        if not ids:
            return True
        _save_offset(max(ids) + 1)
        total += len(ids)
    # Cap reached with the queue still non-empty. Report FAILURE so the caller
    # keeps draining instead of dispatching — otherwise enough spam messages
    # push a real stale command past the cap and into the executing loop.
    logger.warning("telegram: backlog drain hit its page cap at %d updates — "
                   "continuing next cycle", total)
    return False


def poll_loop() -> None:
    timeout = int(getattr(config, "TELEGRAM_COMMANDS_POLL_TIMEOUT_SEC", 30))
    backoff = 1.0
    drained = False
    logger.info("telegram command poller started (long-poll %ss)", timeout)
    while not _stop.is_set():
        try:
            if not drained:
                drained = _drain_backlog()
                if not drained:
                    _stop.wait(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue
                backoff = 1.0
            started = time.time()
            updates = _get_updates(_load_offset(), timeout=timeout)
            if updates is None:
                # 401 / 409 / 429 / network. Back off — spinning here would
                # hammer Telegram AND read SQLite thousands of times a second
                # against the same WAL DB the arena trades on.
                _stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            backoff = 1.0
            if updates:
                process_updates(updates)
            elif time.time() - started < 1.0:
                # A long-poll that returns instantly is misbehaving; don't spin.
                _stop.wait(1.0)
        except Exception as e:
            # Never let a transient blip kill the thread — this loop is the
            # operator's only remote control.
            logger.warning("telegram poll cycle failed: %s", type(e).__name__)
            _stop.wait(backoff)
            backoff = min(backoff * 2, 60.0)


def start_poller() -> bool:
    """Start the long-poll thread once. Returns True if it is now running."""
    global _poller_thread
    # FastAPI TestClient runs the dashboard lifespan. Starting a real
    # getUpdates loop from pytest would drain the live Telegram queue
    # (ack-without-execute on startup) and 409 against the running dashboard.
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("PYTEST_VERSION"):
        return False
    if not getattr(config, "TELEGRAM_COMMANDS_ENABLED", False):
        return False
    if not (_token() and _chat_id()):
        logger.info("telegram commands: no bot token / chat id configured")
        return False
    with _poller_lock:
        if _poller_thread and _poller_thread.is_alive():
            return True
        _stop.clear()
        _poller_thread = threading.Thread(
            target=poll_loop, name="telegram-commands", daemon=True)
        _poller_thread.start()
    return True


def stop_poller() -> None:
    _stop.set()

"""Centralized Risk Engine for the Polymarket Bot Arena.

Single authority for pre-trade risk gates and continuous risk evaluation:

* Per-bot and portfolio **daily loss** limits (net P&L)
* Per-bot and portfolio **max drawdown** (peak-to-trough on bankroll-anchored
  equity curves — full pool capital base + cumulative trade P&L, not
  zero-based P&L and not portfolio-weight micro-books)
* Automatic **size reduction** as drawdown approaches the limit, then **pause**
* Underperformance pause (window P&L floor)
* **Historical VaR** (percentile of recent trade P&Ls) when enough data
* **Kill switch** — dashboard, API, or flag file
* Every decision is **logged** (``risk_events`` table + arena_state snapshot)

Hot path (``pre_trade`` / ``is_killed`` / ``size_multiplier``) is cached for a
few seconds so the 1s trader tick stays cheap. Full evaluation runs on the
evolution-loop host every ``RISK_EVAL_INTERVAL_SEC``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import config
import db

logger = logging.getLogger("arena.risk_engine")

STATE_KEY = "risk_engine"
KILL_STATE_KEY = "kill_switch"  # lightweight flag also mirrored in STATE_KEY

# Hot-path cache: (ts, state_dict, killed)
_cache: tuple = (0.0, {}, False)
# Dedupe block_trade logs: (bot, reason) -> last_ts
_block_log_ts: dict[tuple, float] = {}
_BLOCK_LOG_INTERVAL_SEC = 60.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TradeDecision:
    """Result of a pre-trade risk check."""
    allow: bool
    size_mult: float = 1.0
    reason: str = "ok"
    action: str = "allow"  # allow | reduce | block | pause | kill
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def max_drawdown_pct(
    pnls: Sequence[float],
    *,
    starting_equity: float = 0.0,
) -> float:
    """Peak-to-trough drawdown as a fraction of peak equity (0..inf).

    ``starting_equity`` is the capital base *before* the first P&L (e.g.
    bankroll at window open). Without it, pure-loss series start at peak=0
    and cannot form a meaningful ratio — callers should pass bankroll.
    """
    start = max(0.0, float(starting_equity or 0.0))
    if not pnls:
        return 0.0
    equity = start
    peak = start
    max_dd = 0.0
    for p in pnls:
        equity += float(p)
        peak = max(peak, equity)
        if peak > 1e-12:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def equity_stats(
    pnls: Sequence[float],
    *,
    starting_equity: float = 0.0,
) -> dict[str, float]:
    """Return equity, peak, drawdown_pct for a P&L series (oldest→newest).

    Equity is bankroll-anchored: curve starts at ``starting_equity`` (capital
    before the first trade in ``pnls``), then accumulates trade P&Ls.
    Drawdown is ``(peak − equity) / peak``. A pure-loss window on a $1000
    book is ~1–2% DD, not 100% (the old zero-based curve forced 100% whenever
    cumulative P&L never went positive).
    """
    start = max(0.0, float(starting_equity or 0.0))
    if not pnls:
        return {
            "equity": round(start, 4),
            "peak": round(start, 4),
            "drawdown": 0.0,
            "n": 0,
            "starting_equity": round(start, 4),
        }
    equity = start
    peak = start
    for p in pnls:
        equity += float(p)
        peak = max(peak, equity)
    dd = 0.0
    if peak > 1e-12:
        dd = max(0.0, (peak - equity) / peak)
    return {
        "equity": round(equity, 4),
        "peak": round(peak, 4),
        "drawdown": round(dd, 4),
        "n": len(pnls),
        "starting_equity": round(start, 4),
    }


def _capital_now() -> float:
    """Current risk capital base (paper gross equity = bankroll + realized)."""
    try:
        gross = float(db.get_paper_pool_gross())
        if gross > 0:
            return gross
    except Exception:
        pass
    try:
        bankroll = float(db.get_paper_bankroll())
        if bankroll > 0:
            return bankroll
    except Exception:
        pass
    return float(getattr(config, "PAPER_BANKROLL_DEFAULT", 200.0))


def _window_start_equity(pnls: Sequence[float], capital_now: float) -> float:
    """Reconstruct equity immediately before the first trade in ``pnls``.

    ``capital_now − sum(pnls)`` is the equity at window open when
    ``capital_now`` is bankroll + all realized P&L and ``pnls`` is the
    window's chronological trade series.
    """
    return max(0.0, float(capital_now) - sum(float(p) for p in pnls))


def _bot_capital_weight(bot_name: str) -> float:
    """Fraction of pool capital this bot is sized against (1.0 if allocation off)."""
    try:
        from arena.portfolio import get_weight
        return max(0.0, float(get_weight(bot_name)))
    except Exception:
        return 1.0


def historical_var(pnls: Sequence[float], confidence: float = 0.95) -> Optional[float]:
    """1-period historical VaR as a positive USD loss estimate.

    Uses the empirical ``(1-confidence)`` quantile of trade P&Ls. Returns
    None when sample is too thin. VaR is reported as a positive number
    meaning "loss not expected to exceed this with conf probability".
    """
    clean = [float(p) for p in pnls if p is not None]
    min_n = int(getattr(config, "RISK_VAR_MIN_TRADES", 20))
    if len(clean) < min_n:
        return None
    conf = min(0.999, max(0.5, float(confidence)))
    ordered = sorted(clean)
    # Lower-tail quantile (worst outcomes)
    idx = max(0, min(len(ordered) - 1, int(math.floor((1.0 - conf) * len(ordered)))))
    q = ordered[idx]
    # VaR as positive loss magnitude
    return round(max(0.0, -q), 4)


def _cutoff_hours(hours: float) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(hours=float(hours))
    ).strftime("%Y-%m-%d %H:%M:%S")


def _pnls_for_bots(
    bot_names: Sequence[str],
    *,
    hours: Optional[float] = None,
    today_only: bool = False,
    mode: Optional[str] = None,
) -> dict[str, list[float]]:
    """Ordered (oldest→newest) resolved P&Ls per bot."""
    out: dict[str, list[float]] = {n: [] for n in bot_names}
    if not bot_names:
        return out
    placeholders = ",".join("?" * len(bot_names))
    conds = [
        f"bot_name IN ({placeholders})",
        "outcome IN ('win', 'loss', 'exit_tp', 'exit_sl')",
        "pnl IS NOT NULL",
    ]
    params: list[Any] = list(bot_names)
    if today_only:
        # ET calendar day (00:00 America/New_York), same as dashboard "Today".
        # UTC date() misaligned Day P&L vs Overview when session spans ET midnight.
        conds.append("created_at>=?")
        params.append(db.et_day_start_utc(0))
    elif hours is not None:
        conds.append("created_at>=?")
        params.append(_cutoff_hours(hours))
    if mode is not None:
        conds.append("mode=?")
        params.append(mode)
    where = " AND ".join(conds)
    with db.get_conn() as conn:
        rows = conn.execute(
            f"""SELECT bot_name, pnl FROM trades
                WHERE {where}
                ORDER BY created_at ASC""",
            params,
        ).fetchall()
    for r in rows:
        name = r["bot_name"]
        if name in out:
            out[name].append(float(r["pnl"]))
    return out


def _portfolio_pnls(
    *,
    hours: Optional[float] = None,
    today_only: bool = False,
    mode: Optional[str] = None,
) -> list[float]:
    """Pool-level P&L series: sum of all bots' trades ordered by time.

    Approximates portfolio equity by chronological trade P&Ls across bots.
    """
    conds = [
        "outcome IN ('win', 'loss', 'exit_tp', 'exit_sl')",
        "pnl IS NOT NULL",
    ]
    params: list[Any] = []
    if today_only:
        # ET calendar day — keep Risk Engine Day P&L aligned with Overview Today.
        conds.append("created_at>=?")
        params.append(db.et_day_start_utc(0))
    elif hours is not None:
        conds.append("created_at>=?")
        params.append(_cutoff_hours(hours))
    if mode is not None:
        conds.append("mode=?")
        params.append(mode)
    where = " AND ".join(conds)
    with db.get_conn() as conn:
        rows = conn.execute(
            f"""SELECT pnl FROM trades WHERE {where}
                ORDER BY created_at ASC""",
            params,
        ).fetchall()
    return [float(r["pnl"]) for r in rows]


# ---------------------------------------------------------------------------
# Limits resolution
# ---------------------------------------------------------------------------

def _default_limits(mode: str = "paper") -> dict[str, Any]:
    """Resolve numeric limits from config (+ live vs paper daily loss caps)."""
    bot_daily = getattr(config, "RISK_BOT_DAILY_LOSS", None)
    if bot_daily is None:
        bot_daily = (
            config.LIVE_MAX_DAILY_LOSS_PER_BOT
            if mode == "live"
            else getattr(config, "RISK_PAPER_BOT_DAILY_LOSS", 75.0)
        )
    port_daily = getattr(config, "RISK_PORTFOLIO_DAILY_LOSS", None)
    if port_daily is None:
        port_daily = (
            config.LIVE_MAX_DAILY_LOSS_TOTAL
            if mode == "live"
            else getattr(config, "RISK_PAPER_PORTFOLIO_DAILY_LOSS", 150.0)
        )
    return {
        "bot_daily_loss": float(bot_daily),
        "portfolio_daily_loss": float(port_daily),
        "bot_max_drawdown": float(getattr(config, "RISK_BOT_MAX_DRAWDOWN", 0.35)),
        "portfolio_max_drawdown": float(
            getattr(config, "RISK_PORTFOLIO_MAX_DRAWDOWN", 0.40)),
        "var_confidence": float(getattr(config, "RISK_VAR_CONFIDENCE", 0.95)),
        "var_limit_usd": getattr(config, "RISK_VAR_LIMIT_USD", None),
        "underperform_pnl": float(
            getattr(config, "RISK_UNDERPERFORM_PAUSE_PNL", -40.0)),
        "underperform_hours": float(
            getattr(config, "RISK_UNDERPERFORM_WINDOW_HOURS", 12.0)),
        "dd_reduce_start": float(getattr(config, "RISK_SIZE_REDUCE_DD_FRAC", 0.50)),
        "size_reduce_min": float(getattr(config, "RISK_SIZE_REDUCE_MIN_MULT", 0.25)),
        "drawdown_window_hours": float(
            getattr(config, "RISK_DRAWDOWN_WINDOW_HOURS", 24.0)),
    }


def _merge_limits(stored: Optional[dict], mode: str = "paper") -> dict[str, Any]:
    base = _default_limits(mode)
    if not isinstance(stored, dict):
        return base
    for k, v in stored.items():
        if k not in base:
            continue
        if v is None:
            base[k] = None
            continue
        try:
            base[k] = float(v)
        except (TypeError, ValueError):
            pass
    return base


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

def kill_switch_file_path() -> Path:
    raw = getattr(config, "RISK_KILL_SWITCH_FILE", None)
    if raw:
        return Path(raw)
    return Path(config.LOG_DIR) / "KILL_SWITCH"


def _file_kill_armed() -> bool:
    path = kill_switch_file_path()
    try:
        if not path.is_file():
            return False
        text = path.read_text(encoding="utf-8", errors="ignore").strip().lower()
        # Empty file or "1"/"true"/"kill" arms; "0"/"false"/"off" disarms
        if text in ("0", "false", "off", "no", "clear", "disarm"):
            return False
        return True  # empty or any other content
    except OSError:
        return False


def is_killed() -> bool:
    """True if the global kill switch is armed (state or file)."""
    global _cache
    now = time.time()
    ttl = float(getattr(config, "RISK_HOTPATH_CACHE_SEC", 2.0))
    if (now - _cache[0]) < ttl:
        return bool(_cache[2])
    state = load_state()
    killed = bool(state.get("kill_switch")) or _file_kill_armed()
    _cache = (now, state, killed)
    return killed


def set_kill_switch(armed: bool, reason: str = "", source: str = "api") -> dict:
    """Arm or disarm the kill switch. Logs a risk event."""
    state = load_state()
    prev = bool(state.get("kill_switch"))
    armed = bool(armed)
    state["kill_switch"] = armed
    state["kill_reason"] = (reason or ("armed" if armed else "cleared"))[:500]
    state["kill_source"] = source
    state["kill_at"] = time.time() if armed else None
    # Mirror lightweight key for external tools
    db.set_arena_state(KILL_STATE_KEY, "1" if armed else "0")
    # File mirror: write/remove so operators can use either channel
    path = kill_switch_file_path()
    try:
        if armed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"kill\nreason={state['kill_reason']}\nsource={source}\n",
                            encoding="utf-8")
        elif path.is_file():
            # Only remove if we own it / content is kill-related
            path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("kill-switch file update failed: %s", e)

    save_state(state)
    action = "kill" if armed else "unkill"
    if prev != armed:
        log_event(
            action=action,
            level="critical" if armed else "info",
            reason=state["kill_reason"],
            bot=None,
            detail={"source": source, "prev": prev},
        )
    logger.critical("KILL SWITCH %s source=%s reason=%s",
                    "ARMED" if armed else "CLEARED", source, state["kill_reason"])
    return state


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _default_state() -> dict[str, Any]:
    return {
        "enabled": bool(getattr(config, "RISK_ENGINE_ENABLED", True)),
        "kill_switch": False,
        "kill_reason": None,
        "kill_source": None,
        "kill_at": None,
        "limits": _default_limits("paper"),
        "bots": {},
        "portfolio": {
            "status": "active",
            "daily_pnl": 0.0,
            "drawdown": 0.0,
            "var_1d": None,
            "size_mult": 1.0,
            "reason": None,
        },
        "updated_at": None,
        "last_eval_at": None,
    }


def load_state() -> dict[str, Any]:
    raw = db.get_arena_state(STATE_KEY)
    base = _default_state()
    if not raw:
        # Also check lightweight kill flag
        ks = db.get_arena_state(KILL_STATE_KEY)
        if ks in ("1", "true", "on"):
            base["kill_switch"] = True
        return base
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return base
    if not isinstance(data, dict):
        return base
    base.update(data)
    base["enabled"] = bool(base.get("enabled", True))
    base["kill_switch"] = bool(base.get("kill_switch"))
    if not isinstance(base.get("bots"), dict):
        base["bots"] = {}
    if not isinstance(base.get("portfolio"), dict):
        base["portfolio"] = _default_state()["portfolio"]
    base["limits"] = _merge_limits(base.get("limits"), "paper")
    return base


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    db.set_arena_state(STATE_KEY, json.dumps(state, default=str))
    global _cache
    killed = bool(state.get("kill_switch")) or _file_kill_armed()
    _cache = (time.time(), state, killed)


def bust_cache() -> None:
    global _cache
    _cache = (0.0, {}, False)


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

def log_event(
    *,
    action: str,
    level: str = "info",
    reason: str = "",
    bot: Optional[str] = None,
    detail: Optional[dict] = None,
    force: bool = False,
) -> None:
    """Persist a risk decision for dashboard visibility.

    ``block_trade`` events are rate-limited per (bot, reason) so a paused bot
    does not write one row every trader tick.
    """
    if action == "block_trade" and not force:
        key = (bot or "", reason or "")
        now = time.time()
        last = _block_log_ts.get(key, 0.0)
        if now - last < _BLOCK_LOG_INTERVAL_SEC:
            return
        _block_log_ts[key] = now
    try:
        db.log_risk_event(
            action=action,
            level=level,
            reason=reason,
            bot_name=bot,
            detail=detail or {},
        )
    except Exception as e:
        logger.warning("risk event log failed: %s", e)
    logger.log(
        logging.CRITICAL if level == "critical"
        else logging.WARNING if level == "warn"
        else logging.INFO,
        "risk action=%s bot=%s reason=%s detail=%s",
        action, bot, reason, detail or {},
    )
    # Production alerts (debounced inside alerts.notify)
    if action in ("pause", "kill", "portfolio_paused", "resume", "unkill") or (
            action in ("reduced",) and level in ("warn", "critical")):
        try:
            from arena.alerts import alert_risk
            alert_risk(action, reason or action, bot=bot, level=level)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Size multiplier from drawdown proximity
# ---------------------------------------------------------------------------

def _dd_size_mult(drawdown: float, max_dd: float, start_frac: float,
                  min_mult: float) -> float:
    """Linear size taper between start_frac*max_dd and max_dd.

    Below start → 1.0; at/above max → min_mult (caller may pause separately).
    """
    if max_dd <= 0:
        return 1.0
    start = max_dd * max(0.0, min(1.0, start_frac))
    if drawdown <= start:
        return 1.0
    if drawdown >= max_dd:
        return max(0.0, min_mult)
    # Linear from 1.0 → min_mult over [start, max_dd]
    t = (drawdown - start) / max(max_dd - start, 1e-9)
    return max(min_mult, 1.0 - t * (1.0 - min_mult))


# ---------------------------------------------------------------------------
# Full evaluation (slow path)
# ---------------------------------------------------------------------------

def evaluate(bot_names: Optional[Sequence[str]] = None,
             mode: str = "paper") -> dict[str, Any]:
    """Recompute risk metrics + status for all bots and the portfolio.

    Called from the arena main loop. Idempotent; logs only on status changes.
    """
    state = load_state()
    if bot_names is None:
        try:
            bot_names = [b["bot_name"] for b in db.get_active_bots()
                         if b.get("bot_name")]
        except Exception:
            bot_names = list((state.get("bots") or {}).keys())

    limits = _merge_limits(state.get("limits"), mode)
    state["limits"] = limits
    names = list(dict.fromkeys(bot_names or []))

    dd_hours = float(limits.get("drawdown_window_hours") or 24)
    under_hours = float(limits.get("underperform_hours") or 12)
    bot_daily_lim = float(limits["bot_daily_loss"])
    port_daily_lim = float(limits["portfolio_daily_loss"])
    bot_max_dd = float(limits["bot_max_drawdown"])
    port_max_dd = float(limits["portfolio_max_drawdown"])
    dd_start = float(limits.get("dd_reduce_start") or 0.5)
    min_mult = float(limits.get("size_reduce_min") or 0.25)
    under_pnl = float(limits.get("underperform_pnl") or -40)
    var_conf = float(limits.get("var_confidence") or 0.95)
    var_limit = limits.get("var_limit_usd")

    daily_pnls = _pnls_for_bots(names, today_only=True, mode=None)
    window_pnls = _pnls_for_bots(names, hours=dd_hours, mode=None)
    under_pnls = _pnls_for_bots(names, hours=under_hours, mode=None)

    # Bankroll-anchored equity: reconstruct window-start capital so pure-loss
    # runs report ~loss/capital DD, not a false 100% from a zero-based curve.
    capital_now = _capital_now()
    port_window_for_base = _portfolio_pnls(hours=dd_hours)
    port_start = _window_start_equity(port_window_for_base, capital_now)

    prev_bots = state.get("bots") or {}
    new_bots: dict[str, dict] = {}

    for name in names:
        d_series = daily_pnls.get(name) or []
        w_series = window_pnls.get(name) or []
        u_series = under_pnls.get(name) or []
        daily_pnl = sum(d_series)
        # Per-bot DD uses the FULL pool capital base, not portfolio weight ×
        # pool. Weights only control Kelly sizing; measuring DD against a 5%
        # micro-book ($10 on a $200 pool) made a −$0.50 day look like 35% DD
        # after a small run-up (sniper/momentum false pauses). Portfolio DD
        # still protects shared capital; daily-loss / underperform gates still
        # catch bot-level bleeding.
        bot_start = max(0.0, port_start)
        stats = equity_stats(w_series, starting_equity=bot_start)
        dd = float(stats["drawdown"])
        var = historical_var(w_series, var_conf)
        under_total = sum(u_series)
        auto_reason = None  # reason before manual overrides

        status = "active"
        reason = None
        size_mult = _dd_size_mult(dd, bot_max_dd, dd_start, min_mult)

        # Daily loss (net)
        if daily_pnl <= -bot_daily_lim:
            status = "paused"
            reason = f"bot_daily_loss:{daily_pnl:.2f}<=-{bot_daily_lim:.2f}"
            size_mult = 0.0
        # Drawdown hard stop
        elif dd >= bot_max_dd and stats["n"] >= 5:
            status = "paused"
            reason = f"bot_max_drawdown:{dd:.2%}>= {bot_max_dd:.0%}"
            size_mult = 0.0
        # Underperformance — graduated taper (audit 1e): replace binary
        # pause with progressive size reduction so a bot never fully stops.
        elif under_total <= under_pnl and len(u_series) >= int(
                getattr(config, "RISK_UNDERPERFORM_MIN_TRADES", 15)):
            if getattr(config, "RISK_UNDERPERFORM_GRADUATED", True):
                tiers = getattr(config, "RISK_UNDERPERFORM_GRADUATED_TIERS",
                                ((-20.0, 0.75), (-30.0, 0.50), (-40.0, 0.25)))
                # Walk tiers from worst (most negative) to best;
                # first match sets the mult.
                grad_mult = 1.0
                for floor, mult in sorted(tiers):
                    if under_total <= float(floor):
                        grad_mult = float(mult)
                        break
                size_mult = min(size_mult, grad_mult)
                if grad_mult < 1.0:
                    status = "reduced"
                    reason = (f"underperform:{under_total:.2f}≤"
                              f"graduated tier ×{grad_mult:.2f}"
                              f"/{under_hours:.0f}h")
                else:
                    status = "active"
                    reason = f"underperform:{under_total:.2f}>tiers"
            else:
                status = "paused"
                reason = (f"underperform:{under_total:.2f}<={under_pnl:.2f}"
                          f"/{under_hours:.0f}h")
                size_mult = 0.0
        elif size_mult < 0.999:
            status = "reduced"
            reason = f"drawdown_taper:dd={dd:.2%}/max={bot_max_dd:.0%}"

        auto_reason = reason
        prev = prev_bots.get(name) or {}
        manual_pause = bool(prev.get("manual_pause"))
        # Sticky operator resume: stays until metrics would naturally be active
        # again (so Resume isn't immediately undone by the same auto-pause).
        manual_resume = bool(prev.get("manual_resume")) and not manual_pause

        if manual_pause:
            status = "paused"
            reason = prev.get("reason") or "manual_pause"
            size_mult = 0.0
            manual_resume = False
        elif manual_resume and status == "paused":
            # Force trading at the DD taper floor while override is held.
            size_mult = max(min_mult, _dd_size_mult(dd, bot_max_dd, dd_start, min_mult))
            status = "reduced" if size_mult < 0.999 else "active"
            reason = f"manual_resume_override:{auto_reason or 'paused'}"
        elif manual_resume and status in ("active", "reduced"):
            # Metrics no longer require a hard pause — drop the sticky flag.
            if status == "active":
                manual_resume = False

        bot_state = {
            "status": status,
            "size_mult": round(size_mult, 4),
            "daily_pnl": round(daily_pnl, 4),
            "window_pnl": round(sum(w_series), 4),
            "drawdown": round(dd, 4),
            "peak": stats["peak"],
            "equity": stats["equity"],
            "starting_equity": stats.get("starting_equity", bot_start),
            "var_1d": var,
            "n_window": stats["n"],
            "reason": reason,
            "manual_pause": manual_pause,
            "manual_resume": bool(manual_resume),
            "capital_weight": round(_bot_capital_weight(name), 4),
        }
        new_bots[name] = bot_state

        # Log transitions
        prev_status = prev.get("status")
        if prev_status and prev_status != status:
            log_event(
                action=status if status != "active" else "resume",
                level="warn" if status != "active" else "info",
                reason=reason or f"{prev_status}->{status}",
                bot=name,
                detail={"prev": prev_status, "size_mult": size_mult,
                        "daily_pnl": daily_pnl, "drawdown": dd},
            )

    # Portfolio
    port_daily_series = _portfolio_pnls(today_only=True)
    port_window = port_window_for_base
    port_daily = sum(port_daily_series)
    port_stats = equity_stats(port_window, starting_equity=port_start)
    port_dd = float(port_stats["drawdown"])
    port_var = historical_var(port_window, var_conf)
    port_status = "active"
    port_reason = None
    port_mult = _dd_size_mult(port_dd, port_max_dd, dd_start, min_mult)

    if port_daily <= -port_daily_lim:
        port_status = "paused"
        port_reason = f"portfolio_daily_loss:{port_daily:.2f}<=-{port_daily_lim:.2f}"
        port_mult = 0.0
    elif port_dd >= port_max_dd and port_stats["n"] >= 10:
        port_status = "paused"
        port_reason = f"portfolio_max_drawdown:{port_dd:.2%}>= {port_max_dd:.0%}"
        port_mult = 0.0
    elif var_limit is not None and port_var is not None and port_var >= float(var_limit):
        port_status = "reduced"
        port_reason = f"var_limit:{port_var:.2f}>={float(var_limit):.2f}"
        port_mult = min(port_mult, min_mult)
    elif port_mult < 0.999:
        port_status = "reduced"
        port_reason = f"portfolio_dd_taper:dd={port_dd:.2%}"

    prev_port = state.get("portfolio") or {}
    if prev_port.get("status") != port_status:
        log_event(
            action=f"portfolio_{port_status}",
            level="critical" if port_status == "paused" else "warn",
            reason=port_reason or f"{prev_port.get('status')}->{port_status}",
            bot=None,
            detail={"daily_pnl": port_daily, "drawdown": port_dd,
                    "var_1d": port_var},
        )

    # File kill switch may arm without API
    file_kill = _file_kill_armed()
    if file_kill and not state.get("kill_switch"):
        state["kill_switch"] = True
        state["kill_reason"] = state.get("kill_reason") or "kill_switch_file"
        state["kill_source"] = "file"
        state["kill_at"] = time.time()
        log_event(action="kill", level="critical",
                  reason="kill_switch_file", bot=None,
                  detail={"path": str(kill_switch_file_path())})

    state["bots"] = new_bots
    state["portfolio"] = {
        "status": port_status,
        "daily_pnl": round(port_daily, 4),
        "window_pnl": round(sum(port_window), 4),
        "drawdown": round(port_dd, 4),
        "peak": port_stats["peak"],
        "equity": port_stats["equity"],
        "starting_equity": port_stats.get("starting_equity", port_start),
        "var_1d": port_var,
        "n_window": port_stats["n"],
        "size_mult": round(port_mult, 4),
        "reason": port_reason,
    }
    state["last_eval_at"] = time.time()
    save_state(state)
    return state


def maybe_evaluate() -> Optional[dict[str, Any]]:
    """Evolution-loop entry: evaluate if interval elapsed."""
    state = load_state()
    interval = float(getattr(config, "RISK_EVAL_INTERVAL_SEC", 15))
    last = state.get("last_eval_at") or 0.0
    try:
        last = float(last)
    except (TypeError, ValueError):
        last = 0.0
    if time.time() - last < interval:
        # Still refresh kill-file into cache
        if _file_kill_armed() and not state.get("kill_switch"):
            return evaluate()
        return None
    return evaluate()


# ---------------------------------------------------------------------------
# Hot-path API
# ---------------------------------------------------------------------------

def _cached_state() -> dict[str, Any]:
    global _cache
    now = time.time()
    ttl = float(getattr(config, "RISK_HOTPATH_CACHE_SEC", 2.0))
    if (now - _cache[0]) < ttl and _cache[1]:
        return _cache[1]
    state = load_state()
    killed = bool(state.get("kill_switch")) or _file_kill_armed()
    _cache = (now, state, killed)
    return state


def size_multiplier(bot_name: str) -> float:
    """Combined bot × portfolio size multiplier for sizing (0..1)."""
    state = _cached_state()
    if not state.get("enabled", True):
        return 1.0
    if state.get("kill_switch") or _file_kill_armed():
        return 0.0
    bot = (state.get("bots") or {}).get(bot_name) or {}
    port = state.get("portfolio") or {}
    bm = float(bot.get("size_mult", 1.0))
    pm = float(port.get("size_mult", 1.0))
    if bot.get("status") == "paused" or port.get("status") == "paused":
        return 0.0
    return max(0.0, min(1.0, bm * pm))


def pre_trade(bot_name: str, mode: str = "paper",
              amount: float = 0.0) -> TradeDecision:
    """Gate a trade. Called from ``BaseBot.execute`` / arbitrage.

    Returns allow/block + size multiplier. Logs blocks.
    """
    state = _cached_state()

    # Kill switch always wins (even if engine "disabled" for soft limits)
    if state.get("kill_switch") or _file_kill_armed():
        reason = state.get("kill_reason") or "kill_switch"
        log_event(action="block_trade", level="critical", reason=f"kill:{reason}",
                  bot=bot_name, detail={"amount": amount})
        return TradeDecision(allow=False, size_mult=0.0, reason="kill_switch",
                             action="kill", detail={"kill_reason": reason})

    if not state.get("enabled", True):
        # Fall back to legacy hard caps only
        return _legacy_daily_check(bot_name, mode)

    bot = (state.get("bots") or {}).get(bot_name) or {}
    port = state.get("portfolio") or {}

    if bot.get("status") == "paused":
        reason = bot.get("reason") or "bot_paused_risk"
        log_event(action="block_trade", level="warn", reason=reason,
                  bot=bot_name, detail={"status": "paused"})
        return TradeDecision(allow=False, size_mult=0.0, reason=reason,
                             action="pause", detail=bot)

    if port.get("status") == "paused":
        reason = port.get("reason") or "portfolio_paused_risk"
        log_event(action="block_trade", level="warn", reason=reason,
                  bot=bot_name, detail={"portfolio": port.get("status")})
        return TradeDecision(allow=False, size_mult=0.0, reason=reason,
                             action="pause", detail=port)

    # Live refresh of daily loss (state may be up to RISK_EVAL_INTERVAL stale)
    limits = _merge_limits(state.get("limits"), mode)
    try:
        daily_series = _pnls_for_bots([bot_name], today_only=True).get(bot_name) or []
        daily_pnl = sum(daily_series)
    except Exception:
        daily_pnl = float(bot.get("daily_pnl") or 0.0)

    bot_lim = float(limits["bot_daily_loss"])
    if daily_pnl <= -bot_lim:
        reason = f"bot_daily_loss:{daily_pnl:.2f}"
        log_event(action="block_trade", level="warn", reason=reason, bot=bot_name)
        return TradeDecision(allow=False, size_mult=0.0, reason=reason,
                             action="pause",
                             detail={"daily_pnl": daily_pnl, "limit": bot_lim})

    try:
        port_daily = sum(_portfolio_pnls(today_only=True))
    except Exception:
        port_daily = float(port.get("daily_pnl") or 0.0)
    port_lim = float(limits["portfolio_daily_loss"])
    if port_daily <= -port_lim:
        reason = f"portfolio_daily_loss:{port_daily:.2f}"
        log_event(action="block_trade", level="warn", reason=reason, bot=bot_name)
        return TradeDecision(allow=False, size_mult=0.0, reason=reason,
                             action="pause",
                             detail={"daily_pnl": port_daily, "limit": port_lim})

    mult = size_multiplier(bot_name)
    if mult <= 0:
        return TradeDecision(allow=False, size_mult=0.0,
                             reason=bot.get("reason") or "risk_size_zero",
                             action="pause")
    if mult < 0.999:
        return TradeDecision(allow=True, size_mult=mult,
                             reason=bot.get("reason") or "size_reduced",
                             action="reduce",
                             detail={"size_mult": mult})
    return TradeDecision(allow=True, size_mult=1.0, reason="ok", action="allow")


def _legacy_daily_check(bot_name: str, mode: str) -> TradeDecision:
    """When risk engine is disabled, keep the original daily-loss gates."""
    try:
        daily_loss = db.get_bot_daily_loss(bot_name, mode)
        max_daily = config.get_max_daily_loss_per_bot()
        if daily_loss >= max_daily:
            return TradeDecision(
                allow=False, size_mult=0.0,
                reason="daily_loss_limit", action="pause",
                detail={"daily_loss": daily_loss, "limit": max_daily},
            )
        total_daily = db.get_total_daily_loss(mode)
        max_total = config.get_max_daily_loss_total()
        if total_daily >= max_total:
            return TradeDecision(
                allow=False, size_mult=0.0,
                reason="arena_loss_limit", action="pause",
                detail={"total_daily": total_daily, "limit": max_total},
            )
    except Exception as e:
        logger.warning("legacy risk check failed: %s", e)
    return TradeDecision(allow=True, size_mult=1.0, reason="ok", action="allow")


# ---------------------------------------------------------------------------
# Manual controls
# ---------------------------------------------------------------------------

def pause_bot(bot_name: str, reason: str = "manual_pause") -> dict:
    state = load_state()
    bots = state.setdefault("bots", {})
    entry = dict(bots.get(bot_name) or {})
    entry.update({
        "status": "paused",
        "size_mult": 0.0,
        "manual_pause": True,
        "manual_resume": False,
        "reason": reason,
    })
    bots[bot_name] = entry
    save_state(state)
    log_event(action="pause", level="warn", reason=reason, bot=bot_name,
              detail={"manual": True})
    return entry


def resume_bot(bot_name: str) -> dict:
    """Clear manual pause and force-allow trading until risk metrics recover.

    Without ``manual_resume``, the next ``evaluate()`` immediately re-pauses
    any bot still over an automatic limit (e.g. max DD) — which made the
    dashboard Resume button appear broken.
    """
    state = load_state()
    bots = state.setdefault("bots", {})
    entry = dict(bots.get(bot_name) or {})
    entry["manual_pause"] = False
    entry["manual_resume"] = True
    # Optimistic UI state until evaluate recomputes size_mult / reason
    entry["status"] = "active"
    entry["size_mult"] = float(entry.get("size_mult") or 0.0) or 1.0
    entry["reason"] = "manual_resume"
    bots[bot_name] = entry
    save_state(state)
    log_event(action="resume", level="info", reason="manual_resume", bot=bot_name,
              detail={"manual_resume": True})
    evaluate()
    return (load_state().get("bots") or {}).get(bot_name) or entry


def set_enabled(enabled: bool) -> dict:
    state = load_state()
    state["enabled"] = bool(enabled)
    save_state(state)
    log_event(action="enable" if enabled else "disable", level="info",
              reason="dashboard", bot=None)
    return state


def update_limits(updates: dict) -> dict:
    state = load_state()
    limits = dict(state.get("limits") or {})
    allowed = set(_default_limits().keys())
    for k, v in (updates or {}).items():
        if k not in allowed:
            continue
        if v is None:
            limits[k] = None
        else:
            limits[k] = float(v)
    state["limits"] = _merge_limits(limits, "paper")
    save_state(state)
    log_event(action="limits_update", level="info", reason="dashboard",
              bot=None, detail=updates)
    return evaluate()


def dashboard_snapshot(limit_events: int = 40) -> dict[str, Any]:
    """Full risk view for the dashboard."""
    state = load_state()
    # Ensure at least one evaluation so the card isn't empty on first open
    if not state.get("last_eval_at"):
        try:
            state = evaluate()
        except Exception as e:
            logger.warning("risk evaluate on snapshot failed: %s", e)
    events = []
    try:
        events = db.get_risk_events(limit=limit_events)
    except Exception:
        events = []
    return {
        **state,
        "killed": bool(state.get("kill_switch")) or _file_kill_armed(),
        "kill_file": str(kill_switch_file_path()),
        "kill_file_armed": _file_kill_armed(),
        "events": events,
    }

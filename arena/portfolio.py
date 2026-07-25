"""Portfolio-level capital allocation across active arena bots.

Splits the shared bankroll so each bot sizes against a *fraction* of the pool
instead of the full pool (which multiplies correlated risk when several bots
trade the same 5-minute window).

Methods
-------
equal
    1/N to every active bot (baseline / cold-start).
sharpe
    Weight ∝ max(0, per-trade Sharpe) over the lookback window.
expectancy
    Weight ∝ max(0, average P&L per trade).
kelly_portfolio
    Simplified Kelly: score ∝ expectancy / variance, then shrink by average
    pairwise correlation so highly-overlapping bots don't all get full score.

Rebalance
---------
Hosted by the evolution loop (same cadence home as lane_monitor). Fires on:
  * timer (``PORTFOLIO_REBALANCE_INTERVAL_SEC``)
  * regime change (``regime_detector`` label flip vs last rebalance)
  * manual dashboard "Rebalance now" / weight save

Manual overrides (dashboard sliders) pin a bot's weight; remaining free mass
is distributed across unlocked bots by the chosen method, then renormalized.

State lives in arena_state key ``portfolio_allocation`` (JSON). Hot-path
reads go through ``get_weight(bot_name)`` with a short TTL cache.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import config
import db

logger = logging.getLogger("arena.portfolio")

STATE_KEY = "portfolio_allocation"
METHODS = ("equal", "sharpe", "expectancy", "kelly_portfolio")

# Hot-path cache: (ts, enabled, weights_dict, n_active)
_weight_cache: tuple = (0.0, False, {}, 0)


# ---------------------------------------------------------------------------
# Metrics + correlation
# ---------------------------------------------------------------------------

def _resolved_pnls_by_bot(
    bot_names: Sequence[str],
    hours: float,
    limit_per_bot: int = 500,
) -> dict[str, list[float]]:
    """Ordered P&L series per bot (oldest → newest) over the lookback window."""
    out: dict[str, list[float]] = {n: [] for n in bot_names}
    if not bot_names:
        return out
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=float(hours))
    ).strftime("%Y-%m-%d %H:%M:%S")
    placeholders = ",".join("?" * len(bot_names))
    with db.get_conn() as conn:
        rows = conn.execute(
            f"""SELECT bot_name, pnl, created_at FROM trades
                WHERE bot_name IN ({placeholders})
                  AND outcome IN ('win', 'loss', 'exit_tp', 'exit_sl')
                  AND pnl IS NOT NULL
                  AND created_at >= ?
                ORDER BY created_at ASC""",
            (*bot_names, cutoff),
        ).fetchall()
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        name = r["bot_name"]
        if name not in out:
            continue
        if counts[name] >= limit_per_bot:
            continue
        out[name].append(float(r["pnl"]))
        counts[name] += 1
    return out


def _market_returns_by_bot(
    bot_names: Sequence[str],
    hours: float,
) -> dict[str, dict[str, float]]:
    """bot -> {market_id: signed_pnl} for correlation (one trade per market).

    When a bot traded a market more than once, keep the net P&L.
    """
    out: dict[str, dict[str, float]] = {n: {} for n in bot_names}
    if not bot_names:
        return out
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=float(hours))
    ).strftime("%Y-%m-%d %H:%M:%S")
    placeholders = ",".join("?" * len(bot_names))
    with db.get_conn() as conn:
        rows = conn.execute(
            f"""SELECT bot_name, market_id, SUM(pnl) AS net
                FROM trades
                WHERE bot_name IN ({placeholders})
                  AND outcome IN ('win', 'loss', 'exit_tp', 'exit_sl')
                  AND pnl IS NOT NULL
                  AND market_id IS NOT NULL
                  AND created_at >= ?
                GROUP BY bot_name, market_id""",
            (*bot_names, cutoff),
        ).fetchall()
    for r in rows:
        name = r["bot_name"]
        if name in out:
            out[name][str(r["market_id"])] = float(r["net"] or 0.0)
    return out


def sharpe_ratio(pnls: Sequence[float]) -> float:
    """Per-trade Sharpe (mean / sample sd). 0 when undefined."""
    n = len(pnls)
    if n < 2:
        return 0.0
    mean = sum(pnls) / n
    var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
    sd = math.sqrt(var)
    if sd < 1e-12:
        return 0.0 if abs(mean) < 1e-12 else (10.0 if mean > 0 else -10.0)
    return mean / sd


def expectancy(pnls: Sequence[float]) -> float:
    """Mean P&L per trade (0 if empty)."""
    if not pnls:
        return 0.0
    return sum(pnls) / len(pnls)


def pairwise_correlation(
    market_returns: dict[str, dict[str, float]],
    min_overlap: int = 8,
) -> dict[str, dict[str, float]]:
    """Pearson correlation on overlapping market P&Ls.

    Returns full matrix bot -> bot -> rho in [-1, 1]. Missing / thin overlap
    defaults to 0 (independent) so a new bot is not unfairly penalized.
    """
    names = list(market_returns.keys())
    corr: dict[str, dict[str, float]] = {a: {b: 0.0 for b in names} for a in names}
    for a in names:
        corr[a][a] = 1.0
    for i, a in enumerate(names):
        ra = market_returns[a]
        for b in names[i + 1:]:
            rb = market_returns[b]
            common = [m for m in ra if m in rb]
            if len(common) < min_overlap:
                continue
            xa = [ra[m] for m in common]
            xb = [rb[m] for m in common]
            ma = sum(xa) / len(xa)
            mb = sum(xb) / len(xb)
            da = [x - ma for x in xa]
            db = [x - mb for x in xb]
            cov = sum(u * v for u, v in zip(da, db))
            va = sum(u * u for u in da)
            vb = sum(v * v for v in db)
            if va < 1e-18 or vb < 1e-18:
                rho = 0.0
            else:
                rho = max(-1.0, min(1.0, cov / math.sqrt(va * vb)))
            corr[a][b] = corr[b][a] = rho
    return corr


def compute_metrics(
    bot_names: Sequence[str],
    hours: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-bot performance metrics used by the allocator."""
    hours = float(hours if hours is not None else
                  getattr(config, "PORTFOLIO_WINDOW_HOURS", 24))
    min_trades = int(getattr(config, "PORTFOLIO_MIN_TRADES", 10))
    pnls = _resolved_pnls_by_bot(bot_names, hours)
    metrics: dict[str, dict[str, Any]] = {}
    for name in bot_names:
        series = pnls.get(name) or []
        n = len(series)
        sh = sharpe_ratio(series) if n >= min_trades else 0.0
        exp = expectancy(series) if n >= min_trades else 0.0
        var = 0.0
        if n >= 2:
            mean = sum(series) / n
            var = sum((p - mean) ** 2 for p in series) / (n - 1)
        metrics[name] = {
            "n": n,
            "sharpe": round(sh, 4),
            "expectancy": round(exp, 4),
            "total_pnl": round(sum(series), 4),
            "variance": round(var, 6),
            "ready": n >= min_trades,
        }
    return metrics


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def _raw_scores(
    method: str,
    metrics: dict[str, dict[str, Any]],
    corr: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Non-negative raw scores (pre-normalization) for each bot."""
    names = list(metrics.keys())
    if not names:
        return {}
    if method == "equal":
        return {n: 1.0 for n in names}

    scores: dict[str, float] = {}
    for n in names:
        m = metrics[n]
        if method == "sharpe":
            scores[n] = max(0.0, float(m.get("sharpe") or 0.0))
        elif method == "expectancy":
            scores[n] = max(0.0, float(m.get("expectancy") or 0.0))
        elif method == "kelly_portfolio":
            # f* proxy: E[r] / Var(r). Floor variance so a single lucky trade
            # doesn't explode the score; zero/negative expectancy → 0.
            exp = float(m.get("expectancy") or 0.0)
            var = max(float(m.get("variance") or 0.0), 1e-4)
            scores[n] = max(0.0, exp) / var
        else:
            scores[n] = 1.0

    # Correlation shrink (kelly_portfolio + sharpe/expectancy): a bot that
    # moves in lockstep with high-score peers gets penalized so capital is
    # not stacked on the same risk factor.
    shrink = float(getattr(config, "PORTFOLIO_CORR_SHRINK", 0.65))
    if method in ("kelly_portfolio", "sharpe", "expectancy") and shrink > 0:
        total = sum(scores.values()) or 1.0
        adjusted: dict[str, float] = {}
        for n in names:
            peers = [p for p in names if p != n]
            if not peers:
                adjusted[n] = scores[n]
                continue
            # Score-weighted average of max(0, corr) with peers
            num = 0.0
            den = 0.0
            for p in peers:
                w = scores[p]
                rho = max(0.0, float((corr.get(n) or {}).get(p) or 0.0))
                num += w * rho
                den += w
            avg_corr = (num / den) if den > 0 else 0.0
            adjusted[n] = scores[n] * (1.0 - shrink * avg_corr)
        scores = adjusted

    # Cold-start: bots under the sample floor keep a tiny positive score so
    # they aren't starved of exploration capital.
    floor_score = float(getattr(config, "PORTFOLIO_COLD_START_SCORE", 0.05))
    for n in names:
        if not metrics[n].get("ready"):
            scores[n] = max(scores[n], floor_score)
        elif scores[n] <= 0:
            scores[n] = floor_score * 0.5  # losing but still trading
    return scores


def apply_regime_tilt(
    scores: dict[str, float],
    regime_edges: dict[str, float] | None,
    *,
    max_tilt: float,
    min_weight: float,
) -> dict[str, float]:
    """Multiplicatively tilt allocation scores by per-bot regime edge.

    Each bot's edge is mapped to a bounded multiplier in
    ``[1 - max_tilt, 1 + max_tilt]`` via the sign-scaled position of its edge
    within the current regime's edge range. Bots absent from ``regime_edges``
    are left neutral. The explore floor then keeps every score at least
    ``min_weight * max(score)`` so no bot is starved out of generating the
    future trades the attribution needs.

    Pure: returns a new dict; never mutates ``scores``. Identity (a copy) when
    ``regime_edges`` is falsy — the no-conditioning / no-validated-regime path.
    """
    if not regime_edges:
        return dict(scores)
    vals = list(regime_edges.values())
    hi = max(vals)
    lo = min(vals)
    span = (hi - lo) or 1.0
    out: dict[str, float] = {}
    for bot, s in scores.items():
        e = regime_edges.get(bot)
        if e is None:
            out[bot] = s  # neutral: not attributed in this regime
            continue
        norm = 2.0 * (e - lo) / span - 1.0  # -1 (worst) .. +1 (best)
        out[bot] = s * (1.0 + max_tilt * norm)
    if out:
        floor = min_weight * max(out.values())
        out = {b: max(v, floor) for b, v in out.items()}
    return out


def _normalize(
    scores: dict[str, float],
    *,
    min_w: float,
    max_w: float,
) -> dict[str, float]:
    """Normalize scores to weights in [min_w, max_w] summing to 1.

    Projects proportional scores onto the probability simplex with box
    constraints. Caps (max) are fixed first, then floors (min), re-scaling
    the remaining free set each time so mass left by a capped winner flows
    to peers (who may then clear the floor without being pinned early).
    """
    names = list(scores.keys())
    n = len(names)
    if n == 0:
        return {}
    if n == 1:
        return {names[0]: 1.0}

    # Feasible box for the simplex.
    min_w = max(0.0, min(float(min_w), 1.0 / n))
    max_w = max(float(max_w), 1.0 / n)
    max_w = min(max_w, 1.0 - (n - 1) * min_w)
    min_w = max(min_w, 1.0 - (n - 1) * max_w)
    if max_w < min_w:
        min_w = max_w = 1.0 / n

    total = sum(max(0.0, float(scores[k])) for k in names)
    if total <= 0:
        prop = {k: 1.0 / n for k in names}
    else:
        prop = {k: max(0.0, float(scores[k])) / total for k in names}

    fixed: dict[str, float] = {}
    free = set(names)

    def _free_prop() -> dict[str, float]:
        sub = sum(prop[k] for k in free)
        if sub <= 1e-15:
            return {k: 1.0 for k in free}
        return {k: prop[k] / sub for k in free}

    for _ in range(2 * n + 2):
        if not free:
            break
        rem = 1.0 - sum(fixed.values())
        fp = _free_prop()
        free_sum = sum(fp.values())  # == 1
        trial = {k: fp[k] * rem for k in free}

        # 1) Fix caps first (mass flows to free peers on next iter).
        over = [k for k, tw in trial.items() if tw > max_w + 1e-12]
        if over:
            for k in over:
                fixed[k] = max_w
                free.discard(k)
            continue

        # 2) Then floors.
        under = [k for k, tw in trial.items() if tw < min_w - 1e-12]
        if under:
            for k in under:
                fixed[k] = min_w
                free.discard(k)
            continue

        # All free in bounds.
        out = dict(fixed)
        out.update(trial)
        s = sum(out.values()) or 1.0
        return {k: round(v / s, 6) for k, v in out.items()}

    out = dict(fixed)
    rem = 1.0 - sum(out.values())
    if free:
        fp = _free_prop()
        for k in free:
            out[k] = fp[k] * rem
    s = sum(out.values()) or 1.0
    return {k: round(v / s, 6) for k, v in out.items()}


def allocate(
    bot_names: Sequence[str],
    method: str = "kelly_portfolio",
    *,
    manual_overrides: Optional[dict[str, float]] = None,
    hours: float | None = None,
    regime_edges: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute portfolio weights for ``bot_names``.

    Returns dict with keys: weights, auto_weights, metrics, correlations,
    method, window_hours.
    """
    names = list(dict.fromkeys(bot_names))  # stable unique
    method = method if method in METHODS else "equal"
    hours = float(hours if hours is not None else
                  getattr(config, "PORTFOLIO_WINDOW_HOURS", 24))
    min_w = float(getattr(config, "PORTFOLIO_MIN_WEIGHT", 0.05))
    max_w = float(getattr(config, "PORTFOLIO_MAX_WEIGHT", 0.45))
    overrides = {k: float(v) for k, v in (manual_overrides or {}).items()
                 if k in names and float(v) >= 0}

    metrics = compute_metrics(names, hours)
    market_rets = _market_returns_by_bot(names, hours)
    corr = pairwise_correlation(
        market_rets,
        min_overlap=int(getattr(config, "PORTFOLIO_CORR_MIN_OVERLAP", 8)),
    )

    # Split locked (manual) vs free bots
    locked = {k: overrides[k] for k in overrides}
    locked_sum = sum(locked.values())
    free_names = [n for n in names if n not in locked]

    if locked_sum > 1.0 + 1e-9:
        # Scale locked down to fit
        scale = 1.0 / locked_sum
        locked = {k: v * scale for k, v in locked.items()}
        locked_sum = 1.0
        free_names = []

    free_mass = max(0.0, 1.0 - locked_sum)
    auto_weights: dict[str, float] = {}

    if free_names and free_mass > 0:
        scores = _raw_scores(method, {n: metrics[n] for n in free_names}, corr)
        # Regime-conditioning (Layer 3): bounded, floored tilt by each bot's
        # validated shrunk edge for the CURRENT regime. No-op (identity) when
        # regime_edges is None — the conditioning-off / no-validated-regime path.
        scores = apply_regime_tilt(
            scores, regime_edges,
            max_tilt=float(getattr(config, "REGIME_ALLOC_MAX_TILT", 0.25)),
            min_weight=float(getattr(config, "REGIME_ALLOC_MIN_WEIGHT", 0.05)),
        )
        # Normalize free bots onto a unit simplex with min/max scaled so that
        # after multiplying by free_mass the absolute floors/caps still hold.
        n_free = len(free_names)
        free_min = min(min_w / free_mass, 1.0 / n_free) if free_mass > 0 else 0.0
        free_max = min(max_w / free_mass, 1.0) if free_mass > 0 else 1.0
        free_w = _normalize(scores, min_w=free_min, max_w=free_max)
        auto_weights = {n: free_w.get(n, 0.0) * free_mass for n in free_names}
    elif free_names:
        auto_weights = {n: 0.0 for n in free_names}

    weights = {**auto_weights, **locked}
    # Ensure every bot present
    for n in names:
        weights.setdefault(n, 0.0)
    s = sum(weights.values()) or 1.0
    weights = {k: round(v / s, 6) for k, v in weights.items()}

    # Compact correlation for JSON (upper triangle pairs as "a|b")
    corr_pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = (corr.get(a) or {}).get(b, 0.0)
            if abs(rho) >= 0.05:
                corr_pairs[f"{a}|{b}"] = round(rho, 3)

    return {
        "weights": weights,
        "auto_weights": {k: round(v, 6) for k, v in auto_weights.items()},
        "manual_overrides": {k: round(v, 6) for k, v in locked.items()},
        "metrics": metrics,
        "correlations": corr_pairs,
        "method": method,
        "window_hours": hours,
    }


# ---------------------------------------------------------------------------
# Persistence + rebalance
# ---------------------------------------------------------------------------

def _default_state() -> dict[str, Any]:
    return {
        "enabled": bool(getattr(config, "PORTFOLIO_ALLOCATION_ENABLED", False)),
        "method": getattr(config, "PORTFOLIO_METHOD", "kelly_portfolio"),
        "window_hours": float(getattr(config, "PORTFOLIO_WINDOW_HOURS", 24)),
        "weights": {},
        "auto_weights": {},
        "manual_overrides": {},
        "metrics": {},
        "correlations": {},
        "last_rebalance_at": None,
        "last_regime": None,
        "rebalance_reason": None,
        "n_active": 0,
    }


def load_state() -> dict[str, Any]:
    raw = db.get_arena_state(STATE_KEY)
    base = _default_state()
    if not raw:
        return base
    try:
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return base
    if not isinstance(data, dict):
        return base
    base.update(data)
    # Coerce types
    base["enabled"] = bool(base.get("enabled"))
    method = base.get("method") or "kelly_portfolio"
    base["method"] = method if method in METHODS else "kelly_portfolio"
    try:
        base["window_hours"] = float(base.get("window_hours") or
                                     getattr(config, "PORTFOLIO_WINDOW_HOURS", 24))
    except (TypeError, ValueError):
        base["window_hours"] = float(getattr(config, "PORTFOLIO_WINDOW_HOURS", 24))
    for key in ("weights", "auto_weights", "manual_overrides", "metrics",
                "correlations"):
        if not isinstance(base.get(key), dict):
            base[key] = {}
    return base


def save_state(state: dict[str, Any]) -> None:
    db.set_arena_state(STATE_KEY, json.dumps(state, default=str))
    # Bust hot-path cache so next tick sees new weights
    global _weight_cache
    _weight_cache = (0.0, False, {}, 0)


def active_bot_names() -> list[str]:
    """Names of active bots that participate in capital allocation.

    Evolution-exempt market-neutral bots (arbitrage) still get a weight so
    the dashboard shows a complete picture; they may ignore it in execute.
    """
    bots = db.get_active_bots()
    return [b["bot_name"] for b in bots if b.get("bot_name")]


def _current_regime_label() -> Optional[str]:
    try:
        from signals.regime_detector import get_detector
        snap = get_detector().snapshot()
        return (snap or {}).get("label") or (snap or {}).get("regime")
    except Exception:
        return None


def rebalance(
    *,
    force: bool = False,
    reason: str = "timer",
    bot_names: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Recompute and persist portfolio weights.

    Skips when not due (unless ``force``). Returns the new state.
    """
    state = load_state()
    names = list(bot_names) if bot_names is not None else active_bot_names()
    if not names:
        state["weights"] = {}
        state["n_active"] = 0
        state["rebalance_reason"] = "no_bots"
        save_state(state)
        return state

    now = time.time()
    interval = float(getattr(config, "PORTFOLIO_REBALANCE_INTERVAL_SEC", 1800))
    last = state.get("last_rebalance_at")
    try:
        last_ts = float(last) if last is not None else 0.0
    except (TypeError, ValueError):
        last_ts = 0.0

    regime = _current_regime_label()
    last_regime = state.get("last_regime")
    regime_changed = (
        bool(getattr(config, "PORTFOLIO_REBALANCE_ON_REGIME", True))
        and regime is not None
        and last_regime is not None
        and regime != last_regime
        and regime not in ("unknown",)
    )

    due_timer = (now - last_ts) >= interval
    if not force and not due_timer and not regime_changed:
        return state

    if not force:
        if regime_changed:
            reason = f"regime:{last_regime}->{regime}"
        elif due_timer:
            reason = "timer"

    method = state.get("method") or getattr(config, "PORTFOLIO_METHOD", "kelly_portfolio")
    hours = float(state.get("window_hours") or
                  getattr(config, "PORTFOLIO_WINDOW_HOURS", 24))
    overrides = state.get("manual_overrides") or {}

    # Regime-conditioning (Layer 3): when the toggle is on and the CURRENT
    # regime has been discovered + OOS-validated, fetch its per-bot shrunk
    # edges so allocate() can tilt weights toward what works in this regime.
    # Best-effort: any gap (toggle off, no current_cell yet, unvalidated
    # regime) leaves regime_edges None and the tilt is a no-op.
    regime_edges = None
    try:
        if db.get_regime_conditioning():
            from arena.regime_map import edges_for_cell
            cur_cell = db.get_regime_map().get("current_cell")
            edges = edges_for_cell(tuple(cur_cell)) if cur_cell else None
            if edges:
                regime_edges = {b: e["shrunk_pnl"] for b, e in edges.items()}
    except Exception:
        regime_edges = None

    result = allocate(
        names,
        method=method,
        manual_overrides=overrides,
        hours=hours,
        regime_edges=regime_edges,
    )

    state.update({
        "weights": result["weights"],
        "auto_weights": result["auto_weights"],
        "manual_overrides": result["manual_overrides"],
        "metrics": result["metrics"],
        "correlations": result["correlations"],
        "method": result["method"],
        "window_hours": result["window_hours"],
        "last_rebalance_at": now,
        "last_regime": regime,
        "rebalance_reason": reason,
        "n_active": len(names),
        "enabled": bool(state.get("enabled")),
    })
    save_state(state)
    logger.info(
        "Portfolio rebalance reason=%s method=%s n=%d weights=%s",
        reason, method, len(names),
        {k: round(v, 3) for k, v in result["weights"].items()},
    )
    return state


def maybe_rebalance() -> Optional[dict[str, Any]]:
    """Evolution-loop entry: rebalance if timer/regime due. No-op if disabled.

    When disabled we still refresh metrics occasionally so the dashboard
    shows what *would* be allocated, but weights are not applied to sizing.
    """
    state = load_state()
    # Always allow rebalance computation (dashboard preview); enable flag only
    # gates sizing application in get_weight().
    new_state = rebalance(force=False)
    return new_state


# ---------------------------------------------------------------------------
# Hot-path weight lookup
# ---------------------------------------------------------------------------

def get_weight(bot_name: str) -> float:
    """Capital fraction for ``bot_name`` used in Kelly / sizing.

    When portfolio allocation is **disabled**, returns 1.0 (legacy: every bot
    sizes against the full shared bankroll).

    When **enabled**, returns the bot's portfolio weight in (0, 1], defaulting
    to equal-share if the bot is missing from the last rebalance (e.g. just
    evolved in).
    """
    global _weight_cache
    now = time.time()
    ttl = float(getattr(config, "SIZING_BANKROLL_CACHE_SEC", 5.0))
    if (now - _weight_cache[0]) < ttl:
        enabled, weights, n_active = _weight_cache[1], _weight_cache[2], _weight_cache[3]
    else:
        try:
            state = load_state()
            enabled = bool(state.get("enabled"))
            weights = dict(state.get("weights") or {})
            n_active = int(state.get("n_active") or len(weights) or 0)
        except Exception:
            enabled, weights, n_active = False, {}, 0
        _weight_cache = (now, enabled, weights, n_active)

    if not enabled:
        return 1.0
    if bot_name in weights:
        return max(0.0, float(weights[bot_name]))
    # New bot mid-cycle: equal share of remaining / N
    n = max(n_active, 1)
    return 1.0 / n


def is_enabled() -> bool:
    """True when portfolio capital slices are applied to sizing."""
    try:
        # Prefer hot-path cache (refreshed by get_weight / size_multiplier).
        global _weight_cache
        now = time.time()
        ttl = float(getattr(config, "SIZING_BANKROLL_CACHE_SEC", 5.0))
        if (now - _weight_cache[0]) < ttl:
            return bool(_weight_cache[1])
        return bool(load_state().get("enabled"))
    except Exception:
        return False


def size_multiplier(bot_name: str) -> float:
    """Scale factor for zone bots that size via max_position * pct.

    Equal-weight (weight = 1/N) → multiplier 1.0 so enabling portfolio with
    method=equal does not change zone-bot sizes. Winners (weight > 1/N) size
    up; losers size down. When disabled → 1.0.
    """
    global _weight_cache
    now = time.time()
    ttl = float(getattr(config, "SIZING_BANKROLL_CACHE_SEC", 5.0))
    if (now - _weight_cache[0]) >= ttl:
        # Refresh via get_weight side effect
        get_weight(bot_name)
    enabled, weights, n_active = _weight_cache[1], _weight_cache[2], _weight_cache[3]
    if not enabled:
        return 1.0
    n = max(n_active, len(weights), 1)
    w = get_weight(bot_name)
    return max(0.0, w * n)


def set_enabled(enabled: bool) -> dict[str, Any]:
    state = load_state()
    state["enabled"] = bool(enabled)
    save_state(state)
    if enabled and not state.get("weights"):
        return rebalance(force=True, reason="enable")
    return state


def set_method(method: str) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    state = load_state()
    state["method"] = method
    save_state(state)
    return rebalance(force=True, reason="method_change")


def set_manual_overrides(overrides: dict[str, float], *, merge: bool = False) -> dict[str, Any]:
    """Set manual weight pins. Pass ``{}`` to clear all. Values in [0, 1]."""
    state = load_state()
    cleaned: dict[str, float] = {}
    for k, v in (overrides or {}).items():
        fv = float(v)
        if fv < 0 or fv > 1:
            raise ValueError(f"override for {k} must be in [0, 1], got {fv}")
        if fv > 0:
            cleaned[str(k)] = fv
    if merge:
        cur = dict(state.get("manual_overrides") or {})
        cur.update(cleaned)
        # Allow explicit 0 / missing to mean "remove" when merge and value is 0
        for k, v in list(overrides.items()):
            if float(v) <= 0:
                cur.pop(str(k), None)
        state["manual_overrides"] = cur
    else:
        state["manual_overrides"] = cleaned
    save_state(state)
    return rebalance(force=True, reason="manual_override")


def dashboard_snapshot() -> dict[str, Any]:
    """Full state for the Settings / Portfolio dashboard card."""
    state = load_state()
    # Ensure metrics stay reasonably fresh even if rebalance is not due
    names = active_bot_names()
    if names and (
        not state.get("metrics")
        or set(state.get("weights") or {}) != set(names)
    ):
        # Soft refresh without waiting for timer when roster changed
        state = rebalance(force=True, reason="roster_sync", bot_names=names)
    return state

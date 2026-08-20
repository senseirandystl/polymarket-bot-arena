"""Multi-objective fitness for GA individuals.

Objectives (higher is always better after transform):

1. **pnl**         — total realized P&L over the judgment window
2. **sharpe**      — mean/sd of per-trade P&L (unannualized; 5-min markets)
3. **drawdown**    — 1 − max peak-to-trough drawdown fraction of peak equity
4. **consistency** — fraction of non-overlapping trade blocks with positive P&L

Raw objectives are rank-normalized across the population (0..1) then combined
with configurable weights so scale differences (dollars vs ratios) do not
let one objective dominate.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import config

# Default weights — sum need not be 1; they are re-normalized.
DEFAULT_WEIGHTS = {
    "pnl": 0.35,
    "sharpe": 0.20,
    "drawdown": 0.18,
    "consistency": 0.12,
    "regime_robustness": 0.15,
}


def _trade_pnls(trades: Sequence[dict]) -> list[float]:
    """Extract ordered P&L list from resolved trade rows (oldest first)."""
    resolved = _resolved_trades(trades)
    return [float(t["pnl"]) for t in resolved]


def _resolved_trades(trades: Sequence[dict]) -> list[dict]:
    resolved = [
        t for t in trades
        if t.get("outcome") in ("win", "loss", "exit_tp", "exit_sl")
        and t.get("pnl") is not None
    ]
    try:
        resolved = sorted(resolved, key=lambda t: t.get("created_at") or "")
    except Exception:
        pass
    return resolved


def _parse_trade_ts(t: dict) -> float | None:
    """Best-effort epoch seconds from created_at (str or number)."""
    raw = t.get("created_at")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        from datetime import datetime, timezone
        s = str(raw).replace("Z", "+00:00")
        # SQLite often stores "YYYY-MM-DD HH:MM:SS" without tz — treat as UTC
        if "T" not in s and "+" not in s:
            s = s.replace(" ", "T") + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def weighted_trade_pnls(
    trades: Sequence[dict],
    *,
    current_regime: str | None = None,
    now_ts: float | None = None,
    halflife_hours: float | None = None,
    regime_boost: float | None = None,
) -> list[float]:
    """P&L series with recency + current-regime reweighting.

    Each trade contributes ``pnl * weight`` as a synthetic sample (weight
    rounded into repeated unit samples would bias length; instead we scale
    the P&L value so total/mean emphasize recent + in-regime outcomes while
    keeping one sample per trade for Sharpe/drawdown structure).
    """
    import time as _time

    resolved = _resolved_trades(trades)
    if not resolved:
        return []
    hl = float(halflife_hours if halflife_hours is not None
               else getattr(config, "GA_REGIME_RECENCY_HALFLIFE_H", 6.0))
    boost = float(regime_boost if regime_boost is not None
                  else getattr(config, "GA_REGIME_MATCH_BOOST", 1.5))
    now = float(now_ts if now_ts is not None else _time.time())
    hl_sec = max(hl, 0.25) * 3600.0
    out: list[float] = []
    for t in resolved:
        w = 1.0
        ts = _parse_trade_ts(t)
        if ts is not None and hl_sec > 0:
            age = max(0.0, now - ts)
            # half-life decay: w = 0.5 ** (age / hl)
            w *= 0.5 ** (age / hl_sec)
        if current_regime:
            rid = _regime_from_trade(t)
            if rid and rid == current_regime:
                apply_boost = True
                try:
                    from signals.regime_detector import get_detector
                    snap = get_detector().snapshot() or {}
                    live_rid = snap.get("regime_id") or snap.get("label")
                    if live_rid == current_regime and not snap.get("actionable", False):
                        apply_boost = False
                except Exception:
                    pass
                if apply_boost:
                    w *= boost
        # Floor so ancient trades still count a little
        w = max(0.15, w)
        out.append(float(t["pnl"]) * w)
    return out


def max_drawdown_pct(pnls: Sequence[float]) -> float:
    """Max peak-to-trough drawdown as a fraction of peak equity (0..inf)."""
    if not pnls:
        return 0.0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        if peak > 0:
            max_dd = max(max_dd, dd / peak)
        elif dd > 0 and peak == 0:
            # All losses from zero: treat full loss as 100% DD of |equity|
            max_dd = max(max_dd, 1.0)
    return max_dd


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


def consistency_score(pnls: Sequence[float], block_size: int = 10) -> float:
    """Fraction of non-overlapping blocks with positive sum P&L.

    Falls back to a single-block (0 or 1) when the series is shorter than
    ``block_size``. Empty series → 0.
    """
    if not pnls:
        return 0.0
    if len(pnls) < block_size:
        return 1.0 if sum(pnls) > 0 else 0.0
    wins = 0
    blocks = 0
    for i in range(0, len(pnls) - block_size + 1, block_size):
        block = pnls[i:i + block_size]
        blocks += 1
        if sum(block) > 0:
            wins += 1
    return wins / blocks if blocks else 0.0


def _regime_from_trade(t: dict) -> str | None:
    """Extract regime id stamped into trade_features at decision time."""
    feats = t.get("trade_features")
    if feats is None:
        return None
    if isinstance(feats, str):
        try:
            import json
            feats = json.loads(feats)
        except Exception:
            return None
    if isinstance(feats, list):
        for f in feats:
            if isinstance(f, str) and f.startswith("regime:") and not f.startswith("regime_legacy:"):
                return f.split(":", 1)[1]
    if isinstance(feats, dict):
        return feats.get("regime") or feats.get("regime_id")
    return None


def _current_regime_from_state() -> str | None:
    """Best-effort read of the live regime id from arena_state."""
    try:
        import json
        import db
        raw = db.get_arena_state("regime") or db.get_arena_state("regime_detector")
        if not raw:
            return None
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            return data.get("regime") or data.get("id") or data.get("label")
        if isinstance(data, str):
            return data
    except Exception:
        return None
    return None


def regime_breakdown(trades: Sequence[dict]) -> dict[str, dict[str, float]]:
    """Per-regime multi-objective components from stamped trades."""
    buckets: dict[str, list[float]] = {}
    for t in trades or []:
        if t.get("outcome") not in ("win", "loss", "exit_tp", "exit_sl"):
            continue
        if t.get("pnl") is None:
            continue
        rid = _regime_from_trade(t) or "unknown"
        buckets.setdefault(rid, []).append(float(t["pnl"]))
    return {
        rid: multi_objective_fitness(pnls=pnls, use_recency=False)
        for rid, pnls in buckets.items()
    }


def multi_objective_fitness(
    trades: Sequence[dict] | None = None,
    *,
    pnls: Sequence[float] | None = None,
    block_size: int | None = None,
    regime_condition: bool | None = None,
    current_regime: str | None = None,
    use_recency: bool | None = None,
) -> dict[str, float]:
    """Compute raw objective components from trades or a P&L series.

    Returns dict with keys: pnl, sharpe, drawdown, consistency, n_trades.
    ``drawdown`` is already inverted (higher = better = lower drawdown).

    When ``regime_condition`` is True (default from config.GA_REGIME_CONDITION)
    and trades carry ``regime:*`` feature stamps, the composite also includes
    a cross-regime robustness term: bots that only print in one regime and
    bleed in others are penalized via the worst-regime P&L rank.

    When ``use_recency`` is True (default from config.GA_RECENCY_WEIGHTING),
    P&L samples are reweighted by exponential half-life and a boost for trades
    stamped with ``current_regime`` so the fitness favors the present tape.
    """
    trade_list = list(trades or [])
    if pnls is None:
        do_rec = use_recency
        if do_rec is None:
            do_rec = bool(getattr(config, "GA_RECENCY_WEIGHTING", True))
        if do_rec and trade_list:
            # Resolve current regime from arena_state if not provided
            cr = current_regime
            if cr is None:
                cr = _current_regime_from_state()
            pnls = weighted_trade_pnls(trade_list, current_regime=cr)
        else:
            pnls = _trade_pnls(trade_list)
    else:
        pnls = list(pnls)
    n = len(pnls)
    if n == 0:
        # No evidence → neutral-zero components (not "perfect drawdown").
        return {
            "pnl": 0.0,
            "sharpe": 0.0,
            "drawdown": 0.0,
            "consistency": 0.0,
            "n_trades": 0.0,
            "max_drawdown_pct": 0.0,
            "regime_robustness": 0.0,
            "regime_breakdown": {},
        }
    total_pnl = float(sum(pnls))
    sh = sharpe_ratio(pnls)
    dd_pct = max_drawdown_pct(pnls)
    # Invert drawdown: 0 DD → 1.0, 100%+ DD → 0.0
    dd_score = max(0.0, 1.0 - min(1.0, dd_pct))
    bs = block_size if block_size is not None else int(
        getattr(config, "GA_CONSISTENCY_BLOCK", 10)
    )
    cons = consistency_score(pnls, block_size=bs)

    # Regime-conditioned robustness
    use_rc = regime_condition
    if use_rc is None:
        use_rc = bool(getattr(config, "GA_REGIME_CONDITION", True))
    regime_robust = 0.5  # neutral when not applicable
    breakdown: dict = {}
    if use_rc and trade_list:
        breakdown = regime_breakdown(trade_list)
        # Only count regimes with enough samples
        min_n = int(getattr(config, "GA_REGIME_MIN_TRADES", 5))
        scored = {
            rid: c for rid, c in breakdown.items()
            if rid != "unknown" and c.get("n_trades", 0) >= min_n
        }
        if len(scored) >= 2:
            # Robustness = fraction of regimes with positive P&L, blended
            # with normalized worst-regime P&L (tanh) so a single toxic
            # regime pulls fitness down even if others are fine.
            pos = sum(1 for c in scored.values() if c["pnl"] > 0)
            frac_pos = pos / len(scored)
            worst = min(c["pnl"] for c in scored.values())
            worst_s = 0.5 + 0.5 * math.tanh(worst / 25.0)  # ~0..1
            regime_robust = 0.55 * frac_pos + 0.45 * worst_s
        elif len(scored) == 1:
            only = next(iter(scored.values()))
            regime_robust = 0.5 + 0.5 * math.tanh(only["pnl"] / 25.0)

    return {
        "pnl": total_pnl,
        "sharpe": sh,
        "drawdown": dd_score,
        "consistency": cons,
        "n_trades": float(n),
        "max_drawdown_pct": dd_pct,
        "regime_robustness": float(regime_robust),
        "regime_breakdown": {
            rid: {"pnl": c["pnl"], "n": c["n_trades"], "sharpe": c["sharpe"]}
            for rid, c in breakdown.items()
        },
    }


def _rank_scores(values: Sequence[float], higher_is_better: bool = True) -> list[float]:
    """Average-rank normalize to [0, 1]. Ties share the average rank."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    # Sort indices by value
    order = sorted(range(n), key=lambda i: values[i], reverse=higher_is_better)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # ranks i..j (0-based position in sorted order); higher position = better
        avg_pos = (i + j) / 2.0
        # Map best (pos 0) → 1.0, worst (pos n-1) → 0.0
        score = 1.0 - (avg_pos / (n - 1))
        for k in range(i, j + 1):
            ranks[order[k]] = score
        i = j + 1
    return ranks


def rank_normalize_fitness(
    components_list: Sequence[dict[str, float]],
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Rank-normalize each objective across the population and form a composite.

    Returns one dict per individual with:
      fitness (float), components (raw), ranks (0..1 per objective), weights.
    """
    w = dict(weights or getattr(config, "GA_FITNESS_WEIGHTS", None) or DEFAULT_WEIGHTS)
    # Only score objectives we know
    obj_keys = [k for k in (
        "pnl", "sharpe", "drawdown", "consistency", "regime_robustness",
    ) if k in w]
    w_sum = sum(max(0.0, float(w[k])) for k in obj_keys) or 1.0
    w_norm = {k: max(0.0, float(w[k])) / w_sum for k in obj_keys}

    n = len(components_list)
    if n == 0:
        return []

    rank_map: dict[str, list[float]] = {}
    for key in obj_keys:
        vals = [float(c.get(key, 0.0)) for c in components_list]
        rank_map[key] = _rank_scores(vals, higher_is_better=True)

    results = []
    for i, raw in enumerate(components_list):
        ranks = {k: rank_map[k][i] for k in obj_keys}
        fitness = sum(w_norm[k] * ranks[k] for k in obj_keys)
        results.append({
            "fitness": fitness,
            "components": {
                "pnl": float(raw.get("pnl", 0.0)),
                "sharpe": float(raw.get("sharpe", 0.0)),
                "drawdown": float(raw.get("drawdown", 0.0)),
                "consistency": float(raw.get("consistency", 0.0)),
                "regime_robustness": float(raw.get("regime_robustness", 0.5)),
                "n_trades": float(raw.get("n_trades", 0.0)),
                "max_drawdown_pct": float(raw.get("max_drawdown_pct", 0.0)),
                "regime_breakdown": raw.get("regime_breakdown") or {},
            },
            "ranks": ranks,
            "weights": w_norm,
        })
    return results


def composite_from_raw(
    components: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Single-individual composite without rank normalization (offline / tests).

    Applies a soft tanh on P&L so dollar magnitude does not explode, and uses
    raw sharpe/drawdown/consistency already on roughly comparable scales.
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    obj_keys = [k for k in (
        "pnl", "sharpe", "drawdown", "consistency", "regime_robustness",
    ) if k in w]
    w_sum = sum(max(0.0, float(w[k])) for k in obj_keys) or 1.0
    w_norm = {k: max(0.0, float(w[k])) / w_sum for k in obj_keys}

    pnl = float(components.get("pnl", 0.0))
    # Soft scale: ~$50 maps near ±1
    pnl_s = math.tanh(pnl / 50.0)
    sharpe_s = math.tanh(float(components.get("sharpe", 0.0)))
    dd_s = float(components.get("drawdown", 0.0))  # already 0..1
    cons_s = float(components.get("consistency", 0.0))  # 0..1
    reg_s = float(components.get("regime_robustness", 0.5))  # 0..1
    scaled = {
        "pnl": pnl_s,
        "sharpe": sharpe_s,
        "drawdown": dd_s,
        "consistency": cons_s,
        "regime_robustness": reg_s,
    }
    return sum(w_norm[k] * scaled[k] for k in obj_keys)

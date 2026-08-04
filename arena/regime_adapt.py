"""Regime-adaptive decision adjustments from live ``regime_performance``.

Data-driven set-and-forget controls:

1. **Live WR/P&L → size** (bad regimes shrink bets automatically).
2. **Strategy × regime priors** → how signals are *read* in that regime.
3. **Hard directional stand-down** when a regime is toxic live (WR/P&L bar
   with hysteresis so recovery re-enables trading without a restart).
4. **Mid-band drift floors** raised when the regime is depressed.

Hot-path cached. Fail-open to neutral adjustments on any error.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import config
import db

logger = logging.getLogger("arena.regime_adapt")

_cache: tuple[float, dict[str, float], dict[str, Any], dict[str, bool]] = (
    0.0, {}, {}, {},
)


@dataclass(frozen=True)
class RegimeAdjust:
    """Per-decision knobs for the current regime (+ optional strategy)."""

    size_mult: float = 1.0
    edge_mult: float = 1.0
    flow_full_trust: float | None = None
    mom_lane_scale: float = 1.0
    strat_lane_scale: float = 1.0
    no_edge_mult: float = 1.0
    extra_drift_floor: float = 0.0
    # When True, directional bots should skip (arb/makers exempt at call site).
    block_directional: bool = False
    # Extra min |signed drift| for mid in coin-flip favorite band [0.50, 0.58].
    mid_band_drift_min: float | None = None
    label: str = "unknown"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Strategy-aware priors (structural; live WR then modulates size/block).
_REGIME_STRATEGY_PRIORS: dict[str, dict[str, dict[str, float]]] = {
    "low_vol_trend": {
        "_default": {
            "edge_mult": 1.35,
            "flow_full_trust": 0.38,
            "mom_lane_scale": 0.50,
            "strat_lane_scale": 0.45,
            "no_edge_mult": 1.40,
            "extra_drift_floor": 0.08,
            "mid_band_drift_min": 0.40,
        },
        "momentum": {
            "edge_mult": 1.45,
            "flow_full_trust": 0.40,
            "mom_lane_scale": 0.40,
            "strat_lane_scale": 0.40,
            "no_edge_mult": 1.45,
            "extra_drift_floor": 0.10,
            "mid_band_drift_min": 0.42,
        },
        "mean_reversion": {
            "edge_mult": 1.50,
            "flow_full_trust": 0.35,
            "mom_lane_scale": 0.0,
            "strat_lane_scale": 0.50,
            "no_edge_mult": 1.30,
            "extra_drift_floor": 0.12,
            "mid_band_drift_min": 0.45,
        },
        "mean_reversion_tp": {
            "edge_mult": 1.50,
            "flow_full_trust": 0.35,
            "mom_lane_scale": 0.0,
            "strat_lane_scale": 0.50,
            "no_edge_mult": 1.30,
            "extra_drift_floor": 0.12,
            "mid_band_drift_min": 0.45,
        },
        "hybrid": {
            "edge_mult": 1.40,
            "flow_full_trust": 0.38,
            "mom_lane_scale": 0.45,
            "strat_lane_scale": 0.40,
            "no_edge_mult": 1.50,
            "extra_drift_floor": 0.10,
            "mid_band_drift_min": 0.42,
        },
        "phantom": {
            "edge_mult": 1.50,
            "flow_full_trust": 0.42,
            "mom_lane_scale": 0.35,
            "strat_lane_scale": 0.35,
            "no_edge_mult": 1.55,
            "extra_drift_floor": 0.12,
            "mid_band_drift_min": 0.45,
        },
        "sniper": {
            "edge_mult": 1.20,
            "flow_full_trust": 0.35,
            "mom_lane_scale": 1.0,
            "strat_lane_scale": 1.0,
            "no_edge_mult": 1.45,
            "extra_drift_floor": 0.08,
            "mid_band_drift_min": 0.38,
        },
    },
    "low_vol_range": {
        "_default": {
            "edge_mult": 1.05,
            "flow_full_trust": 0.28,
            "mom_lane_scale": 0.70,
            "strat_lane_scale": 0.75,
            "no_edge_mult": 1.15,
            "extra_drift_floor": 0.02,
            "mid_band_drift_min": 0.28,
        },
        "mean_reversion": {
            "edge_mult": 0.95,
            "mom_lane_scale": 0.0,
            "strat_lane_scale": 0.85,
            "no_edge_mult": 1.10,
            "mid_band_drift_min": 0.25,
        },
        "momentum": {
            "edge_mult": 1.15,
            "mom_lane_scale": 0.60,
            "strat_lane_scale": 0.65,
            "no_edge_mult": 1.20,
            "mid_band_drift_min": 0.30,
        },
    },
    "high_vol_chop": {
        "_default": {
            "edge_mult": 1.10,
            "flow_full_trust": 0.28,
            "mom_lane_scale": 0.65,
            "strat_lane_scale": 0.70,
            "no_edge_mult": 1.20,
            "extra_drift_floor": 0.03,
            "mid_band_drift_min": 0.30,
        },
    },
    "high_vol_trend": {
        "_default": {
            "edge_mult": 0.95,
            "flow_full_trust": 0.22,
            "mom_lane_scale": 1.10,
            "strat_lane_scale": 0.90,
            "no_edge_mult": 1.10,
            "extra_drift_floor": 0.0,
            "mid_band_drift_min": 0.25,
        },
        "momentum": {
            "edge_mult": 0.90,
            "mom_lane_scale": 1.15,
            "strat_lane_scale": 0.95,
        },
    },
    "normal": {
        "_default": {
            "edge_mult": 1.0,
            "mom_lane_scale": 1.0,
            "strat_lane_scale": 1.0,
            "no_edge_mult": 1.15,
            "extra_drift_floor": 0.0,
            "mid_band_drift_min": 0.28,
        },
    },
}

# Directional strategy types subject to hard regime skip (not arb/makers).
_DIRECTIONAL_TYPES = frozenset({
    "momentum", "mean_reversion", "mean_reversion_sl", "mean_reversion_tp",
    "phantom", "hybrid", "sniper", "sentiment",
})


def _load_perf() -> dict[str, Any]:
    raw = db.get_arena_state("regime_performance")
    if not raw:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _size_mult_from_stats(n: int, wins: int, pnl: float) -> float:
    min_n = int(getattr(config, "REGIME_ADAPT_MIN_TRADES", 15))
    if n < min_n:
        return 1.0
    wr = wins / n if n else 0.5
    bad = float(getattr(config, "REGIME_ADAPT_BAD_WR", 0.48))
    good = float(getattr(config, "REGIME_ADAPT_GOOD_WR", 0.62))
    lo = float(getattr(config, "REGIME_ADAPT_SIZE_MIN", 0.35))
    hi = float(getattr(config, "REGIME_ADAPT_SIZE_MAX", 1.15))
    if good <= bad + 1e-9:
        return 1.0
    t = (wr - bad) / (good - bad)
    t = max(0.0, min(1.0, t))
    mult = lo + t * (hi - lo)
    if pnl < 0 and wr < good:
        mult = min(mult, (lo + 1.0) / 2.0)
    return round(mult, 3)


def _rebuild_size_table() -> dict[str, float]:
    perf = _load_perf()
    out: dict[str, float] = {}
    for label, st in perf.items():
        if not isinstance(st, dict):
            continue
        n = int(st.get("n") or 0)
        wins = int(st.get("wins") or 0)
        pnl = float(st.get("pnl") or 0.0)
        out[str(label)] = _size_mult_from_stats(n, wins, pnl)
    return out


def _hard_block_map(prev_blocks: dict[str, bool]) -> dict[str, bool]:
    """Hysteresis: enter block at low WR, clear only after recovery WR."""
    if not getattr(config, "REGIME_HARD_SKIP_ENABLED", True):
        return {}
    min_n = int(getattr(config, "REGIME_HARD_SKIP_MIN_TRADES", 20))
    enter_wr = float(getattr(config, "REGIME_HARD_SKIP_WR", 0.42))
    clear_wr = float(getattr(config, "REGIME_HARD_SKIP_CLEAR_WR", 0.50))
    need_neg = bool(getattr(config, "REGIME_HARD_SKIP_REQUIRE_NEG_PNL", True))
    perf = _load_perf()
    out: dict[str, bool] = {}
    for label, st in perf.items():
        if not isinstance(st, dict):
            continue
        n = int(st.get("n") or 0)
        if n < min_n:
            # Keep previous block state if we still have history but sample
            # temporarily thin after a reset; otherwise clear.
            if prev_blocks.get(str(label)):
                out[str(label)] = True
            continue
        wins = int(st.get("wins") or 0)
        wr = wins / n if n else 0.5
        pnl = float(st.get("pnl") or 0.0)
        was = bool(prev_blocks.get(str(label)))
        if was:
            # Stay blocked until clear bar
            if wr >= clear_wr and (not need_neg or pnl >= 0):
                out[str(label)] = False
            else:
                out[str(label)] = True
        else:
            if wr <= enter_wr and (not need_neg or pnl < 0):
                out[str(label)] = True
            else:
                out[str(label)] = False
    return out


def _live_edge_boost(regime_label: str) -> float:
    perf = _load_perf().get(regime_label) or {}
    n = int(perf.get("n") or 0)
    min_n = int(getattr(config, "REGIME_ADAPT_MIN_TRADES", 15))
    if n < min_n:
        return 1.0
    wins = int(perf.get("wins") or 0)
    wr = wins / n if n else 0.5
    bad = float(getattr(config, "REGIME_ADAPT_BAD_WR", 0.48))
    good = float(getattr(config, "REGIME_ADAPT_GOOD_WR", 0.62))
    if wr <= bad:
        return 1.20
    if wr >= good:
        return 0.92
    return 1.0


def _prior_for(regime_label: str, strategy_type: Optional[str]) -> dict[str, float]:
    block = _REGIME_STRATEGY_PRIORS.get(regime_label) or {}
    base = dict(block.get("_default") or {})
    if strategy_type and strategy_type in block:
        base.update(block[strategy_type])
    return base


def _refresh_cache() -> None:
    global _cache
    now = time.time()
    ttl = float(getattr(config, "REGIME_ADAPT_CACHE_SEC", 30.0))
    if (now - _cache[0]) < ttl:
        return
    try:
        prev_blocks = _cache[3] if isinstance(_cache[3], dict) else {}
        size_table = _rebuild_size_table()
        perf = _load_perf()
        blocks = _hard_block_map(prev_blocks)
        _cache = (now, size_table, perf, blocks)
    except Exception as e:
        logger.debug("regime_adapt rebuild failed: %s", e)


def size_multiplier(regime_label: Optional[str]) -> float:
    """Backward-compatible size mult for the current regime (1.0 if unknown)."""
    adj = adjustments(regime_label, strategy_type=None)
    return float(adj.size_mult)


def adjustments(
    regime_label: Optional[str],
    strategy_type: Optional[str] = None,
) -> RegimeAdjust:
    """Full decision adjustments for regime × strategy."""
    if not getattr(config, "REGIME_ADAPT_ENABLED", True):
        return RegimeAdjust(reason="disabled")
    label = str(regime_label or "unknown")
    if label in ("unknown", ""):
        return RegimeAdjust(label=label, reason="unknown_regime")

    _refresh_cache()
    size_table = _cache[1]
    blocks = _cache[3]
    size_m = float(size_table.get(label, 1.0))
    prior = _prior_for(label, strategy_type)
    live_boost = _live_edge_boost(label)

    edge_m = float(prior.get("edge_mult", 1.0)) * live_boost
    flow_ft = prior.get("flow_full_trust")
    if flow_ft is not None:
        flow_ft = float(flow_ft)

    mid_band = prior.get("mid_band_drift_min")
    if mid_band is not None:
        mid_band = float(mid_band)
    # When size is depressed, raise mid-band floor further
    bad_mid = float(getattr(config, "MID_COINFLIP_DRIFT_MIN_BAD_REGIME", 0.40))
    if size_m <= float(getattr(config, "REGIME_ADAPT_SIZE_MIN", 0.35)) + 0.05:
        mid_band = max(mid_band or 0.0, bad_mid)

    block = bool(blocks.get(label, False))
    # Only block true directionals; callers still check strategy_type
    if strategy_type and strategy_type not in _DIRECTIONAL_TYPES:
        block = False

    reason_parts = [f"size={size_m:.2f}", f"edge×{edge_m:.2f}"]
    if block:
        reason_parts.append("HARD_SKIP")

    return RegimeAdjust(
        size_mult=0.0 if block else size_m,
        edge_mult=max(0.5, min(2.5, edge_m)),
        flow_full_trust=flow_ft,
        mom_lane_scale=float(prior.get("mom_lane_scale", 1.0)),
        strat_lane_scale=float(prior.get("strat_lane_scale", 1.0)),
        no_edge_mult=float(prior.get("no_edge_mult", 1.0)),
        extra_drift_floor=float(prior.get("extra_drift_floor", 0.0)),
        block_directional=block,
        mid_band_drift_min=mid_band,
        label=label,
        reason=" ".join(reason_parts),
    )


def snapshot() -> dict[str, Any]:
    """Dashboard / soak-report view of current multipliers + hard skips."""
    try:
        _refresh_cache()
        table = _cache[1]
        perf = _cache[2]
        blocks = _cache[3]
        rows = []
        for label, mult in sorted(table.items()):
            st = perf.get(label) or {}
            n = int(st.get("n") or 0)
            wins = int(st.get("wins") or 0)
            prior = _prior_for(label, None)
            rows.append({
                "regime": label,
                "n": n,
                "wr": (wins / n) if n else None,
                "pnl": round(float(st.get("pnl") or 0.0), 2),
                "size_mult": mult,
                "hard_skip": bool(blocks.get(label)),
                "prior": prior,
            })
        return {
            "enabled": bool(getattr(config, "REGIME_ADAPT_ENABLED", True)),
            "hard_skip_enabled": bool(
                getattr(config, "REGIME_HARD_SKIP_ENABLED", True)),
            "regimes": rows,
        }
    except Exception as e:
        return {"enabled": False, "error": str(e), "regimes": []}

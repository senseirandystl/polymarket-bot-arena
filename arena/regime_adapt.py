"""Regime-adaptive decision adjustments from live trade stats.

Data-driven controls (2026-08-07: fast dual-window + continuous response):

1. **Live WR/P&L → size** (bad regimes shrink bets automatically).
2. **Strategy × regime priors** as *seeds*; live blended WR dominates.
3. **Continuous edge / drift / mid-band tax** from fast+long WR (before skip).
4. **Dual-path style-skip** — fast (~10 fills / 2.5h) + slow long-window.
5. **Side-aware tax/skip** via strategy×regime×side cells (YES bleed ≠ NO).
6. **Hard directional stand-down** optional emergency-only (default OFF).

Hot-path cached. Fail-open to neutral adjustments on any error.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import config
import db

logger = logging.getLogger("arena.regime_adapt")

# cache: (ts, size_table, perf, regime_blocks, strategy_blocks)
# strategy_blocks: {(regime, strategy_type): bool}
_cache: tuple = (0.0, {}, {}, {}, {})


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
    # Strategy-level style skip (this strategy toxic in this regime only).
    block_strategy: bool = False
    # Extra min |signed drift| for mid in coin-flip favorite band [0.50, 0.58].
    mid_band_drift_min: float | None = None
    # Soft tandem: max distinct bots on same (market, side); None = config default.
    max_bots_side: int | None = None
    # Per-side continuous edge mult (applied after side selection).
    side_edge_mult: dict = field(default_factory=lambda: {"yes": 1.0, "no": 1.0})
    # Binary: skip only this side when strategy×regime×side is fast-toxic.
    block_side: str | None = None  # "yes" | "no" | None
    # Effective blended WR used for continuous knobs (diagnostics).
    wr_eff: float | None = None
    label: str = "unknown"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def side_edge_for(self, side: str) -> float:
        s = (side or "").lower()
        m = self.side_edge_mult or {}
        try:
            return float(m.get(s, 1.0) or 1.0)
        except (TypeError, ValueError):
            return 1.0


# Strategy-aware priors (structural; live WR then modulates size/block).
_REGIME_STRATEGY_PRIORS: dict[str, dict[str, dict[str, float]]] = {
    # Most session fills stamped "normal" — without priors they got no mid-band tax.
    "normal": {
        "_default": {
            "edge_mult": 1.15,
            "flow_full_trust": 0.32,
            "mom_lane_scale": 0.75,
            "strat_lane_scale": 0.70,
            "no_edge_mult": 1.35,
            "extra_drift_floor": 0.06,
            "mid_band_drift_min": 0.40,
        },
        "momentum": {
            "edge_mult": 1.20,
            "mom_lane_scale": 0.65,
            "no_edge_mult": 1.45,
            "extra_drift_floor": 0.08,
            "mid_band_drift_min": 0.42,
        },
        "hybrid": {
            "edge_mult": 1.20,
            "no_edge_mult": 1.45,
            "extra_drift_floor": 0.08,
            "mid_band_drift_min": 0.42,
        },
        "sniper": {
            "edge_mult": 1.10,
            "no_edge_mult": 1.40,
            "extra_drift_floor": 0.08,
            "mid_band_drift_min": 0.40,
        },
    },
    "low_vol_trend": {
        "_default": {
            "edge_mult": 1.20,
            "flow_full_trust": 0.38,
            "mom_lane_scale": 0.65,
            "strat_lane_scale": 0.45,
            "no_edge_mult": 1.40,
            "extra_drift_floor": 0.08,
            "mid_band_drift_min": 0.40,
        },
        "momentum": {
            "edge_mult": 1.25,
            "flow_full_trust": 0.40,
            "mom_lane_scale": 0.55,
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
            "edge_mult": 1.15,
            "flow_full_trust": 0.32,
            "mom_lane_scale": 0.55,
            "strat_lane_scale": 0.60,
            "no_edge_mult": 1.40,
            "extra_drift_floor": 0.08,
            "mid_band_drift_min": 0.40,
        },
        "mean_reversion": {
            "edge_mult": 1.05,
            "mom_lane_scale": 0.0,
            "strat_lane_scale": 0.70,
            "no_edge_mult": 1.25,
            "mid_band_drift_min": 0.40,
        },
        "momentum": {
            "edge_mult": 1.20,
            "mom_lane_scale": 0.50,
            "strat_lane_scale": 0.55,
            "no_edge_mult": 1.45,
            "mid_band_drift_min": 0.42,
        },
        "phantom": {
            "edge_mult": 1.15,
            "mom_lane_scale": 0.50,
            "strat_lane_scale": 0.55,
            "no_edge_mult": 1.40,
            "mid_band_drift_min": 0.42,
        },
        "hybrid": {
            "edge_mult": 1.15,
            "mom_lane_scale": 0.55,
            "strat_lane_scale": 0.60,
            "no_edge_mult": 1.40,
            "mid_band_drift_min": 0.42,
        },
        "sniper": {
            "edge_mult": 1.10,
            "mom_lane_scale": 0.80,
            "strat_lane_scale": 0.85,
            "no_edge_mult": 1.40,
            "mid_band_drift_min": 0.40,
        },
    },
    "high_vol_chop": {
        "_default": {
            "edge_mult": 1.15,
            "flow_full_trust": 0.32,
            "mom_lane_scale": 0.55,
            "strat_lane_scale": 0.65,
            "no_edge_mult": 1.35,
            "extra_drift_floor": 0.05,
            "mid_band_drift_min": 0.35,
        },
        # Trend bots bleed in chop until live data clears them (style, not size).
        "momentum": {
            "edge_mult": 1.25,
            "flow_full_trust": 0.35,
            "mom_lane_scale": 0.40,
            "strat_lane_scale": 0.55,
            "no_edge_mult": 1.50,
            "extra_drift_floor": 0.08,
            "mid_band_drift_min": 0.40,
        },
        "phantom": {
            "edge_mult": 1.25,
            "flow_full_trust": 0.35,
            "mom_lane_scale": 0.40,
            "strat_lane_scale": 0.55,
            "no_edge_mult": 1.50,
            "extra_drift_floor": 0.08,
            "mid_band_drift_min": 0.40,
        },
        "mean_reversion": {
            "edge_mult": 1.10,
            "mom_lane_scale": 0.0,
            "strat_lane_scale": 0.70,
            "no_edge_mult": 1.40,
            "extra_drift_floor": 0.05,
            "mid_band_drift_min": 0.32,
        },
        "hybrid": {
            "edge_mult": 1.05,
            "mom_lane_scale": 0.60,
            "strat_lane_scale": 0.75,
            "no_edge_mult": 1.25,
            "mid_band_drift_min": 0.30,
        },
        "sniper": {
            "edge_mult": 1.0,
            "mom_lane_scale": 1.0,
            "strat_lane_scale": 1.0,
            "no_edge_mult": 1.15,
            "mid_band_drift_min": 0.28,
        },
    },
    "high_vol_trend": {
        "_default": {
            "edge_mult": 1.15,
            "flow_full_trust": 0.22,
            "mom_lane_scale": 0.85,
            "strat_lane_scale": 0.90,
            "no_edge_mult": 1.10,
            "extra_drift_floor": 0.0,
            "mid_band_drift_min": 0.30,
        },
        "momentum": {
            "edge_mult": 1.25,
            "mom_lane_scale": 0.70,
            "strat_lane_scale": 0.95,
        },
        "mean_reversion": {
            "edge_mult": 1.10,
            "mom_lane_scale": 0.0,
            "strat_lane_scale": 0.80,
            "no_edge_mult": 1.20,
            "mid_band_drift_min": 0.30,
        },
        "phantom": {
            "edge_mult": 1.10,
            "mom_lane_scale": 0.85,
            "strat_lane_scale": 0.95,
        },
        "hybrid": {
            "edge_mult": 1.15,
            "mom_lane_scale": 0.80,
            "strat_lane_scale": 0.90,
        },
        "sniper": {
            "edge_mult": 1.20,
            "mom_lane_scale": 0.85,
            "strat_lane_scale": 1.0,
            "no_edge_mult": 1.10,
            "mid_band_drift_min": 0.35,
        },
    },
}

# Directional strategy types subject to hard regime skip (not arb/makers).
_DIRECTIONAL_TYPES = frozenset({
    "momentum", "mean_reversion", "mean_reversion_sl", "mean_reversion_tp",
    "phantom", "hybrid", "sniper",
    "lag_residual", "regime_specialist", "no_lag", "sweeper",
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
    lo = float(getattr(config, "REGIME_ADAPT_SIZE_MIN", 0.85))
    hi = float(getattr(config, "REGIME_ADAPT_SIZE_MAX", 1.15))
    if good <= bad + 1e-9:
        return 1.0
    t = (wr - bad) / (good - bad)
    t = max(0.0, min(1.0, t))
    mult = lo + t * (hi - lo)
    # Style mode: mild taper only — do not crush size when PnL is slightly red.
    primary = str(getattr(config, "REGIME_ADAPT_PRIMARY", "style") or "style")
    if primary == "throttle" and pnl < 0 and wr < good:
        mult = min(mult, (lo + 1.0) / 2.0)
    elif primary == "style" and pnl < 0 and wr < bad:
        mult = min(mult, max(lo, 0.90))
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
    """Hysteresis: enter block at low WR, clear only after recovery WR.

    Default policy (PLAN 2026-08-05): hard-skip is OFF. When
    ``REGIME_HARD_SKIP_ENABLED`` is True and emergency-only, the bar is the
    high min-trades / low WR constants — not the old n=20 freeze.
    """
    try:
        from arena.regime_settings import get_bool as _reg_bool
        hard_on = bool(_reg_bool("hard_skip"))
    except Exception:
        hard_on = bool(getattr(config, "REGIME_HARD_SKIP_ENABLED", False))
    if not hard_on:
        return {}
    min_n = int(getattr(config, "REGIME_HARD_SKIP_MIN_TRADES", 80))
    enter_wr = float(getattr(config, "REGIME_HARD_SKIP_WR", 0.38))
    clear_wr = float(getattr(config, "REGIME_HARD_SKIP_CLEAR_WR", 0.48))
    need_neg = bool(getattr(config, "REGIME_HARD_SKIP_REQUIRE_NEG_PNL", True))
    perf = _load_perf()
    out: dict[str, bool] = {}
    for label, st in perf.items():
        if not isinstance(st, dict):
            continue
        n = int(st.get("n") or 0)
        if n < min_n:
            # Emergency-only: do not sticky-block on thin samples after reset.
            if (not getattr(config, "REGIME_HARD_SKIP_EMERGENCY_ONLY", True)
                    and prev_blocks.get(str(label))):
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


def _style_skip_enabled() -> bool:
    try:
        from arena.regime_settings import get_bool as _reg_bool
        return bool(_reg_bool("style_skip"))
    except Exception:
        return bool(getattr(config, "REGIME_STYLE_SKIP_ENABLED", True))


def _strategy_block_map(
    prev: dict[tuple[str, str], bool],
) -> dict[tuple[str, str], bool]:
    """Per-(regime, strategy) style skip: fast + slow paths + hysteresis.

    * **Fast enter** — enough fills in the fast window with very low WR + red $.
    * **Slow enter** — long-window bar (overnight-stable).
    * **Clear** — prefer long healthy; also clear if fast healthy with mass
      and long is no longer toxic (prevents sticky block after regime flip).
    """
    if not _style_skip_enabled():
        return {}
    try:
        from arena.regime_stats import snapshot, is_toxic_cell, is_healthy_cell
        blob = snapshot()
    except Exception:
        return {}
    by_strat = blob.get("by_strategy") or {}
    min_n = int(getattr(config, "REGIME_STYLE_SKIP_MIN_TRADES", 18))
    enter_wr = float(getattr(config, "REGIME_STYLE_SKIP_WR", 0.42))
    clear_wr = float(getattr(config, "REGIME_STYLE_SKIP_CLEAR_WR", 0.48))
    fast_n = int(getattr(config, "REGIME_STYLE_SKIP_FAST_MIN_N", 10))
    fast_wr = float(getattr(config, "REGIME_STYLE_SKIP_FAST_WR", 0.38))
    exempt = set(getattr(config, "REGIME_STYLE_SKIP_EXEMPT_TYPES", ()) or ())
    out: dict[tuple[str, str], bool] = {}
    keys: set[tuple[str, str]] = set(prev.keys())
    for reg, strats in by_strat.items():
        for st in strats:
            keys.add((str(reg), str(st)))
    for key in keys:
        reg, st = key
        if st in exempt:
            out[key] = False
            continue
        cell = (by_strat.get(reg) or {}).get(st) or {}
        was = bool(prev.get(key))
        slow_toxic = is_toxic_cell(
            cell, min_n=min_n, wr_bar=enter_wr, require_neg_pnl=True, path="long"
        )
        fast_toxic = is_toxic_cell(
            cell, min_n=fast_n, wr_bar=fast_wr, require_neg_pnl=True, path="fast"
        )
        long_healthy = is_healthy_cell(
            cell, min_n=min_n, wr_clear=clear_wr, path="long"
        )
        fast_healthy = is_healthy_cell(
            cell, min_n=fast_n, wr_clear=clear_wr, path="fast"
        )
        if was:
            if long_healthy or (fast_healthy and not slow_toxic):
                out[key] = False
            else:
                # Sticky while still thin on long or still bad
                n = int(cell.get("n") or 0)
                if n < min_n and not fast_healthy:
                    out[key] = True
                else:
                    still_bad = is_toxic_cell(
                        cell, min_n=min_n, wr_bar=clear_wr,
                        require_neg_pnl=False, path="long",
                    ) or (
                        float(cell.get("pnl") or 0) < 0
                        and (cell.get("wr") or 1.0) < clear_wr
                        and n >= min_n
                    ) or fast_toxic
                    out[key] = bool(still_bad)
        else:
            out[key] = bool(slow_toxic or fast_toxic)
    return out


def _wr_to_edge_mult(wr: Optional[float]) -> float:
    """Map blended WR → continuous edge multiplier (1.0 = neutral)."""
    if wr is None:
        return 1.0
    bad = float(getattr(config, "REGIME_ADAPT_BAD_WR", 0.48))
    good = float(getattr(config, "REGIME_ADAPT_GOOD_WR", 0.62))
    lo = float(getattr(config, "REGIME_ADAPT_CONT_EDGE_MAX", 1.55))  # high tax
    hi = float(getattr(config, "REGIME_ADAPT_CONT_EDGE_MIN", 0.95))  # ease
    if good <= bad + 1e-9:
        return 1.0
    # wr <= bad → lo (raise min_edge); wr >= good → hi
    t = (float(wr) - bad) / (good - bad)
    t = max(0.0, min(1.0, t))
    return lo + t * (hi - lo)


def _wr_to_extra_drift(wr: Optional[float]) -> float:
    if wr is None:
        return 0.0
    bad = float(getattr(config, "REGIME_ADAPT_BAD_WR", 0.48))
    good = float(getattr(config, "REGIME_ADAPT_GOOD_WR", 0.62))
    dmax = float(getattr(config, "REGIME_ADAPT_CONT_DRIFT_MAX", 0.10))
    if float(wr) >= good:
        return 0.0
    if float(wr) <= bad:
        # scale further if much worse than bad
        span = max(0.05, bad - 0.30)
        t = min(1.0, max(0.0, (bad - float(wr)) / span))
        return dmax * (0.5 + 0.5 * t)
    # interpolate bad→good: dmax/2 → 0
    t = (float(wr) - bad) / max(1e-9, good - bad)
    return dmax * 0.5 * (1.0 - max(0.0, min(1.0, t)))


def _wr_to_mid_floor(wr: Optional[float], prior_mid: Optional[float]) -> Optional[float]:
    base = float(prior_mid) if prior_mid is not None else float(
        getattr(config, "MID_COINFLIP_DRIFT_MIN", 0.28)
    )
    if wr is None:
        return prior_mid
    bad = float(getattr(config, "REGIME_ADAPT_BAD_WR", 0.48))
    good = float(getattr(config, "REGIME_ADAPT_GOOD_WR", 0.62))
    mmax = float(getattr(config, "REGIME_ADAPT_CONT_MID_MAX", 0.45))
    if float(wr) >= good:
        return prior_mid
    if float(wr) <= bad:
        return max(base, min(mmax, mmax))
    t = (float(wr) - bad) / max(1e-9, good - bad)
    # wr at bad → mmax; wr at good → base
    return max(base, base + (1.0 - max(0.0, min(1.0, t))) * (mmax - base))


def _live_edge_boost(regime_label: str) -> float:
    """Legacy regime-pool boost; mild when continuous path also active."""
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
        return 1.10  # continuous path carries most of the tax
    if wr >= good:
        return 0.95
    return 1.0


def _side_edge_map(
    regime_label: str, strategy_type: Optional[str],
) -> tuple[dict[str, float], Optional[str]]:
    """Per-side continuous mult + optional binary block_side."""
    side_m = {"yes": 1.0, "no": 1.0}
    block_side: Optional[str] = None
    if not strategy_type:
        return side_m, None
    if not bool(getattr(config, "REGIME_SIDE_SKIP_ENABLED", True)):
        return side_m, None
    try:
        from arena.regime_stats import (
            strategy_side_regime_cell, effective_wr, is_toxic_cell,
        )
    except Exception:
        return side_m, None
    cont_min = int(getattr(config, "REGIME_SIDE_CONT_MIN_N", 8))
    fast_n = int(getattr(config, "REGIME_STYLE_SKIP_FAST_MIN_N", 10))
    fast_wr = float(getattr(config, "REGIME_STYLE_SKIP_FAST_WR", 0.38))
    blocked: list[str] = []
    for side in ("yes", "no"):
        cell = strategy_side_regime_cell(regime_label, strategy_type, side)
        wr = effective_wr(cell, min_n_fast=cont_min)
        mult = _wr_to_edge_mult(wr)
        # Slightly stronger side tax than strategy-level (side is finer)
        if wr is not None and wr < float(getattr(config, "REGIME_ADAPT_BAD_WR", 0.48)):
            mult = min(1.70, mult * 1.05)
        side_m[side] = round(max(0.9, min(1.70, mult)), 3)
        if is_toxic_cell(
            cell, min_n=fast_n, wr_bar=fast_wr, require_neg_pnl=True, path="fast"
        ):
            blocked.append(side)
    # Only binary-block a side when the other side is not also toxic
    # (if both bad, strategy-level skip should handle stand-down).
    if len(blocked) == 1:
        block_side = blocked[0]
    return side_m, block_side


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
    if (now - _cache[0]) < ttl and len(_cache) >= 5:
        return
    try:
        prev_blocks = _cache[3] if isinstance(_cache[3], dict) else {}
        prev_strat = _cache[4] if len(_cache) > 4 and isinstance(_cache[4], dict) else {}
        size_table = _rebuild_size_table()
        perf = _load_perf()
        blocks = _hard_block_map(prev_blocks)
        strat_blocks = _strategy_block_map(prev_strat)
        _cache = (now, size_table, perf, blocks, strat_blocks)
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
    """Full decision adjustments for regime × strategy (data-driven)."""
    try:
        from arena.regime_settings import get_bool as _reg_bool, get_adapt_primary
        if not _reg_bool("adapt_enabled"):
            return RegimeAdjust(reason="disabled")
        primary = get_adapt_primary()
    except Exception:
        if not getattr(config, "REGIME_ADAPT_ENABLED", True):
            return RegimeAdjust(reason="disabled")
        primary = str(getattr(config, "REGIME_ADAPT_PRIMARY", "style") or "style")
    label = str(regime_label or "unknown")
    if label in ("unknown", ""):
        return RegimeAdjust(label=label, reason="unknown_regime")
    # If the live detector is on this same label and it is not yet
    # actionable (thin tape / low conf / just flipped), do not apply
    # style-skip or prior taxes. Unit tests that pass a label with no
    # matching live snapshot still get structural priors.
    try:
        from signals.regime_detector import get_detector
        live = get_detector().snapshot() or {}
        if (live.get("regime_id") or live.get("label")) == label:
            if not live.get("actionable", False):
                return RegimeAdjust(label=label, reason="not_actionable")
    except Exception:
        pass

    _refresh_cache()
    size_table = _cache[1]
    blocks = _cache[3]
    strat_blocks = _cache[4] if len(_cache) > 4 else {}
    size_m = float(size_table.get(label, 1.0))
    prior = _prior_for(label, strategy_type)
    live_boost = _live_edge_boost(label)

    # --- Continuous response from strategy×regime blended WR (primary) ---
    wr_eff: Optional[float] = None
    strat_soft_bad = False
    try:
        from arena.regime_stats import (
            regime_cell, is_toxic_cell, side_regime_cell,
            strategy_regime_cell, effective_wr,
        )
        strat_cell = (
            strategy_regime_cell(label, strategy_type) if strategy_type else {}
        )
        wr_eff = effective_wr(strat_cell) if strat_cell else None
        if wr_eff is None:
            # Fall back to whole-regime cell
            wr_eff = effective_wr(regime_cell(label))
        cont_edge = _wr_to_edge_mult(wr_eff)
        cont_drift = _wr_to_extra_drift(wr_eff)
        bad = float(getattr(config, "REGIME_ADAPT_BAD_WR", 0.48))
        cont_min = int(getattr(config, "REGIME_ADAPT_CONT_MIN_N", 8))
        if wr_eff is not None and wr_eff < bad and (
                int(strat_cell.get("fast_n") or 0) >= cont_min
                or int(strat_cell.get("n") or 0) >= cont_min):
            strat_soft_bad = True
    except Exception:
        cont_edge = 1.0
        cont_drift = 0.0
        strat_cell = {}

    # Prior edge (seed) × mild regime boost × continuous live tax
    prior_edge = float(prior.get("edge_mult", 1.0)) * live_boost
    if primary == "style":
        # Flatten structural prior toward 1.0; continuous path owns the tax.
        prior_edge = 1.0 + 0.35 * (prior_edge - 1.0)
        prior_edge = max(0.92, min(1.25, prior_edge))
    edge_m = prior_edge * cont_edge
    edge_m = max(0.90, min(1.70, edge_m))

    flow_ft = prior.get("flow_full_trust")
    if flow_ft is not None:
        flow_ft = float(flow_ft)

    mid_band = prior.get("mid_band_drift_min")
    if mid_band is not None:
        mid_band = float(mid_band)
    mid_band = _wr_to_mid_floor(wr_eff, mid_band)

    bad_mid = float(getattr(config, "MID_COINFLIP_DRIFT_MIN_BAD_REGIME", 0.35))
    size_floor = float(getattr(config, "REGIME_ADAPT_SIZE_MIN", 0.85))
    reg_toxic = False
    no_is_toxic = False
    try:
        from arena.regime_stats import (
            regime_cell, is_toxic_cell, side_regime_cell,
        )
        reg_cell = regime_cell(label)
        reg_toxic = is_toxic_cell(
            reg_cell,
            min_n=int(getattr(config, "REGIME_STYLE_SKIP_MIN_TRADES", 18)),
            wr_bar=float(getattr(config, "REGIME_ADAPT_BAD_WR", 0.48)),
            path="long",
        ) or is_toxic_cell(
            reg_cell,
            min_n=int(getattr(config, "REGIME_STYLE_SKIP_FAST_MIN_N", 10)),
            wr_bar=float(getattr(config, "REGIME_ADAPT_BAD_WR", 0.48)),
            path="fast",
        )
        no_cell = side_regime_cell(label, "no")
        no_min = int(getattr(config, "REGIME_NO_SIDE_MIN_TRADES", 15))
        no_is_toxic = is_toxic_cell(
            no_cell,
            min_n=no_min,
            wr_bar=float(getattr(config, "REGIME_NO_SIDE_WR", 0.42)),
            path="either",
        )
    except Exception:
        pass

    if primary == "throttle":
        if size_m <= size_floor + 0.05:
            mid_band = max(mid_band or 0.0, bad_mid)
    elif reg_toxic or strat_soft_bad or size_m <= size_floor + 0.05:
        mid_band = max(mid_band or 0.0, bad_mid)

    # Auto per-regime rules when strategy is soft-bad in this regime:
    # raise min_edge further and taper size (data-driven, not hand-named).
    if wr_eff is not None and strat_soft_bad:
        size_m = min(size_m, max(size_floor, 0.85))
        edge_m = max(
            edge_m,
            float(getattr(config, "REGIME_ADAPT_CONT_EDGE_MAX", 1.55)) * 0.92,
        )

    mom_prior = float(prior.get("mom_lane_scale", 1.0))
    strat_prior = float(prior.get("strat_lane_scale", 1.0))
    no_edge = float(prior.get("no_edge_mult", 1.0))
    extra_drift = float(prior.get("extra_drift_floor", 0.0))
    live_toxic = reg_toxic or strat_soft_bad
    if primary == "style":
        blend = 0.35 if not live_toxic else 0.75
        mom_scale = 1.0 + blend * (mom_prior - 1.0)
        strat_scale = 1.0 + blend * (strat_prior - 1.0)
        no_edge = min(
            1.60,
            1.0 + blend * (no_edge - 1.0) + (0.15 if live_toxic else 0.0),
        )
        extra_drift = min(0.12, extra_drift * (0.5 + 0.5 * blend))
    else:
        mom_scale = mom_prior
        strat_scale = strat_prior

    # Continuous drift floor from live WR
    extra_drift = max(extra_drift, cont_drift)
    if strat_soft_bad:
        extra_drift = max(
            extra_drift,
            float(getattr(config, "REGIME_ADAPT_CONT_DRIFT_MAX", 0.10)) * 0.6,
        )

    if no_is_toxic:
        no_edge = max(
            no_edge, float(getattr(config, "REGIME_NO_SIDE_EDGE_MULT", 1.55))
        )
        extra_drift = max(
            extra_drift,
            float(getattr(config, "REGIME_NO_SIDE_EXTRA_DRIFT", 0.06)),
        )

    block = bool(blocks.get(label, False))
    block_strat = False
    if strategy_type and strategy_type in _DIRECTIONAL_TYPES:
        block_strat = bool(strat_blocks.get((label, strategy_type), False))
        seeds = getattr(config, "REGIME_STYLE_SKIP_SEEDS", None) or {}
        if not block_strat and (
            seeds.get((label, strategy_type))
            or seeds.get(f"{label}|{strategy_type}")
        ):
            block_strat = True
    else:
        block = False
        block_strat = False

    side_m, block_side = _side_edge_map(label, strategy_type)
    if strategy_type and strategy_type not in _DIRECTIONAL_TYPES:
        block_side = None
        side_m = {"yes": 1.0, "no": 1.0}

    # Tandem: data-driven heat (WR), not regime-name bandaid alone.
    # 0 = unlimited (paper-eval); do not inject a 1-bot clamp over config 0.
    max_bots = None
    if reg_toxic or block_strat or strat_soft_bad:
        mb = int(getattr(config, "MARKET_SIDE_MAX_BOTS_BAD_REGIME", 0) or 0)
        max_bots = mb if mb > 0 else None
    elif label == "high_vol_chop":
        mb = int(getattr(config, "MARKET_SIDE_MAX_BOTS_CHOP", 0) or 0)
        max_bots = mb if mb > 0 else None

    reason_parts = [
        f"size={size_m:.2f}",
        f"edge×{edge_m:.2f}",
        f"mode={primary}",
    ]
    if wr_eff is not None:
        reason_parts.append(f"wr_eff={wr_eff:.2f}")
    if block:
        reason_parts.append("HARD_SKIP")
    if block_strat:
        reason_parts.append(f"STYLE_SKIP:{strategy_type}")
    if block_side:
        reason_parts.append(f"SIDE_SKIP:{block_side}")
    if reg_toxic:
        reason_parts.append("reg_toxic")
    if strat_soft_bad:
        reason_parts.append("strat_soft_bad")

    size_out = 0.0 if (block or block_strat) else size_m
    return RegimeAdjust(
        size_mult=size_out,
        edge_mult=max(0.5, min(2.5, edge_m)),
        flow_full_trust=flow_ft,
        mom_lane_scale=max(0.0, min(1.5, mom_scale)),
        strat_lane_scale=max(0.0, min(1.5, strat_scale)),
        no_edge_mult=max(1.0, min(2.5, no_edge)),
        extra_drift_floor=0.0,
        block_directional=block,
        block_strategy=block_strat,
        mid_band_drift_min=mid_band,
        max_bots_side=max_bots,
        side_edge_mult=side_m,
        block_side=block_side,
        wr_eff=wr_eff,
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
        strat_blocks = _cache[4] if len(_cache) > 4 else {}
        rows = []
        for label, mult in sorted(table.items()):
            st = perf.get(label) or {}
            n = int(st.get("n") or 0)
            wins = int(st.get("wins") or 0)
            prior = _prior_for(label, None)
            blocked_strats = sorted(
                st for (reg, st), on in strat_blocks.items()
                if on and reg == label
            )
            rows.append({
                "regime": label,
                "n": n,
                "wr": (wins / n) if n else None,
                "pnl": round(float(st.get("pnl") or 0.0), 2),
                "size_mult": mult,
                "hard_skip": bool(blocks.get(label)),
                "style_skip_strategies": blocked_strats,
                "prior": prior,
            })
        return {
            "enabled": bool(getattr(config, "REGIME_ADAPT_ENABLED", True)),
            "hard_skip_enabled": bool(
                getattr(config, "REGIME_HARD_SKIP_ENABLED", False)),
            "style_skip_enabled": _style_skip_enabled(),
            "regimes": rows,
        }
    except Exception as e:
        return {"enabled": False, "error": str(e), "regimes": []}

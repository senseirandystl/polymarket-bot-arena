"""Per-regime × strategy lane weight resolution and seeds.

Resolution order for a lane weight (PLAN 2026-08-05 + soak fix 2026-08-06):
  1. lane_overrides[lane].by_regime[regime][strategy] **only if earned**
     (by_regime_meta marks regime-local sample mass) — OR if it differs from
     the global profile *and* no seed applies
  2. REGIME_PROFILE_SEEDS[regime][strategy][lane] when seeds enabled
  3. lane_overrides[lane].profile[strategy] (global core override)
  4. BaseBot class default profile[strategy][lane]
  5. ``default`` arg (usually 1.0 for non-core lanes)

Why earn-gating: the core tuner previously cloned global elevated weights into
every by_regime cell (same mom=0.50 in high_vol_chop as normal). That silent
clone sat above seeds and killed chop-aware damp. Seeds must win until a cell
is *actually* tuned on that regime's own data.
"""

from __future__ import annotations

from typing import Any, Optional

import config

# Seeds: style starting points until the live per-regime tuner earns overrides.
# Only core directional lanes — candidates stay at override profile (0 until approved).
REGIME_PROFILE_SEEDS: dict[str, dict[str, dict[str, float]]] = {
    "low_vol_range": {
        "momentum": {"drift": 0.70, "mom": 0.15, "strat": 0.15},
        "phantom": {"drift": 0.65, "mom": 0.15, "strat": 0.20},
        "mean_reversion": {"drift": 0.85, "mom": 0.0, "strat": 0.15},
        "mean_reversion_tp": {"drift": 0.85, "mom": 0.0, "strat": 0.15},
        "hybrid": {"drift": 0.60, "mom": 0.10, "strat": 0.30},
        "sniper": {"drift": 0.75, "mom": 0.05, "strat": 0.10},
        "lag_residual": {"drift": 0.80, "mom": 0.05, "strat": 0.15},
        "regime_specialist": {"drift": 0.55, "mom": 0.15, "strat": 0.20},
        "no_lag": {"drift": 0.85, "mom": 0.0, "strat": 0.15},
    },
    "low_vol_trend": {
        "momentum": {"drift": 0.60, "mom": 0.25, "strat": 0.15},
        "phantom": {"drift": 0.55, "mom": 0.25, "strat": 0.20},
        "mean_reversion": {"drift": 0.80, "mom": 0.0, "strat": 0.20},
        "mean_reversion_tp": {"drift": 0.80, "mom": 0.0, "strat": 0.20},
        "hybrid": {"drift": 0.55, "mom": 0.20, "strat": 0.25},
        "sniper": {"drift": 0.70, "mom": 0.10, "strat": 0.10},
        "lag_residual": {"drift": 0.75, "mom": 0.10, "strat": 0.15},
        "regime_specialist": {"drift": 0.55, "mom": 0.25, "strat": 0.15},
        "no_lag": {"drift": 0.80, "mom": 0.0, "strat": 0.20},
    },
    "high_vol_trend": {
        "momentum": {"drift": 0.45, "mom": 0.40, "strat": 0.15},
        "phantom": {"drift": 0.40, "mom": 0.35, "strat": 0.25},
        "mean_reversion": {"drift": 0.70, "mom": 0.0, "strat": 0.30},
        "mean_reversion_tp": {"drift": 0.70, "mom": 0.0, "strat": 0.30},
        "hybrid": {"drift": 0.45, "mom": 0.30, "strat": 0.25},
        "sniper": {"drift": 0.55, "mom": 0.20, "strat": 0.15},
        "lag_residual": {"drift": 0.65, "mom": 0.15, "strat": 0.20},
        "regime_specialist": {"drift": 0.50, "mom": 0.30, "strat": 0.20},
        "no_lag": {"drift": 0.75, "mom": 0.0, "strat": 0.25},
    },
    "high_vol_chop": {
        "momentum": {"drift": 0.65, "mom": 0.10, "strat": 0.15},
        "phantom": {"drift": 0.60, "mom": 0.10, "strat": 0.20},
        "mean_reversion": {"drift": 0.80, "mom": 0.0, "strat": 0.20},
        "mean_reversion_tp": {"drift": 0.80, "mom": 0.0, "strat": 0.20},
        "hybrid": {"drift": 0.60, "mom": 0.10, "strat": 0.20},
        "sniper": {"drift": 0.70, "mom": 0.05, "strat": 0.10},
        "lag_residual": {"drift": 0.75, "mom": 0.05, "strat": 0.20},
        "regime_specialist": {"drift": 0.55, "mom": 0.15, "strat": 0.15},
        "no_lag": {"drift": 0.80, "mom": 0.0, "strat": 0.20},
    },
    "normal": {
        # Empty → fall through to class defaults
    },
}


def seed_weight(regime: Optional[str], strategy: str, lane: str) -> Optional[float]:
    try:
        from arena.regime_settings import get_bool
        if not get_bool("profile_seeds"):
            return None
    except Exception:
        if not getattr(config, "REGIME_PROFILE_SEEDS_ENABLED", True):
            return None
    if not regime or regime in ("unknown", "normal"):
        return None
    try:
        from signals.regime_detector import get_detector
        snap = get_detector().snapshot() or {}
        if (snap.get("regime_id") or snap.get("label")) == regime:
            if not snap.get("actionable", False):
                return None
    except Exception:
        pass
    block = REGIME_PROFILE_SEEDS.get(regime) or {}
    strat = block.get(strategy) or {}
    if lane not in strat:
        return None
    return float(strat[lane])


def _by_regime_earned(
    ov: dict,
    regime: str,
    strategy_type: str,
    lane: str,
) -> bool:
    """True when meta says this by_regime cell was tuned on regime-local data."""
    meta_root = ov.get("by_regime_meta") or {}
    reg_meta = meta_root.get(regime) or {}
    # Support nested {strategy: {lane: {...}}} or flat {strategy: {...}}
    cell = reg_meta.get(strategy_type)
    if cell is None:
        return False
    if isinstance(cell, dict) and lane in cell and isinstance(cell[lane], dict):
        cell = cell[lane]
    if not isinstance(cell, dict):
        return False
    if not cell.get("earned"):
        return False
    min_n = int(getattr(config, "CORE_TUNE_MIN_TRADES_REGIME", 40))
    return int(cell.get("n") or 0) >= min_n


def resolve_lane_weight(
    lane: str,
    strategy_type: str,
    regime: Optional[str],
    *,
    profile: dict,
    overrides: Optional[dict] = None,
    default: float = 1.0,
) -> float:
    """Resolve effective weight for one lane under current regime.

    Priority: earned by_regime → seed → global override → class profile → default.
    Unearned / clone by_regime cells do not shadow seeds.
    """
    ov = (overrides or {}).get(lane) if overrides else None
    seed = seed_weight(regime, strategy_type, lane)
    global_w: Optional[float] = None
    if ov and ov.get("enabled"):
        prof = ov.get("profile") or {}
        if strategy_type in prof:
            global_w = float(prof[strategy_type])

    if ov and ov.get("enabled") and regime:
        by_reg = ov.get("by_regime") or {}
        reg_prof = by_reg.get(regime) if isinstance(by_reg, dict) else None
        if isinstance(reg_prof, dict) and strategy_type in reg_prof:
            by_w = float(reg_prof[strategy_type])
            earned = _by_regime_earned(ov, regime, strategy_type, lane)
            if earned:
                return by_w
            # Clone of global (or unearned): do not shadow seed
            is_clone = (
                global_w is not None and abs(by_w - float(global_w)) < 1e-9
            )
            if seed is not None and lane in ("drift", "mom", "strat"):
                # Prefer seed until cell is earned on regime-local data
                return float(seed)
            if not is_clone:
                # Distinct unearned override still usable when no seed
                return by_w
            # clone + no seed → fall through to global

    if seed is not None and lane in ("drift", "mom", "strat"):
        return float(seed)

    if global_w is not None:
        return float(global_w)

    if ov and ov.get("enabled"):
        # Core override that omits strategy → 0
        if ov.get("core"):
            return 0.0
        return 0.0

    if lane in profile:
        return float(profile[lane])
    return float(default)


def resolve_profile_weights(
    strategy_type: str,
    regime: Optional[str],
    lanes: dict,
    profile: dict,
    overrides: Optional[dict] = None,
) -> dict[str, float]:
    """Map every lane key → weight for blend."""
    return {
        k: resolve_lane_weight(
            k, strategy_type, regime,
            profile=profile, overrides=overrides, default=1.0,
        )
        for k in lanes
    }

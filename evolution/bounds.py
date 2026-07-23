"""Sensible parameter bounds for Gaussian mutation clamping.

Unknown numeric keys fall back to a relative band around the current value
so mutation never invents unbounded junk. Booleans and strings are not
mutated by the Gaussian operator (they pass through crossover only).
"""

from __future__ import annotations

from typing import Any

# Absolute bounds for known evolvable params across strategies.
# Keys not listed use relative bounds derived from the current value.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    # Shared sizing / confidence
    "position_size_pct": (0.01, 0.15),
    "min_confidence": (0.05, 0.85),
    # Momentum / phantom / sniper
    "lookback_candles": (3, 40),
    "momentum_threshold": (5e-5, 5e-3),
    "trend_strength_weight": (0.1, 0.95),
    "volume_weight": (0.05, 0.9),
    "regime_conf_weight": (0.0, 0.8),
    # Mean reversion
    "bb_std_dev": (1.0, 3.5),
    "rsi_period": (5, 28),
    "rsi_oversold": (20, 45),
    "rsi_overbought": (55, 80),
    "reversion_threshold": (0.15, 1.5),
    "min_drift": (0.0, 0.45),
    "trending_conf_damp": (0.0, 0.9),
    # Sniper zones
    "min_price_yes": (0.20, 0.55),
    "max_price_yes": (0.55, 0.90),
    "max_price_no": (0.10, 0.45),
    "skip_zone_low": (0.35, 0.55),
    "skip_zone_high": (0.55, 0.75),
    "quiet_drift_bump": (0.0, 0.15),
    # Phantom
    "ema_fast": (3, 20),
    "ema_slow": (10, 60),
    "atr_period": (5, 30),
    "breakout_lookback": (5, 30),
    "min_atr_pct": (5e-5, 2e-3),
    "max_atr_pct": (2e-3, 0.05),
    # Hybrid base weights (if present as flat floats)
    "w_momentum": (0.0, 1.0),
    "w_mean_reversion": (0.0, 1.0),
    "w_phantom": (0.0, 1.0),
    "w_sentiment": (0.0, 1.0),
}

# Relative half-width for unknown numeric keys: value ± RELATIVE_BAND * |value|
# (with a floor so near-zero params still have room to move).
RELATIVE_BAND = 0.50
RELATIVE_FLOOR = 0.01


def bounds_for(key: str, value: float) -> tuple[float, float]:
    """Return (lo, hi) for a param key given its current value."""
    if key in PARAM_BOUNDS:
        return PARAM_BOUNDS[key]
    mag = abs(float(value))
    half = max(RELATIVE_FLOOR, mag * RELATIVE_BAND)
    lo = float(value) - half
    hi = float(value) + half
    # Keep positive-ish for thresholds that are almost always > 0
    if float(value) > 0 and lo <= 0:
        lo = min(RELATIVE_FLOOR * 0.1, mag * 0.05)
    return (lo, hi)


def clamp(key: str, value: Any, reference: Any = None) -> Any:
    """Clamp a mutated value into its bounds; preserve int/float type."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    ref = reference if isinstance(reference, (int, float)) else value
    lo, hi = bounds_for(key, float(ref))
    clamped = max(lo, min(hi, float(value)))
    if isinstance(ref, int) and not isinstance(ref, bool):
        return max(int(lo) if lo >= 1 else 1, int(round(clamped)))
    return round(clamped, 6)


def is_numeric_gene(value: Any) -> bool:
    """True for evolvable numeric genes (not bool)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)

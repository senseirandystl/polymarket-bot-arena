"""Basic regime indicators: trend strength + choppiness (pure, deterministic).

Complements signals/volatility_regime.py (which is wired to the live candle
stream and labels regimes for HybridBot). This module is the pure feature
layer: explicit price-list in, bounded floats out, no state — usable by both
the live builder and the offline harness with identical results.

All outputs are CONTEXT (non-directional, 0..1) — they describe HOW the tape
is moving, never which way. The harness uses them to measure regime-specific
value ("is drift only profitable when trending?"); strategy code may scale
selectivity by them.

Outputs:
- ``regime_trend_10`` / ``regime_trend_30``: Kaufman efficiency ratio (net
  move / path length) over 10 and 30 candles, smooth-ramped so a random walk
  reads ~0 and a straight line reads ~1.
- ``regime_chop``: Choppiness-index analog over CHOP_WINDOW candles —
  log(path / range) / log(n), clamped to 0..1. High = rangebound churn,
  low = directional.
- ``regime_trend``: headline trend-strength — mean of the two ER reads.
"""

import math

from signals.curves import smooth_ramp

TREND_WINDOWS = (10, 30)
CHOP_WINDOW = 14


def efficiency_ratio(prices: list, window: int) -> float:
    """Kaufman ER over the last ``window`` candles, smooth-ramped to 0..1.

    Raw ER = |net move| / path length. A random walk over N steps sits near
    1/sqrt(N); the ramp maps [1/sqrt(N), 0.6] onto [0, 1] so "trend" means
    meaningfully-better-than-random, not merely nonzero.
    """
    w = prices[-(window + 1):]
    if len(w) < 3:
        return 0.0
    net = abs(w[-1] - w[0])
    path = sum(abs(w[i] - w[i - 1]) for i in range(1, len(w)))
    if path <= 0:
        return 0.0
    random_walk = 1.0 / math.sqrt(len(w) - 1)
    return smooth_ramp(net / path, random_walk, 0.6)


def choppiness(prices: list, window: int = CHOP_WINDOW) -> float:
    """Choppiness analog in 0..1: log(path/range)/log(n). High = churn."""
    w = prices[-(window + 1):]
    if len(w) < 3:
        return 0.0
    path = sum(abs(w[i] - w[i - 1]) for i in range(1, len(w)))
    rng = max(w) - min(w)
    if path <= 0 or rng <= 0:
        return 0.0
    val = math.log(path / rng) / math.log(len(w) - 1)
    return max(0.0, min(1.0, val))


def compute(prices: list) -> dict:
    """All regime features from closed 1m closes (oldest first)."""
    clean = [p for p in (prices or []) if p and p > 0]
    ers = {w: efficiency_ratio(clean, w) for w in TREND_WINDOWS}
    return {
        "regime_trend_10": ers[10],
        "regime_trend_30": ers[30],
        "regime_trend": (ers[10] + ers[30]) / 2.0,
        "regime_chop": choppiness(clean),
    }

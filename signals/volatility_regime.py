"""Volatility-regime estimation from the live BTC candle stream.

Pure local computation over the PriceFeed's closed 1m candles — zero network,
safe on the 1s hot path. This is *context*, not a directional signal: it never
says YES or NO, only how violently BTC is moving right now. Consumers:

- HybridBot's regime-switching meta-learner (weights trend-following up in
  trending regimes, the drift/fade book up in chop),
- future sizing/selectivity work (low-vol = no edge to harvest).

Outputs (all smooth, no buckets):
- ``realized_vol``: stdev of 1m log-returns over the window (fraction/min).
- ``atr_pct``: mean absolute 1m move as a fraction of price.
- ``vol_score``: realized vol on a smooth 0..1 scale (0.5 at the calibrated
  typical BTC 1m vol; see VOL_TYPICAL).
- ``trend_score``: 0..1 trendiness — |net move| / path length over the
  window (1.0 = straight line, ~0 = pure chop), smoothed.
- ``regime``: convenience label ("quiet" / "normal" / "trending" /
  "volatile") derived from the smooth scores — for logs/dashboard only;
  strategy code should consume the continuous scores.
"""

import math

from signals.curves import sigmoid, smooth_ramp

WINDOW = 20            # 1m candles considered (20 min lookback)
VOL_TYPICAL = 0.0006   # typical BTC 1m log-return stdev (~0.06%): vol_score 0.5
VOL_STEEPNESS = 3000.0  # sigmoid steepness around VOL_TYPICAL


def compute(prices: list) -> dict:
    """Regime metrics from a list of closed 1m closes (oldest first)."""
    if not prices or len(prices) < 5:
        return {"realized_vol": 0.0, "atr_pct": 0.0, "vol_score": 0.0,
                "trend_score": 0.0, "regime": "unknown"}

    window = [p for p in prices[-WINDOW:] if p and p > 0]
    if len(window) < 5:
        return {"realized_vol": 0.0, "atr_pct": 0.0, "vol_score": 0.0,
                "trend_score": 0.0, "regime": "unknown"}

    rets = [math.log(window[i] / window[i - 1]) for i in range(1, len(window))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    realized_vol = math.sqrt(var)
    atr_pct = sum(abs(r) for r in rets) / len(rets)

    # Trendiness: net displacement over total path length (efficiency ratio).
    net = abs(window[-1] - window[0])
    path = sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))
    efficiency = (net / path) if path > 0 else 0.0
    # Random-walk efficiency over N steps ~ 1/sqrt(N) (~0.23 at N=19); smooth
    # ramp from there to 0.6 (strongly directional) instead of a hard cutoff.
    trend_score = smooth_ramp(efficiency, 0.2, 0.6)

    vol_score = sigmoid(realized_vol, center=VOL_TYPICAL,
                        steepness=VOL_STEEPNESS)

    if vol_score < 0.3:
        regime = "quiet"
    elif trend_score > 0.5:
        regime = "trending"
    elif vol_score > 0.7:
        regime = "volatile"
    else:
        regime = "normal"

    return {
        "realized_vol": realized_vol,
        "atr_pct": atr_pct,
        "vol_score": vol_score,
        "trend_score": trend_score,
        "regime": regime,
    }

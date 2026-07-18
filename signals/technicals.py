"""Advanced technicals over the live BTC 1m candle stream.

Pure local computation (zero network, hot-path safe). Every output is a
smooth, bounded score — no buckets:

- ``macd_score``: MACD histogram (12/26/9 EMAs on 1m closes), soft-saturated
  as a fraction of price so it is scale-free.
- ``bb_score``: Bollinger %B recentred to [-1, 1] (position inside the
  20-period +/-2-sigma band; beyond-band values saturate smoothly).
- ``mtf_score``: multi-timeframe alignment — the tanh-mean of 1m / 3m / 5m
  momentum reads. Near +/-1 only when all horizons agree.

LANE IS KILL-SWITCHED (config.SIGNAL_WEIGHT_TECH = 0) until the offline
harness measures positive NET edge for any of these on the 5-min markets
(price-history indicators were exactly the class the orderflow research
ranked below flow reads — prove them before weighting them).
"""

from signals.curves import soft_saturate

MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_PERIOD, BB_SIGMA = 20, 2.0
MACD_SCALE = 0.0004    # histogram of 0.04% of price reads ~0.76
MTF_SCALE = 0.0015     # 0.15% move over a horizon reads ~0.76


def _ema(values: list, period: int) -> list:
    if not values:
        return []
    k = 2.0 / (period + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1.0 - k))
    return out


def macd_score(prices: list) -> float:
    """MACD histogram as a smooth scale-free score in (-1, 1)."""
    if len(prices) < MACD_SLOW + MACD_SIGNAL:
        return 0.0
    fast = _ema(prices, MACD_FAST)
    slow = _ema(prices, MACD_SLOW)
    macd_line = [f - s for f, s in zip(fast, slow)]
    signal_line = _ema(macd_line, MACD_SIGNAL)
    hist = macd_line[-1] - signal_line[-1]
    price = prices[-1]
    if price <= 0:
        return 0.0
    return soft_saturate(hist / price, MACD_SCALE)


def bollinger_score(prices: list) -> float:
    """Bollinger %B recentred: -1 at the lower band, +1 at the upper."""
    if len(prices) < BB_PERIOD:
        return 0.0
    window = prices[-BB_PERIOD:]
    mean = sum(window) / len(window)
    var = sum((p - mean) ** 2 for p in window) / len(window)
    sd = var ** 0.5
    if sd <= 0:
        return 0.0
    # z / sigma-band: +/-1 at the band edges; tanh keeps beyond-band smooth.
    return soft_saturate((prices[-1] - mean) / (BB_SIGMA * sd), 1.0)


def multi_timeframe_score(prices: list) -> float:
    """Alignment of 1m / 3m / 5m momentum: mean of per-horizon tanh reads."""
    if len(prices) < 6 or prices[-1] <= 0:
        return 0.0
    scores = []
    for horizon in (1, 3, 5):
        if len(prices) > horizon and prices[-1 - horizon] > 0:
            move = (prices[-1] - prices[-1 - horizon]) / prices[-1 - horizon]
            # Longer horizons see proportionally larger moves; scale by sqrt(h)
            # so each horizon is judged against its own typical magnitude.
            scores.append(soft_saturate(move, MTF_SCALE * (horizon ** 0.5)))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def compute(prices: list) -> dict:
    """All technical scores from a list of closed 1m closes (oldest first)."""
    clean = [p for p in (prices or []) if p and p > 0]
    return {
        "macd_score": macd_score(clean),
        "bb_score": bollinger_score(clean),
        "mtf_score": multi_timeframe_score(clean),
    }

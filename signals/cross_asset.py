"""Cross-asset lead/lag context from the shared Binance candle stream.

Majors (ETH, SOL) sometimes front-run or confirm a BTC move. This module
computes a smooth cross-asset confirmation score from the SAME PriceFeed
websocket the BTC lane already uses — zero extra sockets or requests.

Output ``xasset_score`` in (-1, 1): the tanh-mean of ETH and SOL 1-candle
momentum. Positive = the majors are moving up alongside/ahead of BTC.

LANE IS KILL-SWITCHED (config.SIGNAL_WEIGHT_XASSET = 0) until the offline
harness measures positive NET edge — a confirmation read on assets the market
also watches is exactly the "predictive but priced-in" trap (see pm_mom,
BUG #26).
"""

from signals.curves import soft_saturate

XASSET_SCALE = 0.002   # 0.2% 1-candle move on a major reads ~0.76
PEERS = ("eth", "sol")


def compute(price_feed) -> dict:
    """Cross-asset score from a PriceFeed instance (or None)."""
    if price_feed is None:
        return {"xasset_score": 0.0}

    scores = []
    for sym in PEERS:
        sig = price_feed.get_signals(sym)
        prices = sig.get("prices", [])
        latest = sig.get("latest", 0) or 0
        if sig.get("stale") or len(prices) < 1 or prices[-1] <= 0:
            continue
        ref = prices[-2] if len(prices) >= 2 else prices[-1]
        cur = latest if latest > 0 else prices[-1]
        if ref > 0:
            scores.append(soft_saturate((cur - ref) / ref, XASSET_SCALE))

    if not scores:
        return {"xasset_score": 0.0}
    return {"xasset_score": sum(scores) / len(scores)}

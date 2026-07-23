"""Synthetic order books for backtest fills.

Polymarket does not archive historical order-book DEPTH, only the price
history — so backtest fills walk a synthetic ask ladder anchored on the
recorded mid. The ladder shape (half-spread + tiered depth) comes from
config.BACKTEST_* so the assumed liquidity is explicit and tunable; the
walk itself is the production :func:`polymarket_fills.simulate_fill`, so
slippage/fee math is identical to the paper venue.
"""

from __future__ import annotations

import config


def synth_book(side_mid: float) -> dict:
    """Normalized book (asks only) for buying one side priced at ``side_mid``.

    Asks start at ``mid + BACKTEST_HALF_SPREAD`` and step up per depth tier,
    clamped to (0.01, 0.99). Shape matches polymarket_markets' normalized
    books (``valid``/``asks``/``best_ask``/``min_order_size``) so any fill
    helper accepts it.
    """
    if side_mid is None or not (0.0 < side_mid < 1.0):
        return {"valid": False, "asks": [], "bids": []}
    best_ask = side_mid + config.BACKTEST_HALF_SPREAD
    asks = []
    for offset, shares in config.BACKTEST_BOOK_DEPTH:
        price = round(min(0.99, max(0.01, best_ask + offset)), 4)
        asks.append((price, float(shares)))
    return {
        "valid": True,
        "asks": asks,
        "bids": [],
        "best_ask": asks[0][0],
        "best_bid": round(max(0.01, side_mid - config.BACKTEST_HALF_SPREAD), 4),
        "min_order_size": config.POLYMARKET_MIN_SHARES,
    }

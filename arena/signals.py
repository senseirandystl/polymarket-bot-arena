"""Combine signal feeds + market orderflow into the dict bots consume.

Pure function — does not start any threads and does not open any sockets
itself.  The Trader thread calls this on every tick (1s) and the
secondary-bot hook calls it on every discovery cycle (60s); both feed the
result into ``bot.make_decision(market, signals)``.
"""

import logging
from typing import Optional

from signals.price_feed import PriceFeed
from signals.sentiment import SentimentFeed
from signals.polymarket_prices import PolymarketPriceFeed

logger = logging.getLogger(__name__)


def build_combined_signals(
    price_feed: Optional[PriceFeed],
    sentiment_feed: Optional[SentimentFeed],
    polymarket_price_feed: Optional[PolymarketPriceFeed],
    market: Optional[dict],
) -> dict:
    """Build the signals dict ``BaseBot.make_decision`` expects.

    Hoists ``market["orderflow"]`` (set by ``MarketDiscovery`` once per
    scan) up to the top level as ``signals["orderflow"]`` so the bot can
    read it via ``signals.get("orderflow", {})``.  Polymarket in-market
    price momentum is keyed as ``"pm_momentum"`` + ``"pm_prices"`` (same
    shape the bot has been seeing since v7.1).
    """
    price_signals = (
        price_feed.get_signals("btc") if price_feed is not None else {}
    )
    sent_signals = (
        sentiment_feed.get_signals("btc") if sentiment_feed is not None else {}
    )

    pm_data: dict = {}
    if market is not None and polymarket_price_feed is not None:
        yes_token = market.get("polymarket_token_id", "") or ""
        if yes_token:
            pm_data = polymarket_price_feed.get_momentum(yes_token)
    pm_signals = {
        "pm_momentum": float(pm_data.get("momentum", 0.0) or 0.0),
        "pm_prices": pm_data.get("prices", []) or [],
    }

    orderflow = {}
    if market is not None:
        orderflow = market.get("orderflow", {}) or {}

    return {
        **price_signals,
        **sent_signals,
        "orderflow": orderflow,
        **pm_signals,
    }

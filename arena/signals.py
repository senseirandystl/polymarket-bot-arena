"""Combine signal feeds + market orderflow into the dict bots consume.

Pure function — does not start any threads and does not open any sockets
itself.  The Trader thread calls this on every tick (1s) and the
secondary-bot hook calls it on every discovery cycle (20s); both feed the
result into ``bot.make_decision(market, signals)``.
"""

import logging
from typing import Optional

from signals.price_feed import PriceFeed
from signals.sentiment import SentimentFeed
from signals.polymarket_prices import PolymarketPriceFeed
from signals.orderflow_signals import get_cvd_feed
from signals.strike import get_strike_registry, drift_signal

logger = logging.getLogger(__name__)


def build_combined_signals(
    price_feed: Optional[PriceFeed],
    sentiment_feed: Optional[SentimentFeed],
    polymarket_price_feed: Optional[PolymarketPriceFeed],
    market: Optional[dict],
    warm: Optional[dict] = None,
) -> dict:
    """Build the signals dict ``BaseBot.make_decision`` expects.

    Hoists ``market["orderflow"]`` up to the top level as
    ``signals["orderflow"]`` so the bot can read it via
    ``signals.get("orderflow", {})``.  Polymarket in-market price momentum is
    keyed as ``"pm_momentum"`` + ``"pm_prices"``.

    When ``warm`` (a market-data-warmer snapshot) is supplied — the Trader's 1s
    hot path — OBI, CVD and PM momentum are read straight from it, so this
    function makes **zero network calls**. Without ``warm`` (the 20s maker
    hook) it falls back to fetching CVD / PM momentum through their coalescing
    feeds, exactly as before.
    """
    price_signals = (
        price_feed.get_signals("btc") if price_feed is not None else {}
    )
    sent_signals = (
        sentiment_feed.get_signals("btc") if sentiment_feed is not None else {}
    )

    orderflow = {}
    if market is not None:
        orderflow = market.get("orderflow", {}) or {}

    if warm is not None:
        # Warm path: everything is already <=1s fresh from the warmer thread.
        obi = float(warm.get("obi", 0.0) or 0.0)
        cvd = float(warm.get("cvd", 0.0) or 0.0)
        pm_signals = {
            "pm_momentum": float(warm.get("pm_momentum", 0.0) or 0.0),
            "pm_prices": warm.get("pm_prices", []) or [],
        }
    else:
        # Cold path (maker 20s hook): fetch via the coalescing feeds.
        pm_data: dict = {}
        if market is not None and polymarket_price_feed is not None:
            yes_token = market.get("polymarket_token_id", "") or ""
            if yes_token:
                pm_data = polymarket_price_feed.get_momentum(yes_token)
        pm_signals = {
            "pm_momentum": float(pm_data.get("momentum", 0.0) or 0.0),
            "pm_prices": pm_data.get("prices", []) or [],
        }
        obi = float(orderflow.get("obi", 0.0) or 0.0)
        cvd = 0.0
        if market is not None:
            cond = market.get("condition_id") or market.get("id")
            if cond:
                cvd = get_cvd_feed().get_cvd(cond)

    # BTC drift from the window's "price to beat" (strike). Snapshot the strike
    # at first LIVE sighting, then measure how far BTC has drifted. Regime-
    # agnostic directional fundamental; 0.0 until a strike is captured.
    btc_latest = float(price_signals.get("latest", 0.0) or 0.0)
    btc_drift = 0.0
    btc_strike = None
    if market is not None and btc_latest > 0:
        mkt_id = market.get("id") or market.get("market_id")
        tr = market.get("time_remaining_seconds")
        reg = get_strike_registry()
        reg.observe(mkt_id, btc_latest, tr)
        btc_strike = reg.strike(mkt_id)
        btc_drift = drift_signal(btc_strike, btc_latest, tr)

    return {
        **price_signals,
        **sent_signals,
        "orderflow": orderflow,
        "obi": obi,
        "cvd": cvd,
        "btc_drift": btc_drift,
        "btc_strike": btc_strike,
        **pm_signals,
    }

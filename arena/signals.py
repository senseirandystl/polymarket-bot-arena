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
from signals import cross_asset, technicals, volatility_regime
from signals.futures_meta import get_feed as get_futures_feed
from signals.macro_calendar import macro_caution

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

    orderflow: dict = {}
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

    # BTC drift from the window's "price to beat" (official Polymarket openPrice
    # / Chainlink at eventStartTime). Warm path reads the strike the warmer fetched;
    # cold path (maker) fetches via the registry (cached). Regime-agnostic
    # fundamental; 0.0 until a strike is available.
    btc_latest = float(price_signals.get("latest", 0.0) or 0.0)
    btc_drift = 0.0
    btc_strike = None
    if market is not None and btc_latest > 0:
        mkt_id = market.get("id") or market.get("market_id")
        tr = market.get("time_remaining_seconds")
        if warm is not None and warm.get("strike"):
            btc_strike = warm.get("strike")
        else:
            btc_strike = get_strike_registry().get_strike(
                mkt_id, market.get("event_start_time"))
        btc_drift = drift_signal(btc_strike, btc_latest, tr)

    # --- Context + candidate lanes (all local compute or cached feeds) ---
    # Volatility regime + technicals + cross-asset ride the candle stream the
    # feed already holds (zero network); futures meta returns its background
    # thread's cached snapshot. Candidate DIRECTIONAL lanes (futures,
    # technicals, cross-asset) are kill-switched at weight 0 in config until
    # the offline harness validates positive net edge; regime/macro are
    # non-directional context and are consumed directly (hybrid weighting,
    # selectivity).
    btc_prices = price_signals.get("prices", []) or []
    # Base vol/trend scores (pure, local) — still the continuous inputs
    # HybridBot and others read for tilt; the regime detector builds on them.
    vol_base = volatility_regime.compute(btc_prices)
    tech = technicals.compute(btc_prices)
    xasset = cross_asset.compute(price_feed)
    try:
        fut_feed = get_futures_feed()
        fut_feed.start()  # idempotent — first call boots the refresh thread
        futures = fut_feed.get_signals()
    except Exception as e:  # a broken lane must never stall a trading tick
        logger.debug(f"futures meta read failed: {e}")
        futures = {"funding": 0.0, "oi_delta": 0.0, "taker_delta": 0.0,
                   "stale": True}

    # Robust multi-feature regime (online EMA + hysteresis + optional
    # centroids). Continuous: updates every tick, not only at resolution.
    # Soft market_id stamp logs window rollovers without resetting state.
    market_regime: dict = {}
    try:
        from signals.regime_detector import get_detector
        market_regime = get_detector().update(
            btc_prices,
            cvd=cvd,
            obi=obi,
            vol_score=vol_base.get("vol_score"),
            trend_score=vol_base.get("trend_score"),
            realized_vol=vol_base.get("realized_vol"),
            market_id=(
                (market.get("id") or market.get("market_id"))
                if market is not None else None
            ),
        )
    except Exception as e:
        logger.debug(f"regime detector update failed: {e}")
        market_regime = {
            "regime_id": "unknown", "label": "unknown",
            "regime": vol_base.get("regime", "unknown"),
            "legacy": vol_base.get("regime", "unknown"),
            "confidence": 0.0, "features": {},
            "vol_score": vol_base.get("vol_score", 0.0),
            "trend_score": vol_base.get("trend_score", 0.0),
            "known": False,
        }

    # Enrich vol_regime so existing consumers (SignalView.vol_regime,
    # HybridBot, sniper quiet check) see both legacy + rich fields without
    # a second lookup. Detector's legacy maps onto quiet/normal/trending/
    # volatile; regime_id holds the rich label.
    regime = {
        **vol_base,
        "regime": market_regime.get("legacy") or vol_base.get("regime", "unknown"),
        "regime_id": market_regime.get("regime_id", "unknown"),
        "confidence": market_regime.get("confidence", 0.0),
        "features": market_regime.get("features") or {},
        "meta_bucket": market_regime.get("meta_bucket", "mixed"),
        "mom_score": market_regime.get("mom_score", 0.0),
        "flow_score": market_regime.get("flow_score", 0.0),
    }

    return {
        **price_signals,
        **sent_signals,
        "orderflow": orderflow,
        "obi": obi,
        "cvd": cvd,
        "btc_drift": btc_drift,
        "btc_strike": btc_strike,
        "vol_regime": regime,
        "market_regime": market_regime,
        "technicals": tech,
        "xasset": xasset.get("xasset_score", 0.0),
        "futures": futures,
        "macro_caution": macro_caution(),
        **pm_signals,
    }

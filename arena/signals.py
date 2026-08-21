"""Combine signal feeds + market orderflow into the dict bots consume.

Pure function — does not start any threads and does not open any sockets
itself.  The Trader thread calls this on every tick (1s) and the
secondary-bot hook calls it on every discovery cycle (20s); both feed the
result into ``bot.make_decision(market, signals)``.
"""

import logging
import time
from typing import Optional

import config
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

    # --- Context + candidate lanes (all local compute or cached feeds) ---
    # Volatility regime + technicals + cross-asset ride the candle stream the
    # feed already holds (zero network); futures meta returns its background
    # thread's cached snapshot. Candidate DIRECTIONAL lanes (futures,
    # technicals, cross-asset) are kill-switched at weight 0 in config until
    # the offline harness validates positive net edge; regime/macro are
    # non-directional context and are consumed directly (hybrid weighting,
    # selectivity).
    btc_prices = price_signals.get("prices", []) or []
    # Adaptive drift scale: prefer TWAP ticks (same object as moneyness), else
    # spot 1m closes (signals/drift_scale.py). Update *before* btc_drift.
    drift_vol_scale = float(getattr(config, "DRIFT_VOL_SCALE", 0.0022) or 0.0022)
    drift_scale_source = "prior"
    try:
        from signals.drift_scale import (
            get_drift_scale_estimator, update_estimator_from_feeds,
        )
        twap_ticks = []
        if price_feed is not None and hasattr(price_feed, "btc_twap_ticks"):
            try:
                twap_ticks = price_feed.btc_twap_ticks()
            except Exception:
                twap_ticks = []
        drift_vol_scale = float(update_estimator_from_feeds(
            twap_ticks=twap_ticks, spot_prices=btc_prices,
        ))
        drift_scale_source = get_drift_scale_estimator().last_source()
    except Exception as e:
        logger.debug(f"drift scale update failed: {e}")

    # BTC drift from the window's "price to beat" (official Polymarket openPrice
    # / TWAP at eventStartTime). Under TWAP resolution both open and settlement
    # are Chainlink TWAP prints (``TWAP_WINDOW_SEC``, 60s for 5m) — so
    # ``btc_now`` for moneyness is the official rolling TWAP (and a settlement
    # nowcast inside the final window). Warm path reads the strike the warmer
    # fetched; cold path fetches via the registry. Regime-agnostic fundamental;
    # 0.0 until a strike is available.
    btc_spot = float(price_signals.get("latest", 0.0) or 0.0)
    btc_twap = float(price_signals.get("twap", 0.0) or 0.0)
    btc_latest = btc_spot  # keep candle-path "latest" = spot for mom consumers
    btc_drift = 0.0
    btc_strike = None
    btc_drift_pct = 0.0
    btc_implied_yes = 0.5
    btc_drift_z = 0.0
    resolution_meta: dict = {
        "btc_now": 0.0,
        "source": "none",
        "rtds_twap": btc_twap if btc_twap > 0 else None,
        "spot": btc_spot if btc_spot > 0 else None,
        "nowcast": None,
        "in_settlement_window": False,
        "nowcast_coverage": 0.0,
        "nowcast_frac_elapsed": 0.0,
        "twap_certainty": 0.0,
    }
    if market is not None:
        mkt_id = market.get("id") or market.get("market_id")
        tr = market.get("time_remaining_seconds")
        if warm is not None and warm.get("strike"):
            btc_strike = warm.get("strike")
        else:
            btc_strike = get_strike_registry().get_strike(
                mkt_id, market.get("event_start_time"))

        # Build resolution btc_now (TWAP / nowcast / spot fallback).
        try:
            from signals import twap as twap_mod
            now_epoch = time.time()
            expiry_epoch = None
            if tr is not None:
                try:
                    expiry_epoch = now_epoch + float(tr)
                except (TypeError, ValueError):
                    expiry_epoch = None
            # Prefer resolves_at when present (more accurate than rem clock).
            ra = market.get("resolves_at") or market.get("end_time")
            if ra is not None:
                try:
                    if isinstance(ra, (int, float)):
                        expiry_epoch = float(ra)
                        if expiry_epoch > 1e12:
                            expiry_epoch /= 1000.0
                    else:
                        from datetime import datetime, timezone
                        expiry_epoch = datetime.fromisoformat(
                            str(ra).replace("Z", "+00:00")
                        ).timestamp()
                except Exception:
                    pass

            ticks = []
            if price_feed is not None:
                try:
                    # Settlement is a TWAP object — do not mix in denser spot
                    # ticks (that averages the wrong series vs TWAP-open strike).
                    if hasattr(price_feed, "btc_twap_ticks"):
                        ticks = list(price_feed.btc_twap_ticks() or [])
                    if not ticks and hasattr(price_feed, "btc_spot_ticks"):
                        ticks = list(price_feed.btc_spot_ticks() or [])
                except Exception:
                    ticks = []

            resolution_meta = twap_mod.resolution_btc_now(
                rtds_twap=btc_twap if btc_twap > 0 else None,
                spot=btc_spot if btc_spot > 0 else None,
                time_remaining_sec=tr,
                ticks=ticks,
                now_epoch=now_epoch,
                expiry_epoch=expiry_epoch,
            )
            # Only damp σ when adaptive scale came from *spot* (TWAP σ already
            # matches the resolution object). Mult default is 1.0 for TWAP σ.
            if resolution_meta.get("source") in (
                "rtds_twap", "settlement_nowcast"
            ):
                if drift_scale_source == "spot":
                    mult = float(getattr(
                        config, "TWAP_DRIFT_VOL_MULT_SPOT_FALLBACK", 0.92) or 0.92)
                    drift_vol_scale = float(drift_vol_scale) * mult
                else:
                    drift_vol_scale = twap_mod.soft_dampen_vol_scale(
                        drift_vol_scale)
        except Exception as e:
            logger.debug(f"TWAP resolution_btc_now failed: {e}")
            # Fallback: prefer twap then spot
            if btc_twap > 0:
                resolution_meta["btc_now"] = btc_twap
                resolution_meta["source"] = "rtds_twap"
            elif btc_spot > 0:
                resolution_meta["btc_now"] = btc_spot
                resolution_meta["source"] = "spot_fallback"

        btc_now = float(resolution_meta.get("btc_now") or 0.0)
        if btc_now > 0 and btc_strike:
            from signals.strike import drift_pct as _drift_pct
            btc_drift_pct = _drift_pct(btc_strike, btc_now)
            btc_drift = drift_signal(
                btc_strike, btc_now, tr, vol_scale=drift_vol_scale)
            try:
                from signals.strike import implied_up_prob as _imp
                from signals.strike import drift_z as _dz
                btc_implied_yes = float(_imp(
                    btc_strike, btc_now, tr, vol_scale=drift_vol_scale))
                btc_drift_z = float(_dz(
                    btc_strike, btc_now, tr, vol_scale=drift_vol_scale))
            except Exception:
                btc_implied_yes = 0.5
                btc_drift_z = 0.0
        try:
            from signals import twap as twap_mod
            if resolution_meta.get("in_settlement_window"):
                resolution_meta["twap_certainty"] = twap_mod.twap_certainty(
                    float(resolution_meta.get("nowcast_frac_elapsed") or 0.0),
                    float(resolution_meta.get("nowcast_coverage") or 0.0),
                    abs(float(btc_drift or 0.0)),
                )
            _pol = twap_mod.settlement_adjustments(
                time_remaining_sec=tr,
                twap_certainty_val=float(
                    resolution_meta.get("twap_certainty") or 0.0
                ),
                nowcast_frac_elapsed=float(
                    resolution_meta.get("nowcast_frac_elapsed") or 0.0
                ),
                nowcast_coverage=float(
                    resolution_meta.get("nowcast_coverage") or 0.0
                ),
                abs_drift=abs(float(btc_drift or 0.0)),
                in_settlement=bool(
                    resolution_meta.get("in_settlement_window")
                ),
            )
            resolution_meta["settlement_policy"] = _pol
            resolution_meta["market_phase"] = _pol.get("phase") or "unknown"
        except Exception:
            pass

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

    # Volume activity for regime context. BTC *price* is Chainlink (no volume);
    # price_feed fills volumes["btc"] from Binance btcusdt 1m klines
    # (volume-only — never overwrites Chainlink price). Classifier rules still
    # use volatility, not volume.
    vol_series = list(price_signals.get("volumes") or [])
    if not any(v and v > 0 for v in vol_series[-10:]) and price_feed is not None:
        # Cold start / WS not primed yet — read lock-side buffer directly.
        try:
            with price_feed._lock:
                vol_series = list(price_feed.volumes.get("btc") or [])
        except Exception:
            pass

    # Robust multi-feature regime (online EMA + hysteresis + optional
    # centroids). Continuous: updates every tick, not only at resolution.
    # Soft market_id stamp logs window rollovers without resetting state.
    market_regime: dict = {}
    try:
        from signals.regime_detector import get_detector
        # Recent TWAP observations for resolution-aligned regime features
        twap_px = None
        try:
            if price_feed is not None and hasattr(price_feed, "btc_twap_ticks"):
                ticks = price_feed.btc_twap_ticks() or []
                twap_px = [float(v) for _ts, v in ticks[-40:] if v and float(v) > 0]
                if len(twap_px) < 3:
                    twap_px = None
        except Exception:
            twap_px = None
        pm_state = None
        try:
            yes_px = no_px = None
            if market is not None:
                yes_px = market.get("current_price") or market.get("yes_price")
                no_px = market.get("no_price")
            if warm is not None:
                if warm.get("yes_price") is not None:
                    yes_px = warm.get("yes_price")
                if warm.get("no_price") is not None:
                    no_px = warm.get("no_price")
            spread_score = 0.5
            src = warm if warm is not None else market
            if src is not None and src.get("micro_spread_score") is not None:
                spread_score = float(src.get("micro_spread_score") or 0.5)
            yes_f = float(yes_px or 0.0)
            no_f = float(no_px or 0.0)
            # Omit sidecar until both sides exist — a 0.5 default quality
            # would otherwise haircut confidence every tick on a cold book.
            pm_state = (
                {"spread_score": spread_score, "yes_price": yes_f, "no_price": no_f}
                if yes_f > 0 and no_f > 0 else None
            )
        except (TypeError, ValueError):
            pm_state = None
        try:
            xasset_score = float(xasset.get("xasset_score") or 0.0)
        except (TypeError, ValueError, AttributeError):
            xasset_score = None
        market_regime = get_detector().update(
            btc_prices,
            cvd=cvd,
            obi=obi,
            vol_score=vol_base.get("vol_score"),
            trend_score=vol_base.get("trend_score"),
            realized_vol=vol_base.get("realized_vol"),
            volumes=vol_series,
            market_id=(
                (market.get("id") or market.get("market_id"))
                if market is not None else None
            ),
            twap_prices=twap_px,
            pm_state=pm_state,
            xasset_score=xasset_score,
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

    # Multiscale mom / rvol (pure local). Candidate directional: ms_mom_1m.
    try:
        from signals import multiscale
        ms = multiscale.compute(btc_prices)
    except Exception as e:
        logger.debug(f"multiscale compute failed: {e}")
        ms = {}

    # Lag residual: market lag vs *raw* moneyness (not tanh(z)).
    # Using z-score inherited time-scale blow-up and made lag anti-predictive
    # when moderate false drifts fired (2026-08-07 soak). Map drift_pct through
    # a fixed soft scale so lag does not explode with √(1/tr).
    yes_mid = 0.5
    if market is not None:
        try:
            yes_mid = float(market.get("current_price") or 0.5)
        except (TypeError, ValueError):
            yes_mid = 0.5
    try:
        from signals.curves import soft_saturate
        # ~0.15% moneyness soft-sats; independent of remaining-time z.
        _lag_scale = float(getattr(config, "LAG_MONEYNESS_SCALE", 0.0015) or 0.0015)
        drift_for_lag = soft_saturate(float(btc_drift_pct or 0.0), _lag_scale)
    except Exception:
        drift_for_lag = float(btc_drift or 0.0)
    implied_yes = 0.5 + 0.5 * float(drift_for_lag)
    lag_residual = max(-1.0, min(1.0, (implied_yes - yes_mid) * 2.0))

    # Flow / microstructure from warm when available.
    flow_cvd_decay = 0.0
    flow_whale = 0.0
    micro_spread = 0.0
    micro_spread_score = 0.5
    if warm is not None:
        flow_cvd_decay = float(warm.get("flow_cvd_decay", 0.0) or 0.0)
        flow_whale = float(warm.get("flow_whale", 0.0) or 0.0)
        micro_spread = float(warm.get("micro_spread", 0.0) or 0.0)
        micro_spread_score = float(warm.get("micro_spread_score", 0.5) or 0.5)
    elif market is not None:
        micro_spread = float(market.get("micro_spread", 0.0) or 0.0)
        micro_spread_score = float(market.get("micro_spread_score", 0.5) or 0.5)

    # Warm-path current regime cell for portfolio / core tuner (not last-resolved).
    try:
        import db as _db
        rid = market_regime.get("regime_id") or market_regime.get("label")
        if rid and rid != "unknown":
            # Store a lightweight current cell stamp; map rebuild still owns
            # full cell keys, but consumers can read "live" regime here.
            _db.set_arena_state(
                "warm_regime_cell",
                f"{rid}|mid|{float(yes_mid):.2f}|{float(btc_drift or 0.0):+.2f}",
            )
    except Exception:
        pass

    return {
        **price_signals,
        **sent_signals,
        "orderflow": orderflow,
        "obi": obi,
        "cvd": cvd,
        "btc_drift": btc_drift,
        "btc_drift_pct": btc_drift_pct,
        "btc_implied_yes": btc_implied_yes,
        "btc_drift_z": btc_drift_z,
        "btc_strike": btc_strike,
        "btc_strike_source": (
            get_strike_registry().get_source(
                market.get("id") or market.get("market_id")
            ) if market is not None else None
        ),
        "twap_coverage_outage": bool(
            (resolution_meta.get("settlement_policy") or {}).get(
                "coverage_outage"
            )
        ),
        "btc_now": float(resolution_meta.get("btc_now") or 0.0),
        "btc_spot": btc_spot,
        "drift_vol_scale": float(drift_vol_scale or 0.0),
        "drift_scale_source": drift_scale_source,
        "btc_twap": btc_twap,
        "resolution_source": resolution_meta.get("source") or "none",
        "resolution_nowcast": resolution_meta.get("nowcast"),
        "in_settlement_window": bool(
            resolution_meta.get("in_settlement_window")
        ),
        "twap_certainty": float(resolution_meta.get("twap_certainty") or 0.0),
        "nowcast_coverage": float(
            resolution_meta.get("nowcast_coverage") or 0.0
        ),
        "nowcast_frac_elapsed": float(
            resolution_meta.get("nowcast_frac_elapsed") or 0.0
        ),
        "market_phase": resolution_meta.get("market_phase") or "unknown",
        "settlement_policy": resolution_meta.get("settlement_policy") or {},
        "vol_regime": regime,
        "market_regime": market_regime,
        "technicals": tech,
        "xasset": xasset.get("xasset_score", 0.0),
        "futures": futures,
        "macro_caution": macro_caution(),
        "multiscale": ms,
        "ms_mom_1m": float(ms.get("ms_mom_1m", 0.0) or 0.0),
        "lag_residual": lag_residual,
        "flow_cvd_decay": flow_cvd_decay,
        "flow_whale": flow_whale,
        "micro_spread": micro_spread,
        "micro_spread_score": micro_spread_score,
        **pm_signals,
    }

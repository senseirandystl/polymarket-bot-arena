"""Chainlink BTC price feed + strike latch (no live network)."""

from __future__ import annotations

import time

from signals.price_feed import PriceFeed
from signals import strike as strike_mod


def test_chainlink_tick_updates_latest_and_candles():
    feed = PriceFeed(max_candles=50)
    # Two consecutive minutes of ticks
    base = 1_700_000_000  # fixed epoch
    # minute 0
    feed._on_btc_tick({"timestamp": base * 1000, "value": 64000.0})
    feed._on_btc_tick({"timestamp": (base + 30) * 1000, "value": 64010.0})
    # roll to minute 1 → closes previous candle at 64010
    feed._on_btc_tick({"timestamp": (base + 60) * 1000, "value": 64020.0})
    sig = feed.get_signals("btc")
    assert sig["latest"] == 64020.0
    assert sig["source"] == "chainlink"
    assert sig["stale"] is False
    assert len(sig["prices"]) >= 1
    assert sig["prices"][-1] == 64010.0


def test_price_at_latches_open():
    feed = PriceFeed()
    open_ts = 1_700_000_100.0
    feed._on_btc_tick({"timestamp": (open_ts - 5) * 1000, "value": 63900.0})
    feed._on_btc_tick({"timestamp": open_ts * 1000, "value": 63972.5})
    feed._on_btc_tick({"timestamp": (open_ts + 1) * 1000, "value": 63980.0})
    assert feed.price_at(open_ts, tol_sec=2.0) == 63972.5


def test_registry_uses_feed_latch_when_rest_down(monkeypatch):
    feed = PriceFeed()
    start_iso = "2026-07-29T00:20:00Z"
    from datetime import datetime, timezone
    ts = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp()
    feed._on_btc_tick({"timestamp": ts * 1000, "value": 63894.28})

    monkeypatch.setattr(strike_mod, "_fetch_polymarket_open_price", lambda est: None)
    import signals.price_feed as pf
    monkeypatch.setattr(pf, "get_feed", lambda: feed)

    reg = strike_mod.StrikeRegistry(fetcher=None)
    val = reg.get_strike("m1", start_iso)
    assert val == 63894.28
    assert reg.get_source("m1") == "latch"


def test_registry_prefers_rest_open_price(monkeypatch):
    monkeypatch.setattr(
        strike_mod, "_fetch_polymarket_open_price", lambda est: 63894.28,
    )
    monkeypatch.setattr(
        strike_mod, "_fetch_chainlink_feed_latch_strict", lambda est: 99999.0,
    )
    reg = strike_mod.StrikeRegistry(fetcher=None)
    assert reg.get_strike("m1", "2026-07-29T00:20:00Z") == 63894.28
    assert reg.get_source("m1") == "openPrice"

def test_eth_still_accepted_in_get_signals_shape():
    feed = PriceFeed()
    # Without Binance thread, eth is empty but shape is stable
    sig = feed.get_signals("eth")
    assert "latest" in sig and "prices" in sig
    assert sig.get("source") == "binance"

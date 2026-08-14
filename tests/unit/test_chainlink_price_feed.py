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
    monkeypatch.setattr(strike_mod.time, "time", lambda: ts + 1.0)

    reg = strike_mod.StrikeRegistry(fetcher=None)
    val = reg.get_strike("m1", start_iso)
    assert val == 63894.28
    assert reg.get_source("m1") == "spot_latch"


def test_registry_prefers_twap_open_over_rest(monkeypatch):
    """RTDS TWAP-at-open is UI PTB; REST openPrice must not override it."""
    feed = PriceFeed()
    start_iso = "2026-08-07T00:20:00Z"
    from datetime import datetime, timezone
    ts = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp()
    feed._on_btc_twap_tick({
        "timestamp": ts * 1000, "value": 64250.59, "window_s": 30,
    })
    monkeypatch.setattr(
        strike_mod, "_fetch_polymarket_open_price", lambda est: 64248.01,
    )
    import signals.price_feed as pf
    monkeypatch.setattr(pf, "get_feed", lambda: feed)
    monkeypatch.setattr(strike_mod.time, "time", lambda: ts + 5.0)

    reg = strike_mod.StrikeRegistry(fetcher=None)
    assert reg.get_strike("m1", start_iso) == 64250.59
    assert reg.get_source("m1") == "twap_open"


def test_registry_upgrades_rest_to_twap_open(monkeypatch):
    """If REST was stored first, correct to TWAP-at-open when buffer has open."""
    feed = PriceFeed()
    start_iso = "2026-08-07T00:20:00Z"
    from datetime import datetime, timezone
    ts = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp()
    monkeypatch.setattr(
        strike_mod, "_fetch_polymarket_open_price", lambda est: 64248.01,
    )
    monkeypatch.setattr(strike_mod, "_load_persisted_strike", lambda *a, **k: None)
    monkeypatch.setattr(strike_mod, "_persist_strike", lambda *a, **k: None)
    import signals.price_feed as pf
    monkeypatch.setattr(pf, "get_feed", lambda: feed)
    monkeypatch.setattr(strike_mod.time, "time", lambda: ts + 10.0)

    reg = strike_mod.StrikeRegistry(fetcher=None)
    # No TWAP in buffer yet → REST fallback
    assert reg.get_strike("m-up", start_iso) == 64248.01
    assert reg.get_source("m-up") == "openPrice"
    # Open tick arrives (or was always in buffer after reconnect rebuild)
    feed._on_btc_twap_tick({
        "timestamp": ts * 1000, "value": 64250.59, "window_s": 30,
    })
    assert reg.get_strike("m-up", start_iso) == 64250.59
    assert reg.get_source("m-up") == "twap_open"

def test_eth_still_accepted_in_get_signals_shape():
    feed = PriceFeed()
    # Without Binance thread, eth is empty but shape is stable
    sig = feed.get_signals("eth")
    assert "latest" in sig and "prices" in sig
    assert sig.get("source") == "binance"


def test_twap_tick_updates_latest_twap_and_signals():
    feed = PriceFeed()
    # Feed defaults to config TWAP_WINDOW_SEC (60s for 5m) until first tick.
    assert feed.latest_twap()[2] == 60
    base = 1_700_000_000
    feed._on_btc_twap_tick({
        "timestamp": base * 1000,
        "value": 65000.5,
        "window_s": 60,
        "symbol": "btc/usd",
    })
    twap, ts, win = feed.latest_twap()
    assert abs(twap - 65000.5) < 1e-6
    assert abs(ts - base) < 1e-6
    assert win == 60
    sig = feed.get_signals("btc")
    assert abs(sig["twap"] - 65000.5) < 1e-6
    assert sig["resolution_source"] == "rtds_twap"
    assert abs(sig["resolution_price"] - 65000.5) < 1e-6


def test_twap_full_accuracy_value_e18():
    feed = PriceFeed()
    # 65000.5 * 1e18 as integer string
    feed._on_btc_twap_tick({
        "timestamp": 1_700_000_000_000,
        "full_accuracy_value": str(int(65000.5 * 1e18)),
        "value": 65000.5,
        "window_s": 60,
    })
    twap, _, _ = feed.latest_twap()
    assert abs(twap - 65000.5) < 1e-6


def test_twap_at_latches_open():
    feed = PriceFeed()
    open_ts = 1_700_000_100.0
    feed._on_btc_twap_tick({
        "timestamp": (open_ts - 5) * 1000, "value": 63900.0, "window_s": 30,
    })
    feed._on_btc_twap_tick({
        "timestamp": open_ts * 1000, "value": 63972.5, "window_s": 30,
    })
    feed._on_btc_twap_tick({
        "timestamp": (open_ts + 1) * 1000, "value": 63980.0, "window_s": 30,
    })
    assert feed.twap_at(open_ts, tol_sec=2.0) == 63972.5


def test_twap_at_rejects_mid_window_tick():
    """After restart mid-window, do NOT treat current TWAP as open PTB."""
    feed = PriceFeed()
    open_ts = 1_700_000_100.0
    # Only a tick 90s after open (typical mid-window buffer after restart)
    feed._on_btc_twap_tick({
        "timestamp": (open_ts + 90) * 1000, "value": 64111.0, "window_s": 30,
    })
    assert feed.twap_at(open_ts, tol_sec=2.0) is None


def test_price_at_rejects_mid_window_tick():
    feed = PriceFeed()
    open_ts = 1_700_000_100.0
    feed._on_btc_tick({"timestamp": (open_ts + 120) * 1000, "value": 65000.0})
    assert feed.price_at(open_ts, tol_sec=2.0) is None


def test_registry_refuses_mid_window_latch(monkeypatch):
    """Mid-window TWAP must not become provisional strike."""
    feed = PriceFeed()
    start_iso = "2026-08-07T00:20:00Z"
    from datetime import datetime, timezone
    open_ts = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp()
    # Only mid-window sample in buffer
    feed._on_btc_twap_tick({
        "timestamp": (open_ts + 180) * 1000, "value": 64111.11, "window_s": 30,
    })
    monkeypatch.setattr(strike_mod, "_fetch_polymarket_open_price", lambda est: None)
    import signals.price_feed as pf
    monkeypatch.setattr(pf, "get_feed", lambda: feed)
    # Pretend wall clock is mid-window so late-latch guard also fires
    monkeypatch.setattr(strike_mod.time, "time", lambda: open_ts + 180)

    reg = strike_mod.StrikeRegistry(fetcher=None)
    assert reg.get_strike("m-mid", start_iso) is None


def test_registry_prefers_twap_latch_over_spot(monkeypatch):
    feed = PriceFeed()
    start_iso = "2026-08-07T00:20:00Z"
    from datetime import datetime, timezone
    ts = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).timestamp()
    feed._on_btc_twap_tick({
        "timestamp": ts * 1000, "value": 64111.11, "window_s": 30,
    })
    feed._on_btc_tick({"timestamp": ts * 1000, "value": 64222.22})

    monkeypatch.setattr(strike_mod, "_fetch_polymarket_open_price", lambda est: None)
    import signals.price_feed as pf
    monkeypatch.setattr(pf, "get_feed", lambda: feed)
    # Wall clock near open so late-latch guard allows provisional
    monkeypatch.setattr(strike_mod.time, "time", lambda: ts + 1.0)

    reg = strike_mod.StrikeRegistry(fetcher=None)
    val = reg.get_strike("m-twap", start_iso)
    assert val == 64111.11
    assert reg.get_source("m-twap") == "twap_open"

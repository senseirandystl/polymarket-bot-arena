"""BTC 'price to beat' (strike) registry + drift signal.

Production strike is Polymarket's official Chainlink openPrice (the same value
the website shows as Price to Beat). Binance 1m open is a last-resort fallback.
"""

from unittest.mock import MagicMock, patch

import signals.strike as strike


def test_get_strike_fetches_and_caches():
    calls = []

    def fake(est):
        calls.append(est)
        return 100000.0

    reg = strike.StrikeRegistry(fetcher=fake)
    assert reg.get_strike("m1", "2026-07-16T12:00:00Z") == 100000.0
    # second call is served from cache — fetcher not hit again
    assert reg.get_strike("m1", "2026-07-16T12:00:00Z") == 100000.0
    assert len(calls) == 1


def test_get_strike_none_without_event_start():
    reg = strike.StrikeRegistry(fetcher=lambda est: 100000.0)
    assert reg.get_strike("m1", None) is None


def test_get_strike_none_on_fetch_failure():
    reg = strike.StrikeRegistry(fetcher=lambda est: None)
    assert reg.get_strike("m1", "2026-07-16T12:00:00Z") is None


def test_end_iso_from_start_adds_window():
    end = strike._end_iso_from_start("2026-07-29T00:20:00Z")
    assert end == "2026-07-29T00:25:00Z"


def test_fetch_polymarket_open_price_parses_response():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "openPrice": 63894.28472206572,
        "closePrice": None,
        "incomplete": True,
    }
    with patch("signals.strike.http_client.get", return_value=mock_resp) as get:
        val = strike._fetch_polymarket_open_price("2026-07-29T00:20:00Z")
    assert abs(val - 63894.28472206572) < 1e-6
    kwargs = get.call_args.kwargs
    assert kwargs["params"]["symbol"] == "BTC"
    assert kwargs["params"]["eventStartTime"] == "2026-07-29T00:20:00Z"
    assert kwargs["params"]["variant"] == "fiveminute"
    assert kwargs["params"]["endDate"] == "2026-07-29T00:25:00Z"


def test_fetch_polymarket_open_price_none_when_missing():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"openPrice": None, "closePrice": None}
    with patch("signals.strike.http_client.get", return_value=mock_resp):
        assert strike._fetch_polymarket_open_price("2026-07-29T00:20:00Z") is None


def test_fetch_open_at_uses_official():
    with patch(
        "signals.strike._fetch_polymarket_open_price", return_value=63972.43
    ), patch(
        "signals.strike._fetch_binance_open_at", return_value=64051.99
    ) as bin_f:
        assert strike._fetch_open_at("2026-07-29T00:20:00Z") == 63972.43
        bin_f.assert_not_called()


def test_fetch_open_at_no_binance_fallback():
    """Never invent a Binance strike for live trading — wait for openPrice."""
    with patch(
        "signals.strike._fetch_polymarket_open_price", return_value=None
    ), patch(
        "signals.strike._fetch_chainlink_feed_latch", return_value=None
    ), patch(
        "signals.strike._fetch_binance_open_at", return_value=64051.99
    ) as bin_f:
        assert strike._fetch_open_at("2026-07-29T00:20:00Z") is None
        bin_f.assert_not_called()


def test_registry_upgrades_provisional_latch_to_open_price():
    """A latch must not stick for the whole window if openPrice arrives later."""
    calls = {"n": 0}

    def open_price(est):
        calls["n"] += 1
        # First call(s): unavailable; then official appears
        if calls["n"] < 2:
            return None
        return 63895.49

    reg = strike.StrikeRegistry(fetcher=None)
    with patch("signals.strike._fetch_polymarket_open_price", side_effect=open_price), \
         patch("signals.strike._fetch_chainlink_feed_latch_strict", return_value=63903.51):
        # First: provisional latch
        s1 = reg.get_strike("m-new", "2026-07-29T01:05:00Z")
        assert s1 == 63903.51
        assert reg.get_source("m-new") == "latch"
        # Force refresh window to expire
        with reg._lock:
            reg._strikes["m-new"]["ts"] = 0.0
        s2 = reg.get_strike("m-new", "2026-07-29T01:05:00Z")
        assert s2 == 63895.49
        assert reg.get_source("m-new") == "openPrice"
        # Official is sticky
        s3 = reg.get_strike("m-new", "2026-07-29T01:05:00Z")
        assert s3 == 63895.49


def test_drift_positive_when_btc_above_strike():
    assert strike.drift_signal(100000.0, 100200.0, 60) > 0


def test_drift_negative_when_btc_below_strike():
    assert strike.drift_signal(100000.0, 99800.0, 60) < 0


def test_drift_zero_at_the_strike():
    assert strike.drift_signal(100000.0, 100000.0, 60) == 0.0


def test_drift_bounded():
    hi = strike.drift_signal(100000.0, 110000.0, 15)
    lo = strike.drift_signal(100000.0, 90000.0, 15)
    assert -1.0 <= lo < 0 < hi <= 1.0


def test_drift_more_decisive_near_expiry():
    early = strike.drift_signal(100000.0, 100100.0, time_remaining=280)
    late = strike.drift_signal(100000.0, 100100.0, time_remaining=30)
    assert abs(late) > abs(early)


def test_drift_zero_without_strike():
    assert strike.drift_signal(None, 100000.0, 60) == 0.0

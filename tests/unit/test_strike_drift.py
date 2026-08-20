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


def test_registry_prefers_twap_open_over_rest_open_price():
    """TWAP-at-open is UI PTB; REST must not win once TWAP open is known."""
    reg = strike.StrikeRegistry(fetcher=None)
    with patch("signals.strike._twap_open_from_feed", return_value=64250.59), \
         patch("signals.strike._fetch_polymarket_open_price", return_value=64248.01):
        s1 = reg.get_strike("m-new", "2026-08-07T04:10:00Z")
        assert s1 == 64250.59
        assert reg.get_source("m-new") == "twap_open"
        # Sticky — REST not re-consulted as authority
        s2 = reg.get_strike("m-new", "2026-08-07T04:10:00Z")
        assert s2 == 64250.59


def test_registry_corrects_rest_when_twap_open_appears():
    """REST fallback first, then correct to TWAP-at-open (live $2.58 bug)."""
    reg = strike.StrikeRegistry(fetcher=None)
    twap = {"v": None}

    def _twap(est, **kwargs):
        return twap["v"]

    with patch("signals.strike._twap_open_from_feed", side_effect=_twap), \
         patch("signals.strike._fetch_polymarket_open_price", return_value=64248.01), \
         patch("signals.strike._spot_open_from_feed", return_value=None), \
         patch("signals.strike._load_persisted_strike", return_value=None), \
         patch("signals.strike._persist_strike", return_value=None):
        s1 = reg.get_strike("m-fix", "2026-08-07T04:10:00Z")
        assert s1 == 64248.01
        assert reg.get_source("m-fix") == "openPrice"
        twap["v"] = 64250.59
        s2 = reg.get_strike("m-fix", "2026-08-07T04:10:00Z")
        assert s2 == 64250.59
        assert reg.get_source("m-fix") == "twap_open"


def test_five_bp_twap_stays_modest_under_vol_floor():
    """σ floor 0.18% so a 5 bp mid-window move cannot print as 75¢+."""
    import config
    assert config.DRIFT_VOL_SCALE_MIN >= 0.0017
    p = strike.implied_up_prob(100_000.0, 100_050.0, 180.0, vol_scale=0.0005)
    assert 0.52 <= p <= 0.70


def test_implied_up_prob_symmetric():
    k, tr, scale = 100_000.0, 150.0, 0.0022
    up = strike.implied_up_prob(k, 100_080.0, tr, vol_scale=scale)
    dn = strike.implied_up_prob(k, 99_920.0, tr, vol_scale=scale)
    assert abs((up + dn) - 1.0) < 1e-9


def test_drift_positive_when_btc_above_strike():
    # Fixed vol_scale keeps tests independent of adaptive EMA state.
    assert strike.drift_signal(100000.0, 100200.0, 60, vol_scale=0.0015) > 0


def test_drift_negative_when_btc_below_strike():
    assert strike.drift_signal(100000.0, 99800.0, 60, vol_scale=0.0015) < 0


def test_drift_zero_at_the_strike():
    assert strike.drift_signal(100000.0, 100000.0, 60, vol_scale=0.0015) == 0.0


def test_drift_bounded():
    hi = strike.drift_signal(100000.0, 110000.0, 15, vol_scale=0.0015)
    lo = strike.drift_signal(100000.0, 90000.0, 15, vol_scale=0.0015)
    assert -1.0 <= lo < 0 < hi <= 1.0


def test_drift_more_decisive_near_expiry():
    early = strike.drift_signal(
        100000.0, 100100.0, time_remaining=280, vol_scale=0.0015)
    late = strike.drift_signal(
        100000.0, 100100.0, time_remaining=30, vol_scale=0.0015)
    assert abs(late) > abs(early)


def test_drift_zero_without_strike():
    assert strike.drift_signal(None, 100000.0, 60, vol_scale=0.0015) == 0.0

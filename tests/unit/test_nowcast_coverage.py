"""Settlement nowcast must not report coverage=0 when we have a live price."""

import signals.twap as twap


def test_nowcast_from_rem_with_carry_price():
    """Phase uses rem; nowcast must use the same expiry, not a drifted endDate."""
    now = 1_700_000_100.0
    rem = 30.0
    expiry = now + rem
    # Real pre-window tape (not a back-dated live print) → coverage is real.
    ticks = [(now - 90.0, 65000.0), (now - 5.0, 65010.0)]
    nc = twap.settlement_nowcast(
        ticks,
        now_epoch=now,
        expiry_epoch=expiry,
        twap_window_sec=60,
        fill_price=65010.0,
    )
    assert nc["in_window"] is True
    assert nc["coverage"] > 0.05
    assert nc["nowcast"] is not None


def test_resolution_prefers_rem_expiry_when_resolves_at_disagrees(monkeypatch):
    monkeypatch.setattr("config.TWAP_USE_FOR_DRIFT", True)
    monkeypatch.setattr("config.TWAP_NOWCAST_ENABLED", True)
    monkeypatch.setattr("config.TWAP_NOWCAST_MIN_COVERAGE", 0.05)
    now = 1_700_000_100.0
    rem = 20.0
    # Wrong resolves_at 5 minutes in the future — previously zeroed coverage.
    wrong_expiry = now + 300.0
    ticks = [(now - 50.0, 64000.0), (now - 1.0, 64005.0), (now, 64010.0)]
    res = twap.resolution_btc_now(
        rtds_twap=64005.0,
        spot=64010.0,
        time_remaining_sec=rem,
        ticks=ticks,
        now_epoch=now,
        expiry_epoch=wrong_expiry,
        twap_window_sec=60,
        prefer_remaining_expiry=True,
    )
    assert res["in_settlement_window"] is True
    assert res["nowcast_coverage"] > 0.05
    assert res["source"] in ("settlement_nowcast", "rtds_twap")


def test_inject_does_not_backdate_or_fake_coverage():
    now = 1000.0
    expiry = 1030.0
    ticks = twap.ensure_nowcast_ticks([], now_epoch=now, price=65000.0,
                                      expiry_epoch=expiry, twap_window_sec=60)
    # Empty ring gets a live print at *now*, not at window open.
    assert ticks
    assert all(ts > expiry - 60 for ts, _ in ticks)
    nc = twap.settlement_nowcast(
        ticks, now_epoch=now, expiry_epoch=expiry, twap_window_sec=60,
        fill_price=65000.0,
    )
    # Must not report a full-window synthetic TWAP.
    assert nc["coverage"] < 0.10


def test_data_quality_skip_provisional_strike():
    from bots.base_bot import data_quality_skip
    d = data_quality_skip({"btc_strike_source": "openPrice"})
    assert d is not None
    assert d["action"] == "skip"
    assert d["skip_reason"] == "strike_unconfirmed"
    assert data_quality_skip({"btc_strike_source": "twap_open"}) is None
    assert data_quality_skip({}) is None


def test_data_quality_skip_kalshi_floor_ok_when_brti_present():
    from bots.base_bot import data_quality_skip
    assert data_quality_skip({
        "exchange": "kalshi",
        "btc_strike_source": "kalshi_floor",
        "btc_now": 77000.0,
        "resolution_source": "brti_last60",
    }) is None
    assert data_quality_skip({
        "exchange": "kalshi",
        "btc_strike_source": "brti_open",
        "btc_now": 77000.0,
        "resolution_source": "brti_last",
    }) is None


def test_data_quality_skip_kalshi_empty_brti():
    from bots.base_bot import data_quality_skip
    d = data_quality_skip({
        "exchange": "kalshi",
        "btc_strike_source": "kalshi_floor",
        "btc_now": 0.0,
        "resolution_source": "none",
    })
    assert d is not None
    assert d["action"] == "skip"
    assert d["skip_reason"] == "brti_empty"


def test_data_quality_skip_kalshi_rejects_twap_open_source():
    from bots.base_bot import data_quality_skip
    d = data_quality_skip({
        "exchange": "kalshi",
        "btc_strike_source": "twap_open",
        "btc_now": 77000.0,
        "resolution_source": "brti_last",
    })
    assert d is not None
    assert d["skip_reason"] == "strike_unconfirmed"


def test_data_quality_skip_coverage_outage():
    from bots.base_bot import data_quality_skip
    d = data_quality_skip({"settlement_policy": {"coverage_outage": True}})
    assert d is not None
    assert d["skip_reason"] == "twap_coverage"
    d2 = data_quality_skip({"twap_coverage_outage": True})
    assert d2 is not None
    assert d2["skip_reason"] == "twap_coverage"

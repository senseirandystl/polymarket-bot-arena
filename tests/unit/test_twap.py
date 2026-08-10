"""Chainlink / Polymarket TWAP helpers + resolution btc_now selection."""

from __future__ import annotations

import signals.twap as twap


def test_window_seconds_for_5m():
    assert twap.window_seconds_for_market(300) == 30
    assert twap.window_seconds_for_market(None) == 30


def test_window_seconds_for_15m():
    assert twap.window_seconds_for_market(900) == 60


def test_in_settlement_window():
    assert twap.in_settlement_window(30, twap_window_sec=30) is True
    assert twap.in_settlement_window(15, twap_window_sec=30) is True
    assert twap.in_settlement_window(0, twap_window_sec=30) is True
    assert twap.in_settlement_window(31, twap_window_sec=30) is False
    assert twap.in_settlement_window(None) is False


def test_compute_twap_flat():
    # Constant price across window
    ticks = [(100.0, 100.0), (110.0, 100.0), (120.0, 100.0)]
    val, n, cov = twap.compute_twap(ticks, 100.0, 130.0)
    assert val is not None
    assert abs(val - 100.0) < 1e-9
    assert cov > 0.9


def test_compute_twap_step_change():
    # 10s @ 100, then 10s @ 110 → TWAP 105 over [0, 20]
    ticks = [(0.0, 100.0), (10.0, 110.0)]
    val, n, cov = twap.compute_twap(ticks, 0.0, 20.0)
    assert val is not None
    assert abs(val - 105.0) < 1e-6
    assert cov >= 0.99


def test_compute_twap_carry_into_window():
    # Last pre-window tick carries to window start
    ticks = [(90.0, 200.0), (110.0, 220.0)]
    val, n, cov = twap.compute_twap(ticks, 100.0, 120.0)
    assert val is not None
    # 10s @ 200 (carry) + 10s @ 220 → 210
    assert abs(val - 210.0) < 1e-6


def test_settlement_nowcast_mid_window():
    # Settlement window [70, 100]; now=85; ticks rise then flat
    ticks = [
        (70.0, 100.0),
        (80.0, 110.0),
        (85.0, 110.0),
    ]
    out = twap.settlement_nowcast(
        ticks,
        now_epoch=85.0,
        expiry_epoch=100.0,
        twap_window_sec=30,
        fill_price=110.0,
    )
    assert out["in_window"] is True
    assert out["nowcast"] is not None
    assert out["observed_twap"] is not None
    assert 0.4 < out["frac_elapsed"] < 0.6
    # Observed half is blend of 100→110; remaining filled at 110 → nowcast
    # should sit between 100 and 110, closer to 110.
    assert 100.0 < out["nowcast"] <= 110.0 + 1e-9


def test_settlement_nowcast_before_window():
    out = twap.settlement_nowcast(
        [(50.0, 100.0)],
        now_epoch=50.0,
        expiry_epoch=100.0,
        twap_window_sec=30,
    )
    assert out["in_window"] is False
    assert out["nowcast"] is None


def test_resolution_btc_now_prefers_rtds_twap(monkeypatch):
    monkeypatch.setattr("config.TWAP_USE_FOR_DRIFT", True)
    monkeypatch.setattr("config.TWAP_NOWCAST_ENABLED", True)
    monkeypatch.setattr("config.TWAP_FALLBACK_TO_SPOT", True)
    res = twap.resolution_btc_now(
        rtds_twap=65000.0,
        spot=65100.0,
        time_remaining_sec=120.0,  # outside settlement window
    )
    assert res["btc_now"] == 65000.0
    assert res["source"] == "rtds_twap"
    assert res["in_settlement_window"] is False


def test_resolution_btc_now_spot_fallback(monkeypatch):
    monkeypatch.setattr("config.TWAP_USE_FOR_DRIFT", True)
    monkeypatch.setattr("config.TWAP_FALLBACK_TO_SPOT", True)
    res = twap.resolution_btc_now(
        rtds_twap=None,
        spot=64000.0,
        time_remaining_sec=100.0,
    )
    assert res["btc_now"] == 64000.0
    assert res["source"] == "spot_fallback"


def test_resolution_btc_now_nowcast_inside_window(monkeypatch):
    monkeypatch.setattr("config.TWAP_USE_FOR_DRIFT", True)
    monkeypatch.setattr("config.TWAP_NOWCAST_ENABLED", True)
    monkeypatch.setattr("config.TWAP_NOWCAST_MIN_COVERAGE", 0.3)
    # expiry=1000, now=985 → 15s into 30s settlement window
    ticks = [(970.0, 100.0), (980.0, 102.0), (985.0, 102.0)]
    res = twap.resolution_btc_now(
        rtds_twap=101.5,
        spot=102.0,
        time_remaining_sec=15.0,
        ticks=ticks,
        now_epoch=985.0,
        expiry_epoch=1000.0,
        twap_window_sec=30,
    )
    assert res["in_settlement_window"] is True
    assert res["source"] == "settlement_nowcast"
    assert res["btc_now"] > 0
    assert res["nowcast"] is not None


def test_twap_certainty_bounds():
    c0 = twap.twap_certainty(0.0, 0.0, 0.0)
    c1 = twap.twap_certainty(1.0, 1.0, 1.0, min_drift=0.15)
    assert 0.0 <= c0 <= 1.0
    assert 0.0 <= c1 <= 1.0
    assert c1 > c0


def test_soft_dampen_vol_scale(monkeypatch):
    monkeypatch.setattr("config.TWAP_DRIFT_VOL_MULT", 0.85)
    assert abs(twap.soft_dampen_vol_scale(0.0015) - 0.0015 * 0.85) < 1e-12


def test_market_phase_labels():
    assert twap.market_phase(10) == "settlement"
    assert twap.market_phase(30) == "settlement"
    assert twap.market_phase(40) == "pre_settle"
    assert twap.market_phase(150) == "mid"
    assert twap.market_phase(290) == "open"
    assert twap.market_phase(None) == "unknown"


def test_settlement_adjustments_high_cert(monkeypatch):
    monkeypatch.setattr("config.TWAP_SETTLEMENT_POLICY", True)
    monkeypatch.setattr("config.TWAP_RESOLUTION_ENABLED", True)
    monkeypatch.setattr("config.TWAP_SETTLE_CERT_HIGH", 0.55)
    monkeypatch.setattr("config.TWAP_SETTLE_EDGE_MULT_HIGH", 0.92)
    monkeypatch.setattr("config.TWAP_SETTLE_SIZE_MULT_HIGH", 1.12)
    adj = twap.settlement_adjustments(
        time_remaining_sec=10.0,
        twap_certainty_val=0.80,
        nowcast_frac_elapsed=0.7,
        nowcast_coverage=0.8,
        abs_drift=0.40,
    )
    assert adj["phase"] == "settlement"
    assert adj["policy_active"] is True
    assert adj["edge_mult"] < 1.0
    assert adj["size_mult"] > 1.0
    assert adj["mom_damp"] < 1.0
    assert adj["block_fade"] is True


def test_settlement_adjustments_low_cert(monkeypatch):
    monkeypatch.setattr("config.TWAP_SETTLEMENT_POLICY", True)
    monkeypatch.setattr("config.TWAP_RESOLUTION_ENABLED", True)
    monkeypatch.setattr("config.TWAP_SETTLE_CERT_LOW", 0.25)
    monkeypatch.setattr("config.TWAP_SETTLE_EDGE_MULT_LOW", 1.40)
    monkeypatch.setattr("config.TWAP_SETTLE_SIZE_MULT_LOW", 0.80)
    adj = twap.settlement_adjustments(
        time_remaining_sec=20.0,
        twap_certainty_val=0.10,
        nowcast_frac_elapsed=0.2,
        nowcast_coverage=0.3,
        abs_drift=0.05,
    )
    assert adj["phase"] == "settlement"
    assert adj["edge_mult"] > 1.0
    assert adj["size_mult"] < 1.0
    assert adj["block_fade"] is False


def test_settlement_adjustments_mid_window_neutral(monkeypatch):
    monkeypatch.setattr("config.TWAP_SETTLEMENT_POLICY", True)
    adj = twap.settlement_adjustments(
        time_remaining_sec=150.0,
        twap_certainty_val=0.0,
        abs_drift=0.3,
    )
    assert adj["phase"] == "mid"
    assert adj["edge_mult"] == 1.0
    assert adj["size_mult"] == 1.0
    assert adj["mom_damp"] == 1.0


def test_open_phase_keeps_damps_with_zero_coverage(monkeypatch):
    """Open always has nowcast_coverage=0 — must not trip the outage guard."""
    monkeypatch.setattr("config.TWAP_SETTLEMENT_POLICY", True)
    monkeypatch.setattr("config.TWAP_RESOLUTION_ENABLED", True)
    adj = twap.settlement_adjustments(
        time_remaining_sec=290.0,
        twap_certainty_val=0.0,
        nowcast_frac_elapsed=0.0,
        nowcast_coverage=0.0,
        abs_drift=0.0,
    )
    assert adj["phase"] == "open"
    assert adj["policy_active"] is True
    assert adj["mom_damp"] == 0.85
    assert adj["edge_mult"] == 1.05
    assert adj.get("coverage_outage") is not True


def test_pre_settle_keeps_damps_with_zero_coverage(monkeypatch):
    """Pre-settle is before nowcast; zero coverage is expected, not an outage."""
    monkeypatch.setattr("config.TWAP_SETTLEMENT_POLICY", True)
    monkeypatch.setattr("config.TWAP_RESOLUTION_ENABLED", True)
    adj = twap.settlement_adjustments(
        time_remaining_sec=40.0,
        twap_certainty_val=0.0,
        nowcast_frac_elapsed=0.0,
        nowcast_coverage=0.0,
        abs_drift=0.0,
    )
    assert adj["phase"] == "pre_settle"
    assert adj["policy_active"] is True
    assert adj["mom_damp"] == 0.70
    assert adj["edge_mult"] == 1.08
    assert adj["size_mult"] == 0.95
    assert adj.get("coverage_outage") is not True


def test_settlement_zero_coverage_outage_resets_low_cert_penalty(monkeypatch):
    """Real settlement outage: empty tick coverage → no 1.40× edge tax."""
    monkeypatch.setattr("config.TWAP_SETTLEMENT_POLICY", True)
    monkeypatch.setattr("config.TWAP_RESOLUTION_ENABLED", True)
    monkeypatch.setattr("config.TWAP_SETTLE_CERT_LOW", 0.25)
    monkeypatch.setattr("config.TWAP_SETTLE_EDGE_MULT_LOW", 1.40)
    monkeypatch.setattr("config.TWAP_SETTLE_SIZE_MULT_LOW", 0.80)
    monkeypatch.setattr("config.TWAP_SETTLE_MOM_DAMP", 0.40)
    adj = twap.settlement_adjustments(
        time_remaining_sec=15.0,
        twap_certainty_val=0.05,
        nowcast_frac_elapsed=0.5,
        nowcast_coverage=0.0,
        abs_drift=0.02,
    )
    assert adj["phase"] == "settlement"
    assert adj.get("coverage_outage") is True
    assert adj["edge_mult"] == 1.0
    assert adj["size_mult"] == 1.0
    assert adj["mom_damp"] == 1.0
    assert adj["conf_boost"] == 0.0

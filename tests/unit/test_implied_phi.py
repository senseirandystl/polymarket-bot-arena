"""Honest Φ(z) implied probability — never 0.5+0.5·tanh."""

import math

import pytest

from bots.base_bot import implied_side_prob, price_quality_ok


def test_implied_yes_from_btc_implied_yes():
    assert implied_side_prob(side="yes", signals={"btc_implied_yes": 0.62}) == pytest.approx(0.62)
    assert implied_side_prob(side="no", signals={"btc_implied_yes": 0.62}) == pytest.approx(0.38)


def test_implied_from_raw_z():
    # z=0 → 0.5; z≈0.674 → ~0.75
    p = implied_side_prob(side="yes", signals={"btc_drift_z": 0.0})
    assert p == pytest.approx(0.5)
    p_hi = implied_side_prob(side="yes", signals={"btc_drift_z": 0.67449})
    assert p_hi == pytest.approx(0.75, abs=0.01)
    assert implied_side_prob(side="no", signals={"btc_drift_z": 0.67449}) == pytest.approx(
        1.0 - p_hi, abs=1e-9
    )


def test_implied_never_uses_half_plus_half_tanh():
    """tanh 0.58 is NOT 79¢. Φ(artanh(0.58)) is ~72¢."""
    tanh_lie = 0.5 + 0.5 * 0.58  # 0.79
    p = implied_side_prob(side="yes", signed_lane=0.58)
    assert p < tanh_lie - 0.04
    assert 0.68 < p < 0.76


def test_implied_fail_closed_when_missing():
    assert implied_side_prob(side="yes", signals={}) == pytest.approx(0.5)
    assert implied_side_prob(side="no") == pytest.approx(0.5)


def test_implied_no_uses_yes_frame_lane_once():
    """signed_lane is YES-frame. Do not flip twice for NO."""
    p_yes = implied_side_prob(side="yes", signed_lane=0.40)
    p_no = implied_side_prob(side="no", signed_lane=0.40)
    assert p_yes + p_no == pytest.approx(1.0)
    assert p_no < 0.5 < p_yes


def test_implied_nan_fail_closed():
    assert implied_side_prob(side="yes", signals={"btc_implied_yes": float("nan")}) == pytest.approx(0.5)
    assert implied_side_prob(side="yes", signals={"btc_drift_z": float("nan")}, signed_lane=float("nan")) == pytest.approx(0.5)


def test_drift_z_recovers_from_tanh_lane():
    from bots.base_bot import drift_z_from_signals
    z = 0.67
    tanh_z = math.tanh(z)
    assert drift_z_from_signals({}, tanh_z) == pytest.approx(z, abs=1e-6)
    assert drift_z_from_signals({"btc_drift_z": z}, 0.99) == pytest.approx(z)


def test_price_quality_prefers_phi_not_tanh_fallback():
    # Mid-band: tanh-implied 0.5+0.5*0.40=0.70 would fake fat lag vs 56¢.
    # Honest Φ from the lane is tighter; with implied_side=0.58 it must fail.
    assert not price_quality_ok(
        side_mid=0.56, side_ask=0.56, signed_drift=0.40, implied_side=0.58,
    )

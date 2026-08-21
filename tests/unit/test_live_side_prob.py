"""live_side_prob SSoT: Φ(z), never 0.5+0.5·tanh; edge is P−ask−fee."""

import math

import pytest

import polymarket_fills
from signals.prob import (
    directional_net_edge,
    live_side_prob,
    phi_yes_from_signals,
)


def test_phi_yes_z_zero_is_half():
    assert phi_yes_from_signals({"btc_drift_z": 0.0}) == pytest.approx(0.5)


def test_phi_yes_prefers_btc_implied_yes():
    assert phi_yes_from_signals({
        "btc_implied_yes": 0.62,
        "btc_drift_z": 9.0,
    }) == pytest.approx(0.62)


def test_tanh_0_45_is_not_half_plus_half():
    tanh_lie = 0.5 + 0.5 * 0.45
    p = phi_yes_from_signals({}, signed_lane=0.45)
    assert p < tanh_lie - 0.02
    z = math.atanh(0.45)
    expect = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    assert p == pytest.approx(expect)


def test_live_side_prob_phi_when_overlay_empty():
    p, src = live_side_prob(side="yes", signals={"btc_implied_yes": 0.64})
    assert src == "phi"
    assert p == pytest.approx(0.64)
    p_no, src_no = live_side_prob(side="no", signals={"btc_implied_yes": 0.64})
    assert src_no == "phi"
    assert p_no == pytest.approx(0.36)


def test_live_side_prob_missing_is_half():
    p, src = live_side_prob(side="yes", signals={})
    assert src == "half"
    assert p == pytest.approx(0.5)


def test_directional_net_edge_no_trust_tax():
    # 64¢ model vs 50¢ ask; fee = 0.07 * 0.5 * 0.5 = 0.0175
    fee = polymarket_fills.fee_per_share(0.50, is_maker=False)
    assert fee == pytest.approx(0.07 * 0.5 * 0.5)
    edge = directional_net_edge(0.64, 0.50)
    assert edge == pytest.approx(0.64 - 0.50 - fee)
    assert edge == pytest.approx(0.1225)


def test_implied_side_prob_wraps_live_side_prob():
    from bots.base_bot import implied_side_prob
    sig = {"btc_implied_yes": 0.71}
    assert implied_side_prob(side="yes", signals=sig) == live_side_prob(
        side="yes", signals=sig,
    )[0]
    assert implied_side_prob(side="no", signals=sig) == live_side_prob(
        side="no", signals=sig,
    )[0]

"""Regime-adaptive decision adjustments (not just size damp)."""

from arena.regime_adapt import RegimeAdjust, adjustments


def test_unknown_regime_is_neutral():
    a = adjustments("unknown", "momentum")
    assert a.size_mult == 1.0
    assert a.edge_mult == 1.0
    assert a.mom_lane_scale == 1.0


def test_low_vol_trend_tightens_momentum():
    a = adjustments("low_vol_trend", "momentum")
    assert a.edge_mult > 1.0
    assert a.mom_lane_scale < 1.0
    assert a.strat_lane_scale < 1.0
    assert a.no_edge_mult > 1.0
    assert a.flow_full_trust is not None and a.flow_full_trust >= 0.30
    assert a.extra_drift_floor > 0


def test_high_vol_trend_eases_momentum_relative_to_low_vol_trend():
    lo = adjustments("low_vol_trend", "momentum")
    hi = adjustments("high_vol_trend", "momentum")
    assert hi.edge_mult < lo.edge_mult
    assert hi.mom_lane_scale > lo.mom_lane_scale


def test_regime_adjust_to_dict():
    d = RegimeAdjust(size_mult=0.5, label="x").to_dict()
    assert d["size_mult"] == 0.5
    assert d["label"] == "x"

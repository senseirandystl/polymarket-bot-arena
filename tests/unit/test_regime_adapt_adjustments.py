"""Regime-adaptive decision adjustments (not just size damp)."""

from unittest import mock

from arena.regime_adapt import RegimeAdjust, adjustments

_EMPTY = {
    "n": 0, "wins": 0, "pnl": 0.0, "wr": None,
    "fast_n": 0, "fast_wins": 0, "fast_pnl": 0.0, "fast_wr": None,
}


def _no_live_stats():
    """Structural priors only — thin cells so continuous path is neutral."""
    return mock.patch.multiple(
        "arena.regime_stats",
        strategy_regime_cell=mock.Mock(return_value=dict(_EMPTY)),
        regime_cell=mock.Mock(return_value=dict(_EMPTY)),
        side_regime_cell=mock.Mock(return_value=dict(_EMPTY)),
        strategy_side_regime_cell=mock.Mock(return_value=dict(_EMPTY)),
        is_toxic_cell=mock.Mock(return_value=False),
        effective_wr=mock.Mock(return_value=None),
    )


def test_unknown_regime_is_neutral():
    a = adjustments("unknown", "momentum")
    assert a.size_mult == 1.0
    assert a.edge_mult == 1.0
    assert a.mom_lane_scale == 1.0


def test_low_vol_trend_style_mode_mild():
    """Style mode: mild edge nudge; partial structural lane blend."""
    with _no_live_stats():
        a = adjustments("low_vol_trend", "momentum")
    # Flattened edge mult still slightly elevated vs 1.0 for this prior
    assert a.edge_mult >= 0.90
    assert a.edge_mult <= 1.40
    # Style mode blends prior mom scale toward 1.0 (not full throttle damp)
    assert 0.5 <= a.mom_lane_scale <= 1.05
    assert 0.5 <= a.strat_lane_scale <= 1.05
    assert "mode=style" in a.reason


def test_high_vol_trend_eases_edge_relative_to_low_vol_trend():
    """Without live samples, structural prior for HVT eases vs LVT."""
    with _no_live_stats():
        lo = adjustments("low_vol_trend", "momentum")
        hi = adjustments("high_vol_trend", "momentum")
    assert hi.edge_mult <= lo.edge_mult + 1e-9


def test_regime_adjust_to_dict():
    d = RegimeAdjust(size_mult=0.5, label="x").to_dict()
    assert d["size_mult"] == 0.5
    assert d["label"] == "x"

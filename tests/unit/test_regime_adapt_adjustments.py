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


def test_not_actionable_live_match_is_neutral(monkeypatch):
    """When the live detector is on this label but not actionable, no tax."""
    class _Det:
        def snapshot(self):
            return {
                "regime_id": "low_vol_trend",
                "label": "low_vol_trend",
                "actionable": False,
                "confidence": 0.3,
                "held_sec": 2.0,
            }
    monkeypatch.setattr(
        "signals.regime_detector.get_detector", lambda: _Det(), raising=False,
    )
    a = adjustments("low_vol_trend", "momentum")
    assert a.size_mult == 1.0
    assert a.edge_mult == 1.0
    assert a.mom_lane_scale == 1.0
    assert a.reason == "not_actionable"


def test_low_vol_trend_style_mode_mild(monkeypatch):
    """Style mode: mild edge nudge; partial structural lane blend."""
    class _Det:
        def snapshot(self):
            return {
                "regime_id": "low_vol_trend",
                "label": "low_vol_trend",
                "actionable": True,
                "confidence": 0.8,
                "held_sec": 60.0,
            }
    monkeypatch.setattr(
        "signals.regime_detector.get_detector", lambda: _Det(), raising=False,
    )
    with _no_live_stats():
        a = adjustments("low_vol_trend", "momentum")
    # Flattened edge mult still slightly elevated vs 1.0 for this prior
    assert a.edge_mult >= 0.90
    assert a.edge_mult <= 1.40
    # Style mode blends prior mom scale toward 1.0 (not full throttle damp)
    assert 0.5 <= a.mom_lane_scale <= 1.05
    assert 0.5 <= a.strat_lane_scale <= 1.05
    assert "mode=style" in a.reason


def test_normal_regime_deprecated_mid_band_and_extra_drift(monkeypatch):
    class _Det:
        def snapshot(self):
            return {"regime_id": "high_vol_trend", "actionable": True}
    monkeypatch.setattr(
        "signals.regime_detector.get_detector", lambda: _Det(), raising=False,
    )
    with _no_live_stats():
        a = adjustments("normal", "momentum")
    # Phase 1 honesty: these are no longer live levers.
    assert a.mid_band_drift_min is None
    assert a.extra_drift_floor == 0.0


def test_high_vol_trend_taxes_edge_relative_to_low_vol_trend():
    """Soak 2026-08-24: HVT continuation bled; prior must raise min_edge, not ease."""
    with _no_live_stats():
        lo = adjustments("low_vol_trend", "momentum")
        hi = adjustments("high_vol_trend", "momentum")
    assert hi.edge_mult >= lo.edge_mult - 1e-9
    assert hi.edge_mult > 1.0
    assert hi.mom_lane_scale < 1.0


def test_regime_adjust_to_dict():
    d = RegimeAdjust(size_mult=0.5, label="x").to_dict()
    assert d["size_mult"] == 0.5
    assert d["label"] == "x"

"""Regime strategy routing scores."""

from arena.regime_router import boost_type_alloc_scores, score, scores_for_regime


def test_meanrev_scores_higher_in_range_than_momentum():
    mr = score("mean_reversion", "low_vol_range", live={"n": 0, "pnl": 0})
    mo = score("momentum", "low_vol_range", live={"n": 0, "pnl": 0})
    assert mr > mo


def test_momentum_scores_higher_in_high_vol_trend():
    mo = score("momentum", "high_vol_trend", live={"n": 0})
    mr = score("mean_reversion", "high_vol_trend", live={"n": 0})
    assert mo > mr


def test_boost_type_alloc_changes_order():
    base = {"momentum": 1.0, "mean_reversion": 1.0}
    out = boost_type_alloc_scores(base, "low_vol_range", blend=0.5)
    assert out["mean_reversion"] >= out["momentum"]


def test_scores_for_regime_keys():
    s = scores_for_regime("normal")
    assert "momentum" in s

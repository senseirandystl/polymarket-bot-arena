"""High-vol favorite tax + mom-vs-drift fight — paper-eval soak taxes."""

from arena.eval_taxes import high_vol_favorite_mult, mom_drift_fight_mult


def test_high_vol_favorite_taxes_56c_momentum():
    e, s = high_vol_favorite_mult(
        strategy_type="momentum", regime="high_vol_trend", side_mid=0.56,
    )
    assert e >= 1.49
    assert s <= 0.61


def test_high_vol_favorite_spares_cheap_no():
    e, s = high_vol_favorite_mult(
        strategy_type="hybrid", regime="high_vol_trend", side_mid=0.38,
    )
    assert e == 1.0
    assert s == 1.0


def test_high_vol_favorite_spares_normal_regime():
    e, s = high_vol_favorite_mult(
        strategy_type="momentum", regime="normal", side_mid=0.56,
    )
    assert e == 1.0
    assert s == 1.0


def test_high_vol_favorite_spares_meanrev():
    e, s = high_vol_favorite_mult(
        strategy_type="mean_reversion", regime="high_vol_trend", side_mid=0.56,
    )
    assert e == 1.0


def test_mom_drift_fight_aligned_is_one():
    assert mom_drift_fight_mult(mom=0.8, drift=0.4, regime="normal") == 1.0


def test_mom_drift_fight_raises_bar():
    m = mom_drift_fight_mult(mom=-0.83, drift=0.37, regime="normal")
    assert m >= 1.34


def test_mom_drift_fight_stronger_in_high_vol():
    n = mom_drift_fight_mult(mom=-0.83, drift=0.37, regime="normal")
    h = mom_drift_fight_mult(mom=-0.83, drift=0.37, regime="high_vol_trend")
    assert h > n

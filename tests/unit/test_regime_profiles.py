"""Per-regime lane weight resolution."""

from arena.regime_profiles import resolve_lane_weight, seed_weight


def test_seed_low_vol_range_meanrev_high_drift():
    w = seed_weight("low_vol_range", "mean_reversion", "drift")
    assert w is not None and w >= 0.80


def test_seed_high_vol_trend_mom_higher_than_range():
    r = seed_weight("low_vol_range", "momentum", "mom")
    t = seed_weight("high_vol_trend", "momentum", "mom")
    assert t is not None and r is not None
    assert t > r


def test_seed_high_vol_chop_damps_mom():
    w = seed_weight("high_vol_chop", "momentum", "mom")
    assert w is not None and w <= 0.15


def test_earned_by_regime_beats_seed():
    """Earned by_regime meta allows override above seed."""
    overrides = {
        "drift": {
            "enabled": True,
            "core": True,
            "profile": {"momentum": 0.55},
            "by_regime": {"low_vol_range": {"momentum": 0.80}},
            "by_regime_meta": {
                "low_vol_range": {
                    "momentum": {
                        "drift": {"earned": True, "n": 50, "accuracy": 0.70},
                    }
                }
            },
        }
    }
    w = resolve_lane_weight(
        "drift", "momentum", "low_vol_range",
        profile={"drift": 0.55}, overrides=overrides,
    )
    assert w == 0.80


def test_unearned_clone_defers_to_seed():
    """Clone of global profile must not shadow chop seeds (soak 2026-08-06)."""
    overrides = {
        "mom": {
            "enabled": True,
            "core": True,
            "profile": {"momentum": 0.50},
            "by_regime": {
                "high_vol_chop": {"momentum": 0.50},  # clone of global
            },
        }
    }
    w = resolve_lane_weight(
        "mom", "momentum", "high_vol_chop",
        profile={"mom": 0.30}, overrides=overrides,
    )
    # seed for high_vol_chop momentum mom is 0.10
    assert w == 0.10


def test_global_override_when_no_regime_key():
    overrides = {
        "drift": {
            "enabled": True,
            "core": True,
            "profile": {"momentum": 0.60},
        }
    }
    w = resolve_lane_weight(
        "drift", "momentum", "normal",
        profile={"drift": 0.55}, overrides=overrides,
    )
    assert w == 0.60


def test_seed_when_no_override():
    w = resolve_lane_weight(
        "mom", "momentum", "low_vol_range",
        profile={"mom": 0.30}, overrides={},
    )
    # seed for low_vol_range momentum mom is 0.15
    assert w == 0.15

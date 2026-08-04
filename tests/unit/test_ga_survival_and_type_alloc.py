"""GA survival bar, founder protection, type-alloc sentiment exclusion."""

from unittest import mock

import config
from evolution.ga import _survives_legacy_bar
from evolution.type_alloc import (
    _dead_lane_types,
    allocatable_types,
    pick_strategy_type,
)


def test_survives_positive_pnl():
    assert _survives_legacy_bar({
        "trades": 50, "pnl": 5.0, "be_gap": -0.05, "generation": 1,
    })


def test_survives_mild_red_ink():
    """−$3 is noise, not a cull (2026-08 meanrev-v1 was wrongly replaced)."""
    assert _survives_legacy_bar({
        "trades": 50, "pnl": -3.0, "be_gap": -0.01, "generation": 1,
    })


def test_replaceable_when_deeply_underwater():
    assert not _survives_legacy_bar({
        "trades": 50, "pnl": -25.0, "be_gap": -0.05, "generation": 1,
    })


def test_founder_protected_on_moderate_loss():
    assert _survives_legacy_bar({
        "trades": 50, "pnl": -15.0, "be_gap": -0.01, "generation": 0,
    })


def test_founder_cullable_when_deeply_bad():
    assert not _survives_legacy_bar({
        "trades": 50, "pnl": -30.0, "be_gap": -0.05, "generation": 0,
    })


def test_immune_under_min_trades():
    assert _survives_legacy_bar({
        "trades": 10, "pnl": -100.0, "be_gap": -0.2, "generation": 0,
    })


def test_sentiment_excluded_when_lanes_killed():
    assert "sentiment" in _dead_lane_types()
    assert "sentiment" not in allocatable_types()


def test_pick_never_returns_sentiment_when_dead():
    # Even with high fitness bank entries for sentiment, spawn must not pick it.
    inds = [{
        "strategy_type": "sentiment", "fitness": 0.99,
    }]
    for _ in range(20):
        t = pick_strategy_type(
            "phantom", inds, inds, rng=__import__("random").Random(0),
        )
        assert t != "sentiment"


def test_same_type_only_flag(monkeypatch):
    monkeypatch.setattr(config, "GA_TYPE_SAME_TYPE_ONLY", True)
    t = pick_strategy_type(
        "momentum",
        [{"strategy_type": "hybrid", "fitness": 0.9}],
        [],
        rng=__import__("random").Random(1),
    )
    assert t == "momentum"

"""GA judgment floor scales with realized directional trade rate."""

import config
from evolution.ga import effective_min_trades, _survives_legacy_bar


def test_adaptive_floor_uses_leader_when_below_cap(monkeypatch):
    monkeypatch.setattr(config, "GA_MIN_TRADES_ADAPTIVE", True)
    monkeypatch.setattr(config, "GA_MIN_TRADES_FLOOR", 20)
    monkeypatch.setattr(config, "MIN_TRADES_FOR_JUDGMENT", 40)
    # Leader has 25, others are starved — judge the leader, keep the rest immune.
    assert effective_min_trades([25, 25, 5, 1]) == 25


def test_adaptive_floor_stays_at_cap_when_sample_is_rich(monkeypatch):
    monkeypatch.setattr(config, "GA_MIN_TRADES_ADAPTIVE", True)
    monkeypatch.setattr(config, "MIN_TRADES_FOR_JUDGMENT", 40)
    assert effective_min_trades([80, 55, 40, 12]) == 40


def test_adaptive_off_uses_configured_cap(monkeypatch):
    monkeypatch.setattr(config, "GA_MIN_TRADES_ADAPTIVE", False)
    monkeypatch.setattr(config, "MIN_TRADES_FOR_JUDGMENT", 40)
    assert effective_min_trades([13, 5, 1]) == 40


def test_starved_slate_stays_immune_below_floor(monkeypatch):
    monkeypatch.setattr(config, "GA_MIN_TRADES_ADAPTIVE", True)
    monkeypatch.setattr(config, "GA_MIN_TRADES_FLOOR", 20)
    monkeypatch.setattr(config, "MIN_TRADES_FOR_JUDGMENT", 40)
    assert effective_min_trades([5, 1, 0]) == 20
    assert _survives_legacy_bar(
        {"trades": 5, "pnl": -4.0, "be_gap": -0.14, "generation": 0},
        min_n=20,
    )


def test_leader_is_judgeable_at_adaptive_floor(monkeypatch):
    monkeypatch.setattr(config, "GA_EARLY_CULL_ENABLED", False)
    assert _survives_legacy_bar(
        {"trades": 13, "pnl": 4.7, "be_gap": 0.06, "generation": 0},
        min_n=13,
    )


def test_mid_sample_uses_early_cull_not_full_bar(monkeypatch):
    """n between adaptive floor and 40: mild red is not a cull."""
    monkeypatch.setattr(config, "PAPER_GATE_PROFILE", "off")
    monkeypatch.setattr(config, "MIN_TRADES_FOR_JUDGMENT", 40)
    monkeypatch.setattr(config, "GA_EARLY_CULL_ENABLED", True)
    assert _survives_legacy_bar(
        {"trades": 25, "pnl": -13.0, "be_gap": -0.04, "generation": 1},
        min_n=25,
    )
    assert not _survives_legacy_bar(
        {"trades": 25, "pnl": -20.0, "be_gap": -0.12, "generation": 1},
        min_n=25,
    )

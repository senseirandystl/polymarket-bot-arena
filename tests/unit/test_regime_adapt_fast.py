"""Fast-path style-skip + continuous edge mult from dual-window stats."""

from unittest import mock

import pytest

from arena import regime_adapt as ra
from arena.regime_adapt import (
    _strategy_block_map, _wr_to_edge_mult, adjustments,
)


def test_wr_to_edge_mult_monotone():
    good = _wr_to_edge_mult(0.65)
    mid = _wr_to_edge_mult(0.50)
    bad = _wr_to_edge_mult(0.35)
    assert good < mid < bad
    assert bad >= 1.2
    assert good <= 1.05


def test_fast_path_style_skip_enters(monkeypatch):
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: True if name in ("style_skip", "adapt_enabled") else False,
    )
    monkeypatch.setattr(ra.config, "REGIME_STYLE_SKIP_ENABLED", True, raising=False)
    monkeypatch.setattr(ra.config, "REGIME_STYLE_SKIP_FAST_MIN_N", 10, raising=False)
    monkeypatch.setattr(ra.config, "REGIME_STYLE_SKIP_FAST_WR", 0.38, raising=False)
    monkeypatch.setattr(ra.config, "REGIME_STYLE_SKIP_MIN_TRADES", 18, raising=False)
    monkeypatch.setattr(ra.config, "REGIME_STYLE_SKIP_WR", 0.42, raising=False)

    # Long window looks fine; fast window is toxic
    toxic_fast = {
        "by_strategy": {
            "high_vol_chop": {
                "momentum": {
                    "n": 40, "wins": 24, "wr": 0.60, "pnl": 15.0,
                    "fast_n": 12, "fast_wins": 3, "fast_wr": 0.25,
                    "fast_pnl": -30.0,
                },
                "hybrid": {
                    "n": 30, "wins": 18, "wr": 0.60, "pnl": 10.0,
                    "fast_n": 5, "fast_wins": 3, "fast_wr": 0.60,
                    "fast_pnl": 5.0,
                },
            }
        }
    }
    with mock.patch("arena.regime_stats.snapshot", return_value=toxic_fast):
        blocks = _strategy_block_map({})
    assert blocks.get(("high_vol_chop", "momentum")) is True
    assert not blocks.get(("high_vol_chop", "hybrid"))


def test_continuous_edge_from_wr_eff(monkeypatch):
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: True if name in ("style_skip", "adapt_enabled") else (
            False if name == "hard_skip" else True
        ),
    )
    monkeypatch.setattr(
        "arena.regime_settings.get_adapt_primary",
        lambda: "style",
    )
    soft_bad = {
        "n": 20, "wins": 8, "wr": 0.40, "pnl": -12.0,
        "fast_n": 12, "fast_wins": 4, "fast_wr": 0.333, "fast_pnl": -10.0,
    }
    healthy = {
        "n": 30, "wins": 20, "wr": 0.67, "pnl": 20.0,
        "fast_n": 10, "fast_wins": 7, "fast_wr": 0.70, "fast_pnl": 8.0,
    }
    with mock.patch("arena.regime_adapt._refresh_cache"), \
         mock.patch("arena.regime_adapt._cache", (
             0.0, {"high_vol_chop": 1.0}, {}, {}, {}
         )), \
         mock.patch("arena.regime_stats.strategy_regime_cell",
                    return_value=soft_bad), \
         mock.patch("arena.regime_stats.regime_cell", return_value=soft_bad), \
         mock.patch("arena.regime_stats.side_regime_cell",
                    return_value={"n": 0, "wins": 0, "pnl": 0, "wr": None,
                                  "fast_n": 0, "fast_wr": None, "fast_pnl": 0}), \
         mock.patch("arena.regime_stats.strategy_side_regime_cell",
                    return_value={"n": 0, "wins": 0, "pnl": 0, "wr": None,
                                  "fast_n": 0, "fast_wr": None, "fast_pnl": 0}), \
         mock.patch("arena.regime_stats.is_toxic_cell", return_value=False), \
         mock.patch("arena.regime_stats.effective_wr", return_value=0.38):
        a = adjustments("high_vol_chop", "momentum")
    assert a.edge_mult > 1.15
    assert a.wr_eff == pytest.approx(0.38)
    assert a.extra_drift_floor == 0.0
    assert a.max_bots_side == 1  # soft-bad → tandem clamp

    with mock.patch("arena.regime_adapt._refresh_cache"), \
         mock.patch("arena.regime_adapt._cache", (
             0.0, {"normal": 1.0}, {}, {}, {}
         )), \
         mock.patch("arena.regime_stats.strategy_regime_cell",
                    return_value=healthy), \
         mock.patch("arena.regime_stats.regime_cell", return_value=healthy), \
         mock.patch("arena.regime_stats.side_regime_cell",
                    return_value={"n": 0, "wins": 0, "pnl": 0, "wr": None,
                                  "fast_n": 0, "fast_wr": None, "fast_pnl": 0}), \
         mock.patch("arena.regime_stats.strategy_side_regime_cell",
                    return_value={"n": 0, "wins": 0, "pnl": 0, "wr": None,
                                  "fast_n": 0, "fast_wr": None, "fast_pnl": 0}), \
         mock.patch("arena.regime_stats.is_toxic_cell", return_value=False), \
         mock.patch("arena.regime_stats.effective_wr", return_value=0.68):
        b = adjustments("normal", "momentum")
    assert b.edge_mult < a.edge_mult


def test_side_block_yes_only(monkeypatch):
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: True if name in ("style_skip", "adapt_enabled") else (
            False if name == "hard_skip" else True
        ),
    )
    monkeypatch.setattr(
        "arena.regime_settings.get_adapt_primary", lambda: "style",
    )
    monkeypatch.setattr(ra.config, "REGIME_SIDE_SKIP_ENABLED", True, raising=False)

    yes_toxic = {
        "n": 15, "wins": 3, "wr": 0.20, "pnl": -20.0,
        "fast_n": 12, "fast_wins": 2, "fast_wr": 0.167, "fast_pnl": -18.0,
    }
    no_ok = {
        "n": 12, "wins": 8, "wr": 0.67, "pnl": 10.0,
        "fast_n": 8, "fast_wins": 6, "fast_wr": 0.75, "fast_pnl": 6.0,
    }

    def side_cell(reg, strat, side):
        return yes_toxic if side == "yes" else no_ok

    def toxic(cell, **kw):
        path = kw.get("path", "long")
        if path == "fast" and cell is yes_toxic:
            return True
        return False

    with mock.patch("arena.regime_adapt._refresh_cache"), \
         mock.patch("arena.regime_adapt._cache", (
             0.0, {"high_vol_chop": 1.0}, {}, {}, {}
         )), \
         mock.patch("arena.regime_stats.strategy_regime_cell",
                    return_value={"n": 20, "wins": 10, "wr": 0.5, "pnl": -5,
                                  "fast_n": 10, "fast_wr": 0.4, "fast_pnl": -3}), \
         mock.patch("arena.regime_stats.regime_cell",
                    return_value={"n": 40, "wins": 20, "wr": 0.5, "pnl": 0,
                                  "fast_n": 15, "fast_wr": 0.45, "fast_pnl": 0}), \
         mock.patch("arena.regime_stats.side_regime_cell",
                    return_value={"n": 0, "wr": None, "fast_n": 0, "fast_wr": None,
                                  "pnl": 0, "fast_pnl": 0}), \
         mock.patch("arena.regime_stats.strategy_side_regime_cell",
                    side_effect=side_cell), \
         mock.patch("arena.regime_stats.is_toxic_cell", side_effect=toxic), \
         mock.patch("arena.regime_stats.effective_wr",
                    side_effect=lambda c, **k: c.get("fast_wr") or c.get("wr")):
        a = adjustments("high_vol_chop", "sniper")
    assert a.block_side == "yes"
    assert a.side_edge_for("yes") > a.side_edge_for("no")

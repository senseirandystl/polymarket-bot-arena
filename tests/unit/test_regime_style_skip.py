"""Strategy×regime style-skip (data-driven stand-down per strategy)."""

from unittest import mock

from arena.regime_adapt import RegimeAdjust, _strategy_block_map, adjustments


def test_strategy_block_enters_when_toxic(monkeypatch):
    import config
    monkeypatch.setattr(config, "REGIME_STYLE_SKIP_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "REGIME_STYLE_SKIP_MIN_TRADES", 20, raising=False)
    monkeypatch.setattr(config, "REGIME_STYLE_SKIP_WR", 0.40, raising=False)
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: True if name in ("style_skip", "adapt_enabled") else False,
    )
    toxic = {
        "by_strategy": {
            "high_vol_chop": {
                "momentum": {
                    "n": 30, "wins": 9, "pnl": -40.0, "wr": 0.30,
                    "fast_n": 12, "fast_wins": 3, "fast_wr": 0.25,
                    "fast_pnl": -20.0,
                },
                "hybrid": {
                    "n": 25, "wins": 15, "pnl": 10.0, "wr": 0.60,
                    "fast_n": 10, "fast_wins": 7, "fast_wr": 0.70,
                    "fast_pnl": 5.0,
                },
            }
        }
    }
    with mock.patch("arena.regime_stats.snapshot", return_value=toxic):
        blocks = _strategy_block_map({})
    assert blocks.get(("high_vol_chop", "momentum")) is True
    assert not blocks.get(("high_vol_chop", "hybrid"))


def test_adjustments_block_strategy(monkeypatch):
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: True if name in ("style_skip", "adapt_enabled") else (
            False if name == "hard_skip" else True
        ),
    )
    with mock.patch("arena.regime_adapt._refresh_cache"), \
         mock.patch("arena.regime_adapt._cache", (
             0.0,
             {"high_vol_chop": 0.90},
             {"high_vol_chop": {"n": 80, "wins": 30, "pnl": -50}},
             {},
             {("high_vol_chop", "momentum"): True},
         )), \
         mock.patch("arena.regime_stats.regime_cell", return_value={
             "n": 80, "wins": 30, "pnl": -50, "wr": 0.375,
         }), \
         mock.patch("arena.regime_stats.is_toxic_cell", return_value=True), \
         mock.patch("arena.regime_stats.side_regime_cell", return_value={
             "n": 0, "wins": 0, "pnl": 0, "wr": None,
         }):
        a = adjustments("high_vol_chop", "momentum")
        assert a.block_strategy is True
        assert a.size_mult == 0.0
        assert "STYLE_SKIP" in a.reason
        # hybrid not blocked
        b = adjustments("high_vol_chop", "hybrid")
        assert b.block_strategy is False
        assert b.size_mult > 0


def test_chop_prior_damps_momentum_mom_scale():
    a = adjustments("high_vol_chop", "momentum")
    # Even in style mode, chop prior should pull mom scale below 1
    assert a.mom_lane_scale < 1.0
    assert a.mid_band_drift_min is not None
    assert a.mid_band_drift_min >= 0.30

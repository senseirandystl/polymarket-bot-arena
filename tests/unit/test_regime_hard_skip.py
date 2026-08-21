"""Data-driven regime hard-skip + mid-band floors."""

from unittest import mock

from arena.regime_adapt import RegimeAdjust, _hard_block_map, adjustments
from bots.bot_momentum import MomentumBot


def test_hard_block_enters_and_clears_with_hysteresis(monkeypatch):
    import config
    monkeypatch.setattr(config, "REGIME_HARD_SKIP_ENABLED", True)
    monkeypatch.setattr(config, "REGIME_HARD_SKIP_EMERGENCY_ONLY", True)
    monkeypatch.setattr(config, "REGIME_HARD_SKIP_MIN_TRADES", 80)
    monkeypatch.setattr(config, "REGIME_HARD_SKIP_WR", 0.38)
    monkeypatch.setattr(config, "REGIME_HARD_SKIP_CLEAR_WR", 0.48)
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: True if name == "hard_skip" else False,
    )
    # Enter when WR low + neg PnL with full emergency sample
    with mock.patch("arena.regime_adapt._load_perf", return_value={
        "low_vol_trend": {"n": 100, "wins": 30, "pnl": -40.0},  # WR=30%
    }):
        blocks = _hard_block_map({})
        assert blocks.get("low_vol_trend") is True

    # Thin sample must NOT enter hard-skip in emergency mode
    with mock.patch("arena.regime_adapt._load_perf", return_value={
        "low_vol_trend": {"n": 30, "wins": 8, "pnl": -40.0},
    }):
        blocks = _hard_block_map({})
        assert not blocks.get("low_vol_trend")

    # Stay blocked while WR still under clear bar
    with mock.patch("arena.regime_adapt._load_perf", return_value={
        "low_vol_trend": {"n": 100, "wins": 40, "pnl": -10.0},  # WR=40%
    }):
        blocks = _hard_block_map({"low_vol_trend": True})
        assert blocks.get("low_vol_trend") is True

    # Clear when WR recovers
    with mock.patch("arena.regime_adapt._load_perf", return_value={
        "low_vol_trend": {"n": 100, "wins": 55, "pnl": 5.0},  # WR=55%
    }):
        blocks = _hard_block_map({"low_vol_trend": True})
        assert blocks.get("low_vol_trend") is False


def test_adjustments_block_directional_sets_size_zero(monkeypatch):
    import config
    monkeypatch.setattr(config, "REGIME_HARD_SKIP_ENABLED", True)
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: True if name in ("hard_skip", "adapt_enabled") else False,
    )
    with mock.patch("arena.regime_adapt._refresh_cache"), \
         mock.patch("arena.regime_adapt._cache", (
             0.0,
             {"low_vol_trend": 0.85},
             {"low_vol_trend": {"n": 100, "wins": 30, "pnl": -40}},
             {"low_vol_trend": True},
         )):
        a = adjustments("low_vol_trend", "momentum")
        assert a.block_directional is True
        assert a.size_mult == 0.0


def test_arb_not_blocked():
    with mock.patch("arena.regime_adapt._refresh_cache"), \
         mock.patch("arena.regime_adapt._cache", (
             0.0,
             {"low_vol_trend": 0.35},
             {},
             {"low_vol_trend": True},
         )):
        a = adjustments("low_vol_trend", "arbitrage")
        assert a.block_directional is False


def test_mid_band_gate_skips_weak_drift_in_coinflip(monkeypatch):
    """mid 0.52 with mild drift must skip under mid-band lag gate."""
    import bots.base_bot as bb
    monkeypatch.setattr(bb, "_lane_overrides", lambda: {})
    monkeypatch.setattr(bb, "_sizing_bankroll", lambda mode: 200.0)
    monkeypatch.setattr(bb, "_kelly_fraction", lambda: 0.25)
    monkeypatch.setattr(bb, "_portfolio_weight", lambda name: 1.0)
    monkeypatch.setattr(bb, "_risk_size_mult", lambda name: 1.0)

    bot = MomentumBot(name="m-test", generation=0)
    # Force no hard-skip, but mid-band should still demand high drift
    with mock.patch(
        "arena.regime_adapt.adjustments",
        return_value=RegimeAdjust(
            size_mult=1.0, edge_mult=1.0, mid_band_drift_min=0.28,
            label="normal",
        ),
    ):
        d = bot.make_decision(
            {
                "id": "m", "current_price": 0.52, "no_price": 0.48,
                "yes_ask": 0.53, "no_ask": 0.49,
                "time_remaining_seconds": 150,
            },
            {
                "prices": [100.0, 100.1, 100.2], "latest": 100.2,
                "btc_drift": 0.15,  # below mid-band floor
                "orderflow": {}, "pm_momentum": 0.0, "obi": 0.0, "cvd": 0.0,
            },
        )
    assert d["action"] == "skip"
    reason = (d.get("reasoning") or "").lower()
    assert (
        "mid-band" in reason or "lean" in reason or "dead-zone" in reason
        or "price-quality" in reason or "dual-gate" in reason
    )

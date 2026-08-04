"""Maker mid/ask integrity + redesigned entry paths."""

from bots.bot_fee_zone_maker import FeeZoneMakerBot
from bots.bot_late_window_maker import LateWindowMakerBot


def test_fee_zone_rejects_ask_far_below_mid():
    bot = FeeZoneMakerBot()
    # Drift favors YES; mid in zone but ask is a fantasy fill.
    market = {
        "current_price": 0.70,
        "yes_ask": 0.30,
        "time_remaining_seconds": 200,
    }
    signals = {"prices": [100.0, 100.1, 100.3], "btc_drift": 0.40}
    out = bot.analyze(market, signals)
    assert out["action"] == "hold"
    assert "integrity" in (out.get("reasoning") or "").lower() or "gap" in (
        out.get("reasoning") or ""
    ).lower()


def test_fee_zone_buys_when_lag_and_book_ok():
    bot = FeeZoneMakerBot()
    market = {
        "current_price": 0.62,
        "yes_ask": 0.63,
        "time_remaining_seconds": 200,
    }
    signals = {"prices": [100.0, 100.1, 100.3], "btc_drift": 0.45}
    out = bot.analyze(market, signals)
    assert out["action"] == "buy", out.get("reasoning")
    assert out["side"] == "yes"
    assert abs(float(out["entry_price"]) - 0.63) < 1e-6


def test_late_window_rejects_crossed_book():
    bot = LateWindowMakerBot()
    market = {
        "current_price": 0.70,
        "yes_ask": 0.40,
        "time_remaining_seconds": 40,
    }
    signals = {"prices": [100.0, 100.0, 100.2], "btc_drift": 0.50}
    out = bot.analyze(market, signals)
    assert out["action"] == "hold"

"""Sniper v3 — drift-vs-price lag hunter (no zone buckets)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bots.bot_sniper import SniperBot


def _signals(drift=0.50, prices=None, d_pct=None):
    prices = prices or [100.0, 100.1, 100.2]
    if d_pct is None:
        d_pct = 0.0008 if drift >= 0 else -0.0008
    return {
        "btc_drift": drift,
        "btc_drift_pct": d_pct,
        "btc_strike": 100000.0,
        "btc_now": 100000.0 * (1.0 + d_pct),
        "prices": prices,
        "orderflow": {},
        "regime": {"label": "normal", "known": True, "vol_score": 0.5},
    }


def test_snipes_when_market_lags_drift():
    bot = SniperBot(name="sniper-test")
    # Strong YES drift, YES mid still lagging at 0.45
    market = {
        "current_price": 0.45,
        "no_price": 0.55,
        "yes_ask": 0.46,
        "no_ask": 0.56,
        "time_remaining_seconds": 120,
    }
    d = bot.make_decision(market, _signals(drift=0.50, d_pct=0.0008))
    assert d["action"] == "buy"
    assert d["side"] == "yes"
    assert d["edge"] > 0


def test_skips_when_priced_in():
    bot = SniperBot(name="sniper-test")
    # Extreme drift but YES already 0.75 — priced in
    market = {
        "current_price": 0.75,
        "no_price": 0.25,
        "yes_ask": 0.76,
        "no_ask": 0.26,
        "time_remaining_seconds": 60,
    }
    d = bot.make_decision(market, _signals(drift=0.80, d_pct=0.0015))
    assert d["action"] in ("skip", "hold")


def test_skips_flat_drift():
    bot = SniperBot(name="sniper-test")
    market = {
        "current_price": 0.50,
        "no_price": 0.50,
        "yes_ask": 0.51,
        "no_ask": 0.51,
        "time_remaining_seconds": 200,
    }
    d = bot.make_decision(market, _signals(drift=0.02))
    assert d["action"] in ("skip", "hold")


def test_skips_wide_ask_mid_spread():
    """Mid looks lagging but ask has already re-priced — refuse the fill."""
    bot = SniperBot(name="sniper-test")
    market = {
        "current_price": 0.54,   # mid still "lags"
        "no_price": 0.46,
        "yes_ask": 0.75,         # executable ask far above mid
        "no_ask": 0.47,
        "time_remaining_seconds": 120,
    }
    d = bot.make_decision(market, _signals(drift=0.40))
    assert d["action"] in ("skip", "hold")
    assert "ask gap" in (d.get("reasoning") or "").lower() or d["action"] != "buy"

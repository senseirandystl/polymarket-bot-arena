"""Sniper v3 — drift-vs-price lag hunter (no zone buckets)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bots.bot_sniper import SniperBot


def _signals(drift=0.50, prices=None, d_pct=None, implied_yes=None, z=None):
    prices = prices or [100.0, 100.1, 100.2]
    if d_pct is None:
        d_pct = 0.0016 if drift >= 0 else -0.0016
    out = {
        "btc_drift": drift,
        "btc_drift_pct": d_pct,
        "btc_strike": 100000.0,
        "btc_now": 100000.0 * (1.0 + d_pct),
        "prices": prices,
        "orderflow": {},
        "regime": {"label": "normal", "known": True, "vol_score": 0.5},
    }
    if implied_yes is not None:
        out["btc_implied_yes"] = implied_yes
    if z is not None:
        out["btc_drift_z"] = z
    return out


def test_snipes_when_market_lags_drift():
    bot = SniperBot(name="sniper-test")
    # Strong YES drift, YES mid still lagging
    market = {
        "current_price": 0.45,
        "no_price": 0.55,
        "yes_ask": 0.46,
        "no_ask": 0.56,
        "time_remaining_seconds": 180,
    }
    d = bot.make_decision(market, _signals(drift=0.50, d_pct=0.0016, z=0.50))
    assert d["action"] == "buy"
    assert d["side"] == "yes"
    assert d["edge"] > 0


def test_no_lag_uses_yes_frame_implied():
    """Φ(z) is YES-frame; NO implied is 1−iy (overnight inversion bug)."""
    bot = SniperBot(name="sniper-imp")
    market = {
        "current_price": 0.55,
        "no_price": 0.45,
        "yes_ask": 0.56,
        "no_ask": 0.45,
        "time_remaining_seconds": 180,
    }
    sig = _signals(drift=-0.40, d_pct=-0.0016, prices=[100.2, 100.1, 100.0], z=-0.50)
    sig["btc_implied_yes"] = 0.36
    sig["btc_drift_z"] = -0.40
    d = bot.make_decision(market, sig)
    assert d["action"] == "buy"
    assert d["side"] == "no"


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
        "time_remaining_seconds": 180,
    }
    d = bot.make_decision(
        market, _signals(drift=0.40, d_pct=0.0016, implied_yes=0.72, z=0.50),
    )
    assert d["action"] in ("skip", "hold")
    assert "ask gap" in (d.get("reasoning") or "").lower() or d["action"] != "buy"
    if d["action"] == "skip":
        assert d.get("skip_reason") in (
            "ask_quality", "no_lag_edge", "price_quality", "sniper_conviction",
        )


def test_sniper_prices_edge_on_ask_not_mid():
    """BUG #28: mid still lags but the executable ask has eaten the edge."""
    bot = SniperBot(name="sniper-ask")
    market = {
        "current_price": 0.50,
        "no_price": 0.50,
        "yes_ask": 0.57,
        "no_ask": 0.51,
        "time_remaining_seconds": 150,
    }
    # Φ = 0.60 vs mid 50¢ looks like 10¢ lag; vs ask 57¢ after fee it is not.
    d = bot.make_decision(market, _signals(drift=0.40, d_pct=0.0005, implied_yes=0.60))
    assert d["action"] in ("skip", "hold")


def test_sniper_rejects_ask_below_mid():
    """Crossed book: ask < mid bypassed the spread check (ask−mid is negative)."""
    bot = SniperBot(name="sniper-cross")
    market = {
        "current_price": 0.49,
        "no_price": 0.51,
        "yes_ask": 0.50,
        "no_ask": 0.36,
        "time_remaining_seconds": 150,
    }
    d = bot.make_decision(
        market,
        _signals(drift=-0.40, d_pct=-0.0009, implied_yes=0.34),
    )
    assert d["action"] in ("skip", "hold")


def test_sniper_logs_phi_not_tanh_implied():
    bot = SniperBot(name="sniper-log")
    market = {
        "current_price": 0.45,
        "no_price": 0.55,
        "yes_ask": 0.46,
        "no_ask": 0.56,
        "time_remaining_seconds": 120,
    }
    d = bot.make_decision(
        market, _signals(drift=0.58, d_pct=0.0008, implied_yes=0.72),
    )
    if d["action"] == "buy":
        why = d.get("reasoning") or ""
        # tanh lie would print implied=0.79 (0.5+0.5*0.58)
        assert "implied=0.79" not in why
        assert "implied=0.72" in why

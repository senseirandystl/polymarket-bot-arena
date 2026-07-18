"""Fractional-Kelly bet sizing (config.KELLY_FRACTION of f* = edge/(1-price)).

Replaces the flat confidence-scaled %-of-max-position formula, which sized
wins and losses almost identically ($3.83 vs $3.76 avg over 453 trades).
"""

from unittest import mock

import config
import polymarket_fills
from bots import base_bot
from bots.bot_momentum import MomentumBot


def _bot():
    return MomentumBot(name="momentum-test", generation=0)


def _market(yes, tr=150):
    return {"id": "m", "current_price": yes, "no_price": round(1 - yes, 4),
            "polymarket_token_id": "y", "polymarket_no_token_id": "n",
            "time_remaining_seconds": tr}


def _sig(**over):
    base = {"prices": [100.0, 100.0], "latest": 100.0, "orderflow": {},
            "pm_momentum": 0.0, "obi": 0.0, "cvd": 0.0, "btc_drift": 0.0}
    base.update(over)
    return base


def _decide(bankroll, yes, drift, fraction=0.25):
    # Trending BTC tape agreeing with the (positive) drift: under the
    # fidelity profiles the momentum bot needs actual momentum to trade.
    sig = _sig(btc_drift=drift,
               prices=[100.0, 100.05, 100.12, 100.20, 100.30], latest=100.30)
    with mock.patch.object(base_bot, "_sizing_bankroll", lambda mode: bankroll), \
         mock.patch.object(base_bot, "_kelly_fraction", lambda: fraction):
        return _bot().make_decision(_market(yes), sig)


def test_size_scales_with_bankroll():
    small = _decide(100.0, 0.52, 0.5)
    large = _decide(400.0, 0.52, 0.5)
    assert small["action"] == large["action"] == "buy"
    assert large["suggested_amount"] > small["suggested_amount"]


def test_size_scales_with_edge():
    weak = _decide(200.0, 0.52, 0.5)
    strong = _decide(200.0, 0.52, 0.9)
    assert weak["action"] == strong["action"] == "buy"
    assert strong["suggested_amount"] > weak["suggested_amount"]


def test_kelly_fraction_math():
    # PURE Kelly: amount = fraction * edge/(1-price) * bankroll — no per-trade
    # or %-of-balance caps (shares-first rounding aside).
    bankroll = 200.0
    fraction = 0.25
    d = _decide(bankroll, 0.52, 0.9, fraction=fraction)
    assert d["action"] == "buy"
    price = d["entry_price"]
    edge = float(d["reasoning"].split("edge=")[1].split(" ")[0])
    expected = fraction * edge / (1 - price) * bankroll
    assert abs(d["suggested_amount"] - expected) <= max(0.05, price * 0.01) \
        or d["suggested_amount"] >= config.POLYMARKET_MIN_SHARES * price


def test_size_uncapped_scales_past_old_per_trade_limit():
    # Caps removed 2026-07-17: with a big pool and strong edge the bet may
    # exceed the old $50 per-trade / 10%-of-balance limits (the paper venue's
    # shared-pool gate remains the only spend limit).
    d = _decide(10000.0, 0.52, 0.95)
    assert d["action"] == "buy"
    assert d["suggested_amount"] > config.PAPER_MAX_POSITION


def test_kelly_fraction_scales_size_linearly():
    quarter = _decide(200.0, 0.52, 0.9, fraction=0.25)
    full = _decide(200.0, 0.52, 0.9, fraction=1.0)
    assert full["suggested_amount"] > 3.5 * quarter["suggested_amount"]


def test_kelly_fraction_db_roundtrip(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "kelly.db")
    db_module.init_db()
    assert db_module.get_kelly_fraction() == config.KELLY_FRACTION  # default
    db_module.set_kelly_fraction(0.5)
    assert db_module.get_kelly_fraction() == 0.5
    import pytest
    with pytest.raises(ValueError):
        db_module.set_kelly_fraction(1.5)
    with pytest.raises(ValueError):
        db_module.set_kelly_fraction(0.0)


def test_tiny_edge_still_floors_at_min_shares():
    d = _decide(200.0, 0.52, 0.28)
    if d["action"] == "buy":
        assert d["target_shares"] >= config.POLYMARKET_MIN_SHARES

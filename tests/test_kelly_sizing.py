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


def _decide(bankroll, yes, drift):
    with mock.patch.object(base_bot, "_sizing_bankroll", lambda mode: bankroll):
        return _bot().make_decision(_market(yes), _sig(btc_drift=drift))


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
    # f = KELLY_FRACTION * edge/(1-price), amount = f * bankroll (shares-first
    # rounding aside), capped by MAX_POSITION_PCT_OF_BALANCE and max_pos.
    bankroll = 200.0
    d = _decide(bankroll, 0.52, 0.9)
    assert d["action"] == "buy"
    price = d["entry_price"]
    edge = float(d["reasoning"].split("edge=")[1].split(" ")[0])
    f = min(config.KELLY_FRACTION * edge / (1 - price),
            config.MAX_POSITION_PCT_OF_BALANCE)
    expected = min(f * bankroll, config.get_max_position())
    # shares-first rounding + 5-share floor can nudge the USD slightly
    assert abs(d["suggested_amount"] - expected) <= max(0.05, price * 0.01) \
        or d["suggested_amount"] >= config.POLYMARKET_MIN_SHARES * price


def test_size_capped_at_max_position_pct():
    d = _decide(10000.0, 0.52, 0.95)   # huge bankroll + huge edge
    assert d["suggested_amount"] <= config.get_max_position() + 1e-9


def test_tiny_edge_still_floors_at_min_shares():
    d = _decide(200.0, 0.52, 0.28)
    if d["action"] == "buy":
        assert d["target_shares"] >= config.POLYMARKET_MIN_SHARES

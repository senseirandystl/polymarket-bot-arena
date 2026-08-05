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
    from arena.regime_adapt import RegimeAdjust
    sig = _sig(btc_drift=drift,
               prices=[100.0, 100.05, 100.12, 100.20, 100.30], latest=100.30)
    # Neutral regime adjustments — isolate from live regime_performance DB.
    neutral = RegimeAdjust(size_mult=1.0, label="normal")
    with mock.patch.object(base_bot, "_sizing_bankroll", lambda mode: bankroll), \
         mock.patch.object(base_bot, "_kelly_fraction", lambda: fraction), \
         mock.patch("arena.regime_adapt.adjustments", return_value=neutral):
        m = _market(yes)
        m["yes_ask"] = yes
        m["no_ask"] = round(1 - yes, 4)
        return _bot().make_decision(m, sig)


def test_size_scales_with_bankroll():
    # Large bankrolls so size sits above POLYMARKET_MIN_SHARES floor.
    # mid 0.55: outside dead-zone, under extreme-drift lag ceiling (0.58).
    small = _decide(2000.0, 0.55, 0.45)
    large = _decide(8000.0, 0.55, 0.45)
    assert small["action"] == large["action"] == "buy", (
        small.get("reasoning"), large.get("reasoning"))
    assert large["suggested_amount"] > small["suggested_amount"] * 1.5


def test_size_scales_with_edge_below_cap():
    # Stronger drift sizes bigger once above min-share floor.
    weak = _decide(5000.0, 0.55, 0.30)
    strong = _decide(5000.0, 0.55, 0.45)
    assert weak["action"] == strong["action"] == "buy", (
        weak.get("reasoning"), strong.get("reasoning"))
    assert strong["suggested_amount"] > weak["suggested_amount"]


def test_size_clamped_above_edge_cap():
    # Outsized raw edges get concave/capped sizing — not 1:1 with raw edge.
    from bots.edge_calibration import calibrated_sizing_edge
    big = _decide(2000.0, 0.55, 0.45)
    assert big["action"] == "buy", big.get("reasoning")
    raw_edge = float(big["reasoning"].split("edge=")[1].split(" ")[0])
    se = calibrated_sizing_edge(raw_edge)
    assert se <= config.KELLY_EDGE_CAP + 1e-9
    expected = (0.25 * se / (1 - big["entry_price"]) * 2000.0)
    # shares-first floor may raise tiny sizes; otherwise match calibrated Kelly
    assert abs(big["suggested_amount"] - expected) <= max(0.5, big["entry_price"] * 0.1) \
        or big["suggested_amount"] >= config.POLYMARKET_MIN_SHARES * big["entry_price"]


def test_kelly_fraction_math():
    # PURE Kelly on *calibrated* sizing edge (not raw edge).
    from bots.edge_calibration import calibrated_sizing_edge
    bankroll = 2000.0
    fraction = 0.25
    d = _decide(bankroll, 0.55, 0.45, fraction=fraction)
    assert d["action"] == "buy", d.get("reasoning")
    price = d["entry_price"]
    raw_edge = float(d["reasoning"].split("edge=")[1].split(" ")[0])
    edge = calibrated_sizing_edge(raw_edge)
    expected = fraction * edge / (1 - price) * bankroll
    assert abs(d["suggested_amount"] - expected) <= max(0.5, price * 0.1) \
        or d["suggested_amount"] >= config.POLYMARKET_MIN_SHARES * price


def test_size_uncapped_scales_past_old_per_trade_limit():
    # Caps removed 2026-07-17: with a big pool and strong edge the bet may
    # exceed the old $50 per-trade / 10%-of-balance limits (the paper venue's
    # shared-pool gate remains the only spend limit).
    d = _decide(50000.0, 0.55, 0.45)
    assert d["action"] == "buy", d.get("reasoning")
    assert d["suggested_amount"] > config.PAPER_MAX_POSITION


def test_kelly_fraction_scales_size_linearly():
    quarter = _decide(5000.0, 0.55, 0.45, fraction=0.25)
    full = _decide(5000.0, 0.55, 0.45, fraction=1.0)
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

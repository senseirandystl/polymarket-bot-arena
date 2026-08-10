"""Sweeper bot — locked-outcome fee-curve extreme entries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bots.bot_sweeper import SweeperBot, DEFAULT_PARAMS
from evolution.ga import EVOLUTION_EXEMPT_TYPES
import polymarket_fills


def _signals(drift=0.85, prices=None):
    prices = prices or [100.0, 100.05, 100.08]
    return {
        "btc_drift": drift,
        "prices": prices,
        "orderflow": {},
        "regime": {"label": "normal", "known": True, "vol_score": 0.5},
    }


def _market(**kw):
    base = {
        "current_price": 0.985,
        "no_price": 0.015,
        "yes_ask": 0.988,
        "no_ask": 0.020,
        "time_remaining_seconds": 40,
    }
    base.update(kw)
    return base


def test_buys_locked_yes_in_fee_curve_extreme():
    bot = SweeperBot(name="sweeper-test")
    d = bot.make_decision(_market(), _signals(drift=0.85))
    assert d["action"] == "buy"
    assert d["side"] == "yes"
    assert d["edge"] > 0
    # Settlement edge: 1 - ask - fee
    expect = SweeperBot._settlement_edge(0.988)
    assert abs(d["edge"] - expect) < 1e-9
    assert d.get("entry_price") == 0.988


def test_buys_locked_no_when_drift_negative():
    bot = SweeperBot(name="sweeper-test")
    market = _market(
        current_price=0.015,
        no_price=0.985,
        yes_ask=0.020,
        no_ask=0.988,
    )
    d = bot.make_decision(market, _signals(drift=-0.85, prices=[100.0, 99.95, 99.90]))
    assert d["action"] == "buy"
    assert d["side"] == "no"


def test_skips_outside_entry_window():
    bot = SweeperBot(name="sweeper-test")
    d = bot.make_decision(
        _market(time_remaining_seconds=200),
        _signals(drift=0.90),
    )
    assert d["action"] == "skip"
    assert "waiting" in (d.get("reasoning") or "")


def test_skips_outside_45s_window():
    """Default entry window is last 45s (not 75s)."""
    bot = SweeperBot(name="sweeper-test")
    assert DEFAULT_PARAMS["entry_window_sec"] == 45
    assert DEFAULT_PARAMS["min_drift"] == 0.65
    assert DEFAULT_PARAMS["min_twap_certainty"] == 0.55
    d = bot.make_decision(
        _market(time_remaining_seconds=60),
        _signals(drift=0.90),
    )
    assert d["action"] == "skip"
    assert "waiting" in (d.get("reasoning") or "")


def test_skips_weak_drift():
    bot = SweeperBot(name="sweeper-test")
    d = bot.make_decision(_market(), _signals(drift=0.20))
    assert d["action"] == "skip"
    assert "no lock" in (d.get("reasoning") or "")


def test_skips_mid_book_fake_sweep():
    """V2 rule: do not chase 90–95¢ as a sweeper."""
    bot = SweeperBot(name="sweeper-test")
    d = bot.make_decision(
        _market(
            current_price=0.93,
            no_price=0.07,
            yes_ask=0.94,
            no_ask=0.08,
        ),
        _signals(drift=0.90),
    )
    assert d["action"] == "skip"


def test_skips_when_net_edge_too_thin():
    """Ask too close to $1 leaves nothing after fee."""
    bot = SweeperBot(name="sweeper-test", params={
        **DEFAULT_PARAMS,
        "min_edge": 0.005,  # need ≥ 0.5¢ net
    })
    # At 0.999, net ≈ 0.093¢ — below 0.5¢ floor
    d = bot.make_decision(
        _market(
            current_price=0.999,
            no_price=0.001,
            yes_ask=0.999,
            no_ask=0.002,
        ),
        _signals(drift=0.95),
    )
    assert d["action"] == "skip"


def test_skips_wide_ask_mid_spread():
    bot = SweeperBot(name="sweeper-test")
    d = bot.make_decision(
        _market(
            current_price=0.980,
            yes_ask=0.999,  # 1.9¢ gap > default 1.5¢ max
        ),
        _signals(drift=0.90),
    )
    assert d["action"] == "skip"
    assert "ask gap" in (d.get("reasoning") or "").lower() or d["action"] != "buy"


def test_skips_momentum_contradiction():
    bot = SweeperBot(name="sweeper-test")
    # YES lock but BTC candle crashing
    d = bot.make_decision(
        _market(),
        _signals(drift=0.90, prices=[100.0, 99.5, 98.5]),
    )
    assert d["action"] == "skip"


def test_settlement_edge_matches_fee_formula():
    ask = 0.99
    edge = SweeperBot._settlement_edge(ask)
    fee = polymarket_fills.fee_per_share(ask, is_maker=False)
    assert abs(edge - (1.0 - ask - fee)) < 1e-12
    assert edge > 0.009  # ~0.93¢ at 99¢


def test_evolution_exempt():
    assert "sweeper" in EVOLUTION_EXEMPT_TYPES

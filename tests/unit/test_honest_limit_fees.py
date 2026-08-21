"""Directional paper/live fees: price as taker, no invented maker fills."""

import pytest

import config
import polymarket_fills as fills


def _book(bid=0.48, ask=0.52, size=100.0):
    return {
        "valid": True,
        "bids": [(bid, size)],
        "asks": [(ask, size)],
        "min_order_size": 5,
    }


def test_cap_ask_limit_equals_best_ask():
    lim = fills.limit_buy_price(_book(), mid=0.50, mode="cap_ask")
    assert lim == pytest.approx(0.52)


def test_resting_limit_does_not_invent_a_fill(monkeypatch):
    monkeypatch.setattr(config, "LIMIT_PAPER_ASSUME_MAKER_FILL", False)
    f = fills.simulate_limit_buy(_book(ask=0.55, bid=0.48), 10.0, limit_price=0.49)
    assert f["filled"] is False


def test_marketable_cap_ask_is_taker_with_fee(monkeypatch):
    monkeypatch.setattr(config, "LIMIT_PAPER_ASSUME_MAKER_FILL", False)
    book = _book(ask=0.52)
    lim = fills.limit_buy_price(book, mid=0.50, mode="cap_ask")
    f = fills.simulate_limit_buy(book, 10.0, limit_price=lim)
    assert f["filled"] is True
    assert f["is_maker"] is False
    assert f["fee"] > 0
    assert f["avg_price"] == pytest.approx(0.52)


def test_affordable_spend_leaves_room_for_taker_fee():
    avail = 8.0
    spend = fills.affordable_spend(avail, 0.50, is_maker=False)
    sh = spend / 0.50
    assert spend + fills.taker_fee(sh, 0.50) <= avail + 1e-9
    assert fills.affordable_spend(avail, 0.50, is_maker=True) == avail


def test_edge_math_always_subtracts_taker_fee(monkeypatch):
    from bots.bot_momentum import MomentumBot

    monkeypatch.setattr(config, "ORDER_STYLE", "limit")
    monkeypatch.setattr(config, "LIMIT_PRICE_MODE", "cap_ask")
    bot = MomentumBot(name="mom-fee")
    assert bot._assumed_maker() is False
    ey, en = bot._side_net_edges(0.60, 0.52, 0.48)
    taker = fills.fee_per_share(0.52, is_maker=False)
    assert ey == pytest.approx(0.60 - 0.52 - taker)

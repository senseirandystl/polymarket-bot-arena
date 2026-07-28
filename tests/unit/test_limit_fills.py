"""Limit-order fill simulation + maker/taker fee roles."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
import polymarket_fills as fills


def _book(bid=0.48, ask=0.52, size=100.0):
    return {
        "valid": True,
        "bids": [(bid, size)],
        "asks": [(ask, size)],
        "min_order_size": 5,
    }


def test_taker_fee_positive_maker_zero():
    assert fills.taker_fee(10, 0.5) > 0
    assert fills.maker_fee(10, 0.5) == 0.0
    assert fills.fee_per_share(0.5, is_maker=True) == 0.0
    assert fills.fee_per_share(0.5, is_maker=False) > 0


def test_marketable_limit_is_taker(monkeypatch):
    monkeypatch.setattr(config, "LIMIT_PAPER_ASSUME_MAKER_FILL", True)
    book = _book(ask=0.50)
    f = fills.simulate_limit_buy(book, 10.0, limit_price=0.55)
    assert f["filled"] is True
    assert f["is_maker"] is False
    assert f["fee"] > 0


def test_resting_limit_is_maker(monkeypatch):
    monkeypatch.setattr(config, "LIMIT_PAPER_ASSUME_MAKER_FILL", True)
    book = _book(ask=0.55, bid=0.48)
    f = fills.simulate_limit_buy(book, 10.0, limit_price=0.49)
    assert f["filled"] is True
    assert f["is_maker"] is True
    assert f["fee"] == 0.0
    assert f["avg_price"] == pytest.approx(0.49)


def test_passive_mid_price_below_ask():
    book = _book(bid=0.48, ask=0.52)
    lim = fills.limit_buy_price(book, mid=0.50, mode="passive_mid")
    assert lim is not None
    assert lim < 0.52

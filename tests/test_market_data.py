"""Tests for the market-data warmer store + warm-once refresh."""

from unittest import mock

import polymarket_markets
from arena import market_data
from arena.signals import build_combined_signals
from signals import clean_tick


class _FakeDiscovery:
    def __init__(self, market):
        self._m = market

    def current_market_snapshot(self):
        return self._m


class _FakeCvd:
    def get_cvd(self, cond):
        return 0.4


class _FakePm:
    def get_momentum(self, tok):
        return {"momentum": 0.05, "prices": [0.49, 0.50, 0.51]}


def _book(price):
    return {"valid": True, "asks": [(price, 100.0)], "bids": [(price - 0.02, 100.0)],
            "best_ask": price, "best_bid": price - 0.02, "min_order_size": 0}


def test_store_put_get_prune():
    store = market_data.MarketDataStore()
    store.put("m1", {"yes_price": 0.5})
    store.put("m2", {"yes_price": 0.6})
    assert store.get("m1")["yes_price"] == 0.5
    assert store.get("missing") is None
    store.prune(keep_market_id="m2")
    assert store.get("m1") is None
    assert store.get("m2") is not None


def test_warm_once_populates_store():
    clean_tick.reset()
    market = {
        "id": "0xabc", "condition_id": "0xcond",
        "polymarket_token_id": "YES", "polymarket_no_token_id": "NO",
    }
    warmer = market_data.MarketDataWarmer(
        _FakeDiscovery(market), _FakeCvd(), _FakePm(),
    )
    books = {"YES": _book(0.45), "NO": _book(0.50)}
    with mock.patch.object(polymarket_markets, "get_order_book",
                           lambda tok: books[tok]):
        warmer._warm_once()

    snap = market_data.store().get("0xabc")
    assert snap is not None
    assert snap["yes_price"] == 0.44          # midpoint of bid 0.43 / ask 0.45
    assert snap["no_price"] == 0.49           # midpoint of bid 0.48 / ask 0.50
    assert snap["cvd"] == 0.4
    assert snap["pm_momentum"] == 0.05
    assert snap["yes_book"]["valid"] and snap["no_book"]["valid"]
    # OBI is derived from the YES book and lands in [-1, 1].
    assert -1.0 <= snap["obi"] <= 1.0


def test_build_combined_signals_warm_path_uses_warm_values():
    warm = {"obi": 0.3, "cvd": -0.2, "pm_momentum": 0.07, "pm_prices": [0.5]}
    # No feeds / no market network needed on the warm path.
    sig = build_combined_signals(None, None, None, {"id": "m"}, warm=warm)
    assert sig["obi"] == 0.3
    assert sig["cvd"] == -0.2
    assert sig["pm_momentum"] == 0.07

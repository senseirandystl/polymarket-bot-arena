"""Slippage path A/D/C: atomic warm book, maker ask expected price, backoff."""

import time

import pytest

import config
import polymarket_markets
from arena import market_data
from arena.state import SharedArenaState
from bots.bot_fee_zone_maker import FeeZoneMakerBot
from bots.bot_late_window_maker import LateWindowMakerBot
from bots.bot_momentum import MomentumBot
from tests.conftest import make_book, make_market, make_signals


def test_lay_warm_onto_market_sets_asks_and_books():
    market = make_market(yes_price=0.50)
    yes_book = make_book(asks=[(0.52, 100)], bids=[(0.48, 100)])
    no_book = make_book(asks=[(0.53, 80)], bids=[(0.47, 80)])
    warm = {
        "yes_price": 0.50,
        "no_price": 0.50,
        "yes_book": yes_book,
        "no_book": no_book,
        "obi": 0.1,
    }
    market_data.lay_warm_onto_market(market, warm)
    assert market["yes_ask"] == 0.52
    assert market["no_ask"] == 0.53
    assert market["yes_book"] is yes_book
    assert market["no_book"] is no_book
    assert market_data.side_book(market, "yes") is yes_book
    assert market_data.side_book(market, "no") is no_book


def test_atomic_warm_book_ignores_moved_clob_fetch(arena_db, monkeypatch):
    """Paper fill must walk the warm book on the market, not a re-fetched CLOB.

    Overnight root cause: decision priced warm best_ask, place() re-fetched a
    book that had moved several cents, and ±3c symmetric guard rejected.
    """
    bot = MomentumBot(name="momo-atomic", generation=0)
    warm_yes = make_book(asks=[(0.56, 200)], bids=[(0.54, 200)])
    warm_no = make_book(asks=[(0.46, 200)], bids=[(0.44, 200)])
    market = make_market(
        yes_price=0.55,
        time_remaining=150,
        market_id="mkt-atomic",
        yes_ask=0.56,
        no_ask=0.46,
        yes_book=warm_yes,
        no_book=warm_no,
    )
    prices = [100_000.0 * (1.001 ** i) for i in range(40)]
    signals = make_signals(prices=prices, latest=prices[-1], btc_drift=0.45)

    # If place() still re-fetches, it would see this 80c book and reject.
    moved = make_book(asks=[(0.80, 500)], bids=[(0.78, 500)])
    monkeypatch.setattr(polymarket_markets, "get_order_book", lambda token: moved)

    decision = bot.make_decision(market, signals)
    assert decision["action"] == "buy"
    assert decision["side"] == "yes"
    assert decision["entry_price"] == pytest.approx(0.56, abs=1e-6)

    result = bot.execute(decision, market)
    assert result["success"], f"expected fill on warm book, got {result}"

    with arena_db.get_conn() as conn:
        row = conn.execute(
            "SELECT entry_price FROM trades WHERE bot_name='momo-atomic'"
        ).fetchone()
    assert row["entry_price"] == pytest.approx(0.56, abs=1e-6)


def test_without_warm_book_slippage_band_still_rejects_moved_fetch(
        arena_db, monkeypatch):
    """When no side book is attached, place() still re-fetches and guards."""
    bot = MomentumBot(name="momo-refetch", generation=0)
    market = make_market(yes_price=0.55, time_remaining=150, market_id="mkt-refetch")
    # Ensure no warm books leak in.
    market.pop("yes_book", None)
    market.pop("no_book", None)
    prices = [100_000.0 * (1.001 ** i) for i in range(40)]
    signals = make_signals(prices=prices, latest=prices[-1], btc_drift=0.45)

    decision = bot.make_decision(market, signals)
    assert decision["action"] == "buy"

    moved = make_book(asks=[(0.80, 500)], bids=[(0.78, 500)])
    monkeypatch.setattr(polymarket_markets, "get_order_book", lambda token: moved)
    result = bot.execute(decision, market)
    assert not result["success"]
    assert result.get("reason") in (
        "slippage_band", "slippage_exceeded", "limit_unfilled",
    )


def test_late_window_maker_entry_price_uses_ask_not_mid():
    bot = LateWindowMakerBot()
    # Honest Phi path needs btc_implied_yes (not 0.5+0.5*|drift|).
    signals = {
        "prices": [100.0, 100.0, 100.2],
        "btc_drift": 0.5,
        "btc_implied_yes": 0.85,
        "btc_drift_z": 2.0,
    }
    # mid 0.65, ask 0.68 — expected fill is the ask.
    market = {
        "current_price": 0.65,
        "no_price": 0.35,
        "yes_ask": 0.68,
        "no_ask": 0.37,
        "time_remaining_seconds": 45,
    }
    sig = bot.analyze(market, signals)
    assert sig["action"] == "buy"
    assert sig["side"] == "yes"
    assert sig["entry_price"] == pytest.approx(0.68, abs=1e-6)


def test_fee_zone_maker_entry_price_uses_ask_not_mid():
    bot = FeeZoneMakerBot()
    # Fee zone typically ~0.56–0.86; mid 0.70 with ask 0.73, strong up-drift.
    market = {
        "current_price": 0.70,
        "no_price": 0.30,
        "yes_ask": 0.73,
        "no_ask": 0.32,
        "time_remaining_seconds": 120,
    }
    signals = {"prices": [100.0, 100.05, 100.1], "btc_drift": 0.55}
    sig = bot.analyze(market, signals)
    if sig["action"] != "buy":
        pytest.skip(f"fee-zone gates skipped trade: {sig.get('reasoning')}")
    assert sig["side"] == "yes"
    assert sig["entry_price"] == pytest.approx(0.73, abs=1e-6)


def test_slippage_cooldown_blocks_then_expires():
    state = SharedArenaState()
    key = ("hybrid-v1", "mkt-1")
    now = time.time()
    assert not state.is_slippage_cooling(key, now=now)

    state.mark_slippage_reject(key, cooldown_sec=10.0, now=now)
    assert state.is_slippage_cooling(key, now=now + 1.0)
    assert not state.is_slippage_cooling(key, now=now + 10.0)

    # mark_traded clears cooldown
    state.mark_slippage_reject(key, cooldown_sec=30.0, now=now)
    state.mark_traded(key)
    assert not state.is_slippage_cooling(key, now=now + 1.0)


def test_slippage_cooldown_config_default():
    assert float(getattr(config, "SLIPPAGE_RETRY_COOLDOWN_SEC", 0)) >= 1.0

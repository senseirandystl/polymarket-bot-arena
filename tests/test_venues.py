"""Tests for order-book fill simulation, fees, and the venue engines."""

import sys
import types

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "venues_test.db")
    db_module.init_db()
    return db_module


# ---------------------------------------------------------------------------
# Fees + fill simulation (pure)
# ---------------------------------------------------------------------------
def test_taker_fee_symmetric_around_50c():
    from polymarket_fills import taker_fee
    # 30c and 70c cost the same dollar fee (p*(1-p) is symmetric).
    assert taker_fee(100, 0.30) == pytest.approx(taker_fee(100, 0.70))
    # zero at the extremes, max at 0.5
    assert taker_fee(100, 0.0) == 0
    assert taker_fee(100, 0.5) > taker_fee(100, 0.1)


def test_simulate_fill_walks_book_with_slippage():
    from polymarket_fills import simulate_fill
    book = {"valid": True, "asks": [(0.30, 10), (0.31, 10), (0.32, 100)]}
    # $6 budget: 10 sh @0.30 = $3, then 9.677 sh @0.31 = $3 -> avg > 0.30
    fill = simulate_fill(book, 6.0)
    assert fill["filled"] and fill["full"]
    assert fill["cost"] == pytest.approx(6.0)
    assert 0.30 < fill["avg_price"] < 0.31
    assert fill["shares"] == pytest.approx(10 + 3.0 / 0.31, rel=1e-3)


def test_simulate_fill_partial_when_book_thin():
    from polymarket_fills import simulate_fill
    book = {"valid": True, "asks": [(0.40, 5)]}  # only $2 of depth
    fill = simulate_fill(book, 10.0)
    assert fill["filled"] and not fill["full"]
    assert fill["cost"] == pytest.approx(2.0)
    assert fill["shares"] == pytest.approx(5.0)


def test_simulate_fill_empty_book():
    from polymarket_fills import simulate_fill
    assert simulate_fill({"valid": False}, 10.0)["filled"] is False


# ---------------------------------------------------------------------------
# Paper engine (book + shared bankroll)
# ---------------------------------------------------------------------------
def _mock_book(monkeypatch, asks, min_size=5.0):
    import polymarket_markets
    monkeypatch.setattr(polymarket_markets, "get_order_book",
                        lambda tok: {"valid": True, "asks": asks,
                                     "min_order_size": min_size})


def test_paper_place_fills_from_book(db, monkeypatch):
    from venues.paper import PaperEngine
    _mock_book(monkeypatch, [(0.50, 1000)])
    db.set_paper_bankroll(100.0)
    market = {"id": "m1", "question": "BTC", "polymarket_token_id": "up",
              "polymarket_no_token_id": "down"}
    res = PaperEngine().place(bot_name="momentum-v1", side="yes", amount=10.0,
                              market=market, mode="paper")
    assert res.success and res.fill_source == "paper_sim"
    assert res.shares == pytest.approx(20.0)      # $10 / 0.50
    with db.get_conn() as c:
        row = dict(c.execute("SELECT amount, fee, fill_source, trade_id, mode "
                             "FROM trades WHERE market_id='m1'").fetchone())
    assert row["fill_source"] == "paper_sim" and row["trade_id"] is None
    assert row["fee"] > 0


def test_paper_respects_shared_bankroll(db, monkeypatch):
    from venues.paper import PaperEngine
    _mock_book(monkeypatch, [(0.50, 100000)])
    db.set_paper_bankroll(8.0)                    # only $8 in the pool
    market = {"id": "m2", "polymarket_token_id": "up", "polymarket_no_token_id": "down"}
    res = PaperEngine().place(bot_name="b", side="yes", amount=50.0,
                              market=market, mode="paper")
    # Capped to the $8 available (minus nothing yet), so cost <= 8.
    assert res.success and res.entry_price == pytest.approx(0.50)
    with db.get_conn() as c:
        amt = c.execute("SELECT amount FROM trades WHERE market_id='m2'").fetchone()[0]
    assert amt <= 8.0 + 1e-6


def test_paper_skips_when_bankroll_exhausted(db, monkeypatch):
    from venues.paper import PaperEngine
    _mock_book(monkeypatch, [(0.50, 100000)])
    db.set_paper_bankroll(0.0)
    market = {"id": "m3", "polymarket_token_id": "up", "polymarket_no_token_id": "down"}
    res = PaperEngine().place(bot_name="b", side="yes", amount=10.0,
                              market=market, mode="paper")
    assert not res.success and res.reason == "insufficient_bankroll"


def test_paper_skips_below_min_size(db, monkeypatch):
    from venues.paper import PaperEngine
    _mock_book(monkeypatch, [(0.50, 100000)], min_size=100.0)  # need 100 shares
    db.set_paper_bankroll(100.0)
    market = {"id": "m4", "polymarket_token_id": "up", "polymarket_no_token_id": "down"}
    res = PaperEngine().place(bot_name="b", side="yes", amount=5.0,  # ~10 shares
                              market=market, mode="paper")
    assert not res.success and res.reason == "below_min_size"


# ---------------------------------------------------------------------------
# Live engine
# ---------------------------------------------------------------------------
def test_live_place_missing_token_id(db):
    from venues.live import LiveEngine
    res = LiveEngine().place(bot_name="b", side="yes", amount=1.0,
                             market={"id": "m5", "question": "q"}, mode="live")
    assert not res.success and res.reason == "missing_token_id"


def test_live_place_records_fee(db, monkeypatch):
    from venues.live import LiveEngine
    stub = types.ModuleType("polymarket_client")
    stub.place_market_order = lambda **kw: {
        "success": True, "order_id": "0xabc", "price": 0.55, "size": 3.6}
    monkeypatch.setitem(sys.modules, "polymarket_client", stub)
    market = {"id": "m6", "question": "q", "polymarket_token_id": "tok"}
    res = LiveEngine().place(bot_name="b", side="yes", amount=2.0,
                             market=market, mode="live")
    assert res.success and res.fill_source == "polymarket"
    with db.get_conn() as c:
        row = dict(c.execute("SELECT trade_id, fee, venue FROM trades "
                             "WHERE market_id='m6'").fetchone())
    assert row["trade_id"] == "0xabc" and row["venue"] == "polymarket" and row["fee"] > 0

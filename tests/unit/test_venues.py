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


def test_simulate_fill_shares_exact_count_with_slippage():
    from polymarket_fills import simulate_fill_shares
    book = {"valid": True, "asks": [(0.30, 10), (0.31, 10), (0.32, 100)]}
    fill = simulate_fill_shares(book, 15.0)  # 10 @0.30 + 5 @0.31
    assert fill["filled"] and fill["full"]
    assert fill["shares"] == pytest.approx(15.0)
    assert fill["cost"] == pytest.approx(10 * 0.30 + 5 * 0.31)
    assert 0.30 < fill["avg_price"] < 0.31


def test_simulate_fill_shares_partial_when_book_thin():
    from polymarket_fills import simulate_fill_shares
    book = {"valid": True, "asks": [(0.40, 5)]}  # only 5 shares of depth
    fill = simulate_fill_shares(book, 20.0)
    assert fill["filled"] and not fill["full"]
    assert fill["shares"] == pytest.approx(5.0)


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


def test_paper_share_matched_fill_uses_exact_shares(db, monkeypatch):
    from venues.paper import PaperEngine
    _mock_book(monkeypatch, [(0.60, 1000)], min_size=0.0)
    db.set_paper_bankroll(1000.0)
    market = {"id": "ms", "polymarket_token_id": "up", "polymarket_no_token_id": "down"}
    res = PaperEngine().place(bot_name="arbitrage-v1", side="yes", amount=0.0,
                              market=market, mode="paper", target_shares=20.0)
    assert res.success
    assert res.shares == pytest.approx(20.0)          # exact share count, not USD
    assert res.entry_price == pytest.approx(0.60)


def test_paper_share_matched_skips_when_depth_insufficient(db, monkeypatch):
    from venues.paper import PaperEngine
    _mock_book(monkeypatch, [(0.60, 8)], min_size=0.0)   # only 8 shares of depth
    db.set_paper_bankroll(1000.0)
    market = {"id": "md", "polymarket_token_id": "up", "polymarket_no_token_id": "down"}
    res = PaperEngine().place(bot_name="arbitrage-v1", side="yes", amount=0.0,
                              market=market, mode="paper", target_shares=20.0)
    assert not res.success and res.reason == "insufficient_depth"


def test_paper_rejects_fill_above_limit_price(db, monkeypatch):
    """A BUY whose realized avg price exceeds limit_price is rejected — the
    decision→fill slippage guard that keeps thin edges from filling adversely."""
    from venues.paper import PaperEngine
    _mock_book(monkeypatch, [(0.60, 100000)])   # book moved up to 0.60
    db.set_paper_bankroll(100.0)
    market = {"id": "sl1", "polymarket_token_id": "up", "polymarket_no_token_id": "down"}
    res = PaperEngine().place(bot_name="b", side="yes", amount=10.0,
                              market=market, mode="paper", limit_price=0.55)
    # Limit below the ask does not cross — no invented fill, no slippage walk.
    assert not res.success and res.reason in ("slippage_exceeded", "limit_unfilled")
    with db.get_conn() as c:
        assert c.execute("SELECT COUNT(*) FROM trades WHERE market_id='sl1'").fetchone()[0] == 0


def test_paper_allows_fill_at_or_below_limit_price(db, monkeypatch):
    from venues.paper import PaperEngine
    _mock_book(monkeypatch, [(0.50, 100000)])
    db.set_paper_bankroll(100.0)
    market = {"id": "sl2", "polymarket_token_id": "up", "polymarket_no_token_id": "down"}
    res = PaperEngine().place(bot_name="b", side="yes", amount=10.0,
                              market=market, mode="paper", limit_price=0.53)
    assert res.success and res.entry_price == pytest.approx(0.50)


def test_paper_share_matched_rejects_above_limit_price(db, monkeypatch):
    from venues.paper import PaperEngine
    _mock_book(monkeypatch, [(0.60, 100000)], min_size=0.0)
    db.set_paper_bankroll(1000.0)
    market = {"id": "sl3", "polymarket_token_id": "up", "polymarket_no_token_id": "down"}
    res = PaperEngine().place(bot_name="arbitrage-v1", side="yes", amount=0.0,
                              market=market, mode="paper", target_shares=20.0,
                              limit_price=0.55)
    assert not res.success and res.reason == "slippage_exceeded"


def test_paper_uses_passed_book_instead_of_fetching(db, monkeypatch):
    """When a book is supplied (arbitrage atomic fill), the engine fills against
    THAT snapshot and never re-reads the CLOB — so decision and fill can't drift."""
    from venues.paper import PaperEngine
    import polymarket_markets

    def _boom(tok):
        raise AssertionError("engine must not re-fetch the book when one is passed")
    monkeypatch.setattr(polymarket_markets, "get_order_book", _boom)
    db.set_paper_bankroll(1000.0)
    market = {"id": "sl4", "polymarket_token_id": "up", "polymarket_no_token_id": "down"}
    passed = {"valid": True, "asks": [(0.48, 1000)], "min_order_size": 0.0}
    res = PaperEngine().place(bot_name="arbitrage-v1", side="yes", amount=0.0,
                              market=market, mode="paper", target_shares=20.0,
                              book=passed)
    assert res.success and res.entry_price == pytest.approx(0.48)


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
    import polymarket_markets
    stub = types.ModuleType("polymarket_client")
    stub.place_market_order = lambda **kw: {
        "success": True, "order_id": "0xabc", "price": 0.55, "size": 3.6}
    stub.place_limit_order = lambda **kw: {
        "success": True, "order_id": "0xabc", "price": 0.55, "size": 3.6,
        "status": "matched"}
    monkeypatch.setitem(sys.modules, "polymarket_client", stub)
    monkeypatch.setattr(
        polymarket_markets, "get_order_book",
        lambda tok: {"valid": True, "asks": [(0.55, 100)], "bids": [(0.50, 100)]},
    )
    market = {"id": "m6", "question": "q", "polymarket_token_id": "tok",
              "current_price": 0.52}
    res = LiveEngine().place(bot_name="b", side="yes", amount=2.0,
                             market=market, mode="live")
    assert res.success and res.fill_source == "polymarket"
    with db.get_conn() as c:
        row = dict(c.execute("SELECT trade_id, fee, venue, mode FROM trades "
                             "WHERE market_id='m6'").fetchone())
    assert row["trade_id"] == "0xabc" and row["venue"] == "polymarket" and row["fee"] > 0
    assert row["mode"] == "live"


# ---------------------------------------------------------------------------
# Shared paper bankroll top-up
# ---------------------------------------------------------------------------
def test_topup_sets_available_to_entered_amount_after_losses(db, monkeypatch):
    # Simulate a resolved loss and an open position eating into the pool.
    db.set_paper_bankroll(100.0)
    with db.get_conn() as c:
        c.execute("INSERT INTO trades (bot_name, market_id, side, amount, fee, "
                  "venue, mode, outcome, pnl) VALUES "
                  "('b','ml','yes',20.0,0.5,'polymarket','paper','loss',-20.5)")
        c.execute("INSERT INTO trades (bot_name, market_id, side, amount, fee, "
                  "venue, mode, outcome, pnl) VALUES "
                  "('b','mo','yes',30.0,0.7,'polymarket','paper',NULL,NULL)")
    # available = 100 - 20.5 - 30.7 = 48.8
    assert db.get_paper_available() == pytest.approx(48.8)
    # Top up to 200: available should become exactly 200, history preserved.
    db.topup_paper_bankroll(200.0)
    assert db.get_paper_available() == pytest.approx(200.0)
    with db.get_conn() as c:
        n = c.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert n == 2  # trade history untouched


def test_topup_rejects_negative(db):
    with pytest.raises(ValueError):
        db.topup_paper_bankroll(-5.0)

"""End-to-end paper-trading cycle with mocked venue data.

Drives the real decision → execution → resolution path: BaseBot.make_decision
on crafted signals, PaperEngine fill against a mocked CLOB book, the trade row
landing pending in an isolated DB, then market resolution and shared-pool
accounting — everything except live order submission, exactly like paper mode.
"""

import pytest

import polymarket_markets
from bots.bot_momentum import MomentumBot
from tests.conftest import make_market, make_signals, make_book


@pytest.fixture()
def mock_book(monkeypatch):
    """Serve a deep, tight book for every token the venue asks about."""
    book = make_book(asks=[(0.55, 200), (0.57, 400)],
                     bids=[(0.53, 200), (0.51, 400)])
    monkeypatch.setattr(polymarket_markets, "get_order_book", lambda token: book)
    return book


@pytest.fixture()
def bullish_setup(arena_db):
    """A bot + market + signals combination that clears every decision guard."""
    bot = MomentumBot(name="momo-cycle", generation=0)
    market = make_market(yes_price=0.55, time_remaining=150, market_id="mkt-cycle")
    prices = [100_000.0 * (1.001 ** i) for i in range(40)]
    signals = make_signals(prices=prices, latest=prices[-1], btc_drift=0.45)
    return bot, market, signals


def test_full_win_cycle(arena_db, mock_book, bullish_setup):
    bot, market, signals = bullish_setup

    # 1. Decision: strong up-drift + uptrend must produce a YES buy.
    decision = bot.make_decision(market, signals)
    assert decision["action"] == "buy"
    assert decision["side"] == "yes"
    assert decision["suggested_amount"] > 0
    assert decision.get("entry_price") is not None

    # 2. Execution: paper engine fills against the mocked book.
    result = bot.execute(decision, market)
    assert result["success"], f"paper fill failed: {result.get('reason')}"

    with arena_db.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM trades WHERE bot_name='momo-cycle'").fetchone()
    assert row["outcome"] is None                    # pending until resolution
    assert row["fill_source"] == "paper_sim"
    assert row["mode"] == "paper"
    assert row["shares_bought"] > 0
    assert 0.54 <= row["entry_price"] <= 0.58        # book walk from 0.55
    assert row["fee"] > 0                            # taker fee applied

    # 3. Open cost is reserved from the shared pool.
    bankroll = arena_db.get_paper_bankroll()
    open_cost = row["amount"] + row["fee"]
    assert arena_db.get_paper_available() == pytest.approx(
        bankroll - open_cost, abs=0.01)

    # 4. Resolution (market resolved UP → YES wins $1/share).
    pnl = row["shares_bought"] * 1.0 - row["amount"] - row["fee"]
    assert pnl > 0
    arena_db.resolve_trade(row["id"], "win", pnl)

    # 5. Pool releases the reservation and credits the realized P&L.
    assert arena_db.get_paper_available() == pytest.approx(
        bankroll + pnl, abs=0.01)
    perf = arena_db.get_bot_performance("momo-cycle", hours=1)
    assert perf["wins"] == 1 and perf["total_pnl"] == pytest.approx(pnl, abs=0.01)


def test_full_loss_cycle(arena_db, mock_book, bullish_setup):
    bot, market, signals = bullish_setup
    decision = bot.make_decision(market, signals)
    result = bot.execute(decision, market)
    assert result["success"]

    with arena_db.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM trades WHERE bot_name='momo-cycle'").fetchone()

    # Market resolved DOWN → YES shares expire worthless.
    bankroll = arena_db.get_paper_bankroll()
    pnl = -(row["amount"] + row["fee"])
    arena_db.resolve_trade(row["id"], "loss", pnl)
    assert arena_db.get_paper_available() == pytest.approx(
        bankroll + pnl, abs=0.01)
    perf = arena_db.get_bot_performance("momo-cycle", hours=1)
    assert perf["losses"] == 1 and perf["total_pnl"] == pytest.approx(pnl, abs=0.01)


def test_dead_book_never_logs_a_trade(arena_db, monkeypatch, bullish_setup):
    bot, market, signals = bullish_setup
    monkeypatch.setattr(polymarket_markets, "get_order_book",
                        lambda token: {"valid": False})
    decision = bot.make_decision(market, signals)
    assert decision["action"] == "buy"
    result = bot.execute(decision, market)
    assert not result["success"]
    with arena_db.get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) c FROM trades").fetchone()["c"]
    assert n == 0                                    # no phantom fills


def test_slippage_band_rejects_moved_book(arena_db, monkeypatch, bullish_setup):
    """BUG #28: a fill far from the decision price in EITHER direction is
    stale data — the venue must reject rather than log the trade."""
    bot, market, signals = bullish_setup
    decision = bot.make_decision(market, signals)
    # The book gapped way above the decision's expected entry (~0.55).
    moved = make_book(asks=[(0.80, 500)], bids=[(0.78, 500)])
    monkeypatch.setattr(polymarket_markets, "get_order_book", lambda token: moved)
    result = bot.execute(decision, market)
    assert not result["success"]
    with arena_db.get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) c FROM trades").fetchone()["c"]
    assert n == 0


def test_neutral_tape_holds_and_places_nothing(arena_db, mock_book):
    """Flat drift + coin-flip mid sits in the dead zone → no trade attempted."""
    bot = MomentumBot(name="momo-flat", generation=0)
    market = make_market(yes_price=0.50)
    decision = bot.make_decision(market, make_signals())
    assert decision["action"] in ("hold", "skip")

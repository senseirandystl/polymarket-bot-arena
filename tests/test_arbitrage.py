"""Tests for the market-neutral ArbitrageBot decision logic.

Only ``make_decision`` is exercised (no DB / no order placement): it reads the
two best asks off the CLOB book and fires only when YES_ask + NO_ask + both
taker fees clears $1.00 by at least ``config.ARBITRAGE_MIN_MARGIN``.
"""

from unittest import mock

import pytest

import config
import polymarket_markets
from arena import market_data
from bots.bot_arbitrage import ArbitrageBot
from polymarket_fills import taker_fee


@pytest.fixture(autouse=True)
def _clear_warm_store():
    """Isolate from the shared market-data warm cache so these tests exercise
    the arb bot's own (mocked) book reads, not a snapshot another test left."""
    market_data.store().prune(keep_market_id=None)
    yield
    market_data.store().prune(keep_market_id=None)

MARKET = {
    "id": "0xabc",
    "question": "BTC Up?",
    "polymarket_token_id": "YES",
    "polymarket_no_token_id": "NO",
}


def _book(price):
    return {"valid": True, "asks": [(price, 500.0)], "bids": [],
            "best_ask": price, "best_bid": None, "min_order_size": 0}


def _patched(yes_ask, no_ask):
    books = {"YES": _book(yes_ask), "NO": _book(no_ask)}
    return mock.patch.object(polymarket_markets, "get_order_book",
                             lambda tok: books[tok])


def test_fires_when_asks_sum_below_one_with_margin():
    bot = ArbitrageBot()
    with _patched(0.42, 0.48):  # 0.90 + fees ≈ 0.935 → ~6.5c edge/pair
        sig = bot.make_decision(MARKET, {})
    assert sig["action"] == "buy"
    legs = {leg["side"] for leg in sig["legs"]}
    assert legs == {"yes", "no"}  # both legs, market-neutral


def test_skips_when_asks_sum_above_one():
    bot = ArbitrageBot()
    with _patched(0.50, 0.52):  # 1.02 → clearly no arb
        sig = bot.make_decision(MARKET, {})
    assert sig["action"] == "skip"


def test_skips_when_margin_too_thin():
    # Edge exists but is below ARBITRAGE_MIN_MARGIN once fees are counted.
    bot = ArbitrageBot()
    yes_ask, no_ask = 0.49, 0.50  # sum 0.99, fees ≈ 0.035 → net negative
    fee = taker_fee(1.0, yes_ask) + taker_fee(1.0, no_ask)
    assert (1.0 - (yes_ask + no_ask + fee)) < config.ARBITRAGE_MIN_MARGIN
    with _patched(yes_ask, no_ask):
        assert bot.make_decision(MARKET, {})["action"] == "skip"


def test_skips_when_book_missing():
    bot = ArbitrageBot()
    with mock.patch.object(polymarket_markets, "get_order_book",
                           lambda tok: {"valid": False}):
        assert bot.make_decision(MARKET, {})["action"] == "skip"


def test_skips_when_token_ids_absent():
    bot = ArbitrageBot()
    assert bot.make_decision({"id": "x"}, {})["action"] == "skip"


# --- Depth-aware / share-matched regression tests --------------------------
# These lock in the two bugs the arb rewrite fixed: (1) edge must be measured
# from the depth-walked VWAP, not the thin top-of-book; (2) both legs must be
# share-matched so the position is genuinely neutral.

def _multi(levels):
    """Build a book from [(price, size), ...] ask levels (cheapest first)."""
    best = levels[0][0]
    return {"valid": True, "asks": list(levels), "bids": [],
            "best_ask": best, "best_bid": None, "min_order_size": 0}


def _patched_books(yes_book, no_book):
    books = {"YES": yes_book, "NO": no_book}
    return mock.patch.object(polymarket_markets, "get_order_book",
                             lambda tok: books[tok])


def test_skips_when_top_of_book_edge_evaporates_on_depth():
    """A tiny cheap top level (looks like edge) with expensive depth behind it
    must NOT trade — walking to target size costs > $1/pair."""
    bot = ArbitrageBot()
    # Best asks sum to 0.90 (fake edge) but only 1 share each; the real depth
    # for ~20 shares sits at 0.55/0.55 = 1.10/pair → no edge.
    yes = _multi([(0.45, 1.0), (0.55, 1000.0)])
    no = _multi([(0.45, 1.0), (0.55, 1000.0)])
    with _patched_books(yes, no):
        sig = bot.make_decision(MARKET, {})
    assert sig["action"] == "skip"


def test_fires_and_legs_are_share_matched():
    """When real depth clears the margin, both legs carry the SAME share count."""
    bot = ArbitrageBot()
    # Deep books at 0.42 / 0.48 → VWAP sum 0.90, clears margin after fees.
    yes = _multi([(0.42, 1000.0)])
    no = _multi([(0.48, 1000.0)])
    with _patched_books(yes, no):
        sig = bot.make_decision(MARKET, {})
    assert sig["action"] == "buy"
    shares = {leg["shares"] for leg in sig["legs"]}
    assert len(shares) == 1  # identical share count on both legs
    assert next(iter(shares)) > 0


def test_matched_shares_limited_by_thinner_book():
    """The matched size can't exceed the thinner leg's available depth."""
    bot = ArbitrageBot()
    yes = _multi([(0.42, 1000.0)])       # deep
    no = _multi([(0.48, 8.0)])           # only 8 shares available
    with _patched_books(yes, no):
        sig = bot.make_decision(MARKET, {})
    assert sig["action"] == "buy"
    for leg in sig["legs"]:
        assert leg["shares"] <= 8.0 + 1e-9


# --- Atomic re-validation at execute() -------------------------------------
# The real overnight bug: make_decision found a sub-$1 window, but by the time
# execute() filled, YES+NO had drifted back above $1 (its natural resting vig)
# and both legs filled at a guaranteed loss. execute() must RE-READ the books,
# re-check the combined edge, and abort if the edge is gone — never fill a pair
# it hasn't just re-validated.

@pytest.fixture()
def _paper_db(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "arb_exec.db")
    db_module.init_db()
    db_module.set_paper_bankroll(1000.0)
    return db_module


def _good_signal(shares=20.0):
    """A buy signal as make_decision would emit on a profitable snapshot."""
    return {
        "action": "buy", "side": "yes", "confidence": 0.5, "reasoning": "arb",
        "suggested_amount": 19.0, "features": None,
        "legs": [
            {"side": "yes", "shares": shares, "amount": 9.0, "vwap": 0.45},
            {"side": "no", "shares": shares, "amount": 10.0, "vwap": 0.50},
        ],
    }


def test_execute_aborts_when_edge_gone_at_fill(_paper_db):
    """Books moved above $1 combined by fill time → place NOTHING."""
    bot = ArbitrageBot()
    # At execute time YES+NO VWAP sums to 1.04 → no arb anymore.
    yes = _multi([(0.53, 1000.0)])
    no = _multi([(0.51, 1000.0)])
    with _patched_books(yes, no):
        res = bot.execute(_good_signal(), MARKET)
    assert not res["success"]
    assert res["reason"] == "arb_edge_gone"
    with _paper_db.get_conn() as c:
        assert c.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0


def test_execute_fills_both_legs_when_edge_holds(_paper_db):
    """Edge still present at fill → both legs fill against the re-read snapshot."""
    bot = ArbitrageBot()
    yes = _multi([(0.42, 1000.0)])
    no = _multi([(0.48, 1000.0)])
    with _patched_books(yes, no):
        res = bot.execute(_good_signal(), MARKET)
    assert res["success"]
    with _paper_db.get_conn() as c:
        rows = c.execute("SELECT side, shares_bought FROM trades ORDER BY side").fetchall()
    assert len(rows) == 2  # both legs
    yes_sh = [r["shares_bought"] for r in rows if r["side"] == "yes"][0]
    no_sh = [r["shares_bought"] for r in rows if r["side"] == "no"][0]
    assert yes_sh == pytest.approx(no_sh)  # share-matched (true neutral)


def test_execute_aborts_when_pool_cannot_cover_both_legs(_paper_db):
    """Leg 1 affordable but not leg 2 -> place NOTHING (no naked leg).

    Live bug (2026-07-16): the shared pool covered the $13 YES leg but had
    $0.94 left for the NO leg — the resulting naked position lost -$13.32 and
    wiped out every paired arb gain of the session.
    """
    bot = ArbitrageBot()
    yes = _multi([(0.42, 1000.0)])
    no = _multi([(0.48, 1000.0)])
    # Pool covers ONE leg (~$8.40-9.60 at 20sh) but not both (~$18 + fees).
    _paper_db.set_paper_bankroll(10.0)
    with _patched_books(yes, no):
        res = bot.execute(_good_signal(), MARKET)
    assert not res["success"]
    assert res["reason"] == "arb_insufficient_bankroll"
    with _paper_db.get_conn() as c:
        assert c.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0

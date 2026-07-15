"""Tests for two-sided (YES/NO) net-edge side selection in make_decision."""

import polymarket_fills
from bots.bot_momentum import MomentumBot


def _bot():
    return MomentumBot(name="momentum-test", generation=0)


# --- Task 1: pure fair-value + edge helpers ---

def test_compute_fair_yes_identity_with_combined():
    # price_tilt = (0.60-0.5)*1*0.5 = 0.05; alpha=0.02 -> fair = 0.67
    bot = _bot()
    fair = bot._compute_fair_yes(0.60, 1.0, 0.02)
    assert abs(fair - 0.67) < 1e-9


def test_compute_fair_yes_clamped():
    bot = _bot()
    assert bot._compute_fair_yes(0.98, 2.0, 0.5) <= 0.98
    assert bot._compute_fair_yes(0.02, 2.0, -0.5) >= 0.02


def test_side_net_edges_complementary_is_mirror():
    bot = _bot()
    yes_price, no_price = 0.55, 0.45
    fair_yes = 0.62
    edge_yes, edge_no = bot._side_net_edges(fair_yes, yes_price, no_price)
    fee_y = polymarket_fills.taker_fee(1.0, yes_price)
    fee_n = polymarket_fills.taker_fee(1.0, no_price)
    assert abs(edge_yes - (fair_yes - yes_price - fee_y)) < 1e-9
    assert abs(edge_no - ((1 - fair_yes) - no_price - fee_n)) < 1e-9
    # pre-fee mirror
    assert abs((fair_yes - yes_price) + ((1 - fair_yes) - no_price)) < 1e-9


def test_side_net_edges_no_book_divergence_favors_no():
    bot = _bot()
    fair_yes = 0.55
    yes_price = 0.55
    no_price = 0.38
    edge_yes, edge_no = bot._side_net_edges(fair_yes, yes_price, no_price)
    assert edge_no > edge_yes


# --- Task 2: make_decision side selection, guards, sizing ---

def _market(yes=0.55, no=None):
    return {
        "id": "mkt-1",
        "current_price": yes,
        "no_price": (1 - yes) if no is None else no,
        "polymarket_token_id": "yes-tok",
        "polymarket_no_token_id": "no-tok",
        "time_remaining_seconds": 180,
    }


def _signals(**over):
    base = {
        "prices": [100.0, 100.0], "latest": 100.0, "orderflow": {},
        "pm_momentum": 0.0, "obi": 0.0, "cvd": 0.0,
    }
    base.update(over)
    return base


def test_no_ban_is_gone_strong_no_lean_buys_no():
    # Market leans NO (yes 0.45 / no 0.55, sum 1.0) with bearish alpha → buy NO.
    bot = _bot()
    m = _market(yes=0.45, no=0.55)
    s = _signals(pm_momentum=-0.15, obi=-1.0, cvd=-1.0)
    d = bot.make_decision(m, s)
    assert d["action"] == "buy"
    assert d["side"] == "no"


def test_no_trade_sizes_against_no_price():
    bot = _bot()
    m = _market(yes=0.45, no=0.55)
    s = _signals(pm_momentum=-0.15, obi=-1.0, cvd=-1.0)
    d = bot.make_decision(m, s)
    assert d["side"] == "no"
    assert abs(d["entry_price"] - 0.55) < 1e-6


def test_high_price_guard_fires_on_no_price():
    bot = _bot()
    m = _market(yes=0.20, no=0.80)
    s = _signals(pm_momentum=-0.15, obi=-1.0, cvd=-1.0)
    d = bot.make_decision(m, s)
    assert d["action"] == "skip"


def test_consensus_guard_fires_on_low_side_price():
    # Synthetic underpriced NO book (no=0.30 < CONSENSUS_GUARD) chosen via strong
    # NO edge → consensus guard skips it (backstop against fighting consensus).
    bot = _bot()
    m = _market(yes=0.55, no=0.30)
    s = _signals(pm_momentum=-0.15, obi=-1.0, cvd=-1.0)
    d = bot.make_decision(m, s)
    assert d["action"] == "skip"
    assert "onsensus" in d["reasoning"]


def test_no_edge_skips():
    bot = _bot()
    m = _market(yes=0.50)
    s = _signals()
    d = bot.make_decision(m, s)
    assert d["action"] == "skip"


# --- Task 3: YES-parity regression ---

def test_favorite_upswing_still_buys_yes():
    bot = _bot()
    m = _market(yes=0.62, no=0.38)
    s = _signals(pm_momentum=0.15, obi=1.0, cvd=1.0)
    d = bot.make_decision(m, s)
    assert d["action"] == "buy"
    assert d["side"] == "yes"
    assert abs(d["entry_price"] - 0.62) < 1e-6


def test_complementary_mids_reduce_to_sign():
    bot = _bot()
    m = _market(yes=0.58)
    s = _signals(pm_momentum=0.15, obi=1.0, cvd=1.0)
    d = bot.make_decision(m, s)
    if d["action"] == "buy":
        assert d["side"] == "yes"

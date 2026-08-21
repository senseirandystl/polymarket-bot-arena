"""Tests for two-sided (YES/NO) net-edge side selection in make_decision."""

import polymarket_fills
from bots.bot_momentum import MomentumBot


def _bot():
    return MomentumBot(name="momentum-test", generation=0)


# --- Task 1: pure fair-value + edge helpers ---

def test_compute_fair_yes_blend():
    # fair = mid + trust * (P_model - mid): 0.60 + 0.5*(0.70-0.60) = 0.65
    bot = _bot()
    fair = bot._compute_fair_yes(0.60, 0.70, 0.5)
    assert abs(fair - 0.65) < 1e-9


def test_compute_fair_yes_clamped():
    bot = _bot()
    assert bot._compute_fair_yes(0.97, 0.98, 2.0) <= 0.98
    assert bot._compute_fair_yes(0.03, 0.02, 2.0) >= 0.02


def test_side_net_edges_complementary_is_mirror():
    # Per-side anchoring (BUG #27): edge_side = P_side - price - fee.
    # On complementary books the pre-fee model terms mirror exactly.
    bot = _bot()
    p_yes = 0.62
    yes_price, no_price = 0.55, 0.45
    edge_yes, edge_no = bot._side_net_edges(p_yes, yes_price, no_price)
    fee_y = polymarket_fills.taker_fee(1.0, yes_price)
    fee_n = polymarket_fills.taker_fee(1.0, no_price)
    assert abs((edge_yes + fee_y) + (edge_no + fee_n)) < 1e-9


def test_side_net_edges_book_divergence_is_honest_p_minus_ask():
    bot = _bot()
    fee_n = polymarket_fills.taker_fee(1.0, 0.38)
    _ey, edge_no = bot._side_net_edges(0.50, 0.55, 0.38)
    assert abs(edge_no - ((0.50 - 0.38) - fee_n)) < 1e-9


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


def _bearish_signals(drift=-0.5):
    # Falling BTC tape agreeing with the down-drift: the momentum bot needs
    # actual momentum under the fidelity profiles (cvd/pm are killed lanes).
    return _signals(btc_drift=drift,
                    prices=[100.30, 100.20, 100.12, 100.05, 100.0],
                    latest=100.0)


def test_no_ban_is_gone_strong_no_lean_buys_no():
    # Market leans NO (yes 0.45 / no 0.55, sum 1.0) with a genuinely strong
    # bearish model: decisive down-drift + falling BTC tape.
    bot = _bot()
    d = bot.make_decision(_market(yes=0.45, no=0.55), _bearish_signals())
    assert d["action"] == "buy"
    assert d["side"] == "no"


def test_no_trade_sizes_against_no_price():
    bot = _bot()
    d = bot.make_decision(_market(yes=0.45, no=0.55), _bearish_signals())
    assert d["side"] == "no"
    assert abs(d["entry_price"] - 0.55) < 1e-6


def test_high_price_guard_fires_on_no_price():
    bot = _bot()
    m = _market(yes=0.20, no=0.80)
    s = _signals(pm_momentum=-0.15, obi=-1.0, cvd=-1.0)
    d = bot.make_decision(m, s)
    assert d["action"] == "skip"


def test_consensus_guard_fires_on_low_side_price():
    # NO priced below CONSENSUS_GUARD (0.30, books consistent: 0.68+0.30) with
    # a strong bearish model → the consensus guard skips the cheap side
    # (backstop against fighting strong consensus).
    bot = _bot()
    d = bot.make_decision(_market(yes=0.68, no=0.30), _bearish_signals())
    assert d["action"] == "skip"
    assert "onsensus" in d["reasoning"]


def test_no_edge_skips():
    bot = _bot()
    m = _market(yes=0.50)
    s = _signals()
    d = bot.make_decision(m, s)
    assert d["action"] == "skip"


# --- Task 3: YES-parity regression ---

def test_yes_bought_when_market_lags_bullish_model():
    # Bullish drift + flow with the market still near 50c: the model-vs-price
    # gap is real edge -> buy YES. (A 62c favorite with the same signals is
    # priced-in and correctly skipped under the model-blend fair value.)
    bot = _bot()
    m = _market(yes=0.53, no=0.47)
    s = _signals(btc_drift=0.4,
                 prices=[100.0, 100.05, 100.12, 100.20, 100.30], latest=100.30)
    d = bot.make_decision(m, s)
    assert d["action"] == "buy"
    assert d["side"] == "yes"
    assert abs(d["entry_price"] - 0.53) < 1e-6


def test_complementary_mids_reduce_to_sign():
    bot = _bot()
    m = _market(yes=0.58)
    s = _signals(pm_momentum=0.15, obi=1.0, cvd=1.0)
    d = bot.make_decision(m, s)
    if d["action"] == "buy":
        assert d["side"] == "yes"

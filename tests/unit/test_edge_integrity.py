"""Edge-integrity fixes from the 2026-07-17 evening run (BUG #27).

Live evidence (109 trades):
  * model lean < 0.10  -> 28.6% WR, -$78.74; lean >= 0.10 -> 73% WR, +$96.12.
    Conviction-scaled trust DAMPED weak models but still let them trade into
    large market displacement (trust_eff=0.03 trades in the log).
  * edge_no anchored fair on the YES book but paid the NO book: stale or
    inconsistent books (yes+no sum 0.84-0.94) minted phantom edges with ZERO
    model input, and Kelly sized those garbage trades the LARGEST
    (19:31/19:34: 31-34 shares, -$29.15 on two trades).

Fixes under test:
  1. Hard model-lean floor: |P_model - 0.5| < config.MODEL_LEAN_MIN -> skip.
  2. Per-side edge anchoring: each side's edge is trust_eff * (side_model_prob
     - side_price) - fee, measured against the side's OWN book price. A
     cross-book spread gap is never directional edge (it is the arb bot's
     two-legged trade).
  3. Book-consistency gate: |yes + no - 1| > config.BOOK_SUM_TOLERANCE ->
     directional skip (suspect data).
"""

import pytest

import config
import polymarket_fills
from bots.bot_momentum import MomentumBot
from bots.bot_mean_rev import MeanRevBot


def _bot():
    return MomentumBot(name="momentum-test", generation=0)


def _market(yes=0.52, no=None, tr=180):
    return {
        "id": "m", "current_price": yes,
        "no_price": (round(1 - yes, 4)) if no is None else no,
        "polymarket_token_id": "y", "polymarket_no_token_id": "n",
        "time_remaining_seconds": tr,
    }


def _sig(**over):
    base = {"prices": [100.0, 100.0], "latest": 100.0, "orderflow": {},
            "pm_momentum": 0.0, "obi": 0.0, "cvd": 0.0, "btc_drift": 0.0}
    base.update(over)
    return base


def _trending_sig(drift=0.5, up=True):
    """Signals with real BTC momentum + drift agreeing (a legitimate trade)."""
    prices = [100.0, 100.05, 100.12, 100.20, 100.30] if up else \
             [100.30, 100.20, 100.12, 100.05, 100.0]
    return _sig(prices=prices, latest=prices[-1],
                btc_drift=drift if up else -abs(drift))


# --- Fix 1: hard model-lean floor ---

def test_model_lean_min_exists():
    assert 0.05 <= config.MODEL_LEAN_MIN <= 0.15


def test_ignorant_model_never_trades_any_displacement():
    # A model with nothing to say must not trade no matter how far the market
    # is displaced from 0.5 (the residual BUG #26 leak: damped-but-nonzero
    # trust * huge displacement still cleared MIN_EDGE).
    bot = _bot()
    for yes in (0.36, 0.42, 0.58, 0.65):
        d = bot.make_decision(_market(yes=yes), _sig())
        assert d["action"] == "skip", yes


def test_weak_lean_skips_even_with_displacement():
    # Small drift -> lean below the floor -> skip, even though the old math
    # would have found a multi-cent "edge" against a displaced market.
    bot = _bot()
    d = bot.make_decision(_market(yes=0.36), _sig(btc_drift=0.06))
    assert d["action"] == "skip"
    assert "lean" in d["reasoning"].lower()


def test_strong_lean_still_trades():
    # Decisive, agreeing signals with the market lagging -> trade survives.
    bot = _bot()
    d = bot.make_decision(_market(yes=0.52), _trending_sig(drift=0.6, up=True))
    assert d["action"] == "buy"
    assert d["side"] == "yes"


# --- Fix 2: per-side edge anchoring (no spread-gap harvesting) ---

def test_side_edges_anchor_on_own_price():
    bot = _bot()
    p_yes = 0.62
    yes_price, no_price = 0.55, 0.45
    edge_yes, edge_no = bot._side_net_edges(p_yes, yes_price, no_price)
    fee_y = polymarket_fills.taker_fee(1.0, yes_price)
    fee_n = polymarket_fills.taker_fee(1.0, no_price)
    assert abs(edge_yes - ((p_yes - yes_price) - fee_y)) < 1e-9
    assert abs(edge_no - (((1 - p_yes) - no_price) - fee_n)) < 1e-9


def test_edge_has_no_trust_tax():
    """A 6¢ Φ-ask disagreement is not halved by trust=0.50."""
    bot = _bot()
    ey, _en = bot._side_net_edges(0.56, 0.50, 0.50)
    fee = polymarket_fills.taker_fee(1.0, 0.50)
    assert ey == pytest.approx(0.56 - 0.50 - fee)
    assert ey > 0.04  # would be ~0.0125 with a 0.5× trust tax


def test_spread_gap_book_gate_not_edge_helper():
    # Gapped books can still print a large helper edge at P=0.50; make_decision
    # must skip via the book-sum gate, not a trust tax on _side_net_edges.
    bot = _bot()
    _ey, edge_no = bot._side_net_edges(0.50, 0.47, 0.38)
    assert edge_no > 0.05
    d = bot.make_decision(_market(yes=0.47, no=0.38), _sig())
    assert d["action"] == "skip"


# --- Fix 3: book-consistency gate ---

def test_book_sum_tolerance_exists():
    assert 0.02 <= config.BOOK_SUM_TOLERANCE <= 0.08


def test_inconsistent_books_skip_directional():
    # Books summing to 0.85: real 15c gaps are the arbitrage bot's two-legged
    # trade; a directional bot must treat the data as suspect and stand down —
    # even when its model genuinely leans.
    bot = _bot()
    d = bot.make_decision(_market(yes=0.47, no=0.38),
                          _trending_sig(drift=0.6, up=False))
    assert d["action"] == "skip"
    assert "book" in d["reasoning"].lower()


def test_consistent_books_unaffected():
    bot = _bot()
    d = bot.make_decision(_market(yes=0.52, no=0.49),
                          _trending_sig(drift=0.6, up=True))
    assert d["action"] == "buy"


# --- Ask-priced decisions (follow-up: mid-vs-ask fill mismatch) ---
# Decisions used to price edge + entry off the MID while the paper engine
# fills by walking the ASKS — on wide books (3-8c spreads live) the fill
# landed > MAX_FILL_SLIPPAGE above the decision price and the slippage guard
# rejected 5 of 7 attempted trades in an hour. Edge must be measured against
# the price you can actually execute at; the guard then only catches book
# MOVEMENT between decision and fill.

def test_entry_price_uses_executable_ask():
    bot = _bot()
    # Deep underdog (outside 38–50 shallow-lag sit-flat).
    m = _market(yes=0.36, no=0.63)
    m["yes_ask"] = 0.37
    d = bot.make_decision(m, _trending_sig(drift=0.6, up=True))
    assert d["action"] == "buy" and d["side"] == "yes"
    assert abs(d["entry_price"] - 0.37) < 1e-9


def test_wide_spread_kills_marginal_edge_at_decision_time():
    # A trade whose edge only exists at the mid must not fire once the
    # executable ask eats it (this used to fire, then die at the fill).
    # Marginal case: drift-pure meanrev, model ~0.62 — edge at the 0.52 mid,
    # none at a 0.60 ask.
    bot = _bot()
    # Drift high enough that taker-fee-priced edge still clears MIN_EDGE
    # at the tight mid (0.52) — the invariant is ask-vs-mid, not a maker rebate.
    sig = _trending_sig(drift=0.50, up=True)
    m_tight = _market(yes=0.52, no=0.48)
    d_tight = bot.make_decision(m_tight, sig)
    assert d_tight["action"] == "buy"
    m_wide = _market(yes=0.52, no=0.48)
    # 10c above mid: the ask collapses the +0.060 mid-edge to ~+0.001, below
    # even the data-gathering MIN_EDGE floor (0.012). The example spread tracks
    # the current floor — the invariant under test is that edge is priced at
    # the executable ASK (BUG #27/#28), not the mid, so a marginal edge that
    # only exists at the mid dies once the ask eats it.
    m_wide["yes_ask"] = 0.72
    d_wide = bot.make_decision(m_wide, sig)
    assert d_wide["action"] == "skip"


def test_ask_fallback_to_mid_when_absent():
    # Until the warmer primes a market there is no book — mid fallback keeps
    # the bot functional (same behavior as before).
    bot = _bot()
    d = bot.make_decision(_market(yes=0.52, no=0.49),
                          _trending_sig(drift=0.6, up=True))
    assert d["action"] == "buy"
    assert abs(d["entry_price"] - 0.52) < 1e-9


def test_book_sum_gate_still_uses_mids():
    # Ask prices sum > 1 on any normal spread — the consistency gate must
    # keep judging the MIDS, not the asks.
    bot = _bot()
    m = _market(yes=0.36, no=0.63)
    m["yes_ask"], m["no_ask"] = 0.37, 0.65   # asks sum 1.02: fine
    d = bot.make_decision(m, _trending_sig(drift=0.6, up=True))
    assert d["action"] == "buy"

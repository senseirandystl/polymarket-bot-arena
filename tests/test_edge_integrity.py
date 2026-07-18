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
    model_prob, trust_eff = 0.62, 0.5
    yes_price, no_price = 0.55, 0.45
    edge_yes, edge_no = bot._side_net_edges(model_prob, trust_eff,
                                            yes_price, no_price)
    fee_y = polymarket_fills.taker_fee(1.0, yes_price)
    fee_n = polymarket_fills.taker_fee(1.0, no_price)
    assert abs(edge_yes - (trust_eff * (model_prob - yes_price) - fee_y)) < 1e-9
    assert abs(edge_no - (trust_eff * ((1 - model_prob) - no_price) - fee_n)) < 1e-9


def test_spread_gap_is_not_directional_edge():
    # The 19:34 disaster: yes=0.47, no=0.38 (sum 0.85), ignorant model.
    # Old math: edge_no = (1-fair) - 0.38 ~ +0.13 -> Kelly max-sized it.
    # New math: an ignorant model has trust_eff ~ 0 -> both edges ~ -fee.
    bot = _bot()
    edge_yes, edge_no = bot._side_net_edges(0.50, 0.01, 0.47, 0.38)
    assert edge_no < 0.005
    assert edge_yes < 0.005


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

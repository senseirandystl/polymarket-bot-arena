"""Phase 1 root-cause fixes: signal re-weighting, tilt cap, stop-loss removal.

Backed by the overnight run analysis (docs/superpowers/specs/2026-07-15-...):
OBI + learning bias were anti-predictive; CVD is the real edge; price_tilt
manufactured fake edge at extremes; stop-loss is net-harmful in 5-min markets.
"""

import config
from bots.bot_momentum import MomentumBot
from bots.bot_meanrev_sl import MeanRevSLBot


def _bot():
    return MomentumBot(name="momentum-test", generation=0)


def _market(yes=0.52, no=None):
    return {
        "id": "m", "current_price": yes,
        "no_price": (round(1 - yes, 4)) if no is None else no,
        "polymarket_token_id": "y", "polymarket_no_token_id": "n",
        "time_remaining_seconds": 180,
    }


def _sig(**over):
    base = {"prices": [100.0, 100.0], "latest": 100.0, "orderflow": {},
            "pm_momentum": 0.0, "obi": 0.0, "cvd": 0.0}
    base.update(over)
    return base


# --- R1: config weights reflect measured predictiveness ---

def test_obi_restored():
    # OBI was zeroed in Phase 1 (measured anti-predictive on a stale snapshot),
    # then RESTORED once confirmed the warmer computes it fresh every 1s from the
    # best-first book — a true order-book imbalance signal with its natural sign.
    assert config.SIGNAL_WEIGHT_OBI > 0.0


def test_cvd_weight_boosted():
    assert config.SIGNAL_WEIGHT_CVD >= 0.20


def test_learning_disabled_live():
    assert config.LEARNING_ENABLED is False


def test_strategy_weight_raised():
    assert config.STRATEGY_SIGNAL_WEIGHT >= 0.30


# --- R2: price_tilt is capped so it can't manufacture edge at extremes ---

def test_tilt_capped_high():
    bot = _bot()
    # raw tilt = (0.90-0.5)*1.2*K_TILT = 0.24; capped to FAVORITE_EDGE_CAP.
    fair = bot._compute_fair_yes(0.90, 1.2, 0.0)
    assert abs(fair - (0.90 + config.FAVORITE_EDGE_CAP)) < 1e-9


def test_tilt_capped_low():
    bot = _bot()
    fair = bot._compute_fair_yes(0.10, 1.2, 0.0)
    assert abs(fair - (0.10 - config.FAVORITE_EDGE_CAP)) < 1e-9


def test_tilt_uncapped_in_band():
    bot = _bot()
    # small tilt below the cap passes through unchanged: (0.55-0.5)*1.0*0.5=0.025
    fair = bot._compute_fair_yes(0.55, 1.0, 0.0)
    assert abs(fair - 0.575) < 1e-9


# --- R1: OBI no longer moves the decision; CVD still does ---

def test_cvd_still_moves_decision():
    bot = _bot()
    m = _market(yes=0.52, no=0.48)
    d_pos = bot.make_decision(m, _sig(cvd=1.0))
    d_neg = bot.make_decision(m, _sig(cvd=-1.0))
    # CVD is the real edge: flipping its sign must change the fair value / side lean
    assert d_pos != d_neg


# --- R3: stop-loss removed — SL bots hold to resolution ---

def test_sl_bot_holds_to_resolution():
    bot = MeanRevSLBot()
    assert getattr(bot, "exit_strategy", None) is None

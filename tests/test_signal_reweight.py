"""Model-blend fair value: signal weighting + no manufactured edge.

History: the additive stack (fair = mid + tilt + alpha) counted its bonus
lanes as edge BY CONSTRUCTION — the flat +6c favorite tilt alone cleared the
MIN_EDGE gate at window open, so every bot bought the 58-65c favorite in the
first minute (2026-07-16 live run: 107 early trades, 49% WR, -$79.53). Fair is
now a market-vs-model blend: fair = mid + trust * (P_model - mid), so edge
exists only when the bot's model actively disagrees with the price.
"""

import config
from bots.base_bot import BaseBot
from bots.bot_momentum import MomentumBot
from bots.bot_meanrev_sl import MeanRevSLBot


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


# --- Weights reflect measured predictiveness ---

def test_obi_disabled():
    # OBI kill-switch stays 0.0 until a fade-sign OBI is validated offline.
    assert config.SIGNAL_WEIGHT_OBI == 0.0


def test_cvd_weighted_in_every_profile():
    # CVD (executed aggression) is the one validated flow edge — every
    # strategy's model weights it > 0; sentiment weights it heaviest.
    profs = BaseBot.STRATEGY_SIGNAL_PROFILE
    for strat, prof in profs.items():
        assert prof["cvd"] > 0.0, strat
    assert profs["sentiment"]["cvd"] == max(p["cvd"] for p in profs.values())


def test_learning_disabled_live():
    assert config.LEARNING_ENABLED is False


# --- No manufactured edge: the favorite tilt is gone ---

def test_no_edge_from_price_alone():
    # A 63c favorite with ZERO signal must be a skip for every strategy — the
    # old tilt made this an automatic buy (the -$64.55 bucket).
    for cls_name, bot in [("mom", _bot()), ("sl", MeanRevSLBot())]:
        d = bot.make_decision(_market(yes=0.63, tr=290), _sig())
        assert d["action"] == "skip", cls_name


def test_ignorance_never_fades_the_favorite():
    # With no information the model sits at 0.5; blending toward it made the
    # non-favorite look cheap. The model-lean eligibility rule forbids buying
    # a side the model doesn't actively lean toward.
    d = _bot().make_decision(_market(yes=0.63), _sig())
    assert d["action"] == "skip"
    d = _bot().make_decision(_market(yes=0.37), _sig())
    assert d["action"] == "skip"


def test_priced_in_signal_earns_nothing():
    # Strong drift already reflected in the price -> no edge, skip.
    d = _bot().make_decision(_market(yes=0.70, tr=150), _sig(btc_drift=0.5))
    assert d["action"] == "skip"


def test_market_lagging_model_is_the_trade():
    # Same drift, market still near 50c -> the model-vs-price gap IS the edge.
    d = _bot().make_decision(_market(yes=0.52, tr=150), _sig(btc_drift=0.5))
    assert d["action"] == "buy"
    assert d["side"] == "yes"


def test_fair_blend_math():
    bot = _bot()
    # fair = mid + trust * (P_model - mid)
    assert abs(bot._compute_fair_yes(0.50, 0.70, 0.5) - 0.60) < 1e-9
    assert abs(bot._compute_fair_yes(0.60, 0.60, 0.5) - 0.60) < 1e-9  # priced in


# --- Strategy differentiation is real (emphasis, never direction) ---

def test_profiles_have_no_negative_weights():
    for strat, prof in BaseBot.STRATEGY_SIGNAL_PROFILE.items():
        for lane, w in prof.items():
            assert w >= 0.0, (strat, lane)


def test_strategies_diverge_on_momentum_only_input():
    # A REAL BTC-momentum burst (0.2% candle ~ p97; median moves no longer
    # saturate the lane) trades the momentum bot but not the fundamentals-only
    # mean-reversion bot.
    m = _market(yes=0.55, tr=150)
    s = _sig(btc_drift=0.2, prices=[100.0, 100.2], latest=100.2)
    assert _bot().make_decision(m, s)["action"] == "buy"
    assert MeanRevSLBot().make_decision(m, s)["action"] == "skip"


# --- R3: stop-loss removed — SL bots hold to resolution ---

def test_sl_bot_holds_to_resolution():
    bot = MeanRevSLBot()
    assert getattr(bot, "exit_strategy", None) is None


# --- Drift veto + honest momentum normalization (2026-07-16 overnight run) ---

def test_drift_veto_blocks_contradicting_side():
    # Strong flow/momentum pushing YES while drift reads DOWN: live, trades
    # contradicting a non-trivial drift ran 26% WR (-$55). The veto forbids the
    # contradicting side, symmetric in both directions.
    m = _market(yes=0.45, tr=200)
    s = _sig(btc_drift=-0.10, cvd=1.0, pm_momentum=0.15,
             prices=[100.0, 100.1], latest=100.1)
    d = _bot().make_decision(m, s)
    assert not (d["action"] == "buy" and d["side"] == "yes")
    # mirror: up drift forbids NO
    m2 = _market(yes=0.55, tr=200)
    s2 = _sig(btc_drift=0.10, cvd=-1.0, pm_momentum=-0.15,
              prices=[100.1, 100.0], latest=100.0)
    d2 = _bot().make_decision(m2, s2)
    assert not (d2["action"] == "buy" and d2["side"] == "no")


def test_drift_veto_allows_flow_trades_when_drift_flat():
    # Below the veto floor (drift ~ 0) flow-only trades stay allowed — they
    # measured break-even live and are the sentiment bot's identity.
    from bots.bot_sentiment import SentimentBot
    d = SentimentBot(name="s").make_decision(
        _market(yes=0.50, tr=150), _sig(cvd=0.8))
    assert d["action"] == "buy"


def test_momentum_lane_not_saturated_by_median_move():
    # A median-size BTC 1-min move (~0.022%) must NOT saturate the momentum
    # lane (the first normalization saturated at 0.05% — below the median —
    # letting one candle of noise outvote the time-damped drift).
    m = _market(yes=0.50, tr=280)
    s = _sig(prices=[100000.0, 100022.0], latest=100022.0)  # +0.022%
    d = _bot().make_decision(m, s)
    # 0.022% * 500 = 0.11 of saturation; momentum bot weight 0.25 ->
    # ~0.014 prob shift -> nowhere near the min-edge gate on its own.
    assert d["action"] == "skip"

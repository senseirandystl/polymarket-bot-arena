"""PM-lane kill-switch + conviction-scaled trust (2026-07-17 chop-regime leak).

Live run (45 trades, 44% WR, -$26): with drift ~0 in a low-vol chop, the
saturated pm lane (SCALE=80 then /0.15 -> sign(last tick); pegged at +/-1.000
on 79% of trades) manufactured model leans of 0.55-0.66 out of noise, and
edge = trust * (P_model - mid) converted any market displacement into a
phantom 3-7c edge -> systematic underdog fading (26 trades, 38.5% WR).

Harness (300 markets): pm_mom raw is predictive (69.7% follow-WR) but its NET
edge is NEGATIVE (-0.80c/share, avg entry 0.688) — priced in by the time it
is measurable. House rule: no positive net edge, no live weight -> kill
switch, same treatment as OBI.

Conviction-scaled trust: trust_eff = trust * min(1, |P_model-0.5| / SCALE).
A near-ignorant model (lean ~0.01-0.03) gets almost no say against the
market; a decisive model (lean >= SCALE) keeps full trust. Kills the
ignorance-fade structurally while preserving the validated market-lags-drift
trade (+19.5c/share in the harness).
"""

import config
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


# --- PM lane kill-switch ---

def test_pm_lane_killed():
    # Predictive but net-NEGATIVE after the price (harness 2026-07-17):
    # -0.80c/share. Kill-switch 0.0 until a rework shows positive net edge.
    assert config.SIGNAL_WEIGHT_PM == 0.0


def test_saturated_pm_alone_cannot_trade():
    # The live leak: pm pegged at the clamp (+/-0.15 -> lane +/-1.0) with no
    # other information used to buy the displaced side. Must be a skip now.
    for yes, pm in ((0.42, 0.15), (0.58, -0.15)):
        d = _bot().make_decision(_market(yes=yes, tr=150), _sig(pm_momentum=pm))
        assert d["action"] == "skip", (yes, pm)


# --- Conviction-scaled trust ---

def test_conviction_scale_config():
    assert 0.0 < config.MODEL_CONVICTION_SCALE <= 0.2


def test_weak_model_cannot_fade_displaced_market():
    # Live loser replay: drift 0.06 (barely over the veto floor -> model lean
    # ~0.014) vs a market at 0.42. Old math: edge = 0.5*(0.51-0.42) ~ +3.1c,
    # cleared MIN_EDGE, bought YES, 10% WR. Conviction-scaled trust collapses
    # the phantom edge -> skip.
    d = _bot().make_decision(_market(yes=0.42, tr=150), _sig(btc_drift=0.06))
    assert d["action"] == "skip"


def test_weak_model_fade_blocked_symmetrically():
    # Mirror: tiny down-drift vs a market at 0.58 must not buy NO.
    d = _bot().make_decision(_market(yes=0.58, tr=150), _sig(btc_drift=-0.06))
    assert d["action"] == "skip"


def test_decisive_model_still_trades_market_lag():
    # The validated top rule (drift decisive, market lagging near 50c) must
    # SURVIVE conviction scaling: drift 0.5 -> lean 0.1125 >= scale -> full
    # trust, edge intact.
    d = _bot().make_decision(_market(yes=0.52, tr=150), _sig(btc_drift=0.5))
    assert d["action"] == "buy"
    assert d["side"] == "yes"


def test_meanrev_ignorance_fade_blocked():
    # The 0.36c-entry live loser: meanrev (trust 0.6, drift-only profile) with
    # drift 0.049 bought YES at 0.36/0.41. Conviction ~0.03 -> no trade.
    d = MeanRevSLBot().make_decision(_market(yes=0.41, tr=150),
                                     _sig(btc_drift=0.05))
    assert d["action"] == "skip"

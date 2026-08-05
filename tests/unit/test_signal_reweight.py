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


def test_cvd_lane_killed():
    # CVD kill-switch (BUG #27): the live net/total form saturated on thin
    # tapes and measured statistically flat (53.1% WR). Lane weight 0 in
    # every profile + global switch 0 until the volume-floored form shows
    # positive NET edge offline.
    assert config.SIGNAL_WEIGHT_CVD == 0.0
    for strat, prof in BaseBot.STRATEGY_SIGNAL_PROFILE.items():
        assert prof["cvd"] == 0.0, strat


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
    # Under the fidelity profiles this pure-fundamental trade belongs to the
    # drift-anchored meanrev bot (the momentum bot needs actual momentum).
    d = MeanRevSLBot().make_decision(_market(yes=0.52, tr=150),
                                     _sig(btc_drift=0.5))
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


def test_strategies_diverge_on_momentum_only_input(monkeypatch):
    # A REAL BTC-momentum burst (0.2% candle ~ p97; median moves no longer
    # saturate the lane) trades the momentum bot but not the fundamentals-only
    # mean-reversion bot.
    from arena.regime_adapt import RegimeAdjust
    monkeypatch.setattr(
        "arena.regime_adapt.adjustments",
        lambda *a, **k: RegimeAdjust(size_mult=1.0, label="normal"),
    )
    # mid 0.60 is outside the coin-flip band; drift high enough to clear lag gates.
    m = _market(yes=0.60, tr=150)
    m["yes_ask"] = 0.61
    m["no_ask"] = 0.40
    s = _sig(btc_drift=0.35, prices=[100.0, 100.2], latest=100.2)
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


def test_drift_veto_allows_flow_trades_when_drift_flat(monkeypatch):
    # Below the veto floor (drift ~ 0) the drift veto itself does not block a
    # flow-only trade. Underdog band (0.35–0.42) now requires real drift
    # (2026-08), so price just above that band / outside dead-zone.
    # Isolate from live skip-bandit / consensus tighten (soak can raise the
    # floor above 0.35) and use a mid safely above CONSENSUS_GUARD.
    monkeypatch.setattr(
        "arena.learned_rules.skip_softening",
        lambda *_a, **_k: {"soften": 0.0, "factor": 1.0},
        raising=False,
    )
    from bots.bot_momentum import MomentumBot
    # Outside underdog (0.35–0.42) and dead-zone (0.42–0.58): use 0.60 mid.
    d = MomentumBot(name="s").make_decision(
        _market(yes=0.60, tr=150),
        _sig(cvd=0.8, prices=[100.0, 100.4], latest=100.4),
    )
    # Critical contract: flat drift does NOT veto via the drift-veto path.
    # (May still skip for weak lean / no edge when CVD is kill-switched.)
    reason = (d.get("reasoning") or "").lower()
    assert "drift veto" not in reason
    assert not (d["action"] == "buy" and d.get("side") == "no")


# --- Dead-zone gate (2026-07-21): the single biggest live leak ---

def test_dead_zone_gate_blocks_flat_drift_coinflip():
    # A flat-drift opinion against a near-coin-flip market (mid in 0.42-0.58 &
    # |drift| < 0.10) was 59 trades, 39% WR, -$77.83 — gated flat now.
    # With strat-confirm mode, weak lean may fire first when CVD alone cannot
    # move P_model far from 0.5; either skip is correct "sit flat" behaviour.
    from bots.bot_momentum import MomentumBot
    d = MomentumBot(name="s").make_decision(
        _market(yes=0.50, tr=150), _sig(cvd=0.8))
    assert d["action"] == "skip"
    reason = d["reasoning"].lower()
    assert ("dead-zone" in reason or "lean" in reason or "no edge" in reason)


def test_dead_zone_gate_allows_high_drift_in_band():
    # |drift| >= 0.30 in the SAME price band is the profitable "market lags
    # drift" trade (+$30.10, 65.7% WR) and must pass through the gate.
    m = _market(yes=0.50, tr=150)
    s = _sig(btc_drift=0.35, cvd=0.5, prices=[100.0, 100.3], latest=100.3)
    assert _bot().make_decision(m, s)["action"] == "buy"


def test_dead_zone_quiet_regime_raises_drift_floor():
    # Under low_vol_range, mid-band trades need |drift| >= QUIET floor (0.20),
    # not the base 0.10 — weak-moderate drift was the 2026-07-29 mid-band leak.
    m = _market(yes=0.50, tr=150)
    quiet = {
        "label": "low_vol_range",
        "regime_id": "low_vol_range",
        "known": True,
        "trend_score": 0.3,
        "vol_score": 0.2,
    }
    # 0.15 clears the base floor but not the quiet floor → skip
    s_mid = _sig(
        btc_drift=0.15, cvd=0.5, prices=[100.0, 100.15], latest=100.15,
        market_regime=quiet, vol_regime=quiet,
    )
    d = _bot().make_decision(m, s_mid)
    assert d["action"] == "skip"
    assert "dead-zone" in d["reasoning"].lower()
    assert "0.20" in d["reasoning"] or "0.2" in d["reasoning"]

    # Strong drift still clears quiet floor → may trade (buy or other skip ok
    # only if not dead-zone)
    s_hi = _sig(
        btc_drift=0.35, cvd=0.5, prices=[100.0, 100.3], latest=100.3,
        market_regime=quiet, vol_regime=quiet,
    )
    d_hi = _bot().make_decision(m, s_hi)
    assert "dead-zone" not in d_hi["reasoning"].lower()


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

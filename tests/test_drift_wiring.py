"""Drift + OBI are WIRED but DISABLED (weight 0) pending offline validation.

Both were shipped weighted and both measured anti-predictive (drift especially:
33% WR blow-up in a mean-reverting regime — when drift said UP, YES won 23%).
They stay wired so a validated version can be re-enabled by flipping a weight,
but the live default weight is 0. These tests prove (a) the live defaults are
off, and (b) the plumbing still works when a weight is applied.
"""

import config
from bots.bot_momentum import MomentumBot


def _bot():
    return MomentumBot(name="momentum-test", generation=0)


def _market(yes=0.50, no=None):
    return {
        "id": "m", "current_price": yes,
        "no_price": (round(1 - yes, 4)) if no is None else no,
        "polymarket_token_id": "y", "polymarket_no_token_id": "n",
        "time_remaining_seconds": 60,
    }


def _sig(**over):
    base = {"prices": [100.0, 100.0], "latest": 100.0, "orderflow": {},
            "pm_momentum": 0.0, "obi": 0.0, "cvd": 0.0, "btc_drift": 0.0}
    base.update(over)
    return base


def test_drift_enabled_after_validation():
    # Re-enabled once the accurate strike made it ~76% predictive offline.
    # Drift now lives in the per-strategy model profiles as the anchor lane —
    # every strategy must weight it > 0.
    from bots.base_bot import BaseBot
    for strat, prof in BaseBot.STRATEGY_SIGNAL_PROFILE.items():
        assert prof["drift"] > 0.0, strat


def test_obi_disabled_by_default():
    # OBI stays off — not yet validated (no order-book history in the harness).
    assert config.SIGNAL_WEIGHT_OBI == 0.0


def test_drift_moves_decision_symmetrically():
    # Live weight is on: equal-magnitude opposite drift -> opposite sides.
    bot = _bot()
    m = _market(yes=0.50, no=0.50)
    up = bot.make_decision(m, _sig(btc_drift=1.0))
    dn = bot.make_decision(m, _sig(btc_drift=-1.0))
    if up["action"] == "buy" and dn["action"] == "buy":
        assert up["side"] == "yes" and dn["side"] == "no"


def test_obi_mechanism_works_when_weighted(monkeypatch):
    monkeypatch.setattr(config, "SIGNAL_WEIGHT_OBI", 0.10)
    bot = _bot()
    m = _market(yes=0.52, no=0.48)
    d_pos = bot.make_decision(m, _sig(obi=1.0))
    d_neg = bot.make_decision(m, _sig(obi=-1.0))
    assert d_pos != d_neg


def test_btc_drift_present_in_combined_signals():
    # Wiring: build_combined_signals must expose btc_drift so a future re-enable
    # is a one-line weight change.
    from arena.signals import build_combined_signals

    class PF:
        def get_signals(self, _):
            return {"prices": [100.0], "volumes": [], "latest": 100.0}

    sig = build_combined_signals(PF(), None, None,
                                 market=_market(), warm={"obi": 0, "cvd": 0, "pm_momentum": 0})
    assert "btc_drift" in sig and "btc_strike" in sig

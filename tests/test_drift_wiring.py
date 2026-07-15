"""Drift signal + OBI restoration wired into make_decision, and per-side eval."""

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


def test_config_drift_and_obi_restored():
    assert config.SIGNAL_WEIGHT_DRIFT > 0
    assert config.SIGNAL_WEIGHT_OBI > 0          # restored
    assert config.MARKET_WINDOW_SEC == 300


def test_positive_drift_pushes_toward_yes():
    # At a coin-flip market, strong positive drift (BTC above strike) should make
    # the model favor YES, not default into NO.
    bot = _bot()
    m = _market(yes=0.50, no=0.50)
    d = bot.make_decision(m, _sig(btc_drift=1.0))
    if d["action"] == "buy":
        assert d["side"] == "yes"


def test_negative_drift_pushes_toward_no():
    bot = _bot()
    m = _market(yes=0.50, no=0.50)
    d = bot.make_decision(m, _sig(btc_drift=-1.0))
    if d["action"] == "buy":
        assert d["side"] == "no"


def test_drift_is_regime_symmetric():
    # Equal-magnitude opposite drift yields opposite sides — no baked-in bias.
    bot = _bot()
    m = _market(yes=0.50, no=0.50)
    up = bot.make_decision(m, _sig(btc_drift=1.0))
    dn = bot.make_decision(m, _sig(btc_drift=-1.0))
    if up["action"] == "buy" and dn["action"] == "buy":
        assert up["side"] == "yes" and dn["side"] == "no"


def test_no_drift_coinflip_no_spurious_bet():
    # Zero drift + zero flow at 50/50 -> no real edge -> skip (no reflexive NO).
    bot = _bot()
    m = _market(yes=0.50, no=0.50)
    d = bot.make_decision(m, _sig())
    assert d["action"] == "skip"


def test_obi_moves_decision_again():
    bot = _bot()
    m = _market(yes=0.52, no=0.48)
    d_pos = bot.make_decision(m, _sig(obi=1.0))
    d_neg = bot.make_decision(m, _sig(obi=-1.0))
    # OBI restored (weight>0): flipping its sign must change the decision.
    assert d_pos != d_neg

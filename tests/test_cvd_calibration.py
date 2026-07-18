"""CVD lane calibration + kill-switch (BUG #27, part 3).

Live evidence: cvd = net/total over a ~20s tape with no volume floor pegs at
+/-0.8-1.0 on most trades (a thin tape with a few same-side prints saturates
instantly) — the same magnitude disease as pm_mom before its kill-switch.
Ground truth: cvd-driven trades (|cvd| >= 0.8, |drift| < 0.10) ran 53.1% WR —
statistically flat, no net edge.

House rule (validate-before-weighting): no positive net edge in the live form,
no live weight. The lane keeps flowing (logged, harness-validatable) but
contributes 0 to decisions until the calibrated form measures positive net
edge offline.
"""

import config
from signals.orderflow_signals import cvd_from_trades


def _tape(sides_and_sizes):
    return [{"side": "BUY", "outcome": "Up", "size": s} if d > 0 else
            {"side": "SELL", "outcome": "Up", "size": s}
            for d, s in sides_and_sizes]


def test_cvd_kill_switch_is_zero():
    assert config.SIGNAL_WEIGHT_CVD == 0.0


def test_thin_tape_does_not_saturate():
    # 3 small same-side prints (30 shares total) must NOT read as +/-1.0
    # conviction — the volume floor damps thin tapes toward 0.
    cvd = cvd_from_trades(_tape([(+1, 10), (+1, 10), (+1, 10)]))
    assert 0 < cvd < 0.5


def test_deep_one_sided_tape_still_reads_strong():
    # A genuinely heavy one-sided tape (>> floor) keeps its magnitude.
    cvd = cvd_from_trades(_tape([(+1, 500), (+1, 500), (+1, 500)]))
    assert cvd > 0.8


def test_balanced_tape_reads_zero():
    cvd = cvd_from_trades(_tape([(+1, 200), (-1, 200)]))
    assert abs(cvd) < 1e-9


def test_cvd_lane_contributes_nothing_live():
    # With the kill-switch at 0, a saturated CVD alone must not create a lean.
    from bots.bot_momentum import MomentumBot
    bot = MomentumBot(name="momentum-test", generation=0)
    m = {"id": "m", "current_price": 0.45, "no_price": 0.55,
         "polymarket_token_id": "y", "polymarket_no_token_id": "n",
         "time_remaining_seconds": 180}
    s = {"prices": [100.0, 100.0], "latest": 100.0, "orderflow": {},
         "pm_momentum": 0.0, "obi": 0.0, "cvd": 1.0, "btc_drift": 0.0}
    d = bot.make_decision(m, s)
    assert d["action"] == "skip"

"""Core-lane auto-tuner (arena/core_lane_tuner.py).

Bounded per-(strategy, lane) weight nudging on live attribution, gated by the
auto-approve toggle. Seeds a tmp DB with resolved trades whose reasoning carries
lane readings and asserts the tuner's nudges respect the sample floor, the band
around the class default, and the suggest-only toggle.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
import db
from arena import core_lane_tuner as clt
from bots.base_bot import BaseBot


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    yield


def _seed(strategy, n, drift_reading, side="yes", outcome="win",
          bot="momentum-x1"):
    """Register a bot of `strategy` and give it n resolved trades whose
    reasoning logs the given drift reading (mom/strat neutral)."""
    db.save_bot_config(bot, strategy, 0, {"k": 1})
    reasoning = (f"fair=0.55 => {side} edge=+0.05 "
                 f"drift={drift_reading:+.2f} mom=+0.00 pm=+0.00 "
                 f"of(obi=+0.00 cvd=+0.00) cand(fut=+0.10 tech=+0.10 xa=+0.10) "
                 f"strat=+0.00")
    with db.get_conn() as conn:
        for i in range(n):
            conn.execute(
                """INSERT INTO trades
                   (bot_name, market_id, side, amount, venue, mode, outcome,
                    pnl, entry_price, reasoning, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
                (bot, f"m{i}", side, 5.0, "paper", "paper", outcome, 1.0, 0.5,
                 reasoning))


def test_predictive_lane_nudged_up_bounded(tmp_db):
    db.set_auto_approve_lanes(True)
    # drift reads +0.30 on yes/win trades → sign matches UP → 100% accuracy.
    _seed("momentum", config.CORE_TUNE_MIN_TRADES, drift_reading=0.30)
    clt.tune()
    ov = db.get_lane_overrides().get("drift")
    assert ov and ov["enabled"] and ov.get("core") is True
    default = BaseBot.STRATEGY_SIGNAL_PROFILE["momentum"]["drift"]  # 0.35
    w = ov["profile"]["momentum"]
    assert w == pytest.approx(default + config.CORE_TUNE_STEP)  # one step up
    assert w <= default + config.CORE_TUNE_BAND + 1e-9          # never past band


def test_complete_profile_written(tmp_db):
    db.set_auto_approve_lanes(True)
    _seed("momentum", config.CORE_TUNE_MIN_TRADES, drift_reading=0.30)
    clt.tune()
    prof = db.get_lane_overrides()["drift"]["profile"]
    # Every known strategy is present so none is silently zeroed.
    for strat in BaseBot.STRATEGY_SIGNAL_PROFILE:
        assert strat in prof
    # Untouched strategies keep their class default weight.
    assert prof["hybrid"] == pytest.approx(
        BaseBot.STRATEGY_SIGNAL_PROFILE["hybrid"]["drift"])


def test_anti_predictive_lane_nudged_down(tmp_db):
    db.set_auto_approve_lanes(True)
    # drift reads -0.30 on yes/win trades → contradicts UP → 0% accuracy.
    _seed("momentum", config.CORE_TUNE_MIN_TRADES, drift_reading=-0.30)
    clt.tune()
    default = BaseBot.STRATEGY_SIGNAL_PROFILE["momentum"]["drift"]
    w = db.get_lane_overrides()["drift"]["profile"]["momentum"]
    assert w == pytest.approx(default - config.CORE_TUNE_STEP)
    assert w >= default - config.CORE_TUNE_BAND - 1e-9  # floored by the band


def test_below_sample_floor_holds(tmp_db):
    db.set_auto_approve_lanes(True)
    _seed("momentum", config.CORE_TUNE_MIN_TRADES - 5, drift_reading=0.30)
    clt.tune()
    # Not enough samples → no override written for drift.
    assert "drift" not in db.get_lane_overrides()


def test_toggle_off_is_suggest_only(tmp_db):
    db.set_auto_approve_lanes(False)
    _seed("momentum", config.CORE_TUNE_MIN_TRADES, drift_reading=0.30)
    report = clt.tune()
    # No override applied...
    assert "drift" not in db.get_lane_overrides()
    # ...but the suggestion is computed and surfaced.
    assert report["applied"] is False
    cell = report["lanes"]["drift"]["momentum"]
    assert cell["action"] == "up"
    assert cell["suggested"] > cell["current"]

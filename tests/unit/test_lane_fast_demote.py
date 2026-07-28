"""Fast demote for catastrophic live lane accuracy."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
import db
from arena import lane_monitor


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    yield


def _seed(n, tech_reading, side="yes", outcome="win"):
    reasoning = (f"fair=0.55 => {side} edge=+0.05 drift=+0.00 "
                 f"cand(fut=+0.10 tech={tech_reading:+.2f} xa=+0.10)")
    with db.get_conn() as conn:
        for i in range(n):
            conn.execute(
                """INSERT INTO trades
                   (bot_name, market_id, side, amount, venue, mode, outcome,
                    pnl, entry_price, reasoning, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
                (f"b{i}", f"m{i}", side, 5.0, "paper", "paper", outcome, -1.0,
                 0.5, reasoning))


def test_fast_demote_before_full_sample(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "LANE_MONITOR_MIN_TRADES", 50)
    monkeypatch.setattr(config, "LANE_MONITOR_FAST_DEMOTE_MIN_TRADES", 20)
    monkeypatch.setattr(config, "LANE_MONITOR_FAST_DEMOTE_MAX_ACC", 0.45)
    # Anti-predictive tech (+ on yes/loss → wrong)
    _seed(25, tech_reading=0.40, side="yes", outcome="loss")
    db.set_arena_state("lane_overrides", __import__("json").dumps({
        "tech": {"enabled": True, "profile": {"momentum": 0.1},
                 "approved_at": "1970-01-01"},
    }))
    report = lane_monitor.check_lanes()
    assert report["tech"]["verdict"] == "disabled"
    assert report["tech"].get("fast_demote") is True
    assert db.get_lane_overrides().get("tech", {}).get("enabled") is False

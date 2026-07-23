"""Closed-loop auto-approve promoter (arena/lane_promoter.py).

The promoter judges harness-filed pending proposals against LIVE shadow reads
(the cand(...) tokens logged in resolved trades) and, when the toggle is on,
auto-approves lanes that clear the live bar. These tests drive it against a
tmp DB seeded with synthetic resolved trades.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
import db
from arena import lane_promoter


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    yield


def _seed_trades(n, tech_reading, side="yes", outcome="win"):
    """Insert n resolved trades whose reasoning carries a cand(...) read."""
    reasoning = (f"fair=0.55 => {side} edge=+0.05 drift=+0.00 "
                 f"cand(fut=+0.10 tech={tech_reading:+.2f} xa=+0.10)")
    with db.get_conn() as conn:
        for i in range(n):
            conn.execute(
                """INSERT INTO trades
                   (bot_name, market_id, side, amount, venue, mode, outcome,
                    pnl, entry_price, reasoning, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))""",
                (f"b{i}", f"m{i}", side, 5.0, "paper", "paper", outcome, 1.0,
                 0.5, reasoning))


def _pending_tech():
    return db.create_lane_proposal(
        "tech", {"follow_wr": 0.73}, {"profile": {"momentum": 0.1}})


def test_clears_bar_auto_approves_when_toggle_on(tmp_db):
    db.set_auto_approve_lanes(True)
    # tech reads +0.30 on yes/win trades → sign matches market UP → 100% acc.
    _seed_trades(config.AUTO_APPROVE_MIN_TRADES, tech_reading=0.30)
    pid = _pending_tech()
    report = lane_promoter.check_proposals()
    assert report["tech"]["verdict"] == "auto_approved"
    # Proposal is now approved and the override is live.
    assert db.get_lane_overrides().get("tech", {}).get("enabled") is True
    assert not [p for p in db.get_lane_proposals(status="pending")
                if p["id"] == pid]


def test_clears_bar_only_annotates_when_toggle_off(tmp_db):
    db.set_auto_approve_lanes(False)
    _seed_trades(config.AUTO_APPROVE_MIN_TRADES, tech_reading=0.30)
    pid = _pending_tech()
    report = lane_promoter.check_proposals()
    assert report["tech"]["verdict"] == "clears_bar"
    # Not approved — still pending — but live evidence is attached.
    pending = db.get_lane_proposals(status="pending")
    assert any(p["id"] == pid for p in pending)
    prop = next(p for p in pending if p["id"] == pid)
    assert prop["metrics"]["live"]["accuracy"] == pytest.approx(1.0)


def test_below_bar_not_approved(tmp_db):
    db.set_auto_approve_lanes(True)
    # tech reads -0.30 on yes/win trades → sign contradicts UP → 0% accuracy.
    _seed_trades(config.AUTO_APPROVE_MIN_TRADES, tech_reading=-0.30)
    _pending_tech()
    report = lane_promoter.check_proposals()
    assert report["tech"]["verdict"] == "below_bar"
    assert db.get_lane_overrides().get("tech", {}).get("enabled") is not True


def test_insufficient_sample_collects(tmp_db):
    db.set_auto_approve_lanes(True)
    _seed_trades(config.AUTO_APPROVE_MIN_TRADES - 5, tech_reading=0.30)
    _pending_tech()
    report = lane_promoter.check_proposals()
    assert report["tech"]["verdict"] == "collecting"


def test_active_cap_blocks_promotion(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "AUTO_APPROVE_MAX_ACTIVE", 0)
    db.set_auto_approve_lanes(True)
    _seed_trades(config.AUTO_APPROVE_MIN_TRADES, tech_reading=0.30)
    _pending_tech()
    report = lane_promoter.check_proposals()
    # Clears the bar but the cap is full → stays pending.
    assert report["tech"]["verdict"] == "clears_bar"
    assert db.get_lane_overrides().get("tech", {}).get("enabled") is not True


def test_toggle_persists_in_db(tmp_db):
    db.set_auto_approve_lanes(False)
    assert db.get_auto_approve_lanes() is False
    db.set_auto_approve_lanes(True)
    assert db.get_auto_approve_lanes() is True

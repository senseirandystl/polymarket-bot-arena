"""decision_events log + resolve + rollup + learning wiring."""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
import db
from arena import decision_log


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "dec.db")
    db.init_db()
    # Reset module queue / throttle between tests
    with decision_log._queue_lock:
        decision_log._queue.clear()
    with decision_log._throttle_lock:
        decision_log._throttle.clear()
    monkeypatch.setattr(config, "DECISION_LOG_ENABLED", True)
    monkeypatch.setattr(config, "DECISION_LOG_MIN_INTERVAL_SEC", 0.0)
    yield


def _sig(action="skip", side="yes", edge=0.05, **extra):
    s = {
        "action": action,
        "side": side,
        "edge": edge,
        "confidence": 0.2,
        "entry_price": 0.45,
        "reasoning": "fair=0.55 => yes edge=+0.05 drift=+0.30 mom=+0.10 "
                     "cand(fut=+0.20 tech=+0.10 xa=+0.15) strat=+0.05",
        "signals": {
            "drift": 0.30, "mom": 0.10, "strat": 0.05,
            "fut": 0.20, "tech": 0.10, "xasset": 0.15,
            "model_prob": 0.58, "regime": "low_vol_range",
        },
        "features": ["price_neutral", "mom_flat", "regime:low_vol_range"],
    }
    s.update(extra)
    return s


def test_enqueue_flush_and_resolve(tmp_db):
    ok = decision_log.enqueue(
        bot_name="meanrev-v1",
        strategy_type="mean_reversion",
        market_id="mkt-A",
        signal=_sig(action="skip"),
    )
    assert ok is True
    n = decision_log.flush()
    assert n == 1
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM decision_events").fetchone()
    assert row["action"] == "skip"
    assert row["drift"] == pytest.approx(0.30)
    assert row["fut"] == pytest.approx(0.20)
    assert row["market_up"] is None

    # Resolve UP — yes side would win
    n2 = decision_log.resolve_pending({"mkt-A": True})
    assert n2 == 1
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM decision_events").fetchone()
    assert row["market_up"] == 1
    assert row["would_win"] == 1
    assert row["hyp_pnl"] is not None and row["hyp_pnl"] > 0


def test_throttle_non_buy(tmp_db, monkeypatch):
    monkeypatch.setattr(config, "DECISION_LOG_MIN_INTERVAL_SEC", 60.0)
    a = decision_log.enqueue(
        bot_name="b", strategy_type="momentum", market_id="m1",
        signal=_sig())
    b = decision_log.enqueue(
        bot_name="b", strategy_type="momentum", market_id="m1",
        signal=_sig())
    assert a is True
    assert b is False  # throttled
    # Buys always land
    c = decision_log.enqueue(
        bot_name="b", strategy_type="momentum", market_id="m1",
        signal=_sig(action="buy"), force=True)
    assert c is True
    assert decision_log.flush() == 2


def test_core_attribution_from_decisions(tmp_db):
    for i in range(40):
        decision_log.enqueue(
            bot_name="momentum-v1",
            strategy_type="momentum",
            market_id=f"m{i}",
            signal=_sig(action="skip", side="yes"),
            force=True,
        )
    decision_log.flush()
    # All UP → drift +0.30 is correct
    decision_log.resolve_pending({f"m{i}": True for i in range(40)})
    with db.get_conn() as conn:
        attr = decision_log.core_lane_attribution(conn, deadband=0.05)
    assert attr["momentum"]["drift"]["n"] == 40
    assert attr["momentum"]["drift"]["accuracy"] == pytest.approx(1.0)


def test_candidate_attribution_net_edge(tmp_db):
    for i in range(20):
        decision_log.enqueue(
            bot_name="hybrid-v1",
            strategy_type="hybrid",
            market_id=f"c{i}",
            signal=_sig(action="skip"),
            force=True,
        )
    decision_log.flush()
    decision_log.resolve_pending({f"c{i}": True for i in range(20)})
    with db.get_conn() as conn:
        st = decision_log.candidate_lane_attribution(conn, "fut", 0.05)
    assert st["n"] == 20
    assert st["accuracy"] == pytest.approx(1.0)
    assert st["net_edge"] is not None and st["net_edge"] > 0


def test_rollup_persists(tmp_db):
    decision_log.enqueue(
        bot_name="meanrev-v1", strategy_type="mean_reversion",
        market_id="z1", signal=_sig(action="buy"), force=True)
    decision_log.flush()
    decision_log.resolve_pending({"z1": False})  # NO won → yes would lose
    report = decision_log.rollup()
    assert report["n_resolved"] >= 1
    raw = db.get_arena_state("decision_rollup")
    assert raw
    assert "n_resolved" in json.loads(raw)


def test_skip_learning_not_double_count_buy(tmp_db, monkeypatch):
    """Buys must not write bot_learning via decision resolve (trade path owns it)."""
    calls = []

    def fake_record(bot, feats, side, won):
        calls.append((bot, side, won))

    monkeypatch.setattr("learning.record_outcome", fake_record)
    decision_log.enqueue(
        bot_name="b1", strategy_type="momentum", market_id="t1",
        signal=_sig(action="buy"), force=True)
    decision_log.enqueue(
        bot_name="b1", strategy_type="momentum", market_id="t2",
        signal=_sig(action="skip"), force=True)
    decision_log.flush()
    decision_log.resolve_pending({"t1": True, "t2": True})
    # Only the skip should have recorded
    assert len(calls) == 1
    assert calls[0][0] == "b1"


def test_classify_skip_reason():
    assert decision_log.classify_skip_reason("Dead-zone gate: ...") == "dead_zone"
    assert decision_log.classify_skip_reason("No edge: yes edge=...") == "no_edge"
    assert decision_log.classify_skip_reason("Model lean too weak") == "weak_lean"

"""Tuner/promoter attribution must be one row per (strategy, market)."""

import db
from arena.decision_log import (
    candidate_lane_attribution,
    core_lane_attribution,
)


def _row(conn, **kw):
    cols = {
        "bot_name": "momentum-v1",
        "strategy_type": "momentum",
        "market_id": "m1",
        "action": "skip",
        "side": "yes",
        "skip_reason": "weak_lean",
        "drift": 0.40,
        "mom": 0.20,
        "strat": 0.10,
        "fut": 0.0,
        "tech": 0.30,
        "xasset": 0.20,
        "entry_price": 0.50,
        "market_up": 1,
        "would_win": 1,
        "hyp_pnl": 0.02,
    }
    cols.update(kw)
    conn.execute(
        """INSERT INTO decision_events (
               bot_name, strategy_type, market_id, action, side, skip_reason,
               drift, mom, strat, fut, tech, xasset, entry_price,
               market_up, would_win, hyp_pnl
           ) VALUES (
               :bot_name, :strategy_type, :market_id, :action, :side,
               :skip_reason, :drift, :mom, :strat, :fut, :tech, :xasset,
               :entry_price, :market_up, :would_win, :hyp_pnl
           )""",
        cols,
    )


def test_core_attribution_does_not_count_every_tick(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "attr.db")
    db.init_db()
    with db.get_conn() as conn:
        for _ in range(20):
            _row(conn, market_id="m1", drift=0.40)
        for _ in range(5):
            _row(conn, market_id="m2", drift=-0.40, market_up=0, side="no")
        out = core_lane_attribution(conn, 0.05, unique_market=True)
    assert out["momentum"]["drift"]["n"] == 2


def test_candidate_attribution_unique_market(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "attr2.db")
    db.init_db()
    with db.get_conn() as conn:
        for _ in range(10):
            _row(conn, market_id="m1", tech=0.40)
        out = candidate_lane_attribution(conn, "tech", 0.05, unique_market=True)
    assert out["n"] == 1

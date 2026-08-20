"""Unique-market live scorecard (lanes + gates) from decision_events."""

import db
from arena.live_scorecard import build_live_scorecard, unique_market_rows


def _insert_decision(conn, **kw):
    cols = {
        "bot_name": "hybrid-v1",
        "strategy_type": "hybrid",
        "market_id": "m1",
        "action": "skip",
        "side": "yes",
        "skip_reason": "dead_zone",
        "edge": 0.04,
        "confidence": 0.2,
        "entry_price": 0.51,
        "drift": 0.08,
        "mom": 0.1,
        "strat": 0.05,
        "fut": 0.0,
        "tech": 0.2,
        "xasset": 0.3,
        "market_up": 1,
        "would_win": 1,
        "hyp_pnl": 0.16,
    }
    cols.update(kw)
    conn.execute(
        """INSERT INTO decision_events (
               bot_name, strategy_type, market_id, action, side, skip_reason,
               edge, confidence, entry_price, drift, mom, strat, fut, tech,
               xasset, market_up, would_win, hyp_pnl
           ) VALUES (
               :bot_name, :strategy_type, :market_id, :action, :side,
               :skip_reason, :edge, :confidence, :entry_price, :drift, :mom,
               :strat, :fut, :tech, :xasset, :market_up, :would_win, :hyp_pnl
           )""",
        cols,
    )


def test_unique_market_collapses_ticks(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "sc.db")
    db.init_db()
    with db.get_conn() as conn:
        for i in range(8):
            _insert_decision(conn, market_id="mA", skip_reason="dead_zone")
        for i in range(3):
            _insert_decision(
                conn, market_id="mB", skip_reason="dead_zone",
                would_win=0, hyp_pnl=-0.10, market_up=0, side="yes",
            )
        # buy on mC should win the unique row over later skips
        _insert_decision(
            conn, market_id="mC", action="buy", skip_reason=None,
            would_win=1, hyp_pnl=0.05, entry_price=0.45,
        )
        _insert_decision(
            conn, market_id="mC", action="skip", skip_reason="weak_lean",
            would_win=1, hyp_pnl=None,
        )
        rows = unique_market_rows(conn)
    keys = {(r["strategy_type"], r["market_id"]) for r in rows}
    assert keys == {("hybrid", "mA"), ("hybrid", "mB"), ("hybrid", "mC")}
    buy = next(r for r in rows if r["market_id"] == "mC")
    assert buy["action"] == "buy"


def test_scorecard_gates_and_lanes_are_unique_market(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "sc2.db")
    db.init_db()
    with db.get_conn() as conn:
        _insert_decision(conn, market_id="m1", skip_reason="dead_zone",
                         would_win=1, hyp_pnl=0.16, drift=0.08)
        _insert_decision(conn, market_id="m1", skip_reason="dead_zone",
                         would_win=1, hyp_pnl=0.16, drift=0.08)
        _insert_decision(conn, market_id="m2", skip_reason="dead_zone",
                         would_win=0, hyp_pnl=-0.05, drift=0.09, market_up=0)
        _insert_decision(
            conn, market_id="m3", action="buy", skip_reason=None,
            would_win=1, hyp_pnl=0.08, drift=0.40, entry_price=0.44,
        )
    card = build_live_scorecard(hours=None)
    dz = card["gates"]["dead_zone"]
    assert dz["markets"] == 2
    assert dz["ticks"] == 2 or dz["ticks"] >= 2
    assert 0.4 <= dz["wr"] <= 0.6
    drift = card["lanes"]["drift"]
    assert drift["markets"] == 3
    assert drift["accuracy"] is not None
    assert card["meta"]["unique_markets"] == 3

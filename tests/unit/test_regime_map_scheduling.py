import db
from arena import regime_map


def test_rebuild_records_current_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "sc.db")
    db.init_db()
    cell = ("low_vol_range", 2, 3, "us", 0, 0)
    # get_resolved_trades_with_context orders DESC by created_at, so index 0 is
    # the most recent — that is what becomes current_cell.
    trades = [{"bot_name": "a", "pnl": 1.0, "cell": cell,
               "created_at": f"2026-07-20 10:{i:02d}:00", "context": {"x": 1}}
              for i in range(3)]
    monkeypatch.setattr(db, "get_resolved_trades_with_context",
                        lambda hours=None: trades)
    m = regime_map.rebuild()
    assert "current_cell" in m
    assert m["current_cell"] == list(cell)


def test_rebuild_current_cell_none_when_no_trades(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "sc2.db")
    db.init_db()
    monkeypatch.setattr(db, "get_resolved_trades_with_context",
                        lambda hours=None: [])
    m = regime_map.rebuild()
    assert m["current_cell"] is None

import db


def test_log_trade_stores_and_reads_context(tmp_path, monkeypatch):
    # Isolate the DB the way the repo's conftest arena_db fixture does:
    # db.get_conn() reads the module-global DB_PATH, so patch that directly.
    # (setenv + importlib.reload does NOT work: config.DB_PATH is fixed at
    # config import and db re-reads it, so the reload keeps the real path.)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    ctx = {"vol": 0.2, "trend": 0.1, "weekday": 2, "hour_block": 3,
           "session": "us", "macro_prox": 0, "vol_trend_regime": "low_vol_range",
           "btc_trend_slope": 0.0}
    rid = db.log_trade("momentum", "mkt1", "YES", 5.0, "paper", "paper",
                       context=ctx)
    db.resolve_trade(rid, "win", 1.5)
    rows = db.get_resolved_trades_with_context()
    assert len(rows) == 1
    assert rows[0]["context"]["session"] == "us"
    assert isinstance(rows[0]["cell"], tuple)
    assert rows[0]["pnl"] == 1.5


def test_context_column_migration_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t2.db")
    db.init_db()
    db.init_db()  # second call must not raise
    rid = db.log_trade("m", "mkt", "NO", 1.0, "paper", "paper", context=None)
    assert rid > 0

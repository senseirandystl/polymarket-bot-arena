import db
from arena import regime_map


def _mk(bot, pnl, cell, ts):
    return {"bot_name": bot, "pnl": pnl, "cell": cell, "created_at": ts}


def test_under_sampled_cell_not_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "r.db")
    db.init_db()
    cell = ("r", 2, 3, "us", 0, 0)
    trades = [_mk("a", 1.0, cell, "2026-07-20 10:00:00") for _ in range(5)]
    monkeypatch.setattr(db, "get_resolved_trades_with_context", lambda hours=None: trades)
    monkeypatch.setattr(regime_map.config, "REGIME_MIN_SAMPLES", 60, raising=False)
    m = regime_map.rebuild()
    regimes = {r["cell"]: r for r in m["regimes"]}
    assert regimes[list(regimes)[0]]["validated"] is False


def test_well_sampled_consistent_cell_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "r2.db")
    db.init_db()
    cell = ("r", 2, 3, "us", 0, 0)
    # 'a' consistently wins, 'b' consistently loses, 100 each
    trades = ([_mk("a", 2.0, cell, f"2026-07-20 10:{i:02d}:00") for i in range(60)]
              + [_mk("b", -2.0, cell, f"2026-07-20 11:{i:02d}:00") for i in range(60)])
    monkeypatch.setattr(db, "get_resolved_trades_with_context", lambda hours=None: trades)
    monkeypatch.setattr(regime_map.config, "REGIME_MIN_SAMPLES", 60, raising=False)
    m = regime_map.rebuild()
    reg = m["regimes"][0]
    assert reg["validated"] is True
    assert reg["bot_edges"]["a"]["shrunk_pnl"] > reg["bot_edges"]["b"]["shrunk_pnl"]
    # Persisted
    assert db.get_regime_map()["regimes"]

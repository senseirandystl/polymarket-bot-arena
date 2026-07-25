import db
from fastapi.testclient import TestClient

from dashboard import server


def test_regime_map_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "e.db")
    db.init_db()
    db.set_regime_map({
        "regimes": [{"cell": ["r", 2, 3, "us", 0, 0], "n": 80,
                     "validated": True, "bot_edges": {}}],
        "current_cell": ["r", 2, 3, "us", 0, 0],
    })
    client = TestClient(server.app)
    # App has a global Basic-auth dependency (default admin/Thor in tests).
    r = client.get("/api/regime-map", auth=("admin", "Thor"))
    assert r.status_code == 200
    body = r.json()
    assert body["regimes"][0]["validated"] is True
    assert "conditioning_enabled" in body

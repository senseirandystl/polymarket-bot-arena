"""Settings exchange toggles API (Polymarket / Kalshi)."""

from fastapi.testclient import TestClient

import dashboard.server as server


def _client():
    return TestClient(server.app)


def test_exchanges_require_auth():
    r = _client().get("/api/settings/exchanges")
    assert r.status_code == 401


def test_exchanges_roundtrip(tmp_path, monkeypatch):
    import db
    import exchanges as ex
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "ex-api.db")
    db.init_db()
    ex._TOGGLE_CACHE = (0.0, {})
    auth = ("admin", "Thor")
    c = _client()
    g = c.get("/api/settings/exchanges", auth=auth)
    assert g.status_code == 200
    body = g.json()
    assert body["polymarket"] is True
    assert body["kalshi"] is True
    p = c.post(
        "/api/settings/exchanges",
        auth=auth,
        json={"kalshi": False, "polymarket": True},
    )
    assert p.status_code == 200
    assert p.json()["kalshi"] is False
    assert p.json()["polymarket"] is True
    ex._TOGGLE_CACHE = (0.0, {})
    g2 = c.get("/api/settings/exchanges", auth=auth)
    assert g2.json()["kalshi"] is False


def test_price_kalshi_uses_kalshi_books(monkeypatch):
    import kalshi_markets
    import polymarket_markets

    def _pm_boom(*_a, **_k):
        raise AssertionError("PM current_prices must not run for Kalshi ids")

    monkeypatch.setattr(
        kalshi_markets, "current_prices",
        lambda ticker: {"yes": 0.61, "no": 0.39},
    )
    monkeypatch.setattr(polymarket_markets, "current_prices", _pm_boom)
    r = _client().get("/api/price/kalshi:KXBTC15M-1", auth=("admin", "Thor"))
    assert r.status_code == 200
    assert r.json()["yes"] == 0.61
    assert r.json()["no"] == 0.39

"""Unauthenticated /healthz liveness probe (slice G, request item #7)."""

from fastapi.testclient import TestClient

import dashboard.server as server


def _client():
    return TestClient(server.app)


def test_healthz_needs_no_auth():
    r = _client().get("/healthz")
    assert r.status_code == 200
    body = r.json()
    # ok when log is fresh and kill switch clear; degraded when stale/killed
    assert body["status"] in ("ok", "degraded")
    assert "ts" in body
    assert "arena_log_age_sec" in body
    assert "arena_log_stale" in body
    assert "kill_switch" in body


def test_other_endpoints_still_require_auth():
    # The health middleware must not disarm auth on the real routes.
    r = _client().get("/api/status")
    assert r.status_code == 401

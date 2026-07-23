"""Alerts dispatcher + health checks + ops snapshot."""

from unittest import mock

import pytest

from arena import alerts, health
from arena.ops_snapshot import recent_signal_contributions, ops_snapshot


def test_alerts_config_roundtrip(monkeypatch):
    saved = {}
    monkeypatch.setattr(alerts.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(alerts.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    cfg = alerts.save_config({
        "enabled": True,
        "channels": {"telegram": True, "discord": False, "email": False},
        "min_level": "warn",
    })
    assert cfg["enabled"] is True
    assert cfg["channels"]["telegram"] is True
    assert cfg["min_level"] == "warn"
    loaded = alerts.load_config()
    assert loaded["enabled"] is True


def test_notify_respects_disabled(monkeypatch):
    monkeypatch.setattr(alerts, "load_config", lambda: {
        "enabled": False,
        "channels": {"telegram": True, "discord": True, "email": True},
        "events": {e: True for e in alerts.EVENT_TYPES},
        "min_level": "info",
        "debounce_sec": 1,
    })
    r = alerts.notify("evolution", "test", "body", level="info")
    assert r["skipped"] is True
    assert r["sent"] is False


def test_notify_debounces(monkeypatch):
    monkeypatch.setattr(alerts, "load_config", lambda: {
        "enabled": True,
        "channels": {"telegram": False, "discord": False, "email": False},
        "events": {e: True for e in alerts.EVENT_TYPES},
        "min_level": "info",
        "debounce_sec": 600,
    })
    monkeypatch.setattr(alerts, "_append_log", lambda e: None)
    alerts._debounce.clear()
    r1 = alerts.notify("regime_shift", "A→B", key="A->B")
    r2 = alerts.notify("regime_shift", "A→B", key="A->B")
    assert r1["skipped"] is False
    assert r2["skipped"] is True  # debounced


def test_send_test_without_creds(monkeypatch):
    monkeypatch.setattr(alerts, "_cred", lambda k: None)
    monkeypatch.setattr(alerts, "load_config", lambda: {
        "enabled": True,
        "channels": {"telegram": True, "discord": True, "email": True},
        "events": {},
        "min_level": "info",
        "debounce_sec": 1,
    })
    monkeypatch.setattr(alerts, "_append_log", lambda e: None)
    r = alerts.send_test()
    assert r["success"] is True
    assert r["channels"]["telegram"]["ok"] is False


def test_health_arena_log_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(health.config, "LOG_DIR", tmp_path)
    c = health.check_arena_log()
    assert c["ok"] is False
    assert c["level"] == "critical"
    assert "recommend" in c


def test_health_run_report(monkeypatch, tmp_path):
    monkeypatch.setattr(health.config, "LOG_DIR", tmp_path)
    # Create a fresh log so arena_log is OK
    (tmp_path / "arena.log").write_text("ok\n")
    monkeypatch.setattr(health, "check_db", lambda: health._check("database", True, message="ok"))
    monkeypatch.setattr(health, "check_kill_switch",
                        lambda: health._check("kill_switch", True, message="clear"))
    monkeypatch.setattr(health, "check_risk_status",
                        lambda: health._check("risk", True, message="ok"))
    monkeypatch.setattr(health, "check_paper_pool",
                        lambda: health._check("paper_pool", True, message="$200"))
    monkeypatch.setattr(health, "check_session",
                        lambda: health._check("session", True, message="yes"))
    monkeypatch.setattr(health, "check_price_feed",
                        lambda: health._check("price_feed", True, message="ok"))
    monkeypatch.setattr(health, "check_disk",
                        lambda: health._check("disk", True, message="10GB"))
    report = health.run_health_checks()
    assert report["status"] == "healthy"
    assert report["restart"]["needed"] is False
    assert report["counts"]["ok"] >= 6


def test_signal_contributions_parse(monkeypatch):
    rows = [
        {"reasoning": "fair=0.55 P=0.55[drift=+0.12 mom=-0.03 strat=+0.05]",
         "side": "yes", "outcome": "win", "bot_name": "a"},
        {"reasoning": "fair=0.52 P=0.52[drift=+0.08 mom=+0.01]",
         "side": "yes", "outcome": "loss", "bot_name": "b"},
        {"reasoning": "sniper: no blend here", "side": "yes", "outcome": "win",
         "bot_name": "c"},
    ]

    class FakeConn:
        def execute(self, *a, **k):
            return self
        def fetchall(self):
            return rows
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(alerts.db if False else __import__("arena.ops_snapshot", fromlist=["db"]).db,
                        "get_conn", lambda: FakeConn())
    # Direct patch on ops module's db
    import arena.ops_snapshot as ops
    monkeypatch.setattr(ops.db, "get_conn", lambda: FakeConn())
    out = recent_signal_contributions(hours=6)
    assert out["trades_with_blend"] == 2
    lanes = {l["lane"]: l for l in out["lanes"]}
    assert "drift" in lanes
    assert lanes["drift"]["n"] == 2
    assert lanes["drift"]["mean"] > 0


def test_ops_snapshot_smoke(monkeypatch):
    monkeypatch.setattr("arena.ops_snapshot.recent_signal_contributions",
                        lambda **k: {"lanes": [], "trades_scanned": 0,
                                     "trades_with_blend": 0, "hours": 6})
    monkeypatch.setattr("arena.ops_snapshot.db.get_paper_available", lambda: 200.0)
    monkeypatch.setattr("arena.ops_snapshot.db.get_paper_bankroll", lambda: 200.0)
    monkeypatch.setattr("arena.ops_snapshot.db.get_kelly_fraction", lambda: 0.25)
    monkeypatch.setattr("arena.ops_snapshot.db.get_arena_state",
                        lambda k, d=None: None)
    snap = ops_snapshot()
    assert "ts" in snap
    assert "regime" in snap
    assert "risk" in snap
    assert "sizing" in snap
    assert snap["sizing"]["kelly_fraction"] == 0.25

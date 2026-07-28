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
    monkeypatch.setattr(alerts, "_cred", lambda k: None)
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


def test_default_config_on_when_channel_credentials_present(monkeypatch):
    """Master + configured channels default ON; unconfigured stay OFF."""
    monkeypatch.setattr(alerts.db, "get_arena_state", lambda k, d=None: None)
    creds = {
        alerts.CRED_TELEGRAM_TOKEN: "tok",
        alerts.CRED_TELEGRAM_CHAT: "123",
        # discord / email missing
    }
    monkeypatch.setattr(alerts, "_cred", lambda k: creds.get(k))
    cfg = alerts.load_config()
    assert cfg["enabled"] is True
    assert cfg["channels"]["telegram"] is True
    assert cfg["channels"]["discord"] is False
    assert cfg["channels"]["email"] is False


def test_default_config_off_without_credentials(monkeypatch):
    monkeypatch.setattr(alerts.db, "get_arena_state", lambda k, d=None: None)
    monkeypatch.setattr(alerts, "_cred", lambda k: None)
    monkeypatch.setattr(alerts.config, "ALERTS_ENABLED", False)
    cfg = alerts.load_config()
    assert cfg["enabled"] is False
    assert cfg["channels"]["telegram"] is False


def test_format_evolution_summary_includes_spawn_weights():
    report = {
        "skipped": False,
        "elites": ["winner"],
        "replaced": ["loser"],
        "individuals": [
            {"name": "winner", "status": "survivor", "elite": True,
             "pnl": 12.5, "win_rate": 0.6, "trades": 40, "fitness": 0.8},
            {"name": "loser", "status": "replaceable", "elite": False,
             "pnl": -30.0, "win_rate": 0.4, "trades": 35, "fitness": 0.1},
        ],
        "spawned": [{
            "name": "momentum-g2-111",
            "strategy_type": "momentum",
            "parents": ["winner", "other"],
            "replaced": "loser",
            "params": {"min_edge": 0.012, "lookback": 5},
        }],
    }
    body = alerts.format_evolution_summary(report, trigger="timer")
    assert "Culled" in body and "loser" in body
    assert "Survived" in body and "winner" in body
    assert "momentum-g2-111" in body
    assert "signal weights:" in body
    assert "drift=" in body
    assert "params:" in body


def test_format_hourly_report_concise():
    stats = {
        "hour_pnl": 3.5, "hour_n": 10, "hour_wins": 6, "hour_losses": 4,
        "hour_wr": 0.6, "open": 2,
        "day_pnl": 12.0, "day_n": 40, "day_wr": 0.55,
        "bots": [
            {"bot": "a", "n": 5, "wins": 3, "losses": 2, "pnl": 4.0},
            {"bot": "b", "n": 5, "wins": 3, "losses": 2, "pnl": -0.5},
        ],
    }
    title, body = alerts.format_hourly_report(
        stats, pool=200.0, risk_note="portfolio active", mode="Paper")
    assert "Hourly report" in title
    assert "Paper" in title
    assert " ET" in title  # scheduled windows labeled Eastern Time
    assert "UTC" not in title
    assert "Mode: Paper" in body
    assert "Last hour:" in body
    assert "Today:" in body
    assert "Top:" in body
    assert "Pool: $200.00" in body
    assert "Risk: portfolio active" in body


def test_format_hourly_report_live_mode():
    title, body = alerts.format_hourly_report(
        {"hour_pnl": 0, "hour_n": 0, "hour_wins": 0, "hour_losses": 0,
         "hour_wr": 0, "open": 0, "day_pnl": 0, "day_n": 0, "day_wr": 0,
         "bots": []},
        mode="Live",
    )
    assert "Live" in title
    assert "Mode: Live" in body
    assert title.endswith("ET") or " ET" in title


def test_format_daily_report():
    title, body = alerts.format_daily_report(
        {
            "day": "2026-07-24",
            "pnl": 15.5, "n": 40, "wins": 22, "losses": 18, "wr": 0.55,
            "avg_entry": 0.48, "be_gap": 0.07, "open": 1,
            "bots": [{"bot": "meanrev-v1", "n": 20, "pnl": 10.0,
                      "wins": 12, "losses": 8, "wr": 0.6}],
        },
        pool=210.0,
        mode="Paper",
    )
    assert "Daily EOD" in title
    assert "2026-07-24" in title
    assert " ET" in title
    assert "UTC" not in title
    assert "Mode: Paper" in body
    assert "Break-even gap" in body
    assert "meanrev-v1" in body


def test_new_event_types_in_defaults(monkeypatch):
    monkeypatch.setattr(alerts.db, "get_arena_state", lambda k, d=None: None)
    monkeypatch.setattr(alerts, "_cred", lambda k: None)
    monkeypatch.setattr(alerts.config, "ALERTS_ENABLED", False)
    cfg = alerts.load_config()
    for ev in (
        "daily_report", "low_bankroll", "feed_stale", "feed_restored", "live_fill",
        "lane_change", "core_lane_tune", "skip_storm", "resolver_stuck",
        "portfolio_rebalance", "startup", "regime_shift",
    ):
        assert ev in cfg["events"]
        assert cfg["events"][ev] is True


def test_feed_restored_alert_on_recovery(monkeypatch):
    """Stale→healthy edge fires feed_restored once; healthy→healthy is quiet."""
    saved = {alerts._FEED_STALE_FLAG_KEY: "1"}
    notified = []

    monkeypatch.setattr(alerts.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(alerts.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(alerts, "notify",
                        lambda *a, **kw: notified.append((a, kw)) or {"sent": True})
    monkeypatch.setattr(
        alerts, "publish_price_feed_status",
        lambda: {
            "stale": False,
            "symbols": {"btc": {"latest": 95000.0, "stale": False, "age_sec": 1.0}},
        },
    )

    r = alerts.maybe_alert_feed_stale()
    assert r is not None
    assert r.get("restored") is True
    assert any(kw.get("key") == "feed_restored" or (a and a[0] == "feed_restored")
               for a, kw in notified)
    assert saved.get(alerts._FEED_STALE_FLAG_KEY) == "0"

    notified.clear()
    r2 = alerts.maybe_alert_feed_stale()
    assert r2 is None
    assert not notified


def test_event_toggle_roundtrip(monkeypatch):
    """Per-type event filters persist via save_config / load_config."""
    saved = {}
    monkeypatch.setattr(alerts.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(alerts.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(alerts, "_cred", lambda k: None)
    cfg = alerts.save_config({
        "enabled": True,
        "events": {"regime_shift": False, "hourly_report": True},
    })
    assert cfg["events"]["regime_shift"] is False
    assert cfg["events"]["hourly_report"] is True
    # Unmentioned types keep default True
    assert cfg["events"]["evolution"] is True
    loaded = alerts.load_config()
    assert loaded["events"]["regime_shift"] is False


def test_notify_respects_event_type_disabled(monkeypatch):
    """Master On + channel ready still skips when event type is Off."""
    monkeypatch.setattr(alerts, "load_config", lambda: {
        "enabled": True,
        "channels": {"telegram": True, "discord": False, "email": False},
        "events": {
            **{e: True for e in alerts.EVENT_TYPES},
            "regime_shift": False,
        },
        "min_level": "info",
        "debounce_sec": 1,
    })
    monkeypatch.setattr(alerts, "_append_log", lambda e: None)
    monkeypatch.setattr(alerts, "_send_telegram", lambda *a, **k: (True, "ok"))
    alerts._debounce.clear()
    r = alerts.notify("regime_shift", "A→B", key="A->B")
    assert r["skipped"] is True
    assert r["sent"] is False
    r2 = alerts.notify("evolution", "cycle 1", key="evo:1")
    assert r2["skipped"] is False


def test_et_day_bounds_utc_span():
    """ET midnight maps to 05:00 UTC in winter / 04:00 UTC in summer."""
    start, end = alerts._et_day_bounds_utc("2026-01-15")  # EST
    assert start == "2026-01-15 05:00:00"
    assert end == "2026-01-16 05:00:00"
    start_s, end_s = alerts._et_day_bounds_utc("2026-07-15")  # EDT
    assert start_s == "2026-07-15 04:00:00"
    assert end_s == "2026-07-16 04:00:00"


def test_maybe_send_daily_report_uses_et(monkeypatch):
    """EOD fires after the configured ET hour, keyed on ET yesterday."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    # 00:10 ET — past hour=0 + grace=5 → should send
    fixed = datetime(2026, 7, 25, 0, 10, tzinfo=et)
    monkeypatch.setattr(alerts, "_et_now", lambda: fixed)
    monkeypatch.setattr(alerts.config, "ALERT_DAILY_REPORT_HOUR_ET", 0)
    monkeypatch.setattr(alerts.config, "ALERT_DAILY_REPORT_GRACE_MIN", 5)
    saved = {}
    monkeypatch.setattr(alerts.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(alerts.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    called = {}

    def _fake_daily(day=None):
        called["day"] = day
        return {"sent": True, "day": day}

    monkeypatch.setattr(alerts, "alert_daily_report", _fake_daily)
    out = alerts.maybe_send_daily_report()
    assert out is not None
    assert called["day"] == "2026-07-24"
    assert saved.get(alerts.DAILY_STATE_KEY) == "2026-07-24"
    # Second call same ET day is a no-op
    assert alerts.maybe_send_daily_report() is None


def test_maybe_send_daily_report_waits_for_grace(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    fixed = datetime(2026, 7, 25, 0, 2, tzinfo=et)  # before 5 min grace
    monkeypatch.setattr(alerts, "_et_now", lambda: fixed)
    monkeypatch.setattr(alerts.config, "ALERT_DAILY_REPORT_HOUR_ET", 0)
    monkeypatch.setattr(alerts.config, "ALERT_DAILY_REPORT_GRACE_MIN", 5)
    monkeypatch.setattr(alerts.db, "get_arena_state", lambda k, d=None: None)
    assert alerts.maybe_send_daily_report() is None


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

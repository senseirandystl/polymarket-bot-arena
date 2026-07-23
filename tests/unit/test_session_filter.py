"""Tests for the session-timing skip filter."""

from datetime import datetime, timezone

import config
from arena import session_filter as sf


def _utc(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def test_nyse_open_window_is_skipped(monkeypatch):
    monkeypatch.setattr(config, "SESSION_SKIP_ENABLED", True)
    monkeypatch.setattr(config, "SESSION_SKIP_WINDOWS_ET", ["09:30-10:15"])
    monkeypatch.setattr(config, "SESSION_SKIP_WEEKENDS", False)
    # 2026-07-13 is a Monday. 13:45 UTC = 09:45 ET (EDT, -4).
    reason = sf.session_skip(_utc(2026, 7, 13, 13, 45))
    assert reason is not None and "session skip" in reason


def test_outside_window_is_allowed(monkeypatch):
    monkeypatch.setattr(config, "SESSION_SKIP_ENABLED", True)
    monkeypatch.setattr(config, "SESSION_SKIP_WINDOWS_ET", ["09:30-10:15"])
    monkeypatch.setattr(config, "SESSION_SKIP_WEEKENDS", False)
    # 18:00 UTC = 14:00 ET — well clear of the open window.
    assert sf.session_skip(_utc(2026, 7, 13, 18, 0)) is None


def test_master_switch_disables_all(monkeypatch):
    monkeypatch.setattr(config, "SESSION_SKIP_ENABLED", False)
    monkeypatch.setattr(config, "SESSION_SKIP_WINDOWS_ET", ["00:00-23:59"])
    monkeypatch.setattr(config, "SESSION_SKIP_WEEKENDS", True)
    assert sf.session_skip(_utc(2026, 7, 13, 13, 45)) is None


def test_weekend_skip_when_enabled(monkeypatch):
    monkeypatch.setattr(config, "SESSION_SKIP_ENABLED", True)
    monkeypatch.setattr(config, "SESSION_SKIP_WINDOWS_ET", [])
    monkeypatch.setattr(config, "SESSION_SKIP_WEEKENDS", True)
    # 2026-07-11 is a Saturday.
    reason = sf.session_skip(_utc(2026, 7, 11, 18, 0))
    assert reason is not None and "weekend" in reason


def test_weekend_allowed_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "SESSION_SKIP_ENABLED", True)
    monkeypatch.setattr(config, "SESSION_SKIP_WINDOWS_ET", [])
    monkeypatch.setattr(config, "SESSION_SKIP_WEEKENDS", False)
    assert sf.session_skip(_utc(2026, 7, 11, 18, 0)) is None


def test_naive_datetime_is_treated_as_utc(monkeypatch):
    monkeypatch.setattr(config, "SESSION_SKIP_ENABLED", True)
    monkeypatch.setattr(config, "SESSION_SKIP_WINDOWS_ET", ["09:30-10:15"])
    monkeypatch.setattr(config, "SESSION_SKIP_WEEKENDS", False)
    naive = datetime(2026, 7, 13, 13, 45)  # no tzinfo
    assert sf.session_skip(naive) is not None

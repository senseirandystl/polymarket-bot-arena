"""Tests for the clean-tick price guard."""

import config
from signals import clean_tick


def setup_function():
    clean_tick.reset()


def test_first_tick_dropped_when_configured(monkeypatch):
    monkeypatch.setattr(config, "CLEAN_TICK_DROP_FIRST", True)
    # First read seeds state and is dropped (returns None → caller keeps prior).
    assert clean_tick.clean_price("tok", 0.50) is None
    # Second, plausible read is accepted.
    assert clean_tick.clean_price("tok", 0.51) == 0.51


def test_first_tick_kept_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "CLEAN_TICK_DROP_FIRST", False)
    assert clean_tick.clean_price("tok", 0.50) == 0.50


def test_implausible_jump_rejected(monkeypatch):
    monkeypatch.setattr(config, "CLEAN_TICK_DROP_FIRST", False)
    monkeypatch.setattr(config, "CLEAN_TICK_MAX_JUMP", 0.15)
    monkeypatch.setattr(config, "CLEAN_TICK_STALE_SEC", 10.0)
    assert clean_tick.clean_price("tok", 0.50) == 0.50
    # A 0.30 jump in one read is bad data → last good returned.
    assert clean_tick.clean_price("tok", 0.80) == 0.50
    # A small move is accepted.
    assert clean_tick.clean_price("tok", 0.55) == 0.55


def test_persistent_outlier_accepted_after_stale(monkeypatch):
    monkeypatch.setattr(config, "CLEAN_TICK_DROP_FIRST", False)
    monkeypatch.setattr(config, "CLEAN_TICK_MAX_JUMP", 0.15)
    monkeypatch.setattr(config, "CLEAN_TICK_STALE_SEC", 0.0)  # last good instantly stale
    assert clean_tick.clean_price("tok", 0.50) == 0.50
    # With stale_sec=0 the guard never latches — a real fast reprice gets in.
    assert clean_tick.clean_price("tok", 0.90) == 0.90


def test_none_returns_last_good(monkeypatch):
    monkeypatch.setattr(config, "CLEAN_TICK_DROP_FIRST", False)
    assert clean_tick.clean_price("tok", 0.50) == 0.50
    assert clean_tick.clean_price("tok", None) == 0.50
    # Unknown token with None stays None.
    assert clean_tick.clean_price("other", None) is None


def test_tokens_are_independent(monkeypatch):
    monkeypatch.setattr(config, "CLEAN_TICK_DROP_FIRST", False)
    monkeypatch.setattr(config, "CLEAN_TICK_MAX_JUMP", 0.15)
    monkeypatch.setattr(config, "CLEAN_TICK_STALE_SEC", 10.0)
    assert clean_tick.clean_price("a", 0.20) == 0.20
    assert clean_tick.clean_price("b", 0.80) == 0.80  # b not judged against a

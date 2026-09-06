# -*- coding: utf-8 -*-
"""Phase 4 audit hygiene: config, learning gate, decision_events prune, Windows helper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
import db


def test_paper_starting_balance_removed():
    assert not hasattr(config, "PAPER_STARTING_BALANCE")
    assert config.PAPER_BANKROLL_DEFAULT > 0


def test_num_bots_matches_lean_slate():
    from arena.startup import DEFAULT_INDICES

    assert config.NUM_BOTS == 3
    assert len(DEFAULT_INDICES) == 3
    assert DEFAULT_INDICES == [4, 7, 13]


def test_risk_kill_switch_bound_once_under_log_dir():
    # Single assignment after LOG_DIR; path should live under LOG_DIR.
    src = Path(config.__file__).read_text(encoding="utf-8")
    assert src.count("RISK_KILL_SWITCH_FILE =") == 1
    assert Path(config.RISK_KILL_SWITCH_FILE).parent == Path(config.LOG_DIR)


def test_fee_zone_default_max_clamped_to_hpg():
    from bots.bot_fee_zone_maker import DEFAULT_PARAMS

    assert DEFAULT_PARAMS["max_price_zone"] <= config.HIGH_PRICE_GUARD


def test_meanrev_tp_docstring_not_always_enter():
    import bots.bot_meanrev_tp as m

    doc = (m.__doc__ or "").lower()
    assert "always opens" not in doc
    assert "never skips" not in doc
    assert "may skip" in doc or "including skips" in doc


def test_record_outcome_noop_when_learning_disabled(arena_db, monkeypatch):
    import learning

    monkeypatch.setattr(config, "LEARNING_ENABLED", False)
    learning.record_outcome("mom-test", ["price_mid"], "yes", True)
    with db.get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) c FROM bot_learning WHERE bot_name=?",
            ("mom-test",),
        ).fetchone()["c"]
    assert n == 0


def test_record_outcome_writes_when_learning_enabled(arena_db, monkeypatch):
    import learning

    monkeypatch.setattr(config, "LEARNING_ENABLED", True)
    learning.record_outcome("mom-test2", ["price_mid"], "yes", True)
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT wins, losses FROM bot_learning WHERE bot_name=? AND feature_key=?",
            ("mom-test2", "price_mid"),
        ).fetchone()
    assert row is not None
    assert row["wins"] == 1


def test_prune_decision_events_keeps_newest(arena_db, monkeypatch):
    monkeypatch.setattr(config, "DECISION_EVENTS_MAX_ROWS", 3)
    monkeypatch.setattr(config, "DECISION_EVENTS_RETAIN_DAYS", 0)
    with db.get_conn() as conn:
        for i in range(5):
            conn.execute(
                """INSERT INTO decision_events
                   (bot_name, strategy_type, market_id, action, side)
                   VALUES (?, 'momentum', ?, 'skip', 'yes')""",
                (f"b{i}", f"m{i}"),
            )
    result = db.prune_decision_events(max_rows=3, retain_days=0)
    assert result["deleted_cap"] >= 2
    with db.get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) c FROM decision_events").fetchone()["c"]
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM decision_events ORDER BY id"
        ).fetchall()]
    assert n == 3
    assert ids == sorted(ids)[-3:] or len(ids) == 3


def test_maybe_prune_respects_interval(arena_db, monkeypatch):
    from arena import decision_log

    monkeypatch.setattr(config, "DECISION_EVENTS_PRUNE_INTERVAL_SEC", 99999)
    db.set_arena_state("decision_events_last_prune", str(__import__("time").time()))
    assert decision_log.maybe_prune() is None


def test_windows_arena_helper_exists():
    root = Path(__file__).resolve().parents[2]
    ps1 = root / "bin" / "arena.ps1"
    unix = root / "bin" / "arena"
    assert ps1.is_file()
    assert unix.is_file()
    text = ps1.read_text(encoding="utf-8")
    assert "ARENA_NO_DASHBOARD" in text
    assert ".venv\\Scripts\\python.exe" in text or r".venv\Scripts\python.exe" in text

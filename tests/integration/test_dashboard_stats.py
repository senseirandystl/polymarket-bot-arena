"""Tests for dashboard performance stats and Recent-Trades ordering.

Covers three behaviours added in July 2026:

  1. ``get_dashboard_stats`` reports RESOLVED trade counts in ``trades`` and
     unresolved ones separately in ``pending`` (so the UI can render "229 +2").
  2. A "Current Session" period appears only when ``session_start`` is recorded
     in ``arena_state``, and scopes counts to trades since that instant.
  3. The Recent-Trades query surfaces PENDING trades first, so a handful of
     pending rows are never truncated past the LIMIT by hundreds of resolved
     ones (the Active-Bots vs Recent-Trades reconciliation bug).
"""

import importlib

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """A fresh db module pointed at an isolated temp SQLite file."""
    import db as db_module

    test_path = tmp_path / "test_arena.db"
    monkeypatch.setattr(db_module, "DB_PATH", test_path)
    db_module.init_db()
    return db_module


def _insert(db, bot, outcome, pnl, created_at, resolved_at=None):
    with db.get_conn() as conn:
        conn.execute(
            """INSERT INTO trades
               (bot_name, market_id, side, amount, venue, mode,
                shares_bought, outcome, pnl, created_at, resolved_at)
               VALUES (?, 'mkt', 'yes', 1.0, 'simmer', 'paper',
                       2.0, ?, ?, ?, ?)""",
            (bot, outcome, pnl, created_at, resolved_at),
        )


def test_trades_count_excludes_pending(db):
    _insert(db, "a", "win", 1.0, "2026-07-12 10:00:00", "2026-07-12 10:05:00")
    _insert(db, "a", "loss", -1.0, "2026-07-12 10:01:00", "2026-07-12 10:06:00")
    _insert(db, "a", "expired", 0.0, "2026-07-12 10:02:00", "2026-07-12 10:07:00")
    _insert(db, "a", None, None, "2026-07-12 10:03:00")  # pending

    stats = db.get_dashboard_stats()["all_time"]
    assert stats["trades"] == 3      # win + loss + expired (resolved)
    assert stats["pending"] == 1     # the unresolved trade, counted apart
    assert stats["wins"] == 1
    assert stats["losses"] == 1


def test_session_absent_without_session_start(db):
    _insert(db, "a", "win", 1.0, "2026-07-12 10:00:00", "2026-07-12 10:05:00")
    assert db.get_dashboard_stats()["session"] is None


def test_session_scopes_to_session_start(db):
    # Two trades before the session boot, one after.
    _insert(db, "a", "win", 1.0, "2026-07-12 09:00:00", "2026-07-12 09:05:00")
    _insert(db, "a", "loss", -1.0, "2026-07-12 09:30:00", "2026-07-12 09:35:00")
    db.set_arena_state("session_start", "2026-07-12 10:00:00")
    _insert(db, "a", "win", 1.0, "2026-07-12 10:15:00", "2026-07-12 10:20:00")
    _insert(db, "a", None, None, "2026-07-12 10:16:00")  # pending, this session

    session = db.get_dashboard_stats()["session"]
    assert session is not None
    assert session["trades"] == 1    # only the post-boot resolved trade
    assert session["pending"] == 1
    assert db.get_dashboard_stats()["all_time"]["trades"] == 3


def test_current_bots_excludes_retired_roster(db):
    """Current Bots = all-time stats for active bots only (not retired)."""
    with db.get_conn() as conn:
        conn.execute(
            """INSERT INTO bot_configs
               (bot_name, strategy_type, generation, params, active)
               VALUES ('live-a', 'momentum', 0, '{}', 1)"""
        )
        conn.execute(
            """INSERT INTO bot_configs
               (bot_name, strategy_type, generation, params, active, retired_at)
               VALUES ('dead-b', 'hybrid', 0, '{}', 0, '2026-07-12 12:00:00')"""
        )
    _insert(db, "live-a", "win", 5.0, "2026-07-12 10:00:00", "2026-07-12 10:05:00")
    _insert(db, "live-a", "loss", -1.0, "2026-07-12 11:00:00", "2026-07-12 11:05:00")
    _insert(db, "dead-b", "loss", -20.0, "2026-07-12 10:30:00", "2026-07-12 10:35:00")
    _insert(db, "dead-b", None, None, "2026-07-12 11:30:00")  # pending retired

    stats = db.get_dashboard_stats()
    cur = stats["current_bots"]
    assert cur["trades"] == 2
    assert cur["pending"] == 0
    assert cur["pnl"] == pytest.approx(4.0)
    assert cur["wins"] == 1
    assert cur["losses"] == 1
    # All-time still includes the retired bot.
    assert stats["all_time"]["trades"] == 3
    assert stats["all_time"]["pnl"] == pytest.approx(-16.0)


def test_core_vs_lockin_split(db):
    """Performance card splits Core (directional) vs Lock-in (sweeper+arb)."""
    _insert(db, "sniper-v1", "win", 10.0, "2026-07-12 10:00:00", "2026-07-12 10:05:00")
    _insert(db, "hybrid-v1", "loss", -4.0, "2026-07-12 10:01:00", "2026-07-12 10:06:00")
    _insert(db, "sweeper-v1", "win", 0.5, "2026-07-12 10:02:00", "2026-07-12 10:07:00")
    _insert(db, "sweeper-v1", "win", 0.4, "2026-07-12 10:03:00", "2026-07-12 10:08:00")
    _insert(db, "arbitrage-v1", "win", 1.5, "2026-07-12 10:04:00", "2026-07-12 10:09:00")

    all_time = db.get_dashboard_stats()["all_time"]
    assert all_time["trades"] == 5
    assert all_time["pnl"] == pytest.approx(8.4)

    # Core = sniper + hybrid
    assert all_time["core_trades"] == 2
    assert all_time["core_pnl"] == pytest.approx(6.0)
    assert all_time["core_wins"] == 1
    assert all_time["core_losses"] == 1
    # Lock-in = sweeper + arb
    assert all_time["lockin_trades"] == 3
    assert all_time["lockin_pnl"] == pytest.approx(2.4)
    assert all_time["lockin_wins"] == 3
    assert all_time["lockin_losses"] == 0
    # Legacy aliases still filled
    assert all_time["directional_pnl"] == pytest.approx(6.0)
    assert all_time["structural_pnl"] == pytest.approx(2.4)


def test_core_vs_lockin_pending_split(db):
    """Pending counts are split so Core / Lock-in cells can show +N."""
    _insert(db, "sniper-v1", None, None, "2026-07-12 10:00:00")
    _insert(db, "hybrid-v1", None, None, "2026-07-12 10:01:00")
    _insert(db, "sweeper-v1", None, None, "2026-07-12 10:02:00")
    _insert(db, "sniper-v1", "win", 1.0, "2026-07-12 10:03:00", "2026-07-12 10:08:00")

    all_time = db.get_dashboard_stats()["all_time"]
    assert all_time["pending"] == 3
    assert all_time["core_pending"] == 2
    assert all_time["lockin_pending"] == 1
    assert all_time["trades"] == 1


def test_hour_is_rolling_sixty_minutes(db):
    """Performance card includes a rolling last-hour period."""
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    inside = (now - timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S")
    outside = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    _insert(db, "sniper-v1", "win", 3.0, inside, inside)
    _insert(db, "sweeper-v1", None, None, inside)
    _insert(db, "sniper-v1", "win", 9.0, outside, outside)

    stats = db.get_dashboard_stats()
    hour = stats["hour"]
    assert hour["trades"] == 1
    assert hour["pnl"] == pytest.approx(3.0)
    assert hour["pending"] == 1
    assert hour["core_pending"] == 0
    assert hour["lockin_pending"] == 1
    assert stats["all_time"]["trades"] == 2


def test_graveyard_stats_for_retired_bots(db):
    """Graveyard lists retired bots with lifetime P&L, worst first."""
    with db.get_conn() as conn:
        conn.execute(
            """INSERT INTO bot_configs
               (bot_name, strategy_type, generation, params, active, lineage)
               VALUES ('live-ok', 'sniper', 0, '{}', 1, 'live')"""
        )
        conn.execute(
            """INSERT INTO bot_configs
               (bot_name, strategy_type, generation, params, active, retired_at, lineage)
               VALUES ('momentum-v1', 'momentum', 0, '{"lookback_candles":5}', 0,
                       '2026-07-12 12:00:00', 'defaults')"""
        )
        conn.execute(
            """INSERT INTO bot_configs
               (bot_name, strategy_type, generation, params, active, retired_at)
               VALUES ('hybrid-g2', 'hybrid', 2, '{}', 0, '2026-07-12 14:00:00')"""
        )
    _insert(db, "live-ok", "win", 5.0, "2026-07-12 10:00:00", "2026-07-12 10:05:00")
    _insert(db, "momentum-v1", "loss", -15.0, "2026-07-12 10:30:00", "2026-07-12 10:35:00")
    _insert(db, "momentum-v1", "win", 2.0, "2026-07-12 11:00:00", "2026-07-12 11:05:00")
    _insert(db, "hybrid-g2", "loss", -3.0, "2026-07-12 11:30:00", "2026-07-12 11:35:00")

    gy = db.get_graveyard_stats()
    names = [g["bot_name"] for g in gy]
    assert "live-ok" not in names
    assert names == ["momentum-v1", "hybrid-g2"]  # worst P&L first
    mom = gy[0]
    assert mom["strategy_type"] == "momentum"
    assert mom["total_trades"] == 2
    assert mom["wins"] == 1
    assert mom["losses"] == 1
    assert mom["total_pnl"] == pytest.approx(-13.0)
    assert mom["win_rate"] == pytest.approx(0.5)
    assert mom["retired_at"] is not None


def test_recent_trades_orders_pending_first(db):
    # Many resolved trades, then a couple pending ones placed earlier in time.
    for i in range(30):
        _insert(db, "a", "win", 1.0, f"2026-07-12 11:{i:02d}:00",
                f"2026-07-12 12:{i:02d}:00")
    _insert(db, "p1", None, None, "2026-07-12 10:00:00")
    _insert(db, "p2", None, None, "2026-07-12 10:01:00")

    # Mirror the /api/trades ordering with a small LIMIT.
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT bot_name, outcome FROM trades
               ORDER BY
                   CASE WHEN outcome IS NULL THEN 0 ELSE 1 END,
                   COALESCE(resolved_at, created_at) DESC
               LIMIT ?""",
            (5,),
        ).fetchall()

    top_bots = {r["bot_name"] for r in rows[:2]}
    assert top_bots == {"p1", "p2"}         # both pending rows survive the LIMIT
    assert all(r["outcome"] is None for r in rows[:2])

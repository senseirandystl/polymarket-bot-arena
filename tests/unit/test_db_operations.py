"""Database read/write operations against an isolated tmp SQLite DB.

Covers the trade lifecycle (log → resolve → performance rollups), bot config
CRUD, the shared paper pool accounting, kelly-fraction persistence, per-bot
mode + cache invalidation, and arena_state key/value storage.
"""

import pytest


def _trade(db, bot="bot-a", market="mkt-1", side="yes", amount=10.0, **kw):
    return db.log_trade(bot, market, side, amount, venue="polymarket",
                        mode="paper", fill_source="paper_sim", **kw)


# --- trade lifecycle -------------------------------------------------------

def test_log_trade_returns_row_id_and_persists_fields(arena_db):
    tid = _trade(arena_db, confidence=0.4, reasoning="test thesis",
                 shares_bought=20.0, entry_price=0.5, fee=0.35,
                 trade_features={"price_level": "mid"})
    with arena_db.get_conn() as conn:
        row = conn.execute("SELECT * FROM trades WHERE id=?", (tid,)).fetchone()
    assert row["bot_name"] == "bot-a"
    assert row["side"] == "yes"
    assert row["amount"] == pytest.approx(10.0)
    assert row["entry_price"] == pytest.approx(0.5)
    assert row["fee"] == pytest.approx(0.35)
    assert row["fill_source"] == "paper_sim"
    assert row["outcome"] is None            # pending until resolution
    assert "price_level" in row["trade_features"]


def test_resolve_trade_sets_outcome_and_pnl(arena_db):
    tid = _trade(arena_db)
    arena_db.resolve_trade(tid, "win", 9.65)
    with arena_db.get_conn() as conn:
        row = conn.execute("SELECT * FROM trades WHERE id=?", (tid,)).fetchone()
    assert row["outcome"] == "win"
    assert row["pnl"] == pytest.approx(9.65)
    assert row["resolved_at"] is not None


def test_bot_performance_rollup(arena_db):
    for outcome, pnl in [("win", 5.0), ("win", 3.0), ("loss", -10.0)]:
        tid = _trade(arena_db)
        arena_db.resolve_trade(tid, outcome, pnl)
    perf = arena_db.get_bot_performance("bot-a", hours=24)
    assert perf["wins"] == 2
    assert perf["losses"] == 1
    assert perf["total_pnl"] == pytest.approx(-2.0)


# --- bot configs -----------------------------------------------------------

def test_bot_config_save_and_active_roster(arena_db):
    arena_db.save_bot_config("momo-1", "momentum", 0, {"lookback_candles": 5})
    arena_db.save_bot_config("rev-1", "mean_reversion", 0, {"min_drift": 0.1})
    names = [r["bot_name"] for r in arena_db.get_active_bots()]
    assert set(names) == {"momo-1", "rev-1"}


def test_retire_bot_removes_from_active(arena_db):
    arena_db.save_bot_config("momo-1", "momentum", 0, {})
    arena_db.retire_bot("momo-1")
    assert arena_db.get_active_bots() == []


def test_bot_mode_roundtrip_and_cache_invalidation(arena_db):
    arena_db.save_bot_config("momo-1", "momentum", 0, {})
    assert arena_db.get_bot_mode("momo-1") == "paper"   # default + primes cache
    arena_db.set_bot_mode("momo-1", "live")
    # set_bot_mode must bust the cache: the new mode is visible immediately.
    assert arena_db.get_bot_mode("momo-1") == "live"
    with pytest.raises(ValueError):
        arena_db.set_bot_mode("momo-1", "yolo")


# --- shared paper pool -----------------------------------------------------

def test_paper_available_reserves_open_cost_and_credits_pnl(arena_db):
    bankroll = arena_db.get_paper_bankroll()
    tid_open = _trade(arena_db, amount=25.0)            # pending: reserved
    tid_won = _trade(arena_db, amount=10.0)
    arena_db.resolve_trade(tid_won, "win", 8.0)         # resolved: pnl credited
    avail = arena_db.get_paper_available()
    assert avail == pytest.approx(bankroll + 8.0 - 25.0)
    # sanity: the still-open trade is the only reservation
    assert tid_open != tid_won


def test_topup_paper_bankroll_backsolves_available(arena_db):
    _trade(arena_db, amount=30.0)                       # open reservation
    tid = _trade(arena_db, amount=10.0)
    arena_db.resolve_trade(tid, "loss", -10.0)
    arena_db.topup_paper_bankroll(200.0)
    assert arena_db.get_paper_available() == pytest.approx(200.0)


def test_set_paper_bankroll_rejects_negative(arena_db):
    with pytest.raises(ValueError):
        arena_db.set_paper_bankroll(-5)


# --- kelly fraction --------------------------------------------------------

def test_kelly_fraction_default_persist_and_bounds(arena_db):
    import config
    assert arena_db.get_kelly_fraction() == pytest.approx(config.KELLY_FRACTION)
    arena_db.set_kelly_fraction(0.5)
    assert arena_db.get_kelly_fraction() == pytest.approx(0.5)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            arena_db.set_kelly_fraction(bad)


def test_kelly_fraction_ignores_corrupt_state(arena_db):
    import config
    arena_db.set_arena_state("kelly_fraction", "not-a-number")
    assert arena_db.get_kelly_fraction() == pytest.approx(config.KELLY_FRACTION)


# --- arena_state -----------------------------------------------------------

def test_arena_state_roundtrip_and_default(arena_db):
    """Values are stored stringified — callers json.dumps structured values."""
    import json
    assert arena_db.get_arena_state("missing", default="fallback") == "fallback"
    arena_db.set_arena_state("k", json.dumps({"nested": [1, 2]}))
    assert json.loads(arena_db.get_arena_state("k")) == {"nested": [1, 2]}
    arena_db.set_arena_state("k", "second")          # upsert overwrites
    assert arena_db.get_arena_state("k") == "second"


def test_wipe_all_clears_trades_and_configs(arena_db):
    _trade(arena_db)
    arena_db.save_bot_config("momo-1", "momentum", 0, {})
    arena_db.wipe_all()
    with arena_db.get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) c FROM trades").fetchone()["c"] == 0
    assert arena_db.get_active_bots() == []

"""Tests for entry-price-bucket ROI reporting (db.get_entry_price_buckets)."""

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test_arena.db")
    db_module.init_db()
    return db_module


def _insert(db, entry_price, amount, outcome, pnl):
    with db.get_conn() as conn:
        conn.execute(
            """INSERT INTO trades
               (bot_name, market_id, side, amount, venue, mode,
                shares_bought, outcome, pnl, entry_price)
               VALUES ('bot', 'mkt', 'yes', ?, 'polymarket', 'paper',
                       5.0, ?, ?, ?)""",
            (amount, outcome, pnl, entry_price),
        )


def test_buckets_group_and_compute_gap(db):
    # Three 72¢ entries, 2 wins / 1 loss → WR 0.667, avg_entry 0.72.
    _insert(db, 0.72, 3.60, "win", 1.40)
    _insert(db, 0.72, 3.60, "win", 1.40)
    _insert(db, 0.72, 3.60, "loss", -3.60)
    out = db.get_entry_price_buckets(mode="paper")
    row = next(r for r in out if r["bucket"] == "70-75")
    assert row["count"] == 3
    assert row["wins"] == 2
    assert round(row["win_rate"], 3) == 0.667
    assert row["avg_entry"] == 0.72
    # breakeven_gap = WR − avg_entry ≈ 0.667 − 0.72 = -0.053 (underwater).
    assert row["breakeven_gap"] < 0


def test_healthy_low_price_bucket_has_positive_gap(db):
    # 45¢ entries winning 70% → gap = 0.70 − 0.45 = +0.25 (well above break-even).
    for _ in range(7):
        _insert(db, 0.45, 2.25, "win", 2.75)
    for _ in range(3):
        _insert(db, 0.45, 2.25, "loss", -2.25)
    out = db.get_entry_price_buckets(mode="paper")
    row = next(r for r in out if r["bucket"] == "40-55")
    assert row["breakeven_gap"] > 0.2
    assert row["roi"] > 0


def test_unresolved_and_null_entry_excluded(db):
    _insert(db, 0.60, 3.0, None, None)      # pending — excluded
    _insert(db, None, 3.0, "win", 1.0)      # no entry_price — excluded
    _insert(db, 0.60, 3.0, "win", 2.0)      # counted
    out = db.get_entry_price_buckets(mode="paper")
    row = next(r for r in out if r["bucket"] == "55-65")
    assert row["count"] == 1


def test_mode_filter(db):
    _insert(db, 0.50, 2.5, "win", 2.5)
    # A live-mode row should not appear under mode='paper'.
    with db.get_conn() as conn:
        conn.execute(
            """INSERT INTO trades
               (bot_name, market_id, side, amount, venue, mode,
                shares_bought, outcome, pnl, entry_price)
               VALUES ('bot','mkt','yes',2.5,'polymarket','live',5.0,'win',2.5,0.50)""",
        )
    paper = db.get_entry_price_buckets(mode="paper")
    assert sum(r["count"] for r in paper) == 1

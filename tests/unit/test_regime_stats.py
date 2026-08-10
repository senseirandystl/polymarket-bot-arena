"""Live strategy×regime stats + toxic cell helpers."""

import json

import pytest

from arena.regime_stats import (
    is_healthy_cell,
    is_toxic_cell,
    parse_regime_tag,
    snapshot,
    strategy_regime_cell,
)


def test_parse_regime_tag_from_list():
    assert parse_regime_tag(["price_high", "regime:high_vol_chop"]) == "high_vol_chop"
    assert parse_regime_tag(json.dumps(["regime:normal"])) == "normal"
    assert parse_regime_tag(["regime_legacy:volatile"]) is None
    assert parse_regime_tag(None) is None


def test_toxic_cell_requires_samples_and_neg_pnl():
    assert not is_toxic_cell({"n": 5, "wins": 1, "pnl": -10, "wr": 0.2})
    assert is_toxic_cell(
        {"n": 25, "wins": 8, "pnl": -20, "wr": 0.32},
        min_n=20, wr_bar=0.40,
    )
    assert not is_toxic_cell(
        {"n": 25, "wins": 8, "pnl": 5.0, "wr": 0.32},
        min_n=20, wr_bar=0.40,
    )


def test_healthy_cell():
    assert is_healthy_cell(
        {"n": 30, "wins": 18, "pnl": 12.0, "wr": 0.60},
        min_n=20, wr_clear=0.48,
    )
    assert not is_healthy_cell(
        {"n": 30, "wins": 12, "pnl": -5.0, "wr": 0.40},
        min_n=20, wr_clear=0.48,
    )


def test_snapshot_from_trades(tmp_path, monkeypatch):
    import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "rs.db")
    db.init_db()
    db.save_bot_config("mom-1", "momentum", 0, {})
    feats = json.dumps(["regime:high_vol_chop"])
    with db.get_conn() as conn:
        for i in range(12):
            conn.execute(
                """INSERT INTO trades
                   (bot_name, market_id, side, amount, venue, mode, outcome,
                    pnl, trade_features, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))""",
                ("mom-1", f"m{i}", "no", 3.0, "paper", "paper",
                 "loss" if i < 8 else "win",
                 -2.0 if i < 8 else 2.0, feats),
            )
    from arena import regime_stats as rs
    rs.invalidate_cache()
    cell = strategy_regime_cell("high_vol_chop", "momentum")
    assert cell["n"] == 12
    assert cell["wins"] == 4
    assert cell["pnl"] < 0
    assert is_toxic_cell(cell, min_n=10, wr_bar=0.45)

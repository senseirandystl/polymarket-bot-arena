"""Default slate + mid-run deploy helpers."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arena.startup import (
    DEFAULT_INDICES,
    STRATEGY_MENU,
    build_default_bots,
    instantiate_strategy,
    strategy_catalog,
)
from arena.deploy import process_pending_deploys, unique_bot_name


def test_default_indices_include_hybrid_and_sweeper():
    assert DEFAULT_INDICES == [1, 2, 4, 6, 7, 13]
    bots = build_default_bots()
    assert len(bots) == 6
    types = [b.strategy_type for b in bots]
    assert types == [
        "momentum",
        "mean_reversion",
        "sniper",
        "hybrid",
        "arbitrage",
        "sweeper",
    ]
    names = [b.name for b in bots]
    assert names == [
        "momentum-v1",
        "meanrev-v1",
        "sniper-v1",
        "hybrid-v1",
        "arbitrage-v1",
        "sweeper-v1",
    ]


def test_strategy_catalog_covers_menu():
    cat = strategy_catalog()
    assert len(cat) == len(STRATEGY_MENU)
    types = {e["strategy_type"] for e in cat}
    assert "momentum" in types
    assert "sweeper" in types
    assert "late_window_maker" in types


def test_instantiate_strategy_known_and_unknown():
    bot = instantiate_strategy("phantom")
    assert bot.strategy_type == "phantom"
    assert bot.name == "phantom-v1"
    try:
        instantiate_strategy("not_a_real_strategy")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_unique_bot_name():
    assert unique_bot_name("hybrid-v1", set()) == "hybrid-v1"
    assert unique_bot_name("hybrid-v1", {"hybrid-v1"}) == "hybrid-v1-2"
    assert unique_bot_name("hybrid-v1", {"hybrid-v1", "hybrid-v1-2"}) == "hybrid-v1-3"


def test_process_pending_deploys_empty():
    trader = MagicMock()
    pos = MagicMock()
    with patch("arena.deploy.db.get_arena_state", return_value=None):
        t, m, res = process_pending_deploys([], [], trader, pos)
    assert t == []
    assert m == []
    assert res is None
    trader.set_bots.assert_not_called()


def test_process_pending_deploys_adds_trader_bot():
    trader = MagicMock()
    pos = MagicMock()
    payload = json.dumps({"strategies": ["phantom"]})
    saved = []

    def fake_save(name, st, gen, params, lineage=None):
        saved.append((name, st))

    with patch("arena.deploy.db.get_arena_state", return_value=payload), \
         patch("arena.deploy.db.set_arena_state"), \
         patch("arena.deploy.db.get_active_bots", return_value=[]), \
         patch("arena.deploy.db.save_bot_config", side_effect=fake_save), \
         patch("arena.deploy.db.set_bot_mode"), \
         patch("arena.portfolio.rebalance"):
        t, m, res = process_pending_deploys([], [], trader, pos)

    assert res is not None
    assert res["ok"] is True
    assert len(res["deployed"]) == 1
    assert res["deployed"][0]["strategy_type"] == "phantom"
    assert len(t) == 1
    assert t[0].strategy_type == "phantom"
    assert m == []
    trader.set_bots.assert_called_once()
    pos.update_bots.assert_called_once()
    assert saved and saved[0][1] == "phantom"


def test_process_pending_skips_already_active():
    trader = MagicMock()
    pos = MagicMock()
    payload = json.dumps({"strategies": ["momentum"]})
    with patch("arena.deploy.db.get_arena_state", return_value=payload), \
         patch("arena.deploy.db.set_arena_state"), \
         patch("arena.deploy.db.get_active_bots", return_value=[
             {"bot_name": "momentum-v1", "strategy_type": "momentum"},
         ]), \
         patch("arena.deploy.db.save_bot_config") as save, \
         patch("arena.portfolio.rebalance"):
        t, m, res = process_pending_deploys([], [], trader, pos)

    assert res is not None
    assert res["deployed"] == []
    assert any(s["reason"] == "already_active" for s in res["skipped"])
    save.assert_not_called()
    trader.set_bots.assert_not_called()

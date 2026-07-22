"""Tests for the 2026-07-19 set-and-forget fixes:

- Evolution judgment: EVOLUTION_WINDOW_HOURS window, break-even-gap /
  P&L survival rule (the old 2h-window + flat-WR bar made every bot
  permanently immune — zero evolutions in the whole 24h v5 run).
- Live lane monitor: parses the cand(...) readings out of resolved trades
  and auto-disables approved lanes whose live accuracy fails the bar.
- Auto-validation scheduler: market-count cadence, restart persistence,
  no double-spawn while a run is in flight.
"""

import importlib.util
import pathlib
import time
from contextlib import contextmanager
from unittest import mock

import pytest

import config
from arena import lane_monitor
from arena.validation_scheduler import ValidationScheduler

# ``import arena`` resolves to the ``arena/`` package, which shadows the
# top-level ``arena.py`` script that owns ``run_evolution``. Load the script
# explicitly by path (same pattern as test_maker_section.py).
_ARENA_PY = pathlib.Path(__file__).resolve().parent.parent / "arena.py"
_spec = importlib.util.spec_from_file_location("arena_main", _ARENA_PY)
arena = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arena)


# ---------------------------------------------------------------------------
# Evolution classification
# ---------------------------------------------------------------------------

class FakeBot:
    def __init__(self, name, strategy_type, perf):
        self.name = name
        self.strategy_type = strategy_type
        self.generation = 0
        self.strategy_params = {}
        self.lineage = None
        self._perf = perf
        self.reset_calls = 0

    def get_performance(self, hours=None):
        assert hours == config.EVOLUTION_WINDOW_HOURS, (
            "evolution must judge on the WINDOW, not the 2h cycle interval")
        return {"total_pnl": self._perf["pnl"],
                "win_rate": self._perf["wr"],
                "total_trades": self._perf["trades"],
                "breakeven_gap": self._perf.get("gap")}

    def reset_daily(self):
        self.reset_calls += 1


def _run(bots, monkeypatch):
    replaced = []
    monkeypatch.setattr(arena.db, "retire_bot", lambda name: replaced.append(name))
    monkeypatch.setattr(arena.db, "save_bot_config", lambda *a, **k: None)
    monkeypatch.setattr(arena.db, "log_evolution", lambda *a, **k: None)

    def fake_evolved(parent, strategy_type, cycle):
        return FakeBot(f"{parent.name}-g{cycle}", strategy_type,
                       {"pnl": 0, "wr": 0, "trades": 0})
    monkeypatch.setattr(arena, "create_evolved_bot", fake_evolved)
    monkeypatch.setattr(arena, "_validate_bot", lambda b: True)
    result = arena.run_evolution(bots, cycle_number=1)
    return result, replaced


def test_bot_below_min_trades_is_immune(monkeypatch):
    bots = [FakeBot("loser", "momentum",
                    {"pnl": -50.0, "wr": 0.30, "trades": 5, "gap": -0.2})]
    result, replaced = _run(bots, monkeypatch)
    assert replaced == []
    assert [b.name for b in result] == ["loser"]


def test_negative_pnl_and_gap_is_replaced(monkeypatch):
    # momentum-v1's actual v5 run shape: plenty of trades, negative P&L,
    # WR barely above entry — must finally be culled.
    bots = [
        FakeBot("winner", "mean_reversion",
                {"pnl": 47.0, "wr": 0.63, "trades": 30, "gap": 0.13}),
        FakeBot("loser", "momentum",
                {"pnl": -86.0, "wr": 0.508, "trades": 126, "gap": -0.006}),
    ]
    result, replaced = _run(bots, monkeypatch)
    assert replaced == ["loser"]
    names = [b.name for b in result]
    assert "winner" in names and "loser" not in names
    assert any(n.startswith("winner-g1") for n in names)  # mutated replacement


def test_positive_pnl_survives_even_with_thin_gap(monkeypatch):
    bots = [FakeBot("sizer", "hybrid",
                    {"pnl": 21.8, "wr": 0.547, "trades": 64, "gap": 0.01})]
    result, replaced = _run(bots, monkeypatch)
    assert replaced == []


def test_gap_clears_floor_survives_despite_negative_pnl(monkeypatch):
    # Good book, unlucky sizing: gap >= EVOLUTION_BE_GAP_MIN keeps it alive.
    bots = [
        FakeBot("ok", "sniper",
                {"pnl": -2.0, "wr": 0.62, "trades": 20, "gap": 0.05}),
        FakeBot("winner", "mean_reversion",
                {"pnl": 30.0, "wr": 0.60, "trades": 25, "gap": 0.10}),
    ]
    result, replaced = _run(bots, monkeypatch)
    assert replaced == []


def test_flat_65_wr_bar_is_gone(monkeypatch):
    # 63.3% WR + positive pnl — the old MIN_WIN_RATE=0.65 would have culled
    # the arena's most profitable bot.
    bots = [FakeBot("meanrev", "mean_reversion",
                    {"pnl": 47.0, "wr": 0.633, "trades": 30, "gap": 0.13})]
    result, replaced = _run(bots, monkeypatch)
    assert replaced == []
    assert not hasattr(config, "MIN_WIN_RATE")


# ---------------------------------------------------------------------------
# Lane monitor
# ---------------------------------------------------------------------------

def _trade(side, outcome, fut=0.0, tech=0.0, xa=0.0):
    return {"side": side, "outcome": outcome,
            "reasoning": f"... cand(fut={fut:+.2f} tech={tech:+.2f} "
                         f"xa={xa:+.2f}) strat=+0.0"}


def test_lane_accuracy_scores_sign_vs_market_direction():
    rows = [
        _trade("yes", "win", tech=+0.5),   # market UP, tech + -> correct
        _trade("no", "loss", tech=+0.5),   # market UP, tech + -> correct
        _trade("no", "win", tech=+0.5),    # market DOWN, tech + -> wrong
        _trade("yes", "win", tech=+0.02),  # inside deadband -> ignored
    ]
    stats = lane_monitor._lane_accuracy(rows, "tech", deadband=0.05)
    assert stats["n"] == 3
    assert stats["accuracy"] == pytest.approx(2 / 3)


def _patch_monitor_db(monkeypatch, overrides, rows):
    state, disabled = {}, []

    class FakeCursor:
        def fetchall(self):
            return rows

    class FakeConn:
        def execute(self, *a, **k):
            return FakeCursor()

    @contextmanager
    def fake_get_conn():
        yield FakeConn()

    monkeypatch.setattr(lane_monitor.db, "get_lane_overrides", lambda: overrides)
    monkeypatch.setattr(lane_monitor.db, "get_conn", fake_get_conn)
    monkeypatch.setattr(lane_monitor.db, "disable_lane_override",
                        lambda lane: disabled.append(lane))
    monkeypatch.setattr(lane_monitor.db, "set_arena_state",
                        lambda k, v: state.update({k: v}))
    return state, disabled


def test_lane_monitor_disables_failing_lane(monkeypatch):
    monkeypatch.setattr(config, "LANE_MONITOR_MIN_TRADES", 10, raising=False)
    # 10 readings, 5 correct (50%) — below the 53% bar.
    rows = ([_trade("yes", "win", tech=+0.5)] * 5
            + [_trade("no", "win", tech=+0.5)] * 5)
    overrides = {"tech": {"enabled": True, "approved_at": "2026-07-19 03:17:47"}}
    state, disabled = _patch_monitor_db(monkeypatch, overrides, rows)

    report = lane_monitor.check_lanes()
    assert disabled == ["tech"]
    assert report["tech"]["verdict"] == "disabled"
    assert "lane_monitor" in state


def test_lane_monitor_keeps_healthy_lane(monkeypatch):
    monkeypatch.setattr(config, "LANE_MONITOR_MIN_TRADES", 10, raising=False)
    rows = ([_trade("yes", "win", xa=+0.5)] * 8
            + [_trade("no", "win", xa=+0.5)] * 2)  # 80% accurate
    overrides = {"xasset": {"enabled": True, "approved_at": "2026-07-19"}}
    _, disabled = _patch_monitor_db(monkeypatch, overrides, rows)

    report = lane_monitor.check_lanes()
    assert disabled == []
    assert report["xasset"]["verdict"] == "healthy"


def test_lane_monitor_collecting_below_min_trades(monkeypatch):
    monkeypatch.setattr(config, "LANE_MONITOR_MIN_TRADES", 50, raising=False)
    rows = [_trade("yes", "win", fut=+0.5)] * 5
    overrides = {"fut": {"enabled": True, "approved_at": "2026-07-19"}}
    _, disabled = _patch_monitor_db(monkeypatch, overrides, rows)

    report = lane_monitor.check_lanes()
    assert disabled == []
    assert report["fut"]["verdict"] == "collecting"


# ---------------------------------------------------------------------------
# Auto-validation scheduler
# ---------------------------------------------------------------------------

def _patch_sched_db(monkeypatch, saved=None):
    state = {} if saved is None else dict(saved)
    import arena.validation_scheduler as vs
    monkeypatch.setattr(vs.db, "get_arena_state",
                        lambda k, default=None: state.get(k, default))
    monkeypatch.setattr(vs.db, "set_arena_state",
                        lambda k, v: state.update({k: v}))
    return state


def test_scheduler_not_due_immediately_on_fresh_boot(monkeypatch):
    _patch_sched_db(monkeypatch)
    sched = ValidationScheduler()
    assert not sched.due()


def test_scheduler_due_after_market_count_elapses(monkeypatch):
    every = getattr(config, "AUTO_VALIDATE_EVERY_MARKETS", 100)
    past = time.time() - every * 300 - 60
    _patch_sched_db(monkeypatch,
                    {"last_auto_validation_time": str(past)})
    sched = ValidationScheduler()
    assert sched.due()


def test_scheduler_spawns_and_persists(monkeypatch):
    every = getattr(config, "AUTO_VALIDATE_EVERY_MARKETS", 100)
    past = time.time() - every * 300 - 60
    state = _patch_sched_db(monkeypatch,
                            {"last_auto_validation_time": str(past)})
    sched = ValidationScheduler()

    proc = mock.Mock()
    proc.poll.return_value = None
    with mock.patch("arena.validation_scheduler.subprocess.Popen",
                    return_value=proc) as popen, \
         mock.patch("builtins.open", mock.mock_open()):
        assert sched.check() is True
        argv = popen.call_args[0][0]
        assert "--propose" in argv
        assert str(getattr(config, "AUTO_VALIDATE_WINDOW_MARKETS", 300)) in argv

    # Timer persisted -> restart-safe; and no double-spawn while running.
    assert float(state["last_auto_validation_time"]) > past
    assert not sched.due()
    assert sched.check() is False


def test_scheduler_disabled_flag(monkeypatch):
    monkeypatch.setattr(config, "AUTO_VALIDATE_ENABLED", False, raising=False)
    past = time.time() - 10_000_000
    _patch_sched_db(monkeypatch, {"last_auto_validation_time": str(past)})
    sched = ValidationScheduler()
    assert not sched.due()

"""Tests for evolution judgment + lane monitor + validation scheduler.

Evolution judgment now runs through the Genetic Algorithm
(``evolution.ga.run_ga_cycle``). These tests cover the arena-facing
survival rules (window, BE-gap, immune floor) and the lane-monitor /
scheduler utilities that share the evolution host loop.
"""

import importlib.util
import pathlib
import random
import time
from contextlib import contextmanager
from unittest import mock

import pytest

import config
from arena import lane_monitor
from arena.validation_scheduler import ValidationScheduler
from evolution.ga import run_ga_cycle

# ``import arena`` resolves to the ``arena/`` package, which shadows the
# top-level ``arena.py`` script that owns ``run_evolution``. Load the script
# explicitly by path (same pattern as test_maker_section.py).
_ARENA_PY = pathlib.Path(__file__).resolve().parents[2] / "arena.py"
_spec = importlib.util.spec_from_file_location("arena_main", _ARENA_PY)
arena = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arena)


# ---------------------------------------------------------------------------
# Evolution classification (via GA)
# ---------------------------------------------------------------------------

class FakeBot:
    def __init__(self, name, strategy_type, perf, params=None):
        self.name = name
        self.strategy_type = strategy_type
        self.generation = 0
        self.strategy_params = params or {
            "lookback_candles": 5,
            "momentum_threshold": 0.0003,
            "position_size_pct": 0.05,
            "min_confidence": 0.55,
        }
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

    def export_params(self):
        return {"name": self.name, "strategy_type": self.strategy_type,
                "generation": self.generation, "lineage": self.lineage,
                "params": dict(self.strategy_params)}


def _trade_rows(perf):
    n = int(perf["trades"])
    total = float(perf["pnl"])
    if n <= 0:
        return []
    avg = total / n
    return [
        {"pnl": avg, "outcome": "win" if avg >= 0 else "loss",
         "created_at": f"2026-07-20 12:{i:02d}:00"}
        for i in range(n)
    ]


def _patch_db(monkeypatch, bots):
    trade_map = {b.name: _trade_rows(b._perf) for b in bots}
    replaced = []

    class FakeCursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class FakeConn:
        def execute(self, sql, params=None):
            name = params[0] if params else None
            return FakeCursor(trade_map.get(name, []))

    @contextmanager
    def fake_get_conn():
        yield FakeConn()

    monkeypatch.setattr("evolution.ga.db.get_conn", fake_get_conn)
    monkeypatch.setattr("evolution.ga.db.retire_bot",
                        lambda name: replaced.append(name))
    monkeypatch.setattr("evolution.ga.db.save_bot_config", lambda *a, **k: None)
    monkeypatch.setattr("evolution.ga.db.log_evolution", lambda *a, **k: None)
    monkeypatch.setattr("evolution.ga.db.log_ga_generation", lambda *a, **k: None)
    monkeypatch.setattr("evolution.ga.db.set_arena_state", lambda *a, **k: None)
    monkeypatch.setattr("evolution.ga.db.get_arena_state",
                        lambda *a, **k: None)
    return replaced


def _factory(strategy_type, name, params, generation, lineage):
    b = FakeBot(name, strategy_type, {"pnl": 0, "wr": 0, "trades": 0},
                params=params)
    b.generation = generation
    b.lineage = lineage
    return b


def _run(bots, monkeypatch):
    replaced = _patch_db(monkeypatch, bots)
    # Arena run_evolution → run_ga_cycle; patch validate
    monkeypatch.setattr(arena, "_validate_bot", lambda b: True)

    def fake_ga(bots_in, cycle_number, **kwargs):
        return run_ga_cycle(
            bots_in, cycle_number,
            bot_factory=_factory,
            validate_fn=lambda b: True,
            rng=random.Random(0),
        )

    monkeypatch.setattr("evolution.ga.run_ga_cycle", fake_ga)
    # run_evolution imports run_ga_cycle inside the function
    import evolution.ga as ga_mod
    monkeypatch.setattr(ga_mod, "run_ga_cycle",
                        lambda bots_in, cycle_number, **kw: run_ga_cycle(
                            bots_in, cycle_number,
                            bot_factory=_factory,
                            validate_fn=lambda b: True,
                            rng=random.Random(0),
                        ))
    result, _report = arena.run_evolution(bots, cycle_number=1)
    return result, replaced


def test_bot_below_min_trades_is_immune(monkeypatch):
    bots = [FakeBot("loser", "momentum",
                    {"pnl": -50.0, "wr": 0.30, "trades": 5, "gap": -0.2})]
    result, replaced = _run(bots, monkeypatch)
    assert replaced == []
    assert [b.name for b in result] == ["loser"]


def test_negative_pnl_and_gap_is_replaced(monkeypatch):
    bots = [
        FakeBot("winner", "mean_reversion",
                {"pnl": 47.0, "wr": 0.63, "trades": 30, "gap": 0.13},
                params={"lookback_candles": 10, "min_drift": 0.1,
                        "position_size_pct": 0.05, "min_confidence": 0.55,
                        "reversion_threshold": 0.4, "bb_std_dev": 2.0,
                        "rsi_period": 14, "rsi_oversold": 40,
                        "rsi_overbought": 60, "trending_conf_damp": 0.6}),
        FakeBot("loser", "momentum",
                {"pnl": -86.0, "wr": 0.508, "trades": 126, "gap": -0.006}),
    ]
    result, replaced = _run(bots, monkeypatch)
    assert replaced == ["loser"]
    names = [b.name for b in result]
    assert "winner" in names and "loser" not in names
    # Replacement keeps the loser's strategy type under GA
    assert any("momentum" in n for n in names if n != "winner")


def test_positive_pnl_survives_even_with_thin_gap(monkeypatch):
    bots = [FakeBot("sizer", "hybrid",
                    {"pnl": 21.8, "wr": 0.547, "trades": 64, "gap": 0.01})]
    result, replaced = _run(bots, monkeypatch)
    assert replaced == []


def test_gap_clears_floor_survives_despite_negative_pnl(monkeypatch):
    bots = [
        FakeBot("ok", "sniper",
                {"pnl": -2.0, "wr": 0.62, "trades": 20, "gap": 0.05},
                params={"min_price_yes": 0.4, "max_price_yes": 0.78,
                        "max_price_no": 0.25, "skip_zone_low": 0.48,
                        "skip_zone_high": 0.64, "momentum_threshold": 0.0003,
                        "min_drift": 0.15, "quiet_drift_bump": 0.05,
                        "position_size_pct": 0.08, "min_confidence": 0.1}),
        FakeBot("winner", "mean_reversion",
                {"pnl": 30.0, "wr": 0.60, "trades": 25, "gap": 0.10},
                params={"lookback_candles": 10, "min_drift": 0.1,
                        "position_size_pct": 0.05, "min_confidence": 0.55,
                        "reversion_threshold": 0.4, "bb_std_dev": 2.0,
                        "rsi_period": 14, "rsi_oversold": 40,
                        "rsi_overbought": 60, "trending_conf_damp": 0.6}),
    ]
    # MIN_TRADES may be 30 — lower for this test so both are judged
    monkeypatch.setattr(config, "MIN_TRADES_FOR_JUDGMENT", 15, raising=False)
    result, replaced = _run(bots, monkeypatch)
    assert replaced == []


def test_flat_65_wr_bar_is_gone(monkeypatch):
    bots = [FakeBot("meanrev", "mean_reversion",
                    {"pnl": 47.0, "wr": 0.633, "trades": 30, "gap": 0.13})]
    result, replaced = _run(bots, monkeypatch)
    assert replaced == []
    assert not hasattr(config, "MIN_WIN_RATE")


def test_arbitrage_exempt_via_arena_constant():
    assert "arbitrage" in arena.EVOLUTION_EXEMPT_TYPES


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

    assert float(state["last_auto_validation_time"]) > past
    assert not sched.due()
    assert sched.check() is False


def test_scheduler_disabled_flag(monkeypatch):
    monkeypatch.setattr(config, "AUTO_VALIDATE_ENABLED", False, raising=False)
    past = time.time() - 10_000_000
    _patch_sched_db(monkeypatch, {"last_auto_validation_time": str(past)})
    sched = ValidationScheduler()
    assert not sched.due()

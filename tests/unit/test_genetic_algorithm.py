"""Tests for the Genetic Algorithm evolution package.

Covers:
* multi-objective fitness components
* operators (tournament, crossover, mutation, elitism)
* exempt strategy types
* live GA cycle wiring (replacements, elitism, lineage)
* offline GA improves fitness over generations on historical data
"""

from __future__ import annotations

import importlib.util
import pathlib
import random
from contextlib import contextmanager
from unittest import mock

import pytest

import config
from evolution import fitness as fit_mod
from evolution import operators as ops
from evolution.bounds import clamp, bounds_for
from evolution.fitness import (
    composite_from_raw,
    consistency_score,
    max_drawdown_pct,
    multi_objective_fitness,
    rank_normalize_fitness,
    sharpe_ratio,
)
from evolution.ga import EVOLUTION_EXEMPT_TYPES, run_ga_cycle, should_trigger_evolution
from evolution.offline import (
    evaluate_genome_on_history,
    make_historical_markets,
    run_offline_ga,
)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def test_sharpe_and_drawdown_basic():
    pnls = [1.0, 1.0, 1.0, -0.5, 1.0]
    assert sharpe_ratio(pnls) > 0
    assert 0.0 <= max_drawdown_pct(pnls) <= 1.0
    assert consistency_score(pnls, block_size=2) >= 0.0


def test_multi_objective_empty_is_zero():
    c = multi_objective_fitness(pnls=[])
    assert c["pnl"] == 0.0
    assert c["n_trades"] == 0
    assert c["drawdown"] == 0.0


def test_rank_normalize_prefers_higher_pnl():
    a = multi_objective_fitness(pnls=[2.0] * 20)
    b = multi_objective_fitness(pnls=[-1.0] * 20)
    ranked = rank_normalize_fitness([a, b])
    assert ranked[0]["fitness"] > ranked[1]["fitness"]


def test_composite_from_raw_monotone_in_pnl():
    low = composite_from_raw({"pnl": -30, "sharpe": 0, "drawdown": 0.5, "consistency": 0.3})
    high = composite_from_raw({"pnl": 30, "sharpe": 0, "drawdown": 0.5, "consistency": 0.3})
    assert high > low


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def test_crossover_blends_numeric_keys():
    rng = random.Random(1)
    # Use known bounded keys so clamp does not collapse the blend band
    child = ops.crossover(
        {"min_confidence": 0.2, "lookback_candles": 10, "flag": True},
        {"min_confidence": 0.8, "lookback_candles": 20, "flag": False},
        alpha=0.5,
        rng=rng,
    )
    assert child["min_confidence"] == pytest.approx(0.5)
    assert child["lookback_candles"] == 15
    assert child["flag"] is True  # primary parent


def test_mutate_stays_in_bounds():
    rng = random.Random(0)
    params = {"position_size_pct": 0.05, "min_confidence": 0.5, "lookback_candles": 10}
    for _ in range(50):
        out = ops.mutate(params, rate=1.0, sigma=0.5, rng=rng)
        lo, hi = bounds_for("position_size_pct", 0.05)
        assert lo <= out["position_size_pct"] <= hi
        assert isinstance(out["lookback_candles"], int)


def test_tournament_selects_fittest():
    rng = random.Random(2)
    pop = [{"fitness": i, "name": str(i)} for i in range(10)]
    # With k=10, must pick the global best
    winner = ops.tournament_select(pop, k=10, rng=rng)
    assert winner["fitness"] == 9


def test_tournament_select_pair_prefers_distinct():
    rng = random.Random(0)
    pop = [
        {"fitness": 0.9, "name": "a", "params": {"x": 1}},
        {"fitness": 0.5, "name": "b", "params": {"x": 2}},
        {"fitness": 0.3, "name": "c", "params": {"x": 3}},
    ]
    for _ in range(20):
        p1, p2, self_pair = ops.tournament_select_pair(pop, k=2, rng=rng)
        assert not self_pair
        assert p1["name"] != p2["name"]


def test_tournament_select_pair_self_when_monoculture():
    pop = [
        {"fitness": 0.9, "name": "phantom-v1", "params": {"ema_fast": 9}},
        # Gene-bank clone shares the same name → same identity
        {"fitness": 0.9, "name": "phantom-v1", "params": {"ema_fast": 9},
         "from_gene_bank": True},
    ]
    p1, p2, self_pair = ops.tournament_select_pair(pop, k=2, rng=random.Random(1))
    assert self_pair is True
    assert p1["name"] == p2["name"] == "phantom-v1"


def test_elite_indices_top_n():
    fits = [0.1, 0.9, 0.5, 0.8]
    idx = ops.elite_indices(fits, 2)
    assert idx == [1, 3]


def test_clamp_int_and_float():
    assert isinstance(clamp("lookback_candles", 3.7, reference=5), int)
    assert isinstance(clamp("min_confidence", 0.1234567, reference=0.5), float)


# ---------------------------------------------------------------------------
# Exempt types
# ---------------------------------------------------------------------------

def test_arbitrage_and_makers_are_exempt():
    assert "arbitrage" in EVOLUTION_EXEMPT_TYPES
    assert "late_window_maker" in EVOLUTION_EXEMPT_TYPES
    assert "fee_zone_maker" in EVOLUTION_EXEMPT_TYPES
    assert "btc_maker" in EVOLUTION_EXEMPT_TYPES
    assert "sweeper" in EVOLUTION_EXEMPT_TYPES
    assert "momentum" not in EVOLUTION_EXEMPT_TYPES


# ---------------------------------------------------------------------------
# Live GA cycle (mocked DB / trades)
# ---------------------------------------------------------------------------

class FakeBot:
    def __init__(self, name, strategy_type, perf, params=None, generation=0):
        self.name = name
        self.strategy_type = strategy_type
        self.generation = generation
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
        return {
            "total_pnl": self._perf["pnl"],
            "win_rate": self._perf["wr"],
            "total_trades": self._perf["trades"],
            "breakeven_gap": self._perf.get("gap"),
        }

    def reset_daily(self):
        self.reset_calls += 1

    def export_params(self):
        return {
            "name": self.name,
            "strategy_type": self.strategy_type,
            "generation": self.generation,
            "lineage": self.lineage,
            "params": dict(self.strategy_params),
        }


def _pnls_from_perf(perf):
    """Synthetic trade rows so multi-obj fitness has a series to score."""
    n = int(perf["trades"])
    if n <= 0:
        return []
    total = float(perf["pnl"])
    wr = float(perf["wr"])
    wins = max(0, min(n, int(round(wr * n))))
    losses = n - wins
    # Split total P&L across wins/losses roughly
    avg_w = (total / wins) if wins and total > 0 else 1.0
    avg_l = (total / losses) if losses and total < 0 else -1.0
    if wins and losses:
        # Solve roughly: wins*w + losses*l = total
        avg_w = abs(total) / max(wins, 1) + 0.5
        avg_l = -(abs(total) / max(losses, 1) + 0.5)
        if total < 0:
            avg_w, avg_l = 0.5, total / losses
        else:
            avg_w, avg_l = total / wins, -0.5
    rows = []
    for i in range(wins):
        rows.append({"pnl": avg_w, "outcome": "win", "created_at": f"2026-07-20 10:{i:02d}:00"})
    for i in range(losses):
        rows.append({"pnl": avg_l, "outcome": "loss", "created_at": f"2026-07-20 11:{i:02d}:00"})
    return rows


def _patch_ga_db(monkeypatch, bots):
    """Mock db reads/writes used by run_ga_cycle."""
    trade_map = {b.name: _pnls_from_perf(b._perf) for b in bots}
    retired = []
    saved = []
    evo_logs = []
    ga_logs = []
    state = {}

    class FakeCursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class FakeConn:
        def execute(self, sql, params=None):
            bot_name = params[0] if params else None
            rows = trade_map.get(bot_name, [])
            # Return row-like dicts
            return FakeCursor(rows)

    @contextmanager
    def fake_get_conn():
        yield FakeConn()

    monkeypatch.setattr("evolution.ga.db.get_conn", fake_get_conn)
    monkeypatch.setattr("evolution.ga.db.retire_bot", lambda n: retired.append(n))
    monkeypatch.setattr(
        "evolution.ga.db.save_bot_config",
        lambda *a, **k: saved.append(a),
    )
    monkeypatch.setattr(
        "evolution.ga.db.log_evolution",
        lambda *a, **k: evo_logs.append(a),
    )
    monkeypatch.setattr(
        "evolution.ga.db.log_ga_generation",
        lambda *a, **k: ga_logs.append(a),
    )
    monkeypatch.setattr(
        "evolution.ga.db.set_arena_state",
        lambda k, v: state.update({k: v}),
    )
    monkeypatch.setattr(
        "evolution.ga.db.get_arena_state",
        lambda k, default=None: state.get(k, default),
    )
    return retired, saved, evo_logs, ga_logs, state


class SpawnBot(FakeBot):
    """Bot class used by the factory for new individuals."""
    pass


def test_ga_cycle_replaces_loser_keeps_elite(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRADES_FOR_JUDGMENT", 10, raising=False)
    monkeypatch.setattr(config, "GA_ELITE_COUNT", 1, raising=False)
    monkeypatch.setattr(config, "EVOLUTION_BE_GAP_MIN", 0.03, raising=False)
    monkeypatch.setattr(config, "GA_TYPE_ALLOC_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_GATE_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "GA_RECENCY_WEIGHTING", False, raising=False)

    winner = FakeBot("winner", "mean_reversion",
                     {"pnl": 47.0, "wr": 0.63, "trades": 30, "gap": 0.13},
                     params={"lookback_candles": 10, "min_drift": 0.10,
                             "position_size_pct": 0.05, "min_confidence": 0.55,
                             "reversion_threshold": 0.4, "bb_std_dev": 2.0,
                             "rsi_period": 14, "rsi_oversold": 40,
                             "rsi_overbought": 60, "trending_conf_damp": 0.6})
    loser = FakeBot("loser", "momentum",
                    {"pnl": -86.0, "wr": 0.40, "trades": 50, "gap": -0.10})
    arb = FakeBot("arb-v1", "arbitrage",
                  {"pnl": 5.0, "wr": 0.9, "trades": 5})

    bots = [winner, loser, arb]
    retired, saved, evo_logs, ga_logs, _ = _patch_ga_db(monkeypatch, bots)

    def factory(strategy_type, name, params, generation, lineage):
        b = SpawnBot(name, strategy_type,
                     {"pnl": 0, "wr": 0, "trades": 0}, params=params,
                     generation=generation)
        b.lineage = lineage
        return b

    result, report = run_ga_cycle(
        bots, cycle_number=3,
        bot_factory=factory,
        validate_fn=lambda b: True,
        rng=random.Random(0),
    )

    names = [b.name for b in result]
    assert "winner" in names
    assert "loser" not in names
    assert "arb-v1" in names  # exempt
    assert retired == ["loser"]
    assert any(
        s[0].startswith("momentum-g3-") or "fallback" in s[0]
        for s in saved
    )
    assert report["replaced"] == ["loser"]
    assert report["elites"]  # at least one elite
    assert "winner" in report["elites"]
    assert evo_logs and ga_logs
    # Lineage present on spawn
    spawn = report["spawned"][0]
    assert spawn["parents"]
    op = spawn["operator"] or ""
    assert (
        spawn.get("breed_mode") in (
            "crossover+adaptive", "clone+mutate", "defaults+adaptive", "fallback",
        )
        or "crossover" in op
        or "clone" in op
        or "defaults" in op
        or op == "fallback"
    )
    assert spawn["lineage"]
    # Gene bank recorded the elite
    assert report.get("gene_bank_size", 0) >= 1


def test_ga_cycle_immune_below_min_trades(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRADES_FOR_JUDGMENT", 30, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_GATE_ENABLED", False, raising=False)
    thin = FakeBot("thin", "momentum",
                   {"pnl": -50.0, "wr": 0.30, "trades": 5, "gap": -0.2})
    bots = [thin]
    retired, _, _, _, _ = _patch_ga_db(monkeypatch, bots)
    result, report = run_ga_cycle(
        bots, 1,
        bot_factory=lambda *a, **k: SpawnBot(a[1], a[0], {"pnl": 0, "wr": 0, "trades": 0}),
        validate_fn=lambda b: True,
        rng=random.Random(1),
    )
    assert retired == []
    assert report["skipped"] is True
    assert [b.name for b in result] == ["thin"]


def test_ga_cycle_skips_when_all_profitable(monkeypatch):
    monkeypatch.setattr(config, "MIN_TRADES_FOR_JUDGMENT", 10, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_GATE_ENABLED", False, raising=False)
    a = FakeBot("a", "momentum", {"pnl": 10.0, "wr": 0.6, "trades": 40, "gap": 0.1})
    b = FakeBot("b", "hybrid", {"pnl": 5.0, "wr": 0.55, "trades": 40, "gap": 0.05})
    bots = [a, b]
    retired, _, _, _, _ = _patch_ga_db(monkeypatch, bots)
    result, report = run_ga_cycle(
        bots, 2,
        bot_factory=lambda *a, **k: SpawnBot(a[1], a[0], {"pnl": 0, "wr": 0, "trades": 0}),
        validate_fn=lambda b: True,
        rng=random.Random(2),
    )
    assert retired == []
    assert report["skipped"] is True
    assert set(b.name for b in result) == {"a", "b"}


def test_should_trigger_on_pool_bleed(monkeypatch):
    monkeypatch.setattr(config, "GA_PERF_TRIGGER_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "GA_PERF_TRIGGER_PNL", -25.0, raising=False)
    monkeypatch.setattr(config, "GA_PERF_TRIGGER_MIN_TRADES", 20, raising=False)
    bots = [
        FakeBot("a", "momentum", {"pnl": -20.0, "wr": 0.4, "trades": 30, "gap": -0.1}),
        FakeBot("b", "hybrid", {"pnl": -15.0, "wr": 0.4, "trades": 30, "gap": -0.1}),
        FakeBot("arb", "arbitrage", {"pnl": 100.0, "wr": 0.9, "trades": 100}),
    ]
    due, reason = should_trigger_evolution(bots)
    assert due is True
    assert "pool_pnl" in reason


def test_should_not_trigger_when_healthy(monkeypatch):
    monkeypatch.setattr(config, "GA_PERF_TRIGGER_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "GA_PERF_TRIGGER_PNL", -25.0, raising=False)
    monkeypatch.setattr(config, "GA_PERF_TRIGGER_MIN_TRADES", 20, raising=False)
    bots = [
        FakeBot("a", "momentum", {"pnl": 10.0, "wr": 0.6, "trades": 30, "gap": 0.1}),
        FakeBot("b", "hybrid", {"pnl": 5.0, "wr": 0.55, "trades": 30, "gap": 0.05}),
    ]
    due, _ = should_trigger_evolution(bots)
    assert due is False


# ---------------------------------------------------------------------------
# Offline GA: fitness improves over generations on historical data
# ---------------------------------------------------------------------------

def test_offline_ga_improves_best_fitness_on_historical_data():
    """The core acceptance test: GA search climbs a learnable landscape."""
    history = make_historical_markets(n=500, seed=123)
    result = run_offline_ga(
        history,
        pop_size=24,
        generations=18,
        elite_count=2,
        seed=99,
    )
    best = result["best_fitness_curve"]
    mean = result["mean_fitness_curve"]
    # Best fitness of final generation > first generation
    assert best[-1] > best[0], (
        f"best fitness did not improve: start={best[0]:.4f} end={best[-1]:.4f} curve={best}"
    )
    # Mean should also rise (elitism + selection pressure)
    assert mean[-1] > mean[0], (
        f"mean fitness did not improve: start={mean[0]:.4f} end={mean[-1]:.4f}"
    )
    # Elites are never worse than the population mean at the end
    assert result["best_individual"]["fitness"] == pytest.approx(best[-1])
    # Genome stayed within GA/param bounds
    p = result["best_individual"]["params"]
    assert 0.01 <= p["position_size_pct"] <= 0.15
    assert 0.0 < p["momentum_threshold"] <= 0.90
    assert 3 <= int(p["lookback_candles"]) <= 40


def test_offline_history_is_learnable():
    """Sanity: a near-optimal threshold genome beats a noisy overtrader."""
    history = make_historical_markets(n=400, seed=7)
    # Near the signal/noise scale with moderate size — captures edge, pays few fees
    good = {
        "momentum_threshold": 0.30,
        "min_confidence": 0.25,
        "position_size_pct": 0.05,
        "lookback_candles": 8,
    }
    # Overtrades noise with large size — fee drag + bad WR
    bad = {
        "momentum_threshold": 0.02,
        "min_confidence": 0.05,
        "position_size_pct": 0.12,
        "lookback_candles": 3,
    }
    g = composite_from_raw(evaluate_genome_on_history(good, history))
    b = composite_from_raw(evaluate_genome_on_history(bad, history))
    assert g > b, f"good={g:.4f} should beat bad={b:.4f}"


def test_elitism_preserves_best_genome_identity():
    """With zero mutation/crossover noise forced via elite-only pop of 1…"""
    history = make_historical_markets(n=200, seed=1)
    # Manually verify elite_indices keeps the top
    comps = [
        evaluate_genome_on_history(
            {"momentum_threshold": t, "min_confidence": 0.3,
             "position_size_pct": 0.05, "lookback_candles": 8},
            history,
        )
        for t in (0.1, 0.3, 0.5, 0.7)
    ]
    scores = [composite_from_raw(c) for c in comps]
    elites = ops.elite_indices(scores, 1)
    assert elites[0] == scores.index(max(scores))


# ---------------------------------------------------------------------------
# Arena wrapper still imports and delegates
# ---------------------------------------------------------------------------

def test_arena_run_evolution_delegates_to_ga(monkeypatch):
    _ARENA_PY = pathlib.Path(__file__).resolve().parents[2] / "arena.py"
    _spec = importlib.util.spec_from_file_location("arena_main_ga", _ARENA_PY)
    arena = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(arena)

    called = {}

    def fake_cycle(bots, cycle_number, **kwargs):
        called["cycle"] = cycle_number
        called["bots"] = [b.name for b in bots]
        return bots, {"spawned": [], "replaced": [], "elites": []}

    monkeypatch.setattr("evolution.ga.run_ga_cycle", fake_cycle)
    # Also patch the import path used inside run_evolution
    import evolution.ga as ga_mod
    monkeypatch.setattr(ga_mod, "run_ga_cycle", fake_cycle)

    bots = [FakeBot("x", "momentum", {"pnl": 1, "wr": 0.5, "trades": 40, "gap": 0.05})]
    out, report = arena.run_evolution(bots, 5)
    assert called.get("cycle") == 5
    assert out is bots or [b.name for b in out] == ["x"]
    assert isinstance(report, dict)


# ---------------------------------------------------------------------------
# Gene bank / type alloc / backtest gate / frozen genes
# ---------------------------------------------------------------------------

def test_gene_bank_records_and_exposes_parents(monkeypatch):
    from evolution import gene_bank as gb
    state = {}
    monkeypatch.setattr(gb.db, "get_arena_state", lambda k, d=None: state.get(k, d))
    monkeypatch.setattr(gb.db, "set_arena_state", lambda k, v: state.update({k: v}))
    monkeypatch.setattr(config, "GA_GENE_BANK_SIZE", 5, raising=False)

    inds = [{
        "name": "elite-a", "strategy_type": "momentum", "generation": 2,
        "fitness": 0.9, "pnl": 20, "win_rate": 0.6, "trades": 40,
        "params": {"lookback_candles": 8, "momentum_threshold": 0.0003},
        "elite": True, "lineage": None,
    }, {
        "name": "scrub", "strategy_type": "hybrid", "generation": 1,
        "fitness": 0.1, "pnl": -10, "win_rate": 0.4, "trades": 40,
        "params": {"min_confidence": 0.5}, "elite": False,
    }]
    bank = gb.record_elites(inds, cycle=4)
    assert len(bank) == 1
    assert bank[0]["name"] == "elite-a"
    parents = gb.as_parent_records(bank)
    assert parents[0]["from_gene_bank"] is True
    assert parents[0]["fitness"] == 0.9


def test_type_alloc_stickiness_prefers_dead_type(monkeypatch):
    from evolution.type_alloc import pick_strategy_type
    monkeypatch.setattr(config, "GA_TYPE_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "GA_TYPE_STICKINESS", 0.95, raising=False)
    monkeypatch.setattr(config, "GA_TYPE_ALLOC_TEMPERATURE", 0.5, raising=False)
    # Even with a strong hybrid, high stickiness keeps momentum
    inds = [
        {"strategy_type": "hybrid", "fitness": 1.0},
        {"strategy_type": "momentum", "fitness": 0.1},
    ]
    rng = random.Random(0)
    picks = [pick_strategy_type("momentum", inds, [], rng=rng) for _ in range(30)]
    assert picks.count("momentum") >= 20


def test_type_alloc_disabled_returns_dead(monkeypatch):
    from evolution.type_alloc import pick_strategy_type
    monkeypatch.setattr(config, "GA_TYPE_ALLOC_ENABLED", False, raising=False)
    assert pick_strategy_type("sniper", [{"strategy_type": "hybrid", "fitness": 1}], []) == "sniper"


def test_type_alloc_exclude_types(monkeypatch):
    from evolution.type_alloc import pick_strategy_type
    monkeypatch.setattr(config, "GA_TYPE_ALLOC_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "GA_TYPE_STICKINESS", 0.0, raising=False)
    monkeypatch.setattr(config, "GA_TYPE_ALLOC_TEMPERATURE", 0.2, raising=False)
    # Strong hybrid fitness — but hybrid excluded → never pick hybrid
    inds = [
        {"strategy_type": "hybrid", "fitness": 1.0},
        {"strategy_type": "momentum", "fitness": 0.2},
    ]
    rng = random.Random(0)
    picks = [
        pick_strategy_type(
            "momentum", inds, [], rng=rng, exclude_types={"hybrid"},
        )
        for _ in range(40)
    ]
    assert "hybrid" not in picks


def test_ga_cross_type_uses_defaults_not_foreign_params(monkeypatch):
    """Phantom monoculture must not write min_confidence into hybrid children."""
    monkeypatch.setattr(config, "MIN_TRADES_FOR_JUDGMENT", 10, raising=False)
    monkeypatch.setattr(config, "GA_ELITE_COUNT", 1, raising=False)
    monkeypatch.setattr(config, "EVOLUTION_BE_GAP_MIN", 0.03, raising=False)
    monkeypatch.setattr(config, "GA_TYPE_ALLOC_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_GATE_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "GA_RECENCY_WEIGHTING", False, raising=False)
    monkeypatch.setattr(config, "GA_ADAPTIVE_MUTATION", False, raising=False)
    monkeypatch.setattr(config, "GA_MUTATION_RATE", 0.0, raising=False)
    monkeypatch.setattr(config, "GA_MAX_PER_TYPE_PER_CYCLE", 2, raising=False)

    from bots.bot_hybrid import DEFAULT_PARAMS as HYBRID_DEFAULTS

    elite = FakeBot(
        "phantom-v1", "phantom",
        {"pnl": 40.0, "wr": 0.65, "trades": 40, "gap": 0.12},
        params={
            "ema_fast": 9, "ema_slow": 26, "atr_period": 10,
            "breakout_lookback": 10, "min_atr_pct": 0.0002,
            "max_atr_pct": 0.01, "position_size_pct": 0.06,
            "min_confidence": 0.20, "regime_conf_weight": 0.3,
        },
    )
    # Culled hybrid — type alloc off keeps hybrid type
    loser = FakeBot(
        "hybrid-v1", "hybrid",
        {"pnl": -50.0, "wr": 0.40, "trades": 50, "gap": -0.10},
        params=dict(HYBRID_DEFAULTS),
    )
    bots = [elite, loser]
    _patch_ga_db(monkeypatch, bots)

    def factory(strategy_type, name, params, generation, lineage):
        b = SpawnBot(name, strategy_type,
                     {"pnl": 0, "wr": 0, "trades": 0}, params=params,
                     generation=generation)
        b.lineage = lineage
        return b

    result, report = run_ga_cycle(
        bots, cycle_number=4,
        bot_factory=factory,
        validate_fn=lambda b: True,
        rng=random.Random(0),
    )
    spawn = report["spawned"][0]
    assert spawn["strategy_type"] == "hybrid"
    assert spawn.get("breed_mode") == "defaults+adaptive"
    # min_confidence must stay at hybrid default (0.5), NOT phantom's 0.2
    assert spawn["params"]["min_confidence"] == pytest.approx(
        HYBRID_DEFAULTS["min_confidence"]
    )
    assert "defaults" in (spawn["lineage"] or "")


def test_ga_caps_duplicate_types_per_cycle(monkeypatch):
    """Two open slots must not both become hybrid when max_per_type=1."""
    monkeypatch.setattr(config, "MIN_TRADES_FOR_JUDGMENT", 10, raising=False)
    monkeypatch.setattr(config, "GA_ELITE_COUNT", 1, raising=False)
    monkeypatch.setattr(config, "EVOLUTION_BE_GAP_MIN", 0.03, raising=False)
    monkeypatch.setattr(config, "GA_TYPE_ALLOC_ENABLED", True, raising=False)
    # High stickiness to hybrid would otherwise double-spawn hybrid
    monkeypatch.setattr(config, "GA_TYPE_STICKINESS", 0.95, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_GATE_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "GA_RECENCY_WEIGHTING", False, raising=False)
    monkeypatch.setattr(config, "GA_MAX_PER_TYPE_PER_CYCLE", 1, raising=False)

    elite = FakeBot(
        "phantom-v1", "phantom",
        {"pnl": 40.0, "wr": 0.65, "trades": 40, "gap": 0.12},
        params={"min_confidence": 0.2, "position_size_pct": 0.06,
                "ema_fast": 9, "ema_slow": 26},
    )
    losers = [
        FakeBot("hybrid-v1", "hybrid",
                {"pnl": -40.0, "wr": 0.40, "trades": 40, "gap": -0.10}),
        FakeBot("momentum-v1", "momentum",
                {"pnl": -35.0, "wr": 0.42, "trades": 40, "gap": -0.08}),
    ]
    bots = [elite] + losers
    _patch_ga_db(monkeypatch, bots)

    def factory(strategy_type, name, params, generation, lineage):
        b = SpawnBot(name, strategy_type,
                     {"pnl": 0, "wr": 0, "trades": 0}, params=params or {},
                     generation=generation)
        b.lineage = lineage
        return b

    _, report = run_ga_cycle(
        bots, cycle_number=5,
        bot_factory=factory,
        validate_fn=lambda b: True,
        rng=random.Random(42),
    )
    types = [s["strategy_type"] for s in report["spawned"]]
    assert len(types) == 2
    # No strategy type may appear more than once among spawns (cap=1 and
    # phantom already occupies the phantom slot as elite survivor).
    from collections import Counter
    counts = Counter(types)
    assert max(counts.values()) <= 1, counts


def test_frozen_genes_skip_volume_and_kelly():
    from evolution.frozen import evolvable_keys, frozen_genes
    assert "volume_weight" in frozen_genes()
    assert "position_size_pct" in frozen_genes()
    params = {
        "lookback_candles": 10,
        "momentum_threshold": 0.0003,
        "volume_weight": 0.5,
        "position_size_pct": 0.05,
        "min_confidence": 0.5,
    }
    keys = evolvable_keys("momentum", params)
    assert "lookback_candles" in keys
    assert "volume_weight" not in keys
    assert "position_size_pct" not in keys


def test_adaptive_mutate_preserves_frozen(monkeypatch):
    from evolution.param_search import adaptive_mutate
    monkeypatch.setattr(config, "GA_ADAPTIVE_MUTATION", True, raising=False)
    rng = random.Random(1)
    params = {
        "lookback_candles": 10,
        "momentum_threshold": 0.0003,
        "volume_weight": 0.77,
        "position_size_pct": 0.05,
        "min_confidence": 0.5,
        "trend_strength_weight": 0.5,
        "regime_conf_weight": 0.3,
    }
    out = adaptive_mutate(
        params, strategy_type="momentum",
        elite_genomes=[{"lookback_candles": 12, "momentum_threshold": 0.0004,
                        "min_confidence": 0.55, "trend_strength_weight": 0.6,
                        "regime_conf_weight": 0.2}],
        rate=1.0, rng=rng,
    )
    assert out["volume_weight"] == 0.77
    assert out["position_size_pct"] == 0.05


def test_backtest_gate_rejects_worse_child(monkeypatch):
    from evolution.backtest_gate import evaluate_offspring, clear_cache
    clear_cache()
    monkeypatch.setattr(config, "GA_BACKTEST_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_BEAT_BASELINE", True, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_EPS", 0.5, raising=False)

    class FakeData:
        markets = [1, 2, 3]

    child = object()
    base = object()
    gate = evaluate_offspring(
        child, baseline_bot=base,
        load_fn=lambda n: FakeData(),
        run_fn=lambda bot, data: -10.0 if bot is child else 5.0,
    )
    assert gate.passed is False
    assert gate.reason == "worse_than_baseline"


def test_backtest_gate_passes_better_child(monkeypatch):
    from evolution.backtest_gate import evaluate_offspring, clear_cache
    clear_cache()
    monkeypatch.setattr(config, "GA_BACKTEST_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_BEAT_BASELINE", True, raising=False)

    class FakeData:
        markets = [1, 2, 3]

    child = object()
    base = object()
    gate = evaluate_offspring(
        child, baseline_bot=base,
        load_fn=lambda n: FakeData(),
        run_fn=lambda bot, data: 8.0 if bot is child else 2.0,
    )
    assert gate.passed is True
    assert gate.child_pnl == 8.0


def test_weighted_pnls_boost_recent_and_regime():
    from evolution.fitness import weighted_trade_pnls
    import time
    now = time.time()
    trades = [
        {"pnl": 10.0, "outcome": "win", "created_at": now - 3600,
         "trade_features": ["regime:high_vol_trend"]},
        {"pnl": 10.0, "outcome": "win", "created_at": now - 48 * 3600,
         "trade_features": ["regime:low_vol_range"]},
    ]
    w = weighted_trade_pnls(
        trades, current_regime="high_vol_trend", now_ts=now,
        halflife_hours=6.0, regime_boost=2.0,
    )
    assert len(w) == 2
    # Ordered oldest-first: recent+regime-match (index 1) outweights old (0)
    assert abs(w[1]) > abs(w[0])

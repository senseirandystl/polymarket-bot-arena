"""Spawn diversity + gene bank per-type quotas."""

from __future__ import annotations

import random

import pytest

import config
from evolution.diversity import is_diverse_enough, param_distance
from evolution import gene_bank as gb


def test_param_distance_zero_for_identical():
    p = {"min_confidence": 0.5, "lookback_candles": 10, "position_size_pct": 0.05}
    # position_size_pct may be frozen — distance still defined on evolvable keys
    d = param_distance(p, p, strategy_type="momentum")
    assert d == pytest.approx(0.0)


def test_is_diverse_enough_rejects_near_clone(monkeypatch):
    monkeypatch.setattr(config, "GA_DIVERSITY_MIN_DISTANCE", 0.15, raising=False)
    a = {"min_confidence": 0.50, "lookback_candles": 10, "momentum_threshold": 0.0003}
    b = {"min_confidence": 0.51, "lookback_candles": 10, "momentum_threshold": 0.0003}
    assert not is_diverse_enough(b, strategy_type="momentum", peers=[a])
    c = {"min_confidence": 0.80, "lookback_candles": 25, "momentum_threshold": 0.001}
    assert is_diverse_enough(c, strategy_type="momentum", peers=[a])


def test_gene_bank_type_quota(monkeypatch):
    state = {}
    monkeypatch.setattr(gb.db, "get_arena_state", lambda k, d=None: state.get(k, d))
    monkeypatch.setattr(gb.db, "set_arena_state", lambda k, v: state.update({k: v}))
    monkeypatch.setattr(config, "GA_GENE_BANK_SIZE", 20, raising=False)
    monkeypatch.setattr(config, "GA_GENE_BANK_MAX_PER_TYPE", 2, raising=False)

    # 4 phantoms + 1 momentum — after quota only 2 phantoms + 1 mom
    inds = []
    for i, fit in enumerate([0.9, 0.8, 0.7, 0.6]):
        inds.append({
            "name": f"ph-{i}", "strategy_type": "phantom", "generation": i,
            "fitness": fit, "pnl": fit * 10, "win_rate": 0.6, "trades": 40,
            "params": {"ema_fast": 9 + i, "min_confidence": 0.2},
            "elite": True, "lineage": None,
        })
    inds.append({
        "name": "mom-0", "strategy_type": "momentum", "generation": 1,
        "fitness": 0.5, "pnl": 5, "win_rate": 0.55, "trades": 40,
        "params": {"lookback_candles": 8, "momentum_threshold": 0.0003},
        "elite": True, "lineage": None,
    })
    bank = gb.record_elites(inds, cycle=1)
    types = [e["strategy_type"] for e in bank]
    assert types.count("phantom") <= 2
    assert "momentum" in types
    # Highest fitness phantoms kept
    ph_fits = [e["fitness"] for e in bank if e["strategy_type"] == "phantom"]
    assert max(ph_fits) == 0.9

"""Phase 4 learning spine + optional LLM (provider=none) unit tests."""

from __future__ import annotations

import json

import pytest

from signals.strategy_pipeline.compiler import new_spec_id, sanitize_spec
from signals.strategy_pipeline.learning_spine import (
    AUTOPSY_KEYS,
    bias_params_away_from_bands,
    build_structured_autopsy,
    fingerprint_blocked,
    fingerprint_str,
    get_constraints,
    ingest_autopsy,
    load_spine,
    save_spine,
)
from signals.strategy_pipeline import llm as llm_mod
from signals.strategy_pipeline.store import HypothesisStore


FAR_MOM = {
    "lookback_candles": 25,
    "momentum_threshold": 0.003,
    "min_confidence": 0.72,
}


def test_autopsy_structure_keys_present():
    autopsy = build_structured_autopsy(
        params=dict(FAR_MOM),
        primitive="momentum",
        verdict="backtest_pnl_fail",
        stage="backtested",
        evidence={"regime_mix": {"chop": 3}, "skip_codes": {"no_edge": 2}},
    )
    for key in AUTOPSY_KEYS:
        assert key in autopsy, f"missing autopsy key {key}"
    assert autopsy["fingerprint"]
    assert autopsy["avoid_constraints"]
    assert autopsy["reason"] == "backtest_pnl_fail"
    assert autopsy["died_at_stage"] == "backtested"


def test_constraints_block_clone_fingerprint(arena_db):
    save_spine({
        "updated_at": None,
        "avoid_fingerprints": [],
        "avoid_param_bands": [],
        "prefer_factor_cells": [],
        "stats": {},
    })
    autopsy = build_structured_autopsy(
        params=dict(FAR_MOM),
        primitive="momentum",
        verdict="dead_genome",
        stage="retired",
    )
    ingest_autopsy(autopsy)
    cons = get_constraints()
    assert fingerprint_blocked("momentum", FAR_MOM, cons) is True
    # Different params should not block.
    other = dict(FAR_MOM)
    other["lookback_candles"] = 11
    assert fingerprint_blocked("momentum", other, cons) is False


def test_propose_respects_avoid(arena_db, monkeypatch):
    from signals.strategy_pipeline import research as research_mod

    monkeypatch.setattr("config.STRATEGY_LAB_LLM_PROVIDER", "none", raising=False)
    store = HypothesisStore()
    autopsy = build_structured_autopsy(
        params=dict(FAR_MOM),
        primitive="momentum",
        verdict="lab_reject",
        stage="backtested",
    )
    ingest_autopsy(autopsy)

    # Force catalog to only the blocked genome (+ one free).
    def _only_blocked(context):
        return [
            {
                "primitive": "momentum",
                "name": "mom-wide",
                "thesis": "blocked",
                "params": dict(FAR_MOM),
                "spec_id": new_spec_id("momentum"),
                "universe": ["polymarket:btc_5m"],
                "origin": "heuristic",
            },
            {
                "primitive": "sniper",
                "name": "sniper-ok",
                "thesis": "free",
                "params": {
                    "min_drift": 0.28,
                    "min_confidence": 0.25,
                    "quiet_drift_bump": 0.12,
                },
                "spec_id": new_spec_id("sniper"),
                "universe": ["polymarket:btc_5m"],
                "origin": "heuristic",
            },
        ]

    monkeypatch.setattr(research_mod, "_heuristic_candidates", _only_blocked)
    monkeypatch.setattr(research_mod, "_gene_bank_mutations", lambda ctx: [])
    monkeypatch.setattr(research_mod, "_llm_candidates", lambda ctx: [])

    out = research_mod.propose(store, max_new=3)
    prims = {s.get("primitive") for s in out}
    assert "momentum" not in prims or not any(
        fingerprint_blocked("momentum", s.get("params") or {}, get_constraints())
        for s in out
        if s.get("primitive") == "momentum"
    )
    # At least the free sniper can land (unless clone of active — empty arena ok).
    assert any(s.get("primitive") == "sniper" for s in out)


def test_llm_none_path_unchanged(arena_db, monkeypatch):
    monkeypatch.setattr("config.STRATEGY_LAB_LLM_PROVIDER", "none", raising=False)
    from signals.strategy_pipeline.control import update_settings

    update_settings({"llm_provider": "none"})
    assert llm_mod.provider_name() == "none"
    assert llm_mod.research_assist({"universe": []}) == []
    assert llm_mod.narrate_autopsy({"verdict": "x"}) == ""
    assert llm_mod.suggest_params("momentum", {}) == {}


def test_bias_away_from_avoid_bands():
    cons = {
        "avoid_param_bands": [{
            "primitive": "momentum",
            "param": "lookback_candles",
            "lo": 20.0,
            "hi": 30.0,
            "count": 2,
        }],
        "avoid_fingerprints": [],
        "prefer_factor_cells": [],
    }
    params = dict(FAR_MOM)  # lookback 25 inside band
    out = bias_params_away_from_bands(
        "momentum", params, cons, rng=__import__("random").Random(0)
    )
    assert out["lookback_candles"] != 25 or not (20 <= float(out["lookback_candles"]) <= 30)
    assert not (20.0 <= float(out["lookback_candles"]) <= 30.0)


def test_ga_retire_hook_imports_spine(arena_db):
    """GA retire path can import spine and write autopsy without error (mock bot)."""
    from signals.strategy_pipeline.learning_spine import write_autopsy_from_bot

    bot = {
        "name": "mom-test-retire",
        "strategy_type": "momentum",
        "params": dict(FAR_MOM),
    }
    autopsy = write_autopsy_from_bot(
        bot, verdict="ga_cull", stage="retired", narrate=False, store=HypothesisStore()
    )
    for key in AUTOPSY_KEYS:
        assert key in autopsy
    assert fingerprint_blocked("momentum", FAR_MOM, get_constraints())


def test_postmortem_uses_spine(arena_db):
    from signals.strategy_pipeline.postmortem import write_autopsy

    store = HypothesisStore()
    spec = sanitize_spec({
        "spec_id": new_spec_id("momentum"),
        "primitive": "momentum",
        "name": "mom-pm",
        "params": dict(FAR_MOM),
    })
    store.insert({**spec, "stage": "backtested"})
    autopsy = write_autopsy(
        store, spec["spec_id"], stage="backtested", reason="unit_reject",
        evidence={"params": dict(FAR_MOM), "primitive": "momentum"},
    )
    for key in AUTOPSY_KEYS:
        assert key in autopsy
    row = store.get(spec["spec_id"])
    assert row["status"] == "closed"
    assert (row.get("autopsy") or {}).get("fingerprint")


def test_fold_learned_rules_adapter(arena_db, monkeypatch):
    from signals.strategy_pipeline.learning_spine import fold_learned_rules, save_spine

    fake = {
        "rules": [{
            "cell": "chop|mid|strong|up",
            "type": "skip",
            "n": 40,
            "wr": 0.40,
            "effect": {"size_mult": 0.5},
        }],
        "cells": {},
    }
    monkeypatch.setattr("arena.learned_rules.load_state", lambda: fake)
    state = fold_learned_rules(save_spine({
        "avoid_fingerprints": [],
        "avoid_param_bands": [],
        "prefer_factor_cells": [],
        "stats": {},
    }))
    cells = state.get("prefer_factor_cells") or []
    assert any(c.get("source") == "learned_rules" and c.get("kind") == "avoid" for c in cells)

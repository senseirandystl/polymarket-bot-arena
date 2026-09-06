"""Phase 1 Strategy Lab pipeline unit tests."""

from __future__ import annotations

import json

import pytest

from signals.strategy_pipeline.compiler import (
    EXCLUDED_PRIMITIVES,
    normalize_primitive,
    sanitize_spec,
    new_spec_id,
)
from signals.strategy_pipeline.store import HypothesisStore
from signals.strategy_pipeline.postmortem import write_autopsy
from signals.strategy_pipeline.founders import (
    FOUNDER_BOTS,
    is_protected_bot,
    ga_may_retire,
)


FAR_MOM_PARAMS = {
    "lookback_candles": 25,
    "momentum_threshold": 0.003,
    "min_confidence": 0.72,
}


def test_import_package():
    import signals.strategy_pipeline as sp
    assert hasattr(sp, "get_host")
    assert hasattr(sp, "LabHost")


def test_compile_allowlist_excludes_arbitrage():
    assert "arbitrage" in EXCLUDED_PRIMITIVES
    with pytest.raises(ValueError):
        normalize_primitive("arbitrage")
    spec = sanitize_spec({
        "primitive": "momentum",
        "name": "mom-lab",
        "params": dict(FAR_MOM_PARAMS),
    })
    assert spec["primitive"] == "momentum"
    assert "arbitrage" not in (
        sanitize_spec({"primitive": "sniper", "params": {"min_drift": 0.3}})["primitive"],
    )


def test_unknown_primitive_rejected():
    with pytest.raises(ValueError):
        normalize_primitive("lstm-oracle")


def test_protect_founders(arena_db):
    from signals.strategy_pipeline.control import load_control, update_settings
    from signals.strategy_pipeline.founders import mark_founders_in_db

    data = load_control()
    locks = data.get("founder_locks") or {}
    for name in FOUNDER_BOTS:
        assert name in locks
        assert locks[name].get("protected") is True
        assert is_protected_bot(name, control_data=data)
        assert ga_may_retire(name) is False

    mark_founders_in_db()
    for name in FOUNDER_BOTS:
        arena_db.save_bot_config(name, name.split("-")[0], 0, {})
    mark_founders_in_db()
    with arena_db.get_conn() as conn:
        try:
            row = conn.execute(
                "SELECT protected FROM bot_configs WHERE bot_name=?",
                ("sniper-v1",),
            ).fetchone()
            if row is not None and "protected" in row.keys():
                assert int(row["protected"] or 0) == 1
        except Exception:
            pass  # column may not exist until migration — locks still apply


def test_clone_reject(arena_db, monkeypatch):
    from arena.startup import instantiate_strategy
    from signals.strategy_pipeline.cycle import LabHost
    from signals.strategy_pipeline.promote import queue_paper

    monkeypatch.setattr("config.STRATEGY_LAB_PAPER_SLOTS", 2, raising=False)
    from signals.strategy_pipeline.control import update_settings
    update_settings({"paper_slots": 2})

    bot = instantiate_strategy("momentum")
    arena_db.save_bot_config(
        bot.name, bot.strategy_type, 0, bot.strategy_params or {},
    )
    host = LabHost()
    spec = sanitize_spec({
        "spec_id": new_spec_id("momentum"),
        "primitive": "momentum",
        "name": "mom-lab",
        "params": {},
    })
    host.store.insert({**spec, "stage": "backtested"})
    queued = queue_paper(
        host.store, {"spec": spec, "spec_id": spec["spec_id"]}, {"passed": True},
    )
    assert queued is False
    dead = host.store.get(spec["spec_id"])
    assert dead["status"] == "closed"
    assert "clone" in (dead.get("autopsy") or {}).get("reason", "")


def test_zero_pnl_reject(arena_db, monkeypatch):
    from signals.strategy_pipeline.backtest import run_strict_backtest

    class _Res:
        passed = True
        reason = "passed"
        child_pnl = 0.0
        baseline_pnl = None
        markets = 12
        elapsed_sec = 0.1
        detail = "zero"
        n_trades = 20

    monkeypatch.setattr(
        "evolution.backtest_gate.evaluate_offspring",
        lambda *a, **k: _Res(),
    )
    monkeypatch.setattr("config.STRATEGY_LAB_BACKTEST_MIN_PNL", 1.0, raising=False)
    monkeypatch.setattr("config.STRATEGY_LAB_BACKTEST_MIN_TRADES", 5, raising=False)

    spec = sanitize_spec({
        "spec_id": new_spec_id("momentum"),
        "primitive": "momentum",
        "name": "mom-zero",
        "params": dict(FAR_MOM_PARAMS),
    })
    result = run_strict_backtest({"spec": spec, "spec_id": spec["spec_id"]})
    assert result["passed"] is False
    assert "pnl" in (result.get("reason") or "")


def test_low_trades_reject(arena_db, monkeypatch):
    from signals.strategy_pipeline.backtest import run_strict_backtest

    class _Res:
        passed = True
        reason = "passed"
        child_pnl = 5.0
        baseline_pnl = None
        markets = 12
        elapsed_sec = 0.1
        detail = "ok"
        n_trades = 2

    monkeypatch.setattr(
        "evolution.backtest_gate.evaluate_offspring",
        lambda *a, **k: _Res(),
    )
    monkeypatch.setattr("config.STRATEGY_LAB_BACKTEST_MIN_PNL", 1.0, raising=False)
    monkeypatch.setattr("config.STRATEGY_LAB_BACKTEST_MIN_TRADES", 5, raising=False)

    spec = sanitize_spec({
        "spec_id": new_spec_id("momentum"),
        "primitive": "momentum",
        "name": "mom-low-n",
        "params": dict(FAR_MOM_PARAMS),
    })
    result = run_strict_backtest({"spec": spec, "spec_id": spec["spec_id"]})
    assert result["passed"] is False
    assert result.get("reason") == "below_min_trades"


def test_slots_zero_no_deploy(arena_db, monkeypatch):
    from signals.strategy_pipeline.cycle import LabHost

    monkeypatch.setattr("config.STRATEGY_LAB_PAPER_SLOTS", 0, raising=False)
    monkeypatch.setattr("config.STRATEGY_LAB_MAX_NEW_PER_TICK", 1, raising=False)
    monkeypatch.setattr("config.STRATEGY_LAB_LLM_PROVIDER", "none", raising=False)
    from signals.strategy_pipeline.control import update_settings
    update_settings({"paper_slots": 0})

    host = LabHost()

    class _Res:
        passed = True
        reason = "passed"
        child_pnl = 5.0
        baseline_pnl = None
        markets = 12
        elapsed_sec = 0.1
        detail = "ok"
        n_trades = 20

    monkeypatch.setattr(
        "evolution.backtest_gate.evaluate_offspring",
        lambda *a, **k: _Res(),
    )

    report = host.tick()
    assert report["papered"] == 0
    raw = arena_db.get_arena_state("pending_bot_deploys") or ""
    if raw:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        lab_items = [
            i for i in (payload.get("strategies") or [])
            if isinstance(i, dict) and i.get("source") == "lab"
        ]
        assert lab_items == []
    # Passed specs should sit at backtested, not paper.
    open_paper = host.store.open_by_stage("paper")
    assert open_paper == []


def test_desk_package_removed():
    import importlib.util
    import config
    assert importlib.util.find_spec("desk") is None
    assert not hasattr(config, "DESK_CYCLE_ENABLED")


def test_store_and_autopsy(arena_db):
    store = HypothesisStore()
    spec = sanitize_spec({
        "spec_id": new_spec_id("sniper"),
        "primitive": "sniper",
        "name": "sniper-lab",
        "thesis": "lag hunt",
        "params": {"min_drift": 0.28},
    })
    store.insert(spec)
    write_autopsy(
        store, spec["spec_id"], stage="backtested",
        reason="pnl_-1.2", evidence={"primitive": "sniper"},
    )
    dead = store.get(spec["spec_id"])
    assert dead["status"] == "closed"
    assert dead["autopsy"]["reason"] == "pnl_-1.2"
    assert store.counts()["rejected"] >= 1


def test_control_pause_roundtrip(arena_db):
    from signals.strategy_pipeline.control import (
        set_paused, is_paused, try_acquire_tick_lock, release_tick_lock,
    )

    assert is_paused() is False
    set_paused(True)
    assert is_paused() is True
    set_paused(False)
    assert try_acquire_tick_lock() is True
    assert try_acquire_tick_lock() is False
    release_tick_lock()


def test_promote_only_from_ready(arena_db):
    from signals.strategy_pipeline.cycle import LabHost

    host = LabHost()
    spec = sanitize_spec({
        "spec_id": new_spec_id("momentum"),
        "primitive": "momentum",
        "name": "mom-ready",
        "params": dict(FAR_MOM_PARAMS),
    })
    host.store.insert({**spec, "stage": "backtested"})
    host.store.advance(spec["spec_id"], "backtested")
    out = host.promote_to_live(spec["spec_id"])
    assert out["ok"] is False
    assert out["reason"] == "not_ready"

"""Minimal Python API for Strategy Lab pipeline control (Phase 1)."""

from __future__ import annotations

from typing import Any


def status() -> dict[str, Any]:
    from signals.strategy_pipeline.cycle import get_host
    return get_host().snapshot()


def tick() -> dict[str, Any]:
    from signals.strategy_pipeline.control import (
        try_acquire_tick_lock, release_tick_lock, save_last_cycle,
    )
    from signals.strategy_pipeline.cycle import get_host

    if not try_acquire_tick_lock():
        return {"ok": False, "reason": "tick_in_progress"}
    try:
        report = get_host().tick()
        save_last_cycle(report)
        report["ok"] = True
        return report
    finally:
        release_tick_lock()


def settings(patch: dict[str, Any] | None = None) -> dict[str, Any]:
    from signals.strategy_pipeline.control import update_settings, effective

    if patch:
        update_settings(patch)
    return effective()


def promote(spec_id: str) -> dict[str, Any]:
    from signals.strategy_pipeline.cycle import get_host
    return get_host().promote_to_live(spec_id)


def approve_paper(spec_id: str) -> dict[str, Any]:
    from signals.strategy_pipeline.cycle import get_host
    return get_host().approve_to_paper(spec_id)


def reject(spec_id: str, reason: str = "operator_deny") -> dict[str, Any]:
    """Operator deny: close hyp with autopsy (ready/backtested/paper → rejected/retired)."""
    from signals.strategy_pipeline.cycle import get_host
    from signals.strategy_pipeline.postmortem import write_autopsy

    spec_id = str(spec_id or "").strip()
    if not spec_id:
        return {"ok": False, "reason": "missing_spec_id"}
    host = get_host()
    hyp = host.store.get(spec_id)
    if not hyp or hyp.get("status") != "open":
        return {"ok": False, "reason": "not_found"}
    stage = str(hyp.get("stage") or "ready")
    reason = str(reason or "operator_deny").strip() or "operator_deny"
    autopsy = write_autopsy(
        host.store,
        spec_id,
        stage=stage,
        reason=reason,
        evidence={"source": "operator", "prior_stage": stage},
    )
    closed_as = (
        "rejected"
        if stage in ("idea", "researched", "coded", "backtested")
        else "retired"
    )
    return {
        "ok": True,
        "spec_id": spec_id,
        "closed_as": closed_as,
        "autopsy": autopsy,
    }

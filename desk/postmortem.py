"""Stage 6 — write an autopsy so research does not repeat the same death."""

from __future__ import annotations

from typing import Any


def write_autopsy(
    store,
    spec_id: str,
    *,
    stage: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    autopsy = {
        "reason": reason,
        "died_at_stage": stage,
        "evidence": evidence or {},
        "lesson": _lesson(reason, evidence or {}),
    }
    store.advance(
        spec_id,
        "rejected" if stage in ("idea", "researched", "coded", "backtested") else "retired",
        status="closed",
        autopsy=autopsy,
    )
    store.log(spec_id, "reviewer", "postmortem", "autopsy", reason)
    from desk.roles import set_role

    set_role("reviewer", "done", f"{spec_id}: {reason[:80]}")
    return autopsy


def _lesson(reason: str, evidence: dict[str, Any]) -> str:
    reason_l = (reason or "").lower()
    if "edge" in reason_l or "pnl" in reason_l:
        return "Do not promote on follow-WR; require fee-aware net edge."
    if "drift" in reason_l:
        return "Do not fade or fight a signed drift that already pays the crowd."
    if "sample" in reason_l or "trades" in reason_l:
        return "Hold the candidate in paper until n clears the promotion bar."
    if "data" in reason_l:
        return "Backtest data missing is not an edge. Retry, do not skip the gate."
    prim = evidence.get("primitive")
    if prim:
        return f"Archive failure of primitive={prim}; next spec must change params or lane mix."
    return "Record the failure regime and do not resample the same spec."

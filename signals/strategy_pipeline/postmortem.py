"""Write an autopsy so research does not repeat the same death.

Phase 4: delegates structured autopsy + durable constraints to learning_spine.
"""

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
    """Close a hyp with a structured autopsy and ingest into the learning spine."""
    from signals.strategy_pipeline.learning_spine import write_autopsy_for_spec

    return write_autopsy_for_spec(
        store,
        spec_id,
        stage=stage,
        reason=reason,
        evidence=evidence,
        narrate=True,
    )

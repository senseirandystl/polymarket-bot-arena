"""Named desk roles — UI + heartbeat only.

These are *views* of the six-stage cycle, not separate processes that trade.
Execution still goes through venues/ + risk_engine.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ROLES = (
    {
        "id": "desk_lead",
        "title": "Desk Lead",
        "job": "Routes the cycle, never trades",
        "stage": "orchestrate",
    },
    {
        "id": "researcher",
        "title": "Research",
        "job": "Find something worth testing",
        "stage": "research",
    },
    {
        "id": "coder",
        "title": "Coder",
        "job": "Bind a thesis to a primitive + params",
        "stage": "code",
    },
    {
        "id": "backtester",
        "title": "Backtest",
        "job": "Attack the spec with resolved history",
        "stage": "backtest",
    },
    {
        "id": "risk",
        "title": "Risk",
        "job": "Caps, kill switch, promotion bars",
        "stage": "risk",
    },
    {
        "id": "trader",
        "title": "Execution",
        "job": "Paper then live fills only",
        "stage": "live",
    },
    {
        "id": "reviewer",
        "title": "Reviewer",
        "job": "Autopsy every death; feed the graph",
        "stage": "postmortem",
    },
)


@dataclass
class RoleStatus:
    role_id: str
    title: str
    job: str
    state: str = "idle"  # idle | working | blocked | done
    detail: str = ""
    updated_ts: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.role_id,
            "title": self.title,
            "job": self.job,
            "state": self.state,
            "detail": self.detail,
            "updated_ts": self.updated_ts,
        }


@dataclass
class FloorSnapshot:
    roles: list[RoleStatus] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    pipeline_counts: dict[str, int] = field(default_factory=dict)
    last_cycle: dict | None = None
    provider: str = "none"
    factory_mode: bool = False
    universe_phase: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "roles": [r.as_dict() for r in self.roles],
            "hypotheses": self.hypotheses,
            "pipeline_counts": self.pipeline_counts,
            "last_cycle": self.last_cycle,
            "provider": self.provider,
            "factory_mode": self.factory_mode,
            "universe_phase": self.universe_phase,
        }


_floor: dict[str, RoleStatus] = {
    r["id"]: RoleStatus(role_id=r["id"], title=r["title"], job=r["job"])
    for r in ROLES
}


def set_role(role_id: str, state: str, detail: str = "", ts: float | None = None) -> None:
    import time

    role = _floor.get(role_id)
    if role is None:
        return
    role.state = state
    role.detail = detail
    role.updated_ts = float(ts if ts is not None else time.time())


def get_floor() -> dict[str, RoleStatus]:
    return _floor

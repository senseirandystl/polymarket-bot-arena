"""Stage 1 — find something worth testing.

Heuristic path always works. Optional LLM may rewrite/rank the candidate list.
"""

from __future__ import annotations

import logging
from typing import Any

from desk.compiler import PRIMITIVE_TYPES, new_spec_id, sanitize_spec
from desk.universe import tradable_slots

logger = logging.getLogger("desk.research")

SYSTEM = (
    "You design parameterized trading-bot specs for binary crypto Up/Down "
    "prediction markets (Polymarket 5m, Kalshi 15m). Reply with JSON only: "
    "{\"candidates\": [{\"primitive\": \"momentum|mean_reversion|sniper|"
    "phantom|hybrid|sweeper\", \"name\": \"short-name\", \"thesis\": \"...\", "
    "\"lane_weights\": {\"drift\":0.55,\"mom\":0.3,\"strat\":0.15}, "
    "\"params\": {}, \"universe\": [\"polymarket:btc_5m\"]} ]}. "
    "Do not invent primitives. Do not emit source code. Prefer theses that "
    "avoid failures listed in the autopsies."
)


def propose(store, *, max_new: int = 3) -> list[dict[str, Any]]:
    """Return new sanitized specs inserted into the store."""
    from desk.roles import set_role

    set_role("researcher", "working", "reading autopsies and tape")
    context = _context(store)
    raw_candidates = _heuristic_candidates(context)
    llm_overlay = _llm_candidates(context)
    if llm_overlay:
        raw_candidates = llm_overlay + raw_candidates

    existing = {h.get("name") for h in store.list(limit=200)}
    out: list[dict[str, Any]] = []
    for raw in raw_candidates:
        if len(out) >= max_new:
            break
        try:
            spec = sanitize_spec(raw)
        except ValueError:
            continue
        if spec["name"] in existing:
            spec["name"] = f"{spec['name']}-{spec['spec_id'][-4:]}"
        spec["stage"] = "researched"
        spec["origin"] = raw.get("origin") or spec.get("origin") or "heuristic"
        store.insert(spec)
        existing.add(spec["name"])
        out.append(spec)

    set_role(
        "researcher",
        "done" if out else "idle",
        f"proposed {len(out)} spec(s)",
    )
    return out


def _context(store) -> dict[str, Any]:
    autopsies = store.recent_autopsies(limit=12)
    slots = [s.as_dict() for s in tradable_slots()]
    dead_primitives = [
        a.get("primitive") for a in autopsies if a.get("stage") in ("rejected", "retired")
    ]
    return {
        "universe": slots,
        "autopsies": autopsies,
        "avoid_primitives": dead_primitives[-5:],
        "notes": (
            "Drift vs strike dominates 5m binaries. Contrarian fades against "
            "strong drift historically lose. Promote on net edge, not accuracy. "
            "Dead-zone mids 0.42-0.58 without drift are expensive."
        ),
    }


def _heuristic_candidates(context: dict[str, Any]) -> list[dict[str, Any]]:
    avoid = set(context.get("avoid_primitives") or [])
    universe = [u["slot_id"] for u in context.get("universe") or []] or [
        "polymarket:btc_5m"
    ]
    catalog = [
        {
            "primitive": "momentum",
            "name": "mom-desk",
            "thesis": "Ride short BTC impulse only when drift agrees; sit out late window.",
            "lane_weights": {"drift": 0.55, "mom": 0.30, "strat": 0.15},
        },
        {
            "primitive": "mean_reversion",
            "name": "fade-desk",
            "thesis": "Drift picks the side; enter a TWAP retrace toward this window's strike.",
            "lane_weights": {"drift": 0.75, "mom": 0.0, "strat": 0.25},
        },
        {
            "primitive": "sniper",
            "name": "sniper-desk",
            "thesis": "Hunt lag vs strike; dual-gate inside 50-58c, sit out unconfirmed flow.",
            "lane_weights": {"drift": 0.70, "mom": 0.10, "strat": 0.20},
        },
        {
            "primitive": "hybrid",
            "name": "hybrid-desk",
            "thesis": "Regime switch: trend→mom/phantom, chop→fade. Ensemble, not a new model.",
            "lane_weights": {"drift": 0.55, "mom": 0.20, "strat": 0.25},
        },
        {
            "primitive": "sweeper",
            "name": "sweep-desk",
            "thesis": "Buy locked outcomes still under $1 on the fee-curve extreme.",
            "lane_weights": {"drift": 0.40, "mom": 0.10, "strat": 0.50},
        },
    ]
    out = []
    for c in catalog:
        if c["primitive"] in avoid:
            continue
        c = dict(c)
        c["spec_id"] = new_spec_id(c["primitive"])
        c["universe"] = universe
        c["origin"] = "heuristic"
        c["parent_spec_ids"] = [
            a.get("spec_id") for a in (context.get("autopsies") or [])[:2]
            if a.get("spec_id")
        ]
        out.append(c)
    return out


def _llm_candidates(context: dict[str, Any]) -> list[dict[str, Any]]:
    from desk import llm

    if llm.configured_provider() in ("", "none", "off", "heuristic"):
        return []
    import json

    user = json.dumps(context, default=str)[:6000]
    obj = llm.complete_json(SYSTEM, user)
    if not obj:
        return []
    rows = obj.get("candidates") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return []
    out = []
    allowed = set(PRIMITIVE_TYPES)
    for row in rows:
        if not isinstance(row, dict):
            continue
        prim = str(row.get("primitive") or "").lower()
        if prim not in allowed:
            continue
        row = dict(row)
        row["origin"] = "llm"
        row["spec_id"] = new_spec_id(prim)
        out.append(row)
    return out

"""Compile a StrategySpec onto an existing bot primitive.

No generated source is executed. The spec may only name a known primitive
and a params dict that is intersected with that primitive's DEFAULT_PARAMS.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

logger = logging.getLogger("desk.compiler")

# Primitive → strategy_type used by startup.instantiate_strategy / GA.
PRIMITIVE_TYPES = {
    "momentum": "momentum",
    "mean_reversion": "mean_reversion",
    "meanrev": "mean_reversion",
    "sniper": "sniper",
    "phantom": "phantom",
    "hybrid": "hybrid",
    "lag_residual": "lag_residual",
    "regime_specialist": "regime_specialist",
    "no_lag": "no_lag",
    "sweeper": "sweeper",
    "arbitrage": "arbitrage",
}

LANE_KEYS = ("drift", "mom", "strat")


def normalize_primitive(name: str) -> str:
    key = (name or "").strip().lower()
    if key not in PRIMITIVE_TYPES:
        raise ValueError(f"unknown primitive: {name}")
    return PRIMITIVE_TYPES[key]


def new_spec_id(primitive: str) -> str:
    return f"{normalize_primitive(primitive)[:8]}-{int(time.time()) % 10_000_000:07d}"


def sanitize_spec(raw: dict[str, Any]) -> dict[str, Any]:
    primitive = normalize_primitive(str(raw.get("primitive") or "momentum"))
    spec_id = str(raw.get("spec_id") or new_spec_id(primitive))
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    lanes = raw.get("lane_weights") if isinstance(raw.get("lane_weights"), dict) else {}
    universe = raw.get("universe") if isinstance(raw.get("universe"), list) else []
    constraints = raw.get("constraints") if isinstance(raw.get("constraints"), dict) else {}
    parents = raw.get("parent_spec_ids") if isinstance(raw.get("parent_spec_ids"), list) else []

    clean_lanes = {}
    for k in LANE_KEYS:
        if k in lanes:
            try:
                clean_lanes[k] = float(lanes[k])
            except (TypeError, ValueError):
                continue
    if clean_lanes:
        total = sum(abs(v) for v in clean_lanes.values()) or 1.0
        clean_lanes = {k: abs(v) / total for k, v in clean_lanes.items()}

    name = str(raw.get("name") or f"{primitive}-desk")
    name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)
    while "--" in name:
        name = name.replace("--", "-")
    name = name.strip("-_")[:48] or f"{primitive}-desk"
    return {
        "spec_id": spec_id,
        "name": name,
        "primitive": primitive,
        "params": params,
        "lane_weights": clean_lanes,
        "universe": [str(u) for u in universe][:8],
        "constraints": constraints,
        "thesis": str(raw.get("thesis") or "")[:800],
        "parent_spec_ids": [str(p) for p in parents][:8],
        "origin": str(raw.get("origin") or "desk"),
        "stage": str(raw.get("stage") or "coded"),
    }


def compile_bot(spec: dict[str, Any], *, generation: int = 0):
    """Return a bot instance bound to this spec."""
    spec = sanitize_spec(spec)
    from evolution.ga import _default_params_for
    from arena.startup import instantiate_strategy

    base = _default_params_for(spec["primitive"])
    merged = copy.deepcopy(base)
    for k, v in (spec.get("params") or {}).items():
        if k in merged:
            merged[k] = v

    bot = instantiate_strategy(
        spec["primitive"],
        name=spec["name"],
    )
    bot.generation = generation
    bot.lineage = f"desk:{spec['spec_id']}"
    if hasattr(bot, "strategy_params"):
        bot.strategy_params.update(merged)
    # Lane profile nudge lives on BaseBot if present.
    lanes = spec.get("lane_weights") or {}
    profile = getattr(bot, "signal_profile", None)
    if isinstance(profile, dict) and lanes:
        for k, v in lanes.items():
            if k in profile:
                profile[k] = v
    return bot, spec

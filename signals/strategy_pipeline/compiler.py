"""Compile StrategySpecs onto allowlisted primitives (no arbitrage in Lab v1)."""

from __future__ import annotations

import copy
import secrets
import time
from typing import Any

# Lab allowlist minus arbitrage for pipeline v1.
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
}

EXCLUDED_PRIMITIVES = frozenset({"arbitrage"})
# When STRATEGY_LAB_PAPER_SLOTS==0, invent still runs but only proposes
# live-compatible genomes (default roster + lag_residual) — not culled
# momentum/meanrev/hybrid/phantom types that cannot paper-deploy.
LIVE_COMPAT_PRIMITIVES = frozenset({
    "sniper", "sweeper", "lag_residual",
})
LANE_KEYS = ("drift", "mom", "strat")


def normalize_primitive(name: str) -> str:
    key = (name or "").strip().lower()
    if key in EXCLUDED_PRIMITIVES:
        raise ValueError(f"primitive excluded from lab pipeline: {name}")
    if key not in PRIMITIVE_TYPES:
        raise ValueError(f"unknown primitive: {name}")
    return PRIMITIVE_TYPES[key]


def new_spec_id(primitive: str) -> str:
    return (
        f"{normalize_primitive(primitive)[:8]}-"
        f"{int(time.time()) % 10_000_000:07d}-{secrets.token_hex(3)}"
    )


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

    name = str(raw.get("name") or f"{primitive}-lab")
    name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name)
    while "--" in name:
        name = name.replace("--", "-")
    name = name.strip("-_")[:48] or f"{primitive}-lab"
    if name.endswith("-desk"):
        name = name[:-5] + "-lab" if len(name) > 5 else f"{primitive}-lab"

    origin = str(raw.get("origin") or "lab")
    if origin == "desk":
        origin = "lab"

    # lane_weights stay on the spec as a thesis record. They are NOT applied
    # to the bot: core tuner owns the live blend (see compile_bot).
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
        "origin": origin,
        "stage": str(raw.get("stage") or "coded"),
    }


def compile_bot(spec: dict[str, Any], *, generation: int = 0):
    """Return a bot instance bound to this lab spec (allowlist enforced)."""
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
    bot.lineage = f"lab:{spec['spec_id']}"
    if hasattr(bot, "strategy_params"):
        bot.strategy_params.update(merged)
    return bot, spec


__all__ = [
    "PRIMITIVE_TYPES",
    "EXCLUDED_PRIMITIVES",
    "LANE_KEYS",
    "normalize_primitive",
    "new_spec_id",
    "sanitize_spec",
    "compile_bot",
]

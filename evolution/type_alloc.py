"""Strategy-type allocation for GA spawns (tier-1 evolution).

Instead of always replacing a culled bot with the *same* strategy_type, sample
a type from a softmax over live + gene-bank mean fitness, blended with
stickiness toward the culled slot's type. That lets the roster rebalance
toward types that are actually printing without a full redesign of the slate.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

import config

# Types the GA may assign to a directional slot.
ALLOCATABLE_TYPES = (
    "momentum",
    "mean_reversion",
    "mean_reversion_tp",
    "phantom",
    "hybrid",
    "sniper",
    "sentiment",
)


def _dead_lane_types() -> frozenset[str]:
    """Types whose edge depends on kill-switched lanes — do not spawn them.

    Sentiment's thesis is pm/cvd-heavy; until those kill-switches re-open,
    spawning sentiment (2026-08: phantom→sentiment-g13) only adds noise.
    """
    blocked: set[str] = set(getattr(config, "GA_SPAWN_EXCLUDE_TYPES", ()) or ())
    try:
        pm = float(getattr(config, "SIGNAL_WEIGHT_PM", 0) or 0)
        cvd = float(getattr(config, "SIGNAL_WEIGHT_CVD", 0) or 0)
        if pm <= 0 and cvd <= 0:
            blocked.add("sentiment")
    except Exception:
        blocked.add("sentiment")
    return frozenset(blocked)


def allocatable_types() -> tuple[str, ...]:
    dead = _dead_lane_types()
    return tuple(t for t in ALLOCATABLE_TYPES if t not in dead)


def _type_scores(
    individuals: list[dict],
    bank_parents: list[dict],
) -> dict[str, float]:
    """Mean fitness per strategy_type (live + bank)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for ind in list(individuals) + list(bank_parents or []):
        st = ind.get("strategy_type")
        if st not in ALLOCATABLE_TYPES:
            continue
        buckets[st].append(float(ind.get("fitness") or 0.0))
    # Prior mild mass so empty types stay reachable
    prior = 0.15
    live_types = allocatable_types()
    scores = {st: prior for st in live_types}
    for st, vals in buckets.items():
        if st in scores and vals:
            scores[st] = prior + sum(vals) / len(vals)
    return scores


def pick_strategy_type(
    dead_type: str,
    individuals: list[dict],
    bank_parents: list[dict] | None = None,
    *,
    rng: random.Random | None = None,
    exclude_types: set[str] | frozenset[str] | None = None,
) -> str:
    """Sample a strategy_type for the replacement slot.

    When ``GA_TYPE_ALLOC_ENABLED`` is False, always returns ``dead_type``
    (unless that type is excluded, in which case the first non-excluded
    allocatable type wins).

    ``exclude_types`` removes types already saturated in this spawn batch
    (see ``GA_MAX_PER_TYPE_PER_CYCLE``) so a monoculture elite cannot fill
    every open slot with identical hybrids.
    """
    rng = rng or random.Random()
    exclude = set(exclude_types or ()) | set(_dead_lane_types())
    allowed = allocatable_types()

    # Hard same-type path (or type-alloc off): keep the slot's identity.
    same_only = bool(getattr(config, "GA_TYPE_SAME_TYPE_ONLY", False))
    if same_only or not getattr(config, "GA_TYPE_ALLOC_ENABLED", True):
        if dead_type in allowed and dead_type not in exclude:
            return dead_type
        for t in allowed:
            if t not in exclude:
                return t
        return dead_type if dead_type in ALLOCATABLE_TYPES else "momentum"

    stick = float(getattr(config, "GA_TYPE_STICKINESS", 0.80))
    stick = max(0.0, min(0.95, stick))
    temp = float(getattr(config, "GA_TYPE_ALLOC_TEMPERATURE", 0.35))
    temp = max(0.05, temp)

    scores = _type_scores(individuals, bank_parents or [])
    types = [t for t in allowed if t not in exclude]
    if not types:
        # Every allocatable type is saturated — fall back to full live set.
        types = list(allowed) or list(ALLOCATABLE_TYPES)

    logits = [scores.get(t, 0.15) / temp for t in types]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps) or 1.0
    probs = [e / z for e in exps]

    # Blend stickiness onto dead_type (only if still available)
    if dead_type in types:
        di = types.index(dead_type)
        sticky = [0.0] * len(types)
        sticky[di] = 1.0
        probs = [
            (1.0 - stick) * p + stick * s
            for p, s in zip(probs, sticky)
        ]
        s = sum(probs) or 1.0
        probs = [p / s for p in probs]

    # Sample
    r = rng.random()
    cum = 0.0
    for t, p in zip(types, probs):
        cum += p
        if r <= cum:
            return t
    return types[-1]

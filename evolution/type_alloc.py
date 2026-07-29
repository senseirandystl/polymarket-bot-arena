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
    scores = {st: prior for st in ALLOCATABLE_TYPES}
    for st, vals in buckets.items():
        if vals:
            scores[st] = prior + sum(vals) / len(vals)
    return scores


def pick_strategy_type(
    dead_type: str,
    individuals: list[dict],
    bank_parents: list[dict] | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Sample a strategy_type for the replacement slot.

    When ``GA_TYPE_ALLOC_ENABLED`` is False, always returns ``dead_type``.
    """
    rng = rng or random.Random()
    if not getattr(config, "GA_TYPE_ALLOC_ENABLED", True):
        return dead_type if dead_type in ALLOCATABLE_TYPES else "momentum"

    stick = float(getattr(config, "GA_TYPE_STICKINESS", 0.40))
    stick = max(0.0, min(0.95, stick))
    temp = float(getattr(config, "GA_TYPE_ALLOC_TEMPERATURE", 0.35))
    temp = max(0.05, temp)

    scores = _type_scores(individuals, bank_parents or [])
    # Softmax over scores / temperature
    types = list(ALLOCATABLE_TYPES)
    logits = [scores.get(t, 0.15) / temp for t in types]
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps) or 1.0
    probs = [e / z for e in exps]

    # Blend stickiness onto dead_type
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

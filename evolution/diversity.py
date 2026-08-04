"""Spawn diversity: reject near-clone genomes within the same strategy type.

When the live roster already has a momentum bot at params P, a new momentum
spawn whose numeric genes are within ``GA_DIVERSITY_MIN_DISTANCE`` of P is
rejected so the batch explores different regions of the param space.
"""

from __future__ import annotations

from typing import Any

import config
from evolution.bounds import bounds_for, is_numeric_gene
from evolution.frozen import evolvable_keys


def param_distance(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    strategy_type: str,
) -> float:
    """Normalized L1 distance over evolvable numeric genes (0 = identical).

    Each gene is scaled by its allowable span so lookback (ints 3–40) and
    min_confidence (0–1) contribute comparably. Missing keys count as max
    distance on that gene.
    """
    keys = evolvable_keys(strategy_type, {**a, **b})
    if not keys:
        return 0.0
    total = 0.0
    n = 0
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if not (is_numeric_gene(va) and is_numeric_gene(vb)):
            continue
        lo, hi = bounds_for(k, float(va))
        span = max(hi - lo, 1e-12)
        total += abs(float(va) - float(vb)) / span
        n += 1
    if n == 0:
        return 0.0
    return total / n


def is_diverse_enough(
    child_params: dict[str, Any],
    *,
    strategy_type: str,
    peers: list[dict[str, Any]],
    min_distance: float | None = None,
) -> bool:
    """True if child is at least ``min_distance`` from every peer genome."""
    if not peers:
        return True
    min_d = (
        min_distance
        if min_distance is not None
        else float(getattr(config, "GA_DIVERSITY_MIN_DISTANCE", 0.08))
    )
    if min_d <= 0:
        return True
    for peer in peers:
        if not peer:
            continue
        if param_distance(child_params, peer, strategy_type=strategy_type) < min_d:
            return False
    return True

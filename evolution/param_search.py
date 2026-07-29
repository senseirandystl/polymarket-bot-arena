"""Tier-2 continuous param search: elite-cloud adaptive mutation.

Full CMA-ES is heavy for a ~5-bot live roster; this is a lightweight TPE /
elite-neighborhood style operator:

1. Collect parent + gene-bank params of the *same strategy_type*.
2. For each evolvable gene, estimate mean/std from that elite cloud.
3. With probability ``GA_ELITE_SAMPLE_RATE``, sample ``N(mean, std)`` (explore
   near proven genomes); else fall back to Gaussian mutation around the child
   value (standard operator).

Frozen / non-evolvable genes are never touched.
"""

from __future__ import annotations

import copy
import random
import statistics
from typing import Any

import config
from evolution.bounds import clamp, is_numeric_gene
from evolution.frozen import evolvable_keys
from evolution.operators import mutate as base_mutate


def _cloud_stats(
    genomes: list[dict[str, Any]],
    keys: set[str],
) -> dict[str, tuple[float, float]]:
    """Per-key (mean, std) over genomes that contain the key."""
    buckets: dict[str, list[float]] = {k: [] for k in keys}
    for g in genomes:
        for k in keys:
            v = g.get(k)
            if is_numeric_gene(v):
                buckets[k].append(float(v))
    out: dict[str, tuple[float, float]] = {}
    for k, vals in buckets.items():
        if len(vals) >= 2:
            mu = statistics.fmean(vals)
            sd = statistics.pstdev(vals)
            out[k] = (mu, max(sd, 1e-12))
        elif len(vals) == 1:
            out[k] = (vals[0], abs(vals[0]) * 0.1 + 1e-6)
    return out


def adaptive_mutate(
    params: dict[str, Any],
    *,
    strategy_type: str,
    elite_genomes: list[dict[str, Any]] | None = None,
    rate: float | None = None,
    sigma: float | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Mutate only evolvable genes; bias samples toward elite cloud when possible."""
    rng = rng or random
    rate = rate if rate is not None else float(
        getattr(config, "GA_MUTATION_RATE", 0.20)
    )
    sigma = sigma if sigma is not None else float(
        getattr(config, "GA_MUTATION_SIGMA", 0.12)
    )
    elite_rate = float(getattr(config, "GA_ELITE_SAMPLE_RATE", 0.55))
    use_adaptive = bool(getattr(config, "GA_ADAPTIVE_MUTATION", True))

    child = copy.deepcopy(params)
    keys = evolvable_keys(strategy_type, child)
    if not keys:
        return child

    cloud = _cloud_stats(elite_genomes or [], keys) if use_adaptive else {}

    for key in keys:
        if rng.random() > rate:
            continue
        val = child[key]
        if not is_numeric_gene(val):
            continue
        if use_adaptive and key in cloud and rng.random() < elite_rate:
            mu, sd = cloud[key]
            # Inflate cloud std slightly so we don't collapse to a point
            noise = rng.gauss(mu, sd * (1.0 + sigma))
            child[key] = clamp(key, noise, reference=val)
        else:
            # Standard relative Gaussian around current value
            from evolution.bounds import bounds_for
            lo, hi = bounds_for(key, float(val))
            span = max(hi - lo, 1e-12)
            noise = float(val) + rng.gauss(0.0, sigma * span)
            child[key] = clamp(key, noise, reference=val)
    return child


def mutate_evolvable_only(
    params: dict[str, Any],
    *,
    strategy_type: str,
    rate: float | None = None,
    sigma: float | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Base Gaussian mutate restricted to evolvable keys (no elite cloud)."""
    rng = rng or random
    allow = evolvable_keys(strategy_type, params)
    # Mutate full dict then restore frozen keys
    frozen_vals = {k: copy.deepcopy(v) for k, v in params.items() if k not in allow}
    out = base_mutate(params, rate=rate, sigma=sigma, rng=rng)
    for k, v in frozen_vals.items():
        out[k] = v
    # Also drop accidental mutation of non-evolvable that base_mutate touched
    for k in list(out.keys()):
        if k not in allow and k in params:
            out[k] = copy.deepcopy(params[k])
    return out

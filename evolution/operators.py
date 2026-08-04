"""GA operators: tournament selection, intelligent crossover, Gaussian mutation."""

from __future__ import annotations

import copy
import random
from typing import Any

import config
from evolution.bounds import clamp, is_numeric_gene


def tournament_select(
    population: list[dict],
    *,
    k: int | None = None,
    fitness_key: str = "fitness",
    rng: random.Random | None = None,
) -> dict:
    """Return the fittest of ``k`` randomly sampled individuals.

    Each individual is a dict that must include ``fitness_key``.
    """
    rng = rng or random
    k = k or int(getattr(config, "GA_TOURNAMENT_K", 3))
    k = max(1, min(k, len(population)))
    contenders = rng.sample(population, k)
    return max(contenders, key=lambda ind: float(ind.get(fitness_key, 0.0)))


def _parent_identity(ind: dict) -> str:
    """Stable id for distinct-parent checks (name preferred; params fallback)."""
    name = ind.get("name")
    if name:
        # Gene-bank clones of a live elite share the same name intentionally —
        # treat them as the same parent so we do not fake "diversity".
        return f"name:{name}"
    st = ind.get("strategy_type") or ""
    params = ind.get("params") or {}
    try:
        keys = sorted(params.keys())
        body = "|".join(f"{k}={params[k]!r}" for k in keys)
        return f"params:{st}|{body}"
    except Exception:
        return f"id:{id(ind)}"


def tournament_select_pair(
    population: list[dict],
    *,
    k: int | None = None,
    fitness_key: str = "fitness",
    rng: random.Random | None = None,
) -> tuple[dict, dict, bool]:
    """Select two parents, preferring distinct identities.

    Returns ``(p1, p2, is_self_pair)``. When the pool has ≥2 unique identities,
    ``p2`` is drawn from everyone *except* ``p1``'s identity (second tournament
    on that reduced pool). When only one unique parent exists, both slots are
    the same individual and ``is_self_pair`` is True (caller should label the
    spawn as clone+mutate, not crossover).
    """
    if not population:
        raise ValueError("tournament_select_pair requires a non-empty population")
    rng = rng or random
    p1 = tournament_select(population, k=k, fitness_key=fitness_key, rng=rng)
    id1 = _parent_identity(p1)
    others = [p for p in population if _parent_identity(p) != id1]
    if not others:
        return p1, p1, True
    p2 = tournament_select(others, k=k, fitness_key=fitness_key, rng=rng)
    # Convention at call sites: p1 is the fitter / primary parent
    if float(p2.get(fitness_key, 0.0)) > float(p1.get(fitness_key, 0.0)):
        p1, p2 = p2, p1
    return p1, p2, False


def crossover(
    parent_a: dict[str, Any],
    parent_b: dict[str, Any],
    *,
    alpha: float | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Blend two param dicts into a child.

    * Numeric keys present in both parents: ``alpha * a + (1-alpha) * b``
      with ``alpha ~ U(blend_lo, blend_hi)`` (intelligent blending biased
      toward the midpoint rather than a hard cut).
    * Keys only in one parent: inherited from that parent.
    * Non-numeric (bool/str/list/dict): taken from parent_a (the primary /
      fitter parent by convention at the call site).
    """
    rng = rng or random
    blend_lo = float(getattr(config, "GA_CROSSOVER_ALPHA_LO", 0.30))
    blend_hi = float(getattr(config, "GA_CROSSOVER_ALPHA_HI", 0.70))
    if alpha is None:
        alpha = rng.uniform(blend_lo, blend_hi)
    alpha = max(0.0, min(1.0, float(alpha)))

    child: dict[str, Any] = {}
    keys = set(parent_a) | set(parent_b)
    for key in keys:
        in_a = key in parent_a
        in_b = key in parent_b
        if in_a and in_b:
            va, vb = parent_a[key], parent_b[key]
            if is_numeric_gene(va) and is_numeric_gene(vb):
                blended = alpha * float(va) + (1.0 - alpha) * float(vb)
                # Clamp into the UNION of both parents' bounds so a parent at
                # 0.0 does not collapse the allowable band to ±ε around zero.
                from evolution.bounds import bounds_for
                lo_a, hi_a = bounds_for(key, float(va))
                lo_b, hi_b = bounds_for(key, float(vb))
                lo, hi = min(lo_a, lo_b), max(hi_a, hi_b)
                blended = max(lo, min(hi, blended))
                # Preserve int type when both parents are ints
                if isinstance(va, int) and isinstance(vb, int) and not (
                    isinstance(va, bool) or isinstance(vb, bool)
                ):
                    child[key] = int(round(blended))
                else:
                    child[key] = round(blended, 6)
            else:
                # Prefer primary parent for non-numeric
                child[key] = copy.deepcopy(va)
        elif in_a:
            child[key] = copy.deepcopy(parent_a[key])
        else:
            child[key] = copy.deepcopy(parent_b[key])
    return child


def mutate(
    params: dict[str, Any],
    *,
    rate: float | None = None,
    sigma: float | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Gaussian mutation inside sensible bounds.

    Each numeric gene is mutated independently with probability ``rate``.
    Noise is ``N(0, sigma * (hi-lo))`` so the step size scales with the
    parameter's allowed range. Values are clamped via :mod:`evolution.bounds`.
    """
    rng = rng or random
    rate = rate if rate is not None else float(
        getattr(config, "GA_MUTATION_RATE", getattr(config, "MUTATION_RATE_DIRECTED", 0.07))
    )
    sigma = sigma if sigma is not None else float(
        getattr(config, "GA_MUTATION_SIGMA", 0.15)
    )
    new_params = copy.deepcopy(params)
    for key, val in list(new_params.items()):
        if not is_numeric_gene(val):
            continue
        if rng.random() > rate:
            continue
        from evolution.bounds import bounds_for
        lo, hi = bounds_for(key, float(val))
        span = max(hi - lo, 1e-12)
        noise = rng.gauss(0.0, sigma * span)
        mutated = float(val) + noise
        new_params[key] = clamp(key, mutated, reference=val)
    return new_params


def elite_indices(fitnesses: list[float], n_elite: int) -> list[int]:
    """Indices of the top ``n_elite`` individuals (stable on ties: lower index wins)."""
    n_elite = max(0, min(n_elite, len(fitnesses)))
    if n_elite == 0:
        return []
    ranked = sorted(range(len(fitnesses)), key=lambda i: (-fitnesses[i], i))
    return ranked[:n_elite]

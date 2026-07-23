"""Offline GA against historical (or synthetic historical) decision data.

Used by tests to prove the GA improves fitness over generations without
touching the live arena DB. A simple threshold strategy is evaluated on a
fixed market history; the fitness landscape has a known optimum so the
search can be verified.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

from evolution.fitness import composite_from_raw, multi_objective_fitness
from evolution.operators import crossover, elite_indices, mutate, tournament_select


# Genome keys for the offline threshold strategy
OFFLINE_GENOME_KEYS = (
    "momentum_threshold",
    "min_confidence",
    "position_size_pct",
    "lookback_candles",
)


def make_historical_markets(
    n: int = 400,
    *,
    seed: int = 42,
    signal_noise: float = 0.15,
) -> list[dict]:
    """Build a fixed historical market series with a learnable signal.

    Each row has:
      * ``momentum`` — noisy reading of the latent direction
      * ``price`` — entry price for the favored side
      * ``outcome`` — ``"up"`` / ``"down"`` (true resolution)
      * ``edge_signal`` — clean latent in [-1, 1] (not visible to the bot)

    The latent is predictive: sign(edge_signal) matches the outcome ~70% of
    the time. A bot that thresholds momentum near the true scale captures it.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        # Latent edge: positive → UP more likely
        edge = rng.uniform(-1.0, 1.0)
        # Outcome biased by edge (sigmoid)
        p_up = 1.0 / (1.0 + math.exp(-2.2 * edge))
        outcome = "up" if rng.random() < p_up else "down"
        # Observed momentum = edge + noise (this is what the bot sees)
        momentum = edge + rng.gauss(0.0, signal_noise)
        # Entry prices: favorites cost more
        if outcome == "up":
            price = rng.uniform(0.45, 0.65)
        else:
            price = rng.uniform(0.35, 0.55)
        rows.append({
            "momentum": momentum,
            "price": price,
            "outcome": outcome,
            "edge_signal": edge,
        })
    return rows


def evaluate_genome_on_history(
    params: dict[str, Any],
    history: list[dict],
    *,
    stake: float = 5.0,
    fee_per_trade: float = 0.35,
) -> dict[str, float]:
    """Simulate a simple threshold strategy on historical rows → fitness components.

    Rule: if |momentum| >= threshold and conf proxy >= min_confidence, buy the
    side of the momentum sign at ``price``. Win/loss is binary payoff on the
    cost basis, minus a flat fee. Overtrading (threshold near 0) bleeds fees on
    noise; undertrading (threshold near 1) leaves edge on the table. The
    learnable optimum sits near the signal/noise scale (~0.20–0.40).
    """
    thr = float(params.get("momentum_threshold", 0.25))
    min_conf = float(params.get("min_confidence", 0.4))
    size_pct = float(params.get("position_size_pct", 0.05))
    # lookback is a mild regularizer: extreme values slightly tax size
    lookback = int(params.get("lookback_candles", 5))
    lookback_tax = 1.0 - 0.02 * abs(lookback - 8)  # optimum near 8
    lookback_tax = max(0.7, lookback_tax)

    # Size soft-cap: oversized bets amplify variance (hurts sharpe/drawdown)
    size_mult = min(size_pct / 0.05, 1.5)
    amount = stake * size_mult * lookback_tax
    pnls: list[float] = []
    for row in history:
        mom = float(row["momentum"])
        # Confidence proxy: |momentum| scaled
        conf = min(1.0, abs(mom))
        if abs(mom) < thr or conf < min_conf:
            continue
        side_up = mom > 0
        price = float(row["price"])
        won = (side_up and row["outcome"] == "up") or (
            (not side_up) and row["outcome"] == "down"
        )
        if won:
            # Binary: share pays $1, cost was price → profit per $ is (1-p)/p
            pnl = amount * ((1.0 - price) / max(price, 0.05)) - fee_per_trade
        else:
            pnl = -amount - fee_per_trade
        pnls.append(pnl)

    return multi_objective_fitness(pnls=pnls)


def random_genome(rng: random.Random) -> dict[str, Any]:
    return {
        "momentum_threshold": rng.uniform(0.05, 0.80),
        "min_confidence": rng.uniform(0.10, 0.80),
        "position_size_pct": rng.uniform(0.02, 0.12),
        "lookback_candles": rng.randint(3, 20),
    }


def run_offline_ga(
    history: list[dict] | None = None,
    *,
    pop_size: int = 20,
    generations: int = 15,
    elite_count: int = 2,
    seed: int = 7,
    fitness_fn: Callable[[dict], dict] | None = None,
) -> dict[str, Any]:
    """Run a full offline GA; return history of best/mean fitness per gen.

    ``fitness_fn(params) -> components dict`` defaults to evaluating on
    ``history`` (built if None).
    """
    rng = random.Random(seed)
    if history is None:
        history = make_historical_markets(n=400, seed=seed + 100)
    if fitness_fn is None:
        def fitness_fn(params: dict) -> dict:
            return evaluate_genome_on_history(params, history)

    # Initial population
    pop: list[dict] = []
    for i in range(pop_size):
        params = random_genome(rng)
        comps = fitness_fn(params)
        pop.append({
            "id": f"g0-{i}",
            "params": params,
            "components": comps,
            "fitness": composite_from_raw(comps),
            "generation": 0,
            "lineage": "seed",
            "operator": "seed",
            "parents": [],
        })

    best_curve: list[float] = []
    mean_curve: list[float] = []
    elite_curve: list[float] = []

    for gen in range(1, generations + 1):
        fitnesses = [ind["fitness"] for ind in pop]
        best_curve.append(max(fitnesses))
        mean_curve.append(sum(fitnesses) / len(fitnesses))
        elites_idx = elite_indices(fitnesses, elite_count)
        elite_curve.append(
            sum(fitnesses[i] for i in elites_idx) / max(1, len(elites_idx))
        )

        # Next generation: keep elites, fill rest via select → crossover → mutate
        next_pop: list[dict] = []
        for ei, idx in enumerate(elites_idx):
            elite = dict(pop[idx])
            elite["operator"] = "elite"
            elite["generation"] = gen
            elite["id"] = f"g{gen}-elite{ei}"
            # Elites keep params/fitness; refresh lineage tag
            elite["lineage"] = f"elite:{pop[idx]['id']}"
            elite["parents"] = [pop[idx]["id"]]
            next_pop.append(elite)

        while len(next_pop) < pop_size:
            p1 = tournament_select(pop, rng=rng)
            p2 = tournament_select(pop, rng=rng)
            child_params = crossover(p1["params"], p2["params"], rng=rng)
            child_params = mutate(child_params, rng=rng)
            comps = fitness_fn(child_params)
            child_id = f"g{gen}-{len(next_pop)}"
            next_pop.append({
                "id": child_id,
                "params": child_params,
                "components": comps,
                "fitness": composite_from_raw(comps),
                "generation": gen,
                "lineage": f"{p1['id']}+{p2['id']} -> {child_id}",
                "operator": "crossover+mutate",
                "parents": [p1["id"], p2["id"]],
            })
        pop = next_pop

    # Final metrics after last generation
    fitnesses = [ind["fitness"] for ind in pop]
    best_curve.append(max(fitnesses))
    mean_curve.append(sum(fitnesses) / len(fitnesses))

    best = max(pop, key=lambda ind: ind["fitness"])
    return {
        "best_fitness_curve": best_curve,
        "mean_fitness_curve": mean_curve,
        "elite_fitness_curve": elite_curve,
        "final_population": pop,
        "best_individual": best,
        "generations": generations,
        "pop_size": pop_size,
    }

"""Genetic Algorithm evolution for directional bot configurations.

Replaces the old single-parent mutate-from-winner loop with a proper GA:

* multi-objective fitness (P&L, Sharpe, drawdown, consistency)
* tournament selection, intelligent crossover, Gaussian mutation in bounds
* elitism so top performers are never lost
* lineage tracking (parents + operator)
* arbitrage + pure makers remain evolution-exempt

Public entry point used by the arena coordinator::

    from evolution.ga import run_ga_cycle
"""

from evolution.ga import run_ga_cycle, should_trigger_evolution
from evolution.fitness import multi_objective_fitness, rank_normalize_fitness

__all__ = [
    "run_ga_cycle",
    "should_trigger_evolution",
    "multi_objective_fitness",
    "rank_normalize_fitness",
]

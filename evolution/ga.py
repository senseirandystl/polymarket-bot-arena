"""Live-arena Genetic Algorithm cycle.

Orchestrates fitness evaluation, elitism, selection, crossover, and mutation
over the current directional bot slate. Arbitrage and pure maker strategies
are evolution-exempt and pass through untouched.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from typing import Any, Callable

import config
import db
from evolution.fitness import multi_objective_fitness, rank_normalize_fitness
from evolution.operators import crossover, elite_indices, mutate, tournament_select

logger = logging.getLogger("arena")

# Strategy types never culled or mutated by the GA.
EVOLUTION_EXEMPT_TYPES = frozenset({
    "arbitrage",
    "late_window_maker",
    "fee_zone_maker",
    "btc_maker",
    "copy_trade",
})


def _default_params_for(strategy_type: str) -> dict:
    """Strategy-type default param dicts (lazy imports to avoid cycles)."""
    from bots.bot_momentum import DEFAULT_PARAMS as MOMENTUM_DEFAULTS
    from bots.bot_mean_rev import DEFAULT_PARAMS as MEANREV_DEFAULTS
    from bots.bot_hybrid import DEFAULT_PARAMS as HYBRID_DEFAULTS
    from bots.bot_sentiment import DEFAULT_PARAMS as SENTIMENT_DEFAULTS
    from bots.bot_sniper import DEFAULT_PARAMS as SNIPER_DEFAULTS
    from bots.bot_phantom import DEFAULT_PARAMS as PHANTOM_DEFAULTS

    mapping = {
        "momentum": MOMENTUM_DEFAULTS,
        "mean_reversion": MEANREV_DEFAULTS,
        "mean_reversion_sl": MEANREV_DEFAULTS,
        "mean_reversion_tp": MEANREV_DEFAULTS,
        "sniper": SNIPER_DEFAULTS,
        "phantom": PHANTOM_DEFAULTS,
        "sentiment": SENTIMENT_DEFAULTS,
        "hybrid": HYBRID_DEFAULTS,
    }
    base = mapping.get(strategy_type, MOMENTUM_DEFAULTS)
    return copy.deepcopy(base)


def _resolved_trades(bot_name: str, hours: float) -> list[dict]:
    """Pull resolved trades for fitness (no hard LIMIT — full window)."""
    with db.get_conn() as conn:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        rows = conn.execute(
            """SELECT pnl, outcome, entry_price, created_at, side, trade_features
               FROM trades
               WHERE bot_name=? AND created_at>=?
                 AND outcome IN ('win', 'loss', 'exit_tp', 'exit_sl')
               ORDER BY created_at ASC""",
            (bot_name, cutoff),
        ).fetchall()
        return [dict(r) for r in rows]


def evaluate_population(
    bots: list,
    *,
    hours: float | None = None,
) -> list[dict]:
    """Score every bot; return individual records with multi-objective fitness."""
    hours = hours if hours is not None else float(config.EVOLUTION_WINDOW_HOURS)
    raw_components: list[dict] = []
    meta: list[dict] = []

    for bot in bots:
        trades = _resolved_trades(bot.name, hours)
        comps = multi_objective_fitness(trades)
        # Also pull summary fields the dashboard already shows
        perf = bot.get_performance(hours=hours)
        raw_components.append(comps)
        meta.append({
            "name": bot.name,
            "strategy_type": bot.strategy_type,
            "generation": bot.generation,
            "lineage": getattr(bot, "lineage", None),
            "params": copy.deepcopy(getattr(bot, "strategy_params", {}) or {}),
            "bot": bot,
            "pnl": perf.get("total_pnl", comps["pnl"]),
            "win_rate": perf.get("win_rate", 0.0),
            "trades": perf.get("total_trades", int(comps["n_trades"])),
            "be_gap": perf.get("breakeven_gap"),
            "components": comps,
        })

    scored = rank_normalize_fitness(raw_components)
    individuals = []
    for m, s in zip(meta, scored):
        individuals.append({
            **m,
            "fitness": s["fitness"],
            "ranks": s["ranks"],
            "weights": s["weights"],
            "components": s["components"],
        })
    return individuals


def _survives_legacy_bar(ind: dict) -> bool:
    """Keep the pre-GA survival safety: positive P&L or BE-gap floor.

    Used only to decide who is *eligible for replacement* (not for ranking).
    A bot that fails this bar is a replacement candidate; elites among those
    who pass (or are immune) are protected.
    """
    if ind["trades"] < config.MIN_TRADES_FOR_JUDGMENT:
        return True  # immune
    if ind["pnl"] > 0:
        return True
    gap = ind.get("be_gap")
    return gap is not None and gap >= config.EVOLUTION_BE_GAP_MIN


def should_trigger_evolution(
    bots: list,
    *,
    hours: float | None = None,
    last_trigger_pnl: float | None = None,
) -> tuple[bool, str]:
    """Performance-triggered early evolution.

    Returns (should_run, reason). Time-based cadence is handled by the
    coordinator; this only flags *early* runs when the pool is bleeding.
    """
    if not getattr(config, "GA_PERF_TRIGGER_ENABLED", True):
        return False, "disabled"
    hours = hours if hours is not None else float(config.EVOLUTION_WINDOW_HOURS)
    # Sum P&L of non-exempt bots
    total = 0.0
    n = 0
    for bot in bots:
        if bot.strategy_type in EVOLUTION_EXEMPT_TYPES:
            continue
        perf = bot.get_performance(hours=hours)
        total += float(perf.get("total_pnl") or 0.0)
        n += int(perf.get("total_trades") or 0)
    min_trades = int(getattr(config, "GA_PERF_TRIGGER_MIN_TRADES", 40))
    if n < min_trades:
        return False, f"insufficient_trades({n}<{min_trades})"
    threshold = float(getattr(config, "GA_PERF_TRIGGER_PNL", -25.0))
    if total <= threshold:
        return True, f"pool_pnl={total:.2f}<=threshold={threshold:.2f}"
    # Optional drawdown vs last snapshot
    dd_thr = getattr(config, "GA_PERF_TRIGGER_DROP", None)
    if dd_thr is not None and last_trigger_pnl is not None:
        drop = last_trigger_pnl - total
        if drop >= float(dd_thr):
            return True, f"pool_drop={drop:.2f}>={dd_thr}"
    return False, f"pool_pnl={total:.2f}_ok"


def run_ga_cycle(
    bots: list,
    cycle_number: int,
    *,
    bot_factory: Callable[..., Any] | None = None,
    validate_fn: Callable[[Any], bool] | None = None,
    class_map: dict | None = None,
    rng: random.Random | None = None,
) -> tuple[list, dict]:
    """Run one GA generation over the live bot slate.

    Parameters
    ----------
    bots :
        Current bot instances (may include exempt types).
    cycle_number :
        Generation / cycle index (persisted by the coordinator).
    bot_factory :
        ``(strategy_type, name, params, generation, lineage) -> bot``.
        Defaults to looking up ``class_map``.
    validate_fn :
        Smoke test for new bots; failures fall back to pure defaults.
    class_map :
        strategy_type → Bot class (required if bot_factory is None).
    rng :
        Optional RNG for deterministic tests.

    Returns
    -------
    (new_bots, report)
        ``new_bots`` is the post-evolution slate (exempt bots re-appended).
        ``report`` is a structured summary for DB / dashboard.
    """
    rng = rng or random.Random()
    logger.info("=== GA Evolution Cycle %s ===", cycle_number)

    exempt = [b for b in bots if b.strategy_type in EVOLUTION_EXEMPT_TYPES]
    evolving = [b for b in bots if b.strategy_type not in EVOLUTION_EXEMPT_TYPES]

    if not evolving:
        logger.info("  No evolvable bots — skipping GA")
        report = {
            "cycle": cycle_number,
            "skipped": True,
            "reason": "no_evolvable_bots",
            "individuals": [],
            "elites": [],
            "replaced": [],
            "spawned": [],
        }
        return list(bots), report

    individuals = evaluate_population(evolving)
    # Sort by fitness descending for logging / elitism
    individuals.sort(key=lambda ind: ind["fitness"], reverse=True)

    n_elite = int(getattr(config, "GA_ELITE_COUNT", 1))
    n_elite = max(1, min(n_elite, len(individuals)))  # at least 1 elite safety net

    # Classify: immune (too few trades), replaceable (fails survival bar), rest
    for ind in individuals:
        if ind["trades"] < config.MIN_TRADES_FOR_JUDGMENT:
            ind["status"] = "immune"
        elif _survives_legacy_bar(ind):
            ind["status"] = "survivor"
        else:
            ind["status"] = "replaceable"

    # Safety net: if EVERYONE is replaceable, promote the fittest to survivor
    if all(ind["status"] == "replaceable" for ind in individuals):
        individuals[0]["status"] = "survivor"
        logger.info(
            "  Safety net: keeping %s (fitness=%.3f) as sole survivor",
            individuals[0]["name"], individuals[0]["fitness"],
        )

    # Elites = top-N by fitness among non-replaceable? No — elitism is
    # absolute: the top-N by fitness are NEVER lost, even if the legacy bar
    # would have culled them. That is the point of elitism.
    fitnesses = [ind["fitness"] for ind in individuals]
    elite_idx_set = set(elite_indices(fitnesses, n_elite))
    for i, ind in enumerate(individuals):
        if i in elite_idx_set:
            ind["elite"] = True
            if ind["status"] == "replaceable":
                ind["status"] = "elite_protected"
        else:
            ind["elite"] = False

    # Who gets replaced: non-elite + replaceable (failed bar) with enough trades
    to_replace = [
        ind for ind in individuals
        if (not ind["elite"]) and ind["status"] == "replaceable"
    ]
    keep = [ind for ind in individuals if ind not in to_replace]

    logger.info(
        "  Rankings (%sh window, multi-objective fitness):",
        config.EVOLUTION_WINDOW_HOURS,
    )
    for ind in individuals:
        tag = "ELITE" if ind["elite"] else ind["status"].upper()
        c = ind["components"]
        logger.info(
            "  %s: fit=%.3f pnl=$%.2f sharpe=%.2f dd=%.0f%% cons=%.0f%% "
            "WR=%.1f%% n=%d [%s]",
            ind["name"], ind["fitness"], c["pnl"], c["sharpe"],
            c["max_drawdown_pct"] * 100, c["consistency"] * 100,
            ind["win_rate"] * 100, ind["trades"], tag,
        )

    report: dict[str, Any] = {
        "cycle": cycle_number,
        "skipped": False,
        "reason": None,
        "individuals": [
            {
                "name": ind["name"],
                "strategy_type": ind["strategy_type"],
                "generation": ind["generation"],
                "fitness": ind["fitness"],
                "components": ind["components"],
                "ranks": ind["ranks"],
                "pnl": ind["pnl"],
                "win_rate": ind["win_rate"],
                "trades": ind["trades"],
                "be_gap": ind["be_gap"],
                "status": ind["status"],
                "elite": ind["elite"],
                "lineage": ind.get("lineage"),
            }
            for ind in individuals
        ],
        "elites": [ind["name"] for ind in individuals if ind["elite"]],
        "replaced": [],
        "spawned": [],
        "operators": [],
    }

    if not to_replace:
        logger.info("  No bots below threshold — skipping replacement (elites hold)")
        report["skipped"] = True
        report["reason"] = "no_replacements"
        for bot in evolving:
            bot.reset_daily()
        # Persist generation snapshot even when no replacement
        _persist_ga_state(cycle_number, report)
        return evolving + exempt, report

    # Breeding pool: elites + survivors (immune count as keepers but poor parents
    # if they have ~0 trades — prefer those with trades for tournament)
    parent_pool = [
        ind for ind in keep
        if ind["trades"] >= max(5, config.MIN_TRADES_FOR_JUDGMENT // 3)
        or ind["elite"]
    ]
    if not parent_pool:
        parent_pool = keep[:]

    # Build factory
    if bot_factory is None:
        if not class_map:
            raise ValueError("class_map or bot_factory required")
        def bot_factory(strategy_type, name, params, generation, lineage):
            cls = class_map.get(strategy_type)
            if cls is None:
                from bots.bot_momentum import MomentumBot
                cls = MomentumBot
            return cls(
                name=name, params=params,
                generation=generation, lineage=lineage,
            )

    keep_names = {ind["name"] for ind in keep}
    new_bots = [ind["bot"] for ind in individuals if ind["name"] in keep_names]
    for b in new_bots:
        b.reset_daily()

    for dead in to_replace:
        # Two parents via tournament
        p1 = tournament_select(parent_pool, rng=rng)
        p2 = tournament_select(parent_pool, rng=rng)
        # Prefer fitter as primary parent for non-numeric inheritance
        if p2["fitness"] > p1["fitness"]:
            p1, p2 = p2, p1

        # Start from strategy defaults so type-specific keys are always present,
        # then overlay crossover of parent params on overlapping keys.
        base = _default_params_for(dead["strategy_type"])
        # Inherit shared keys from parents via crossover, then fill gaps
        # from the strategy defaults.
        blended = crossover(p1["params"], p2["params"], rng=rng)
        child_params = dict(base)
        for k, v in blended.items():
            if k in child_params or k in base:
                child_params[k] = v
            elif k in p1["params"] or k in p2["params"]:
                # Only keep keys that belong on this strategy (intersection with
                # defaults) — avoids polluting meanrev with sniper price zones.
                if k in base:
                    child_params[k] = v
        # Also copy overlapping keys that defaults have
        for k in list(child_params.keys()):
            if k in blended:
                child_params[k] = blended[k]

        child_params = mutate(child_params, rng=rng)

        child_name = f"{dead['strategy_type']}-g{cycle_number}-{rng.randint(100, 999)}"
        lineage = (
            f"{p1['name']}+{p2['name']} -> {child_name} "
            f"(crossover+mutate; fit={p1['fitness']:.3f}/{p2['fitness']:.3f})"
        )
        evolved = bot_factory(
            dead["strategy_type"], child_name, child_params, cycle_number, lineage,
        )

        if validate_fn is not None and not validate_fn(evolved):
            logger.warning(
                "  %s failed validation, recreating with pure defaults", child_name,
            )
            fallback_params = _default_params_for(dead["strategy_type"])
            child_name = f"{p1['name']}-g{cycle_number}-fallback"
            lineage = f"{p1['name']} -> {child_name} (fallback)"
            evolved = bot_factory(
                dead["strategy_type"], child_name, fallback_params,
                cycle_number, lineage,
            )
            operator = "fallback"
        else:
            operator = "crossover+mutate"

        db.retire_bot(dead["name"])
        db.save_bot_config(
            evolved.name, evolved.strategy_type, evolved.generation,
            evolved.strategy_params, evolved.lineage,
        )
        new_bots.append(evolved)

        spawn_rec = {
            "name": evolved.name,
            "strategy_type": evolved.strategy_type,
            "generation": evolved.generation,
            "lineage": evolved.lineage,
            "parents": [p1["name"], p2["name"]],
            "operator": operator,
            "replaced": dead["name"],
            "params": copy.deepcopy(evolved.strategy_params),
        }
        report["spawned"].append(spawn_rec)
        report["replaced"].append(dead["name"])
        report["operators"].append(spawn_rec)
        logger.info(
            "  Spawned %s from %s × %s (replaced %s)",
            evolved.name, p1["name"], p2["name"], dead["name"],
        )

    # Rankings payload for legacy evolution_events + dashboard
    rankings = []
    for ind in individuals:
        rankings.append({
            "name": ind["name"],
            "strategy_type": ind["strategy_type"],
            "generation": ind["generation"],
            "pnl": ind["pnl"],
            "win_rate": ind["win_rate"],
            "trades": ind["trades"],
            "be_gap": ind.get("be_gap"),
            "fitness": ind["fitness"],
            "components": ind["components"],
            "ranks": ind["ranks"],
            "status": ind["status"],
            "elite": ind["elite"],
            "lineage": ind.get("lineage"),
        })

    survivors = [ind["name"] for ind in keep]
    new_names = [s["name"] for s in report["spawned"]]
    db.log_evolution(
        cycle_number,
        survivors,
        report["replaced"],
        new_names,
        rankings,
    )
    # Also log GA-specific detail (lineage + operators + fitness curve point)
    db.log_ga_generation(cycle_number, report)
    _persist_ga_state(cycle_number, report)

    return new_bots + exempt, report


def _persist_ga_state(cycle_number: int, report: dict) -> None:
    """Write a compact GA snapshot into arena_state for the dashboard."""
    try:
        individuals = report.get("individuals") or []
        best_fit = max((i.get("fitness", 0.0) for i in individuals), default=0.0)
        mean_fit = (
            sum(i.get("fitness", 0.0) for i in individuals) / len(individuals)
            if individuals else 0.0
        )
        snapshot = {
            "cycle": cycle_number,
            "best_fitness": best_fit,
            "mean_fitness": mean_fit,
            "elites": report.get("elites") or [],
            "replaced": report.get("replaced") or [],
            "spawned": [s.get("name") for s in report.get("spawned") or []],
            "skipped": report.get("skipped", False),
            "reason": report.get("reason"),
            "n_individuals": len(individuals),
        }
        db.set_arena_state("ga_last_cycle", json.dumps(snapshot))
        # Append to fitness history (capped)
        hist_raw = db.get_arena_state("ga_fitness_history")
        hist = json.loads(hist_raw) if hist_raw else []
        if not isinstance(hist, list):
            hist = []
        hist.append({
            "cycle": cycle_number,
            "best_fitness": best_fit,
            "mean_fitness": mean_fit,
            "n_replaced": len(report.get("replaced") or []),
        })
        hist = hist[-50:]  # keep last 50 cycles
        db.set_arena_state("ga_fitness_history", json.dumps(hist))
    except Exception as e:
        logger.warning("Failed to persist GA state: %s", e)

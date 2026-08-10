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
from evolution.operators import crossover, elite_indices, tournament_select_pair
from evolution import gene_bank as gene_bank_mod
from evolution.type_alloc import pick_strategy_type
from evolution.param_search import adaptive_mutate
from evolution.backtest_gate import evaluate_offspring
from evolution.diversity import is_diverse_enough

logger = logging.getLogger("arena")

# Strategy types never culled or mutated by the GA.
EVOLUTION_EXEMPT_TYPES = frozenset({
    "arbitrage",
    "late_window_maker",
    "fee_zone_maker",
    "btc_maker",
    "true_maker",
    "copy_trade",
    "sweeper",  # structural fee-curve / settlement edge — not directional GA
})


def _default_params_for(strategy_type: str) -> dict:
    """Strategy-type default param dicts (lazy imports to avoid cycles)."""
    from bots.bot_momentum import DEFAULT_PARAMS as MOMENTUM_DEFAULTS
    from bots.bot_mean_rev import DEFAULT_PARAMS as MEANREV_DEFAULTS
    from bots.bot_hybrid import DEFAULT_PARAMS as HYBRID_DEFAULTS
    from bots.bot_sniper import DEFAULT_PARAMS as SNIPER_DEFAULTS
    from bots.bot_phantom import DEFAULT_PARAMS as PHANTOM_DEFAULTS
    from bots.bot_lag_residual import DEFAULT_PARAMS as LAG_DEFAULTS
    from bots.bot_regime_specialist import DEFAULT_PARAMS as REGIME_DEFAULTS
    from bots.bot_no_lag import DEFAULT_PARAMS as NO_LAG_DEFAULTS
    from bots.bot_sweeper import DEFAULT_PARAMS as SWEEPER_DEFAULTS

    mapping = {
        "momentum": MOMENTUM_DEFAULTS,
        "mean_reversion": MEANREV_DEFAULTS,
        "mean_reversion_sl": MEANREV_DEFAULTS,
        "mean_reversion_tp": MEANREV_DEFAULTS,
        "sniper": SNIPER_DEFAULTS,
        "phantom": PHANTOM_DEFAULTS,
        "hybrid": HYBRID_DEFAULTS,
        "lag_residual": LAG_DEFAULTS,
        "regime_specialist": REGIME_DEFAULTS,
        "no_lag": NO_LAG_DEFAULTS,
        "sweeper": SWEEPER_DEFAULTS,
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
    """Who is *eligible for replacement* (not for ranking).

    2026-08 redesign: small negative P&L is noise, not a cull signal. Only
    replace bots that are clearly underwater on both P&L and BE gap after a
    full sample. Gen-0 founders get an extra loss floor so defaults are not
    swapped for cold mutants on a −$3 dip.
    """
    n = int(ind.get("trades") or 0)
    if n < int(config.MIN_TRADES_FOR_JUDGMENT):
        return True  # immune — not enough data
    pnl = float(ind.get("pnl") or 0.0)
    gap = ind.get("be_gap")
    gap_f = float(gap) if gap is not None else None

    # Clear survival
    if pnl > 0:
        return True
    if gap_f is not None and gap_f >= float(config.EVOLUTION_BE_GAP_MIN):
        return True

    # Soft floor: mild red ink is not replaceable
    cull_pnl = float(getattr(config, "EVOLUTION_PNL_CULL_MAX", -12.0))
    if pnl > cull_pnl:
        return True  # e.g. −$3 > −$12 → keep

    # Founder / gen-0 protection: only cull when deeply bad
    gen = int(ind.get("generation") or 0)
    if gen == 0 and bool(getattr(config, "GA_PROTECT_FOUNDERS", True)):
        founder_pnl = float(getattr(config, "GA_FOUNDER_CULL_PNL", -20.0))
        founder_gap = float(getattr(config, "GA_FOUNDER_CULL_BE_GAP", -0.02))
        if pnl > founder_pnl:
            return True
        if gap_f is not None and gap_f >= founder_gap:
            return True

    return False


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
    # Audit 3c: zombie check — a bot that is immune AND paused by the risk
    # engine is dead weight (paused, unkillable). Force-replace it.
    zombie_replaced = []
    for ind in individuals:
        if ind["trades"] < config.MIN_TRADES_FOR_JUDGMENT:
            bot = ind.get("bot")
            if bot is not None and getattr(bot, "_paused", False):
                ind["status"] = "replaceable"
                ind["zombie"] = True
                zombie_replaced.append(ind["name"])
                logger.warning(
                    "  Zombie bot %s: immune (n=%d) + paused — auto-retiring",
                    ind["name"], ind["trades"],
                )
            else:
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

    # Always deposit elites into the shadow gene bank (even if no culls).
    try:
        bank = gene_bank_mod.record_elites(individuals, cycle_number)
        report["gene_bank_size"] = len(bank)
    except Exception as e:
        logger.warning("gene bank update failed: %s", e)
        bank = gene_bank_mod.load_bank()
        report["gene_bank_size"] = len(bank)

    if not to_replace:
        logger.info("  No bots below threshold — skipping replacement (elites hold)")
        report["skipped"] = True
        report["reason"] = "no_replacements"
        for bot in evolving:
            bot.reset_daily()
        # Persist generation snapshot even when no replacement — write both the
        # arena_state GA status AND a ga_generations row so the dashboard
        # evolution log shows every cycle (not only culling events).
        try:
            rankings = [{
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
            } for ind in individuals]
            survivors = [ind["name"] for ind in keep]
            db.log_evolution(cycle_number, survivors, [], [], rankings)
        except Exception as e:
            logger.warning("Failed to log skipped evolution_events: %s", e)
        _persist_ga_state(cycle_number, report)
        try:
            db.log_ga_generation(cycle_number, report)
        except Exception as e:
            logger.warning("Failed to log skipped ga_generation: %s", e)
        return evolving + exempt, report

    # Breeding pool: live keepers + gene-bank elites (shadow parents)
    parent_pool = [
        ind for ind in keep
        if ind["trades"] >= max(5, config.MIN_TRADES_FOR_JUDGMENT // 3)
        or ind["elite"]
    ]
    if not parent_pool:
        parent_pool = keep[:]
    bank_parents = gene_bank_mod.as_parent_records(bank)
    if bank_parents:
        parent_pool = parent_pool + bank_parents
        logger.info("  Gene bank parents available: %d", len(bank_parents))

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

    max_attempts = max(1, int(getattr(config, "GA_SPAWN_ATTEMPTS", 3)))
    max_per_type = max(1, int(getattr(config, "GA_MAX_PER_TYPE_PER_CYCLE", 1)))
    # Count types already kept on the live roster so we do not spawn a second
    # hybrid when hybrid-v1 already survived, etc.
    type_counts: dict[str, int] = {}
    for ind in keep:
        st = ind.get("strategy_type")
        if st:
            type_counts[st] = type_counts.get(st, 0) + 1

    for dead in to_replace:
        evolved = None
        spawn_meta: dict[str, Any] = {}
        operator = "fallback"
        for attempt in range(max_attempts):
            # --- Tier 1: strategy-type allocation (respect per-type cap) ---
            saturated = {
                t for t, n in type_counts.items() if n >= max_per_type
            }
            child_type = pick_strategy_type(
                dead["strategy_type"], individuals, bank_parents,
                rng=rng, exclude_types=saturated,
            )
            if type_counts.get(child_type, 0) >= max_per_type:
                # Softmax still returned a saturated type (fallback path) —
                # force any unsaturated allocatable type, else proceed.
                unsaturated = [
                    t for t in (
                        "momentum", "mean_reversion", "mean_reversion_tp",
                        "phantom", "hybrid", "sniper",
                        "lag_residual", "regime_specialist", "no_lag",
                    )
                    if type_counts.get(t, 0) < max_per_type
                ]
                if unsaturated:
                    child_type = rng.choice(unsaturated)

            # Same-type parents only. Never overlay a phantom genome onto a
            # hybrid/sniper/meanrev child — that only shares min_confidence /
            # position_size_pct and silently degrades type-specific defaults.
            typed_pool = [
                p for p in parent_pool
                if p.get("strategy_type") == child_type
            ]
            cross_type = not typed_pool
            if typed_pool:
                p1, p2, is_self_pair = tournament_select_pair(
                    typed_pool, rng=rng,
                )
            else:
                # No same-type parents: seed from type defaults + mutate.
                # Parent labels still record the best available genomes for
                # lineage transparency (often the monoculture elite).
                p1, p2, is_self_pair = tournament_select_pair(
                    parent_pool, rng=rng,
                )
                is_self_pair = True  # param path is clone-of-defaults, not xover

            # --- Tier 2: param construction ---
            base = _default_params_for(child_type)
            if cross_type:
                # Defaults of the child type only — do NOT blend foreign keys.
                child_params = dict(base)
                breed_mode = "defaults+adaptive"
            elif is_self_pair:
                # Single unique same-type parent → clone its params, then mutate.
                child_params = dict(base)
                for k, v in (p1.get("params") or {}).items():
                    if k in base:
                        child_params[k] = copy.deepcopy(v)
                breed_mode = "clone+mutate"
            else:
                blended = crossover(
                    p1.get("params") or {}, p2.get("params") or {}, rng=rng,
                )
                child_params = dict(base)
                for k, v in blended.items():
                    if k in base:
                        child_params[k] = v
                breed_mode = "crossover+adaptive"

            elite_genomes = [
                p["params"] for p in typed_pool
                if p.get("params")
            ]
            child_params = adaptive_mutate(
                child_params,
                strategy_type=child_type,
                elite_genomes=elite_genomes,
                rng=rng,
            )

            # Spawn diversity: reject near-clones of live same-type keepers
            # or already-accepted spawns this cycle.
            peer_params = [
                ind.get("params") or {}
                for ind in keep
                if ind.get("strategy_type") == child_type
            ]
            peer_params.extend(
                s.get("params") or {}
                for s in report["spawned"]
                if s.get("strategy_type") == child_type
            )
            if not is_diverse_enough(
                child_params, strategy_type=child_type, peers=peer_params,
            ):
                logger.info(
                    "  diversity reject type=%s attempt=%d (too close to peer)",
                    child_type, attempt + 1,
                )
                # Nudge harder and retry on next attempt
                child_params = adaptive_mutate(
                    child_params,
                    strategy_type=child_type,
                    elite_genomes=elite_genomes,
                    rate=1.0,
                    sigma=float(getattr(config, "GA_MUTATION_SIGMA", 0.12)) * 1.5,
                    rng=rng,
                )
                if not is_diverse_enough(
                    child_params, strategy_type=child_type, peers=peer_params,
                ):
                    continue

            child_name = f"{child_type}-g{cycle_number}-{rng.randint(100, 999)}"
            if breed_mode == "defaults+adaptive":
                lineage = (
                    f"defaults({child_type})+mutate -> {child_name} "
                    f"(no same-type parents; seed elite={p1['name']}; "
                    f"fit={p1['fitness']:.3f}; attempt={attempt + 1})"
                )
            elif breed_mode == "clone+mutate":
                lineage = (
                    f"{p1['name']} -> {child_name} "
                    f"(type={child_type}; clone+mutate; "
                    f"fit={p1['fitness']:.3f}; attempt={attempt + 1})"
                )
            else:
                lineage = (
                    f"{p1['name']}+{p2['name']} -> {child_name} "
                    f"(type={child_type}; crossover+adaptive; "
                    f"fit={p1['fitness']:.3f}/{p2['fitness']:.3f}; "
                    f"attempt={attempt + 1})"
                )
            candidate = bot_factory(
                child_type, child_name, child_params, cycle_number, lineage,
            )

            if validate_fn is not None and not validate_fn(candidate):
                logger.warning(
                    "  %s failed smoke validation (attempt %d)", child_name, attempt + 1,
                )
                continue

            # Backtest gate vs the culled bot (same type when possible)
            baseline = dead.get("bot")
            if baseline is not None and getattr(baseline, "strategy_type", None) != child_type:
                # Different type — build a defaults bot of child_type as baseline
                try:
                    baseline = bot_factory(
                        child_type,
                        f"baseline-{child_type}",
                        _default_params_for(child_type),
                        0, "baseline",
                    )
                except Exception:
                    baseline = None
            gate = evaluate_offspring(candidate, baseline_bot=baseline)
            parent_names = (
                [p1["name"]] if breed_mode in ("clone+mutate", "defaults+adaptive")
                else [p1["name"], p2["name"]]
            )
            if breed_mode == "defaults+adaptive":
                parent_names = [f"defaults:{child_type}", p1["name"]]
            spawn_meta = {
                "parents": parent_names,
                "child_type": child_type,
                "breed_mode": breed_mode,
                "gate_reason": gate.reason,
                "gate_child_pnl": gate.child_pnl,
                "gate_baseline_pnl": gate.baseline_pnl,
                "gate_markets": gate.markets,
            }
            if not gate.passed:
                logger.info(
                    "  %s failed backtest gate (%s) attempt %d — retrying",
                    child_name, gate.reason, attempt + 1,
                )
                continue

            evolved = candidate
            operator = f"{breed_mode}+backtest"
            if gate.reason in ("disabled", "data_unavailable", "no_markets",
                               "run_failed_soft"):
                operator = f"{breed_mode}({gate.reason})"
            break

        if evolved is None:
            # Last resort: pure defaults of the culled type (slot must fill)
            child_type = dead["strategy_type"]
            child_name = f"{child_type}-g{cycle_number}-fallback"
            lineage = f"{dead['name']} -> {child_name} (fallback defaults)"
            evolved = bot_factory(
                child_type, child_name,
                _default_params_for(child_type),
                cycle_number, lineage,
            )
            operator = "fallback"
            spawn_meta = {
                "parents": [dead["name"]],
                "child_type": child_type,
                "breed_mode": "fallback",
                "gate_reason": "all_attempts_failed",
            }

        db.retire_bot(dead["name"])
        db.save_bot_config(
            evolved.name, evolved.strategy_type, evolved.generation,
            evolved.strategy_params, evolved.lineage,
        )
        new_bots.append(evolved)
        type_counts[evolved.strategy_type] = (
            type_counts.get(evolved.strategy_type, 0) + 1
        )

        spawn_rec = {
            "name": evolved.name,
            "strategy_type": evolved.strategy_type,
            "generation": evolved.generation,
            "lineage": evolved.lineage,
            "parents": list(spawn_meta.get("parents") or [dead["name"]]),
            "operator": operator,
            "breed_mode": spawn_meta.get("breed_mode"),
            "replaced": dead["name"],
            "params": copy.deepcopy(evolved.strategy_params),
            "gate": {
                "reason": spawn_meta.get("gate_reason"),
                "child_pnl": spawn_meta.get("gate_child_pnl"),
                "baseline_pnl": spawn_meta.get("gate_baseline_pnl"),
                "markets": spawn_meta.get("gate_markets"),
            },
        }
        report["spawned"].append(spawn_rec)
        report["replaced"].append(dead["name"])
        report["operators"].append(spawn_rec)
        logger.info(
            "  Spawned %s (%s) replacing %s op=%s gate=%s",
            evolved.name, evolved.strategy_type, dead["name"],
            operator, spawn_meta.get("gate_reason"),
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
    # Persist first so report gains mean_raw_fitness, then DB row uses it.
    _persist_ga_state(cycle_number, report)
    db.log_ga_generation(cycle_number, report)

    return new_bots + exempt, report


def _persist_ga_state(cycle_number: int, report: dict) -> None:
    """Write a compact GA snapshot into arena_state for the dashboard.

    Rank-normalized fitness averages ~0.5 by construction in any symmetric
    population (mean of ranks on [0,1]). Dashboard history therefore also
    records a **raw** composite mean/best from components so the curve can
    actually move when the pool improves.
    """
    try:
        from evolution.fitness import composite_from_raw

        individuals = report.get("individuals") or []
        best_fit = max((i.get("fitness", 0.0) for i in individuals), default=0.0)
        mean_fit = (
            sum(i.get("fitness", 0.0) for i in individuals) / len(individuals)
            if individuals else 0.0
        )
        raw_composites = []
        for ind in individuals:
            comps = ind.get("components") or {}
            try:
                raw_composites.append(float(composite_from_raw(comps)))
            except Exception:
                pass
        best_raw = max(raw_composites) if raw_composites else 0.0
        mean_raw = (
            sum(raw_composites) / len(raw_composites) if raw_composites else 0.0
        )
        # Prefer raw composites for displayed mean/best (rank mean ≈ 0.5 always).
        snapshot = {
            "cycle": cycle_number,
            "best_fitness": best_raw if raw_composites else best_fit,
            "mean_fitness": mean_raw if raw_composites else mean_fit,
            "best_rank_fitness": best_fit,
            "mean_rank_fitness": mean_fit,
            "best_raw_fitness": best_raw,
            "mean_raw_fitness": mean_raw,
            "elites": report.get("elites") or [],
            "replaced": report.get("replaced") or [],
            "spawned": [s.get("name") for s in report.get("spawned") or []],
            "skipped": report.get("skipped", False),
            "reason": report.get("reason"),
            "n_individuals": len(individuals),
        }
        # So log_ga_generation can store raw means in the DB columns.
        report["best_raw_fitness"] = best_raw
        report["mean_raw_fitness"] = mean_raw
        db.set_arena_state("ga_last_cycle", json.dumps(snapshot))
        # Append to fitness history (capped)
        hist_raw = db.get_arena_state("ga_fitness_history")
        hist = json.loads(hist_raw) if hist_raw else []
        if not isinstance(hist, list):
            hist = []
        hist.append({
            "cycle": cycle_number,
            "best_fitness": best_raw if raw_composites else best_fit,
            "mean_fitness": mean_raw if raw_composites else mean_fit,
            "best_raw_fitness": best_raw,
            "mean_raw_fitness": mean_raw,
            "n_replaced": len(report.get("replaced") or []),
        })
        hist = hist[-50:]  # keep last 50 cycles
        db.set_arena_state("ga_fitness_history", json.dumps(hist))
    except Exception as e:
        logger.warning("Failed to persist GA state: %s", e)

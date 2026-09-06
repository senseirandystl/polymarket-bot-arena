"""Polymarket Bot Arena — thin coordinator over four background threads.

This module used to be a single monolithic ``main_loop`` that did everything
on a 15s cadence: market discovery, signal harvesting, bot evaluation,
trade execution, resolution, evolution, copy-trade polling, maker
quoting, SL/TP monitoring.  After the refactor it has been split:

    ┌───────────────────────┬──────────────────────────────────────────────┐
    │ MarketDiscovery       │ 20s tick — scans the Gamma BTC-5m series and  │
    │ (arena/discovery.py)  │ selects the live window. Owns ``current_market│
    │                       │ `` under a snapshot lock. No speculative next.│
    ├───────────────────────┼──────────────────────────────────────────────┤
    │ MarketDataWarmer      │ 1s tick — single owner of ALL per-market      │
    │ (arena/market_data.py)│ network reads (YES+NO books, prices, OBI,     │
    │                       │ CVD, PM momentum) into a shared warm cache.   │
    ├───────────────────────┼──────────────────────────────────────────────┤
    │ Trader                │ 1s tick — runs bot ``make_decision`` +       │
    │ (arena/trader.py)     │ ``execute`` against ``current_market``,      │
    │                       │ reading warm data. Zero network IO per tick. │
    ├───────────────────────┼──────────────────────────────────────────────┤
    │ TradeResolver         │ 60s tick — reads Polymarket closed events for │
    │ (arena/resolver.py)   │ resolved markets, writes outcomes + P&L.     │
    ├───────────────────────┼──────────────────────────────────────────────┤
    │ PositionMonitorThread │ 0.5s tick — SL/TP exit engine against open  │
    │ (arena/position...py) │ positions on bots with ``exit_strategy``.    │
    └───────────────────────┴──────────────────────────────────────────────┘

This file (root ``arena.py``) is now strictly the coordinator.  It builds
the bots, boots the worker threads, runs the periodic evolution cycle on
its main thread, registers one ``on_cycle_complete`` hook that drives the
maker section, and wires Ctrl-C cleanly to all workers.  Every actual
piece of trading logic lives in the ``arena/`` package next door.
"""

import argparse
import atexit
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
import db
import learning
from bots.bot_momentum import MomentumBot
from bots.bot_mean_rev import MeanRevBot
from bots.bot_hybrid import HybridBot
from bots.bot_meanrev_sl import MeanRevSLBot
from bots.bot_meanrev_tp import MeanRevTPBot
from bots.bot_sniper import SniperBot
from bots.bot_phantom import PhantomBot
from bots.bot_late_window_maker import LateWindowMakerBot
from bots.bot_fee_zone_maker import FeeZoneMakerBot
from bots.bot_arbitrage import ArbitrageBot
from bots.bot_lag_residual import LagResidualBot
from bots.bot_regime_specialist import RegimeSpecialistBot
from bots.bot_no_lag import NoLagBot
from bots.bot_sweeper import SweeperBot
from bots.bot_cross_venue_lag import CrossVenueLagBot
from bots.bot_true_maker import TrueMakerBot
from signals.price_feed import get_feed as get_price_feed
from signals.sentiment import get_feed as get_sentiment_feed
from signals.polymarket_prices import get_feed as get_pm_price_feed

from arena.discovery import MarketDiscovery
from arena.market_data import MarketDataWarmer
from arena.maker_thread import MakerThread
from arena.trader import Trader
from arena.resolver import TradeResolver
from arena.position_monitor import PositionMonitorThread
from arena.signals import build_combined_signals
from arena.state import SharedArenaState
from signals.orderflow_signals import get_cvd_feed

from arena.log_setup import configure_logging, log_event
from evolution.ga import EVOLUTION_EXEMPT_TYPES

# Structured logging: JSON when ARENA_LOG_JSON is set, else the same text
# format as before (byte-identical console/file output). See arena/log_setup.py.
configure_logging(config.LOG_DIR / "arena.log", level=logging.INFO)
logger = logging.getLogger("arena")
maker_logger = logging.getLogger("arena.maker")


# Strategy types that the trader loop should never try to evaluate —
# the maker bots and copy-trade bots run on a separate MakerThread cadence.
MAKER_TYPES = {
    "late_window_maker", "fee_zone_maker", "btc_maker", "true_maker", "copy_trade",
}

TAKER_BOT_CLASSES = {
    "momentum": MomentumBot,
    "mean_reversion": MeanRevBot,
    "mean_reversion_sl": MeanRevSLBot,
    "mean_reversion_tp": MeanRevTPBot,
    "sniper": SniperBot,
    "phantom": PhantomBot,
    "hybrid": HybridBot,
    "arbitrage": ArbitrageBot,
    "lag_residual": LagResidualBot,
    "regime_specialist": RegimeSpecialistBot,
    "no_lag": NoLagBot,
    "sweeper": SweeperBot,
    "cross_venue_lag": CrossVenueLagBot,
}

# Maker strategy types that have a concrete default instance. Used to rebuild
# the maker slate from DB configs on the 'continue' path (where create_default_bots
# filters maker rows out of the trader list).
MAKER_BOT_CLASSES = {
    "late_window_maker": LateWindowMakerBot,
    "fee_zone_maker": FeeZoneMakerBot,
    "true_maker": TrueMakerBot,
}

# EVOLUTION_EXEMPT_TYPES imported from evolution.ga — arbitrage (market-neutral)
# and pure makers are never culled or mutated by the GA.


# ----------------------------------------------------------------------
# Bot lifecycle
# ----------------------------------------------------------------------

def create_default_bots():
    """Create the 4 (or N) taker bots from active DB configs.

    Falls back to a single-run default slate on a fresh DB.  Maker-type
    strategy rows are intentionally excluded — the maker section is run
    separately from the on_cycle_complete hook.
    """
    active = db.get_active_bots()
    if active:
        bots = []
        for cfg in active:
            if cfg["strategy_type"] in MAKER_TYPES:
                continue
            cls = TAKER_BOT_CLASSES.get(cfg["strategy_type"], MomentumBot)
            params = cfg["params"]
            if isinstance(params, str):
                params = json.loads(params)
            bots.append(cls(
                name=cfg["bot_name"],
                params=params,
                generation=cfg["generation"],
                lineage=cfg.get("lineage"),
            ))
        if bots:
            # Faithful to the DB: whatever was active last run comes back exactly
            # as it was (this is the 'continue' path). The arbitrage bot is only
            # part of the DEFAULT slate — it is not force-injected here, so a
            # manually-selected slate that excluded it stays excluded on restart.
            return bots
    # First-run fallback (empty DB, non-interactive): lean default slate
    # (founders / DEFAULT_INDICES via startup.build_default_bots). Lab pipeline
    # invents additional genomes; it does not own the empty-roster path.
    from arena import startup
    return startup.build_default_bots()


def _validate_bot(bot) -> bool:
    """Smoke-test ``bot.make_decision`` with dummy data."""
    dummy_market = {"current_price": 0.52, "id": "test", "question": "test"}
    dummy_signals = {"prices": [97000, 97050, 97100], "latest": 97100}
    try:
        result = bot.make_decision(dummy_market, dummy_signals)
        return result.get("action") in ("buy", "skip")
    except Exception as e:
        logger.error(f"  VALIDATION FAILED for {bot.name}: {e}")
        return False


def create_evolved_bot(winner, loser_type: str, gen_number: int):
    """Legacy helper — single-parent mutate. Prefer :func:`run_evolution` (GA).

    Kept for callers/tests that construct one mutant without a full cycle.
    Uses Gaussian mutation inside sensible bounds (evolution.operators.mutate).
    """
    from evolution.operators import mutate as ga_mutate
    from evolution.ga import _default_params_for
    import random as _random

    base_params = _default_params_for(loser_type)
    winner_params = winner.export_params()["params"]
    for key in base_params:
        if key in winner_params:
            base_params[key] = winner_params[key]
    new_params = ga_mutate(
        base_params,
        rate=getattr(config, "GA_MUTATION_RATE", config.MUTATION_RATE_DIRECTED),
    )
    name = f"{loser_type}-g{gen_number}-{_random.randint(100, 999)}"
    cls = TAKER_BOT_CLASSES.get(loser_type, MomentumBot)
    return cls(
        name=name,
        params=new_params,
        generation=gen_number,
        lineage=f"{winner.name} -> {name} (mutate)",
    )


def run_evolution(bots, cycle_number):
    """Run one Genetic Algorithm generation over the directional slate.

    Multi-objective fitness (P&L + Sharpe + low drawdown + consistency),
    tournament selection, crossover, Gaussian mutation, and elitism.
    Evolution-exempt bots (arbitrage + pure makers) pass through untouched.

    Returns
    -------
    (new_bots, report)
        ``report`` is the structured GA summary (elites / replaced / spawned)
        used for evolution notifications and diagnostics.
    """
    from evolution.ga import run_ga_cycle

    new_bots, report = run_ga_cycle(
        bots,
        cycle_number,
        class_map=TAKER_BOT_CLASSES,
        validate_fn=_validate_bot,
    )
    for spawn in report.get("spawned") or []:
        log_event(
            logger, logging.INFO,
            f"  GA spawn {spawn.get('name')} parents={spawn.get('parents')} "
            f"op={spawn.get('operator')}",
            event_type="evolution", action="spawn", cycle=cycle_number,
            bot=spawn.get("name"), parents=spawn.get("parents"),
            strategy=spawn.get("strategy_type"),
            generation=spawn.get("generation"),
            retired=spawn.get("replaced"),
            operator=spawn.get("operator"),
        )
    for bot in new_bots:
        if bot.strategy_type in EVOLUTION_EXEMPT_TYPES:
            continue
        logger.info(
            f"  Post-GA: {bot.name} ({bot.strategy_type}) "
            f"params_keys={list(bot.strategy_params.keys())} "
            f"lineage={getattr(bot, 'lineage', None)}"
        )
    return new_bots, report


# ----------------------------------------------------------------------
# Secondary bots: maker section + copy-trade bots.
# Run on MakerThread (~20s), decoupled from discovery Gamma scans.
# ----------------------------------------------------------------------

def _resolve_maker_bots(slate: list) -> list:
    """The maker bots to run this session, drawn from the launched slate.

    Maker bots are now first-class members of the default lineup (see
    ``startup.STRATEGY_MENU``), so on a fresh/interactive launch they arrive
    inside ``slate`` and are simply partitioned out here. On the 'continue'
    path ``create_default_bots`` filters maker rows out of the trader list, so
    we rebuild them from the DB's active maker configs instead. Their configs
    are persisted so the dashboard's Active Bots roster stays in sync.
    """
    maker_bots = [b for b in slate if b.strategy_type in MAKER_TYPES]
    if not maker_bots:
        for cfg in db.get_active_bots():
            cls = MAKER_BOT_CLASSES.get(cfg["strategy_type"])
            if cls is not None:
                maker_bots.append(cls(
                    name=cfg["bot_name"], generation=cfg["generation"]
                ))

    existing = {b["bot_name"] for b in db.get_active_bots()}
    for bot in maker_bots:
        if bot.name not in existing:
            db.save_bot_config(
                bot.name, bot.strategy_type, bot.generation, bot.strategy_params
            )
            logger.info(f"Registered maker bot: {bot.name} ({bot.strategy_type})")
    return maker_bots


def _make_secondary_hook(maker_bots, copy_bots, signal_feeds, state):
    """Wrap the secondary-bot tick in an exception-bounded hook ready to
    pass to ``MarketDiscovery(on_cycle_complete=...)``."""

    def hook(discovery: MarketDiscovery):
        try:
            _run_secondary_bots(discovery, maker_bots, copy_bots, signal_feeds, state)
        except Exception as e:
            logger.error(f"Secondary bot tick error: {e}")

    return hook


def _publish_maker_state(discovery, maker_targets):
    """Persist the maker section's current target/mode to ``arena_state``.

    Powers the Maker Section card on the dashboard's Overview tab.
    Called once per discovery cycle (20s cadence) -- cheap because it's
    a single upsert into a small key/value table.  Idempotent so the
    dashboard just reads the latest row.

    Mode is derived from the same LIVE-vs-fallback rule the maker
    section uses (a target equal to the live ``current_market`` is
    LIVE; any other target is PRE-WINDOW; empty list of targets is
    IDLE).    Keeping the labelling logic here avoids the dashboard
    having to re-discover markets just to classify them.
    """
    live_market = discovery.current_market_snapshot()
    live_market_id = live_market.get("id") if live_market else None

    if not maker_targets:
        snap = {
            "mode": "IDLE",
            "market_id": None,
            "market_question": None,
            "time_remaining_seconds": None,
            "resolves_at": None,
            "target_count": 0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        target = maker_targets[0]
        is_live = (
            live_market_id is not None
            and target.get("id") == live_market_id
        )
        mode = "LIVE" if is_live else "PRE-WINDOW"
        tr = target.get("time_remaining_seconds") or 0
        snap = {
            "mode": mode,
            "market_id": target.get("id"),
            "market_question": (target.get("question") or "")[:200],
            "time_remaining_seconds": int(round(tr)),
            "resolves_at": target.get("resolves_at"),
            "target_count": len(maker_targets),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    db.set_arena_state("maker_state", json.dumps(snap))


def _run_secondary_bots(discovery, maker_bots, copy_bots, signal_feeds, state):
    """Maker section + copy bots, all once per discovery cycle."""
    all_markets = discovery.all_markets_snapshot()
    maker_targets = discovery.maker_target_markets_snapshot()

    # --- Publish maker target state for the dashboard card. ---
    # Best-effort: a DB write failure here MUST NOT kill the secondary tick.
    try:
        _publish_maker_state(discovery, maker_targets)
    except Exception as e:
        logger.debug(f"publish maker state failed: {e}")
    live_market = discovery.current_market_snapshot()
    live_market_id = live_market.get("id") if live_market else None

    # --- Maker section: live market preferred, ≤20-min upcoming fallback. ---
    # Restores the pre-refactor behavior of the monolithic main_loop:
    # when no market currently contains the wall clock, fall back to
    # the soonest upcoming market resolving within MAKER_UPCOMING_WINDOW_SEC
    # so the maker can begin quoting bid/ask in the pre-window ramp.
    # Deliberately separate from the Trader's "swap only on actual
    # rollover" policy -- see arena/trader.py for that strict rule.
    if maker_bots and maker_targets:
        from arena import market_data as _mdata
        for target in maker_targets:
            is_live = (
                live_market_id is not None
                and target.get("id") == live_market_id
            )
            mode = "LIVE" if is_live else "PRE-WINDOW"
            tr = target.get("time_remaining_seconds", 0)
            # Lay warm mids/asks/books so maker decisions + paper fills share
            # one snapshot (same atomic-book path as the 1s trader).
            mid = target.get("id") or target.get("market_id")
            warm = _mdata.store().get(mid) if mid else None
            _mdata.lay_warm_onto_market(target, warm)
            signals = build_combined_signals(
                *signal_feeds, market=target, warm=warm)
            maker_trades = 0
            for mb in maker_bots:
                try:
                    if _run_maker_section(mb, target, signals, state):
                        maker_trades += 1
                except Exception as e:
                    maker_logger.error(f"[{mb.name}] tick failed: {e}")
            if maker_trades > 0:
                maker_logger.info(
                    f"Maker section ({mode}, time_remaining={tr:.0f}s) "
                    f"placed {maker_trades} paper trade(s) this cycle"
                )

    # Copy bots are disabled (Simmer removed) — _create_copy_bots returns [].


def _run_maker_section(maker_bot, market: dict, signals: dict, state) -> bool:
    """One BtcMakerBot paper cycle on a single market.

    ``trading_mode`` is forced to ``"paper"`` so even if the DB row were
    toggled to ``"live"`` by mistake no real Polymarket orders are placed.
    """
    maker_bot.trading_mode = "paper"
    market_id = market.get("id") or market.get("market_id")
    key = (maker_bot.name, market_id)
    if state.is_traded(key):
        return False
    if state.is_slippage_cooling(key):
        state.note_skip("slippage_cooldown")
        return False

    try:
        signal = maker_bot.analyze(market, signals)
        market_price = market.get("current_price", 0.5)
        maker_bid = signal.get("maker_bid")
        maker_ask = signal.get("maker_ask")
        maker_mid = signal.get("maker_mid")
        maker_side = signal.get("maker_side", "both")
        edge_bps = (
            abs((maker_mid or market_price) - market_price) * 10000
            if maker_mid is not None else 0.0
        )

        maker_logger.info(
            f"[{maker_bot.name}] market={str(market_id)[:12]}... "
            f"price={market_price:.3f} "
            f"bid={maker_bid:.3f} ask={maker_ask:.3f} mid={maker_mid:.3f} "
            f"edge={edge_bps:.1f}bps lean={maker_side} "
            f"conf={signal.get('confidence', 0.0):.3f}"
        )

        if signal.get("action") == "hold":
            # A HOLD is NOT a trade — do NOT add it to the dedup set. The maker
            # section sees each market up to MAKER_UPCOMING_WINDOW_SEC (20 min)
            # early, in PRE-WINDOW mode. Time-gated makers like LateWindowMaker
            # deliberately hold until the final ~90s; marking that early hold as
            # "traded" locked the market out of the dedup set forever, so the
            # bot never got to re-evaluate during its actual entry window
            # (=> zero trades). Return without marking so the next discovery
            # cycle re-runs analyze() with fresh time_remaining / price.
            return False

        result = maker_bot.execute(signal, market)
        if result.get("success"):
            state.mark_traded(key)
            maker_logger.info(
                f"[{maker_bot.name}] PAPER {signal['side'].upper()} "
                f"${signal.get('suggested_amount', 0):.2f} "
                f"on {market.get('question', '')[:50]}"
            )
            return True
        reason = result.get("reason")
        if reason in ("slippage_band", "slippage_exceeded"):
            cd = float(getattr(config, "SLIPPAGE_RETRY_COOLDOWN_SEC", 10.0))
            state.mark_slippage_reject(key, cd)
            state.note_skip("slippage")
        # Failed execute is NOT marked traded — allow retry after cooldown /
        # next discovery cycle (was permanent lockout, which killed makers on
        # one slippage miss).
        maker_logger.debug(
            f"[{maker_bot.name}] paper execute skipped: {reason}"
        )
        return False

    except Exception as e:
        maker_logger.error(f"[{maker_bot.name}] Maker section error: {e}")
        state.mark_traded(key)
        return False


# ----------------------------------------------------------------------
# Evolution check on the main coordinator thread
# ----------------------------------------------------------------------

def _evolution_check_loop(bots, state, pos_monitor, trader, maker_bots=None):
    """Main-thread periodic evolution check.

    Runs every 30s (the trader thread is already at 1s, so we don't need
    to be any faster than this to catch the boundary cleanly).  Body is
    wrapped in try/except so a single bad evolution cycle can't silently
    crash the main coordinator while the four daemon workers continue
    independently without any clear log.

    This loop also hosts the two slow-cadence maintenance jobs (they need a
    periodic home, not their own threads): the live lane monitor
    (arena/lane_monitor.py — auto-demotes approved lanes whose live accuracy
    falls below the bar) and the auto-validation scheduler
    (arena/validation_scheduler.py — spawns the harness every
    AUTO_VALIDATE_EVERY_MARKETS windows so Signal Lab proposals appear
    without any manual run).

    ``maker_bots`` is the live maker list held by reference for MakerThread;
    mid-run deploys may append to it in place.
    """
    from arena import lane_monitor, lane_promoter, core_lane_tuner, portfolio
    from arena import regime_map
    from arena import risk_engine
    from arena.validation_scheduler import ValidationScheduler

    if maker_bots is None:
        maker_bots = []

    evolution_interval = config.EVOLUTION_INTERVAL_HOURS * 3600
    ga_min_interval = float(getattr(config, "GA_MIN_INTERVAL_SEC", 1800))
    validation_scheduler = ValidationScheduler()
    lane_monitor_interval = getattr(config, "LANE_MONITOR_INTERVAL_SEC", 1800)
    # Core / live-lane tuner on a faster cadence so weights track 5m tape
    # (style-skip uses regime_stats 15s cache; tuner was stuck on 30m host).
    core_tune_interval = float(getattr(config, "CORE_TUNE_INTERVAL_SEC", 300))
    portfolio_interval = float(
        getattr(config, "PORTFOLIO_REBALANCE_INTERVAL_SEC", 1800))
    regime_map_interval = float(getattr(config, "REGIME_MAP_INTERVAL_SEC", 900))
    last_lane_check = 0.0
    last_core_tune = 0.0
    last_portfolio_check = 0.0
    last_regime_map_check = 0.0
    last_pool_pnl = None  # for performance-trigger drop detection

    saved_cycle = db.get_arena_state("evolution_cycle", "0")
    cycle_number = int(saved_cycle)
    saved_last_evo = db.get_arena_state("last_evolution_time")
    if saved_last_evo:
        last_evolution = float(saved_last_evo)
        elapsed_hours = (time.time() - last_evolution) / 3600
        logger.info(
            f"Restored evolution timer: cycle {cycle_number}, "
            f"{elapsed_hours:.1f}h since last evolution"
        )
    else:
        last_evolution = time.time()
        db.set_arena_state("last_evolution_time", str(last_evolution))
        db.set_arena_state("evolution_cycle", "0")
        logger.info("No saved evolution state, starting fresh timer (persisted)")

    while True:
        # Dashboard mid-run deploy queue (Bots tab) — fold into live slate.
        try:
            from arena.deploy import process_pending_deploys
            bots, maker_bots, deploy_result = process_pending_deploys(
                bots, maker_bots, trader, pos_monitor,
                maker_types=MAKER_TYPES,
            )
            if deploy_result and deploy_result.get("deployed"):
                log_event(
                    logger, logging.INFO,
                    f"Mid-run deploy: {[d['bot_name'] for d in deploy_result['deployed']]}",
                    event_type="deploy",
                    deployed=deploy_result.get("deployed"),
                    skipped=deploy_result.get("skipped"),
                )
        except Exception as e:
            log_event(
                logger, logging.ERROR, f"Mid-run deploy error (caught): {e}",
                exc_info=True, event_type="error", where="mid_run_deploy",
            )

        try:
            now = time.time()
            elapsed = now - last_evolution
            due_timer = elapsed >= evolution_interval
            due_perf = False
            perf_reason = ""
            # Performance trigger only after the anti-thrash floor
            if (not due_timer) and elapsed >= ga_min_interval:
                from evolution.ga import should_trigger_evolution
                due_perf, perf_reason = should_trigger_evolution(
                    bots, last_trigger_pnl=last_pool_pnl,
                )
            if due_timer or due_perf:
                trigger = "timer" if due_timer else f"performance:{perf_reason}"
                cycle_number += 1
                logger.info(
                    "GA cycle %s trigger=%s (elapsed=%.1fh)",
                    cycle_number, trigger, elapsed / 3600,
                )
                bots, evo_report = run_evolution(bots, cycle_number)
                last_evolution = time.time()
                db.set_arena_state("evolution_cycle", str(cycle_number))
                db.set_arena_state("last_evolution_time", str(last_evolution))
                db.set_arena_state("last_evolution_trigger", trigger)

                # Snapshot pool P&L for next drop check
                try:
                    total = 0.0
                    for b in bots:
                        if b.strategy_type in EVOLUTION_EXEMPT_TYPES:
                            continue
                        total += float(
                            b.get_performance(
                                hours=config.EVOLUTION_WINDOW_HOURS
                            ).get("total_pnl") or 0.0
                        )
                    last_pool_pnl = total
                except Exception:
                    pass

                try:
                    from arena.alerts import alert_evolution
                    alert_evolution(
                        cycle_number, trigger, report=evo_report,
                    )
                except Exception:
                    pass

                state.reset()
                trader.set_bots(bots)
                pos_monitor.update_bots(bots)

                # Roster changed → rebalance capital weights immediately so
                # Capital Allocation / Kelly slices match the new slate
                # (otherwise the dashboard keeps retired -v1 names until the
                # 30m timer or a regime flip).
                if evo_report and (
                    evo_report.get("replaced") or evo_report.get("spawned")
                ):
                    try:
                        portfolio.rebalance(force=True, reason="evolution")
                    except Exception as pe:
                        log_event(
                            logger, logging.WARNING,
                            f"Post-evolution portfolio rebalance failed: {pe}",
                            event_type="error", where="portfolio_rebalance",
                            cycle=cycle_number,
                        )
        except Exception as e:
            log_event(logger, logging.ERROR, f"Evolution cycle error (caught): {e}",
                      exc_info=True, event_type="error", where="evolution_cycle",
                      cycle=cycle_number)
            try:
                from arena.alerts import alert_error
                alert_error("evolution_cycle", str(e))
            except Exception:
                pass

        try:
            if time.time() - last_lane_check >= lane_monitor_interval:
                # Demote live candidate lanes that decayed, then judge pending
                # proposals against live shadow evidence (auto-approve when the
                # toggle is on). Order matters: demotion frees an active-lane
                # slot before the promoter checks the concentration cap.
                lane_monitor.check_lanes()
                lane_promoter.check_proposals()
                # Offline rollup of decision_events (skips + buys) for
                # counterfactual lane/strategy fine-tuning.
                try:
                    from arena.decision_log import maybe_rollup, maybe_prune, flush
                    flush()
                    maybe_rollup()
                    maybe_prune()
                    try:
                        from arena.live_scorecard import maybe_refresh
                        maybe_refresh()
                    except Exception as se:
                        logger.debug("live_scorecard: %s", se)
                    try:
                        from arena.gate_tuner import maybe_tune
                        maybe_tune()
                    except Exception as ge:
                        logger.debug("gate_tuner: %s", ge)
                    try:
                        from arena.combo_explorer import maybe_refresh as combo_refresh
                        combo_refresh()
                    except Exception as ce:
                        logger.debug("combo_explorer: %s", ce)
                    try:
                        from arena.learned_rules import mine_and_update
                        mine_and_update()
                    except Exception as le:
                        logger.debug("learned_rules mine: %s", le)
                except Exception as e:
                    logger.debug("decision rollup: %s", e)
                last_lane_check = time.time()
        except Exception as e:
            log_event(logger, logging.ERROR, f"Lane monitor/promoter error (caught): {e}",
                      exc_info=True, event_type="error", where="lane_pipeline")

        try:
            # Core + live-candidate weight nudges (faster than demote loop).
            if time.time() - last_core_tune >= core_tune_interval:
                core_lane_tuner.tune()
                last_core_tune = time.time()
        except Exception as e:
            log_event(logger, logging.ERROR, f"Core-lane tuner error (caught): {e}",
                      exc_info=True, event_type="error", where="core_lane_tuner")

        try:
            # Regime discovery (Layer 2): recompute the per-bot context
            # attribution map + validated regimes, and publish the current
            # cell. Best-effort — never raises into the arena loop. The
            # portfolio allocator and core-lane tuner read this map.
            if time.time() - last_regime_map_check >= regime_map_interval:
                regime_map.rebuild()
                last_regime_map_check = time.time()
        except Exception as e:
            log_event(logger, logging.ERROR, f"Regime map rebuild error (caught): {e}",
                      exc_info=True, event_type="error", where="regime_map")

        try:
            validation_scheduler.check()
        except Exception as e:
            log_event(logger, logging.ERROR, f"Auto-validation scheduler error (caught): {e}",
                      exc_info=True, event_type="error", where="validation_scheduler")

        try:
            # Portfolio capital allocation: rebalance on timer and/or regime
            # change (arena/portfolio.py). Weights feed Kelly bankroll slices
            # and zone-bot size multipliers; dashboard can force a rebalance.
            if time.time() - last_portfolio_check >= min(60.0, portfolio_interval / 4):
                portfolio.maybe_rebalance()
                last_portfolio_check = time.time()
        except Exception as e:
            log_event(logger, logging.ERROR, f"Portfolio rebalance error (caught): {e}",
                      exc_info=True, event_type="error", where="portfolio_rebalance")

        try:
            # Risk engine: recompute drawdowns / daily loss / VaR / size
            # multipliers; honor kill-switch file. Runs every ~15s (interval
            # gated inside maybe_evaluate). Hot-path pre_trade reads the
            # cached state on every execute.
            risk_engine.maybe_evaluate()
            # Sync bot._paused flags from risk state so UI/status matches
            risk_state = risk_engine.load_state()
            for b in bots:
                st = (risk_state.get("bots") or {}).get(b.name) or {}
                if st.get("status") == "paused" or risk_state.get("kill_switch"):
                    b._paused = True
                elif st.get("status") in ("active", "reduced") and not st.get(
                        "manual_pause"):
                    # Only auto-clear risk pauses (not daily-loss legacy)
                    if getattr(b, "_paused", False) and (
                            st.get("reason") or "").startswith(
                            ("bot_daily", "bot_max", "underperform",
                             "portfolio_")):
                        b._paused = False
        except Exception as e:
            log_event(logger, logging.ERROR, f"Risk engine error (caught): {e}",
                      exc_info=True, event_type="error", where="risk_engine")

        try:
            # Health checks + optional alerts when status worsens (gated)
            from arena import health as health_mod
            import time as _t
            _hi = float(getattr(config, "HEALTH_EVAL_INTERVAL_SEC", 60))
            _last_h = float(db.get_arena_state("health_last_eval") or 0)
            if _t.time() - _last_h >= _hi:
                health_mod.maybe_alert_on_health()
                db.set_arena_state("health_last_eval", str(_t.time()))
        except Exception as e:
            log_event(logger, logging.ERROR, f"Health check error (caught): {e}",
                      exc_info=True, event_type="error", where="health")

        try:
            # Periodic digests + ops threshold alerts (hourly/daily/bankroll/
            # feed/skips/resolver). Event-driven alerts fire at their sources.
            from arena.alerts import run_periodic_alerts
            run_periodic_alerts()
        except Exception as e:
            log_event(logger, logging.ERROR, f"Periodic alert error (caught): {e}",
                      exc_info=True, event_type="error", where="periodic_alerts")

        time.sleep(30)


# ----------------------------------------------------------------------
# Bootstrap (signal feeds) + main loop (thread startup) + main entry
# ----------------------------------------------------------------------

def _dashboard_is_up(port: int, timeout: float = 1.0) -> bool:
    """True if *something* is already serving on the dashboard port.

    Any HTTP response — including a 401 from the Basic-auth gate — means the
    server is live.  Only a connection-level failure counts as "down".  This
    keeps the check independent of the dashboard credentials.
    """
    url = f"http://127.0.0.1:{port}/api/status"
    try:
        urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 (localhost)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def _terminate_dashboard(proc: subprocess.Popen) -> None:
    """Best-effort shutdown of a dashboard child we spawned (atexit hook)."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def start_dashboard() -> None:
    """Ensure the dashboard is serving, then open it (when interactive).

    Cross-platform: if the dashboard port is already answering (e.g. a
    launchd/systemd service, or a manually-started ``dashboard/server.py``),
    we leave it alone.  Otherwise we spawn ``dashboard/server.py`` using the
    *same* interpreter running the arena — ``sys.executable`` resolves to the
    project venv on every OS (``.venv\\Scripts\\python.exe`` on Windows,
    ``.venv/bin/python3`` on macOS/Linux), so no per-OS command is needed.

    Set ``ARENA_NO_DASHBOARD=1`` to skip auto-spawn (e.g. when the dashboard
    is managed by its own service and you don't want a duplicate).
    """
    port = config.DASHBOARD_PORT
    url = f"http://localhost:{port}/"

    if _dashboard_is_up(port):
        logger.info(f"Dashboard already running at {url}")
    elif os.environ.get("ARENA_NO_DASHBOARD"):
        logger.info(
            f"Dashboard not running and ARENA_NO_DASHBOARD is set — "
            f"start it yourself: {sys.executable} dashboard/server.py"
        )
    else:
        server_path = Path(__file__).resolve().parent / "dashboard" / "server.py"
        log_path = config.LOG_DIR / "dashboard.log"
        logger.info(f"Starting dashboard server: {server_path} (logs → {log_path})")
        try:
            log_file = open(log_path, "a", encoding="utf-8")
            proc = subprocess.Popen(
                [sys.executable, str(server_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(server_path.parent.parent),
            )
        except Exception as exc:  # spawn failed outright — don't kill the arena
            logger.warning(f"Could not launch dashboard server: {exc}")
        else:
            atexit.register(_terminate_dashboard, proc)
            # Wait (up to ~30s) for uvicorn to bind before we report/open.
            # The child re-imports the heavy trading deps (py-clob-client,
            # cryptography, ...) before binding, so cold start can be ~15-20s.
            for _ in range(60):
                if proc.poll() is not None:
                    logger.warning(
                        f"Dashboard server exited early (code {proc.returncode}) "
                        f"— see {log_path}"
                    )
                    break
                if _dashboard_is_up(port):
                    logger.info(f"Dashboard is up at {url}")
                    break
                time.sleep(0.5)
            else:
                logger.warning(
                    f"Dashboard did not become ready within ~30s — see {log_path}"
                )

    # Open a browser only for interactive runs, and only once the server is
    # actually answering (avoids the old "browser opens on a blind timer
    # before uvicorn has bound" race).
    if sys.stdin.isatty() and _dashboard_is_up(port):
        webbrowser.open(url)


def main_loop(bots):
    """Wire up everything: feeds, shared state, secondary bots, four
    worker threads, then drive the evolution check on this main thread."""
    lab_host = None
    # Market data + resolution come from Polymarket (public, no keys). Paper
    # mode simulates against real order books; live mode needs Polymarket CLOB
    # credentials, checked lazily when a bot is flipped to live.
    logger.info(
        f"Paper mode: shared virtual bankroll ${db.get_paper_bankroll():.2f} "
        f"(set in dashboard Settings); {len(bots)} bots trade real Polymarket books"
    )

    # Signal feeds are daemons with their own threads; the trader reads
    # their cached state on every 1s tick.
    price_feed = get_price_feed()
    sentiment_feed = get_sentiment_feed()
    pm_price_feed = get_pm_price_feed()
    price_feed.start()
    sentiment_feed.start()
    signal_feeds = (price_feed, sentiment_feed, pm_price_feed)

    state = SharedArenaState()
    with db.get_conn() as conn:
        loaded = state.load_from_db(conn)
    logger.info(
        f"Loaded {loaded} recent trade keys from DB (dedup across restarts)"
    )

    # The launched slate mixes 1s-trader bots and discovery-cadence maker bots.
    # Partition so the Trader / position-monitor / evolution only ever see the
    # trader bots, while the maker section drives the maker bots on its own
    # cadence. (Makers are now part of the default lineup — see startup.py.)
    maker_bots = _resolve_maker_bots(bots)
    maker_names = {b.name for b in maker_bots}
    trader_bots = [b for b in bots if b.name not in maker_names]
    copy_bots = []  # Copy-trading was Simmer-based and has been removed.
    logger.info(
        f"Trader bots: {[b.name for b in trader_bots]} | "
        f"Maker bots: {[b.name for b in maker_bots]}"
    )

    pos_monitor = PositionMonitorThread()
    pos_monitor.update_bots(trader_bots)
    pos_monitor.start()

    # Discovery no longer hosts makers (would delay Gamma scan on paper fills).
    discovery = MarketDiscovery(on_cycle_complete=None)
    discovery.start()

    maker_thread = MakerThread(
        discovery,
        _make_secondary_hook(maker_bots, copy_bots, signal_feeds, state),
    )
    maker_thread.start()

    # Market-data warmer: single owner of all per-market network reads (YES+NO
    # books, prices, OBI, CVD, PM momentum) refreshed every ~1s into a shared
    # warm cache, so the trader hot path and the arbitrage bot never touch the
    # network on their 1s tick.
    warmer = MarketDataWarmer(discovery, get_cvd_feed(), pm_price_feed)
    warmer.start()
    try:
        from signals.brti import start_brti_feed
        start_brti_feed()
        logger.info("BRTI feed started (Kalshi settlement index)")
    except Exception:
        logger.warning("BRTI feed failed to start", exc_info=True)

    resolver = TradeResolver()
    resolver.start()

    # Live mode requires at least one alert channel (set-and-forget ops).
    try:
        any_live = any(
            (getattr(b, "trading_mode", None) or db.get_bot_mode(b.name)) == "live"
            for b in (trader_bots + maker_bots)
        )
        if any_live and getattr(config, "LIVE_REQUIRE_ALERTS", True):
            from arena.alerts import alerts_configured
            if not alerts_configured():
                raise SystemExit(
                    "LIVE_REQUIRE_ALERTS: configure Telegram/Discord alerts "
                    "before running any bot in live mode "
                    "(TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID or DISCORD_WEBHOOK_URL)."
                )
    except SystemExit:
        raise
    except Exception as e:
        logger.debug("live alert gate check failed: %s", e)

    try:
        from arena.alerts import alert_startup
        alert_startup(trader_bots + maker_bots)
    except Exception as e:
        logger.debug("startup alert failed: %s", e)

    trader = Trader(
        discovery=discovery,
        state=state,
        price_feed=price_feed,
        sentiment_feed=sentiment_feed,
        polymarket_price_feed=pm_price_feed,
    )
    trader.set_bots(trader_bots)
    trader.start()


    if getattr(config, "STRATEGY_LAB_ENABLED", False):
        try:
            from signals.strategy_pipeline.cycle import get_host as get_lab_host
            lab_host = get_lab_host()
            lab_host.start()
        except Exception:
            logger.exception("strategy lab cycle host failed to start")

    # Mark the start of this session so the dashboard's "Current Session"
    # performance row can scope stats to trades placed since this boot.
    # Stored in the same UTC "%Y-%m-%d %H:%M:%S" format as trades.created_at
    # so a plain string comparison (created_at >= session_start) works.
    db.set_arena_state(
        "session_start", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )
    try:
        from arena import portfolio as _port
        _port.rebalance(force=True, reason="startup")
    except Exception as e:
        logger.debug("startup portfolio rebalance skipped: %s", e)

    logger.info(
        f"Arena started with {len(trader_bots)} trader + {len(maker_bots)} maker "
        f"bots in {config.get_current_mode()} mode"
    )
    logger.info(f"Bots: {[b.name for b in bots]}")
    logger.info(f"Evolution every {config.EVOLUTION_INTERVAL_HOURS}h")

    try:
        # Evolution only culls/mutates the directional trader bots; makers (and
        # the arbitrage bot, via EVOLUTION_EXEMPT_TYPES) are left untouched.
        # maker_bots is the same list MakerThread holds — mid-run deploys append
        # in place so makers appear without restart.
        _evolution_check_loop(
            trader_bots, state, pos_monitor, trader, maker_bots=maker_bots,
        )
    except KeyboardInterrupt:
        logger.info("Arena stopped by user")
    finally:
        for w in (trader, resolver, discovery, warmer, pos_monitor, maker_thread):
            w.stop()
        if lab_host is not None:
            try:
                lab_host.stop()
            except Exception:
                pass
        time.sleep(0.5)
        logger.info("All workers stopped.")



# ----------------------------------------------------------------------
# Single-instance lock (main() only — importing arena modules stays free)
# ----------------------------------------------------------------------

_INSTANCE_LOCK_FD = None
_INSTANCE_LOCK_PATH = None


def _pid_alive(pid: int) -> bool:
    """True if *pid* looks like a live OS process (Windows-safe)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid),
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _release_instance_lock() -> None:
    global _INSTANCE_LOCK_FD, _INSTANCE_LOCK_PATH
    fd = _INSTANCE_LOCK_FD
    _INSTANCE_LOCK_FD = None
    _INSTANCE_LOCK_PATH = None
    if fd is None:
        return
    try:
        if os.name == "nt":
            import msvcrt
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    except Exception:
        pass
    try:
        os.close(fd)
    except Exception:
        pass


def _acquire_instance_lock() -> None:
    """Exclusive lock under LOG_DIR so two arena mains cannot run together."""
    global _INSTANCE_LOCK_FD, _INSTANCE_LOCK_PATH
    lock_dir = Path(
        getattr(config, "LOG_DIR", Path(__file__).resolve().parent / "logs")
    )
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    path = lock_dir / "arena.instance.lock"
    _INSTANCE_LOCK_PATH = path
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = "?"
        try:
            with open(path, "r", encoding="utf-8") as fh:
                holder = (fh.read() or "").strip() or "?"
        except Exception:
            pass
        try:
            os.close(fd)
        except Exception:
            pass
        alive = False
        try:
            alive = _pid_alive(int(str(holder).strip()))
        except Exception:
            alive = False
        msg = (
            f"Another arena instance appears to be running "
            f"(lock={path}, pid={holder}"
            f"{'' if alive else ', possibly stale'}"
            f"). Exit that process first."
        )
        logger.error(msg)
        raise SystemExit(msg)
    try:
        os.ftruncate(fd, 0)
        os.write(fd, ("%d\n" % os.getpid()).encode("utf-8"))
    except Exception:
        pass
    _INSTANCE_LOCK_FD = fd
    atexit.register(_release_instance_lock)



def main() -> None:
    _acquire_instance_lock()
    parser = argparse.ArgumentParser(description="Polymarket Bot Arena")
    parser.add_argument(
        "--mode", choices=["paper", "live"], default=None,
        help="Trading mode (default: from config)",
    )
    args = parser.parse_args()

    if args.mode:
        if args.mode == "live":
            if not sys.stdin.isatty():
                logger.error(
                    "Refusing to switch to LIVE mode from a non-interactive "
                    "context."
                )
                logger.error(
                    "Run the arena manually (terminal attached) to enable "
                    "live trading."
                )
                sys.exit(2)
            confirm = input(
                "You are switching to LIVE trading with real USDC. "
                "Type YES to confirm: "
            )
            if confirm.strip() != "YES":
                print("Cancelled. Staying in paper mode.")
                sys.exit(0)
        config.set_trading_mode(args.mode)
        logger.info(f"Trading mode set to: {args.mode}")

    # Terminal launches get the interactive startup flow (continue-vs-fresh,
    # then default-vs-manual bot selection). It returns an explicit bot slate,
    # or None to mean "use the existing DB configuration" (continue / launchd).
    from arena import startup
    selected = startup.interactive_startup()
    bots = selected if selected is not None else create_default_bots()

    existing = {b["bot_name"] for b in db.get_active_bots()}
    for bot in bots:
        if bot.name not in existing:
            db.save_bot_config(
                bot.name, bot.strategy_type, bot.generation, bot.strategy_params
            )
    for bot in bots:
        bot.trading_mode = db.get_bot_mode(bot.name)

    active_names = [b.name for b in bots]
    backfilled = learning.backfill_from_resolved_trades(bot_names=active_names)
    if backfilled:
        logger.info(
            f"Backfilled learning from {backfilled} trades for active bots: "
            f"{active_names}"
        )

    start_dashboard()
    main_loop(bots)


if __name__ == "__main__":
    main()

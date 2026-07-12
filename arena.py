"""Polymarket Bot Arena — thin coordinator over four background threads.

This module used to be a single monolithic ``main_loop`` that did everything
on a 15s cadence: market discovery, signal harvesting, bot evaluation,
trade execution, resolution, evolution, copy-trade polling, maker
quoting, SL/TP monitoring.  After the refactor it has been split:

    ┌───────────────────────┬──────────────────────────────────────────────┐
    │ MarketDiscovery       │ 60s tick — scans Simmer SDK + public endpoint│
    │ (arena/discovery.py)  │ and refreshes orderflow context for the live │
    │                       │ market only. Owns ``current_market`` under a │
    │                       │ snapshot lock. No speculative next-market.    │
    ├───────────────────────┼──────────────────────────────────────────────┤
    │ Trader                │ 1s tick — runs bot ``make_decision`` +       │
    │ (arena/trader.py)     │ ``execute`` against ``current_market``.      │
    │                       │ Zero network IO per tick.                    │
    ├───────────────────────┼──────────────────────────────────────────────┤
    │ TradeResolver         │ 60s tick — checks Simmer for resolved        │
    │ (arena/resolver.py)   │ markets, writes outcomes + P&L, sweeps      │
    │                       │ stale-pending >1h trades.                   │
    ├───────────────────────┼──────────────────────────────────────────────┤
    │ PositionMonitorThread │ 0.5s tick — SL/TP exit engine against open  │
    │ (arena/position...py) │ positions on bots with ``exit_strategy``.    │
    └───────────────────────┴──────────────────────────────────────────────┘

This file (root ``arena.py``) is now strictly the coordinator.  It builds
the bots, boots the four worker threads, runs the periodic evolution
cycle on its main thread, registers one ``on_cycle_complete`` hook that
drives the maker section + copy-trade bots, and wires Ctrl-C cleanly to
all four workers.  Every actual piece of trading logic lives in the
``arena/`` package next door.
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
from bots.bot_sentiment import SentimentBot
from bots.bot_hybrid import HybridBot
from bots.bot_meanrev_sl import MeanRevSLBot
from bots.bot_meanrev_tp import MeanRevTPBot
from bots.bot_sniper import SniperBot
from bots.bot_phantom import PhantomBot
from bots.bot_late_window_maker import LateWindowMakerBot
from bots.bot_fee_zone_maker import FeeZoneMakerBot
from bots.bot_copy import CopyBot
from signals.price_feed import get_feed as get_price_feed
from signals.sentiment import get_feed as get_sentiment_feed
from signals.polymarket_prices import get_feed as get_pm_price_feed
from signals.wallet_monitor import WalletMonitor

from arena.discovery import MarketDiscovery
from arena.trader import Trader
from arena.resolver import TradeResolver
from arena.position_monitor import PositionMonitorThread
from arena.signals import build_combined_signals
from arena.state import SharedArenaState

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_DIR / "arena.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("arena")
maker_logger = logging.getLogger("arena.maker")


# Strategy types that the trader loop should never try to evaluate —
# the maker bots and copy-trade bots run on a separate cadence from
# the discovery thread's on_cycle_complete hook.
MAKER_TYPES = {"late_window_maker", "fee_zone_maker", "btc_maker", "copy_trade"}

TAKER_BOT_CLASSES = {
    "momentum": MomentumBot,
    "mean_reversion": MeanRevBot,
    "mean_reversion_sl": MeanRevSLBot,
    "mean_reversion_tp": MeanRevTPBot,
    "sniper": SniperBot,
    "phantom": PhantomBot,
    "sentiment": SentimentBot,
    "hybrid": HybridBot,
}


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
            return bots
    # First-run fallback
    return [
        MomentumBot(name="momentum-v1", generation=0),
        HybridBot(name="hybrid-v1", generation=0),
        MeanRevSLBot(name="meanrev-sl25-v1", generation=0),
        PhantomBot(name="phantom-v1", generation=0),
    ]


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
    """Create an evolved bot: loser's strategy defaults, winner's shared
    params wherever they overlap, then mutation."""
    from bots.bot_momentum import DEFAULT_PARAMS as MOMENTUM_DEFAULTS
    from bots.bot_mean_rev import DEFAULT_PARAMS as MEANREV_DEFAULTS
    from bots.bot_hybrid import DEFAULT_PARAMS as HYBRID_DEFAULTS
    from bots.bot_sentiment import DEFAULT_PARAMS as SENTIMENT_DEFAULTS
    from bots.bot_sniper import DEFAULT_PARAMS as SNIPER_DEFAULTS
    from bots.bot_phantom import DEFAULT_PARAMS as PHANTOM_DEFAULTS

    default_params_map = {
        "momentum": MOMENTUM_DEFAULTS,
        "mean_reversion": MEANREV_DEFAULTS,
        "mean_reversion_sl": MEANREV_DEFAULTS,
        "mean_reversion_tp": MEANREV_DEFAULTS,
        "sniper": SNIPER_DEFAULTS,
        "phantom": PHANTOM_DEFAULTS,
        "sentiment": SENTIMENT_DEFAULTS,
        "hybrid": HYBRID_DEFAULTS,
    }
    base_params = default_params_map.get(loser_type, MOMENTUM_DEFAULTS).copy()
    winner_params = winner.export_params()["params"]
    for key in base_params:
        if key in winner_params:
            base_params[key] = winner_params[key]

    new_params = winner.mutate(base_params)
    import random
    name = f"{loser_type}-g{gen_number}-{random.randint(100, 999)}"

    cls = TAKER_BOT_CLASSES.get(loser_type, MomentumBot)
    return cls(
        name=name,
        params=new_params,
        generation=gen_number,
        lineage=f"{winner.name} -> {name}",
    )


def run_evolution(bots, cycle_number):
    """Run evolution cycle — kill bots below WR threshold, mutate from survivors."""
    logger.info(f"=== Evolution Cycle {cycle_number} ===")

    rankings = []
    for bot in bots:
        perf = bot.get_performance(hours=config.EVOLUTION_INTERVAL_HOURS)
        rankings.append({
            "name": bot.name,
            "strategy_type": bot.strategy_type,
            "generation": bot.generation,
            "pnl": perf["total_pnl"],
            "win_rate": perf["win_rate"],
            "trades": perf["total_trades"],
        })

    rankings.sort(key=lambda x: x["win_rate"], reverse=True)

    immune = []
    above = []
    below = []
    for r in rankings:
        if r["trades"] < config.MIN_TRADES_FOR_JUDGMENT:
            immune.append(r)
        elif r["win_rate"] >= config.MIN_WIN_RATE:
            above.append(r)
        else:
            below.append(r)

    logger.info("Rankings (WR-based):")
    for r in rankings:
        if r in immune: status = "IMMUNE"
        elif r in above: status = "SURVIVES"
        else: status = "REPLACED"
        logger.info(
            f"  {r['name']}: WR={r['win_rate']:.1%}, "
            f"P&L=${r['pnl']:.2f}, Trades={r['trades']} [{status}]"
        )

    if not immune and not above and below:
        best = below.pop(0)
        above.append(best)
        logger.info(
            f"  Safety net: keeping {best['name']} "
            f"(best WR {best['win_rate']:.1%}) as sole survivor"
        )

    if not below:
        logger.info("  No bots below threshold — skipping evolution")
        for bot in bots:
            bot.reset_daily()
        return bots

    survivor_names = {r["name"] for r in immune + above}
    replaced_names = {r["name"] for r in below}

    new_bots = [b for b in bots if b.name in survivor_names]
    for b in new_bots:
        b.reset_daily()

    import random
    winners = [b for b in bots if b.name in survivor_names]
    replaced = [b for b in bots if b.name in replaced_names]

    for dead_bot in replaced:
        parent = random.choice(winners)
        evolved = create_evolved_bot(parent, dead_bot.strategy_type, cycle_number)

        if hasattr(dead_bot, "_api_key_slot"):
            evolved._api_key_slot = dead_bot._api_key_slot
            logger.info(
                f"  {evolved.name} inherits slot {dead_bot._api_key_slot} "
                f"from {dead_bot.name}"
            )

        if not _validate_bot(evolved):
            logger.warning(
                f"  {evolved.name} failed validation, recreating with pure defaults"
            )
            from bots.bot_momentum import DEFAULT_PARAMS as MOMENTUM_DEFAULTS
            from bots.bot_mean_rev import DEFAULT_PARAMS as MEANREV_DEFAULTS
            from bots.bot_hybrid import DEFAULT_PARAMS as HYBRID_DEFAULTS
            from bots.bot_sentiment import DEFAULT_PARAMS as SENTIMENT_DEFAULTS
            from bots.bot_sniper import DEFAULT_PARAMS as SNIPER_DEFAULTS
            from bots.bot_phantom import DEFAULT_PARAMS as PHANTOM_DEFAULTS
            fallback_map = {
                "momentum": MOMENTUM_DEFAULTS, "mean_reversion": MEANREV_DEFAULTS,
                "mean_reversion_sl": MEANREV_DEFAULTS,
                "mean_reversion_tp": MEANREV_DEFAULTS,
                "sniper": SNIPER_DEFAULTS, "phantom": PHANTOM_DEFAULTS,
                "sentiment": SENTIMENT_DEFAULTS, "hybrid": HYBRID_DEFAULTS,
            }
            cls = TAKER_BOT_CLASSES.get(dead_bot.strategy_type, MomentumBot)
            fallback_params = fallback_map.get(
                dead_bot.strategy_type, MOMENTUM_DEFAULTS,
            ).copy()
            evolved = cls(
                name=f"{parent.name}-g{cycle_number}-fallback",
                params=fallback_params,
                generation=cycle_number,
                lineage=f"{parent.name} -> fallback",
            )
            if hasattr(dead_bot, "_api_key_slot"):
                evolved._api_key_slot = dead_bot._api_key_slot

        db.retire_bot(dead_bot.name)
        db.save_bot_config(
            evolved.name, evolved.strategy_type, evolved.generation,
            evolved.strategy_params, evolved.lineage,
        )

        new_bots.append(evolved)
        logger.info(
            f"  Created {evolved.name} (from {parent.name}): "
            f"{json.dumps(evolved.strategy_params)[:200]}"
        )

    db.log_evolution(
        cycle_number,
        list(survivor_names),
        list(replaced_names),
        [b.name for b in new_bots if b.name not in survivor_names],
        rankings,
    )

    for bot in new_bots:
        slot = getattr(bot, "_api_key_slot", None)
        logger.info(
            f"  Post-evolution: {bot.name} ({bot.strategy_type}) "
            f"slot={slot} params_keys={list(bot.strategy_params.keys())}"
        )

    return new_bots


# ----------------------------------------------------------------------
# Credentials / slot assignment
# ----------------------------------------------------------------------

def load_api_key() -> str:
    """Read the Simmer default key from the encrypted credentials store."""
    return config.get_credential("simmer_api_key")


def load_bot_keys() -> dict:
    """Read the per-bot bot_keys map from the encrypted credentials store."""
    raw = config.get_credential("simmer_bot_keys")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def assign_bot_slots(bots, bot_keys: dict, default_key: str) -> None:
    """Assign each bot to a Simmer account slot (slot_0..slot_3).

    Bots that already carry a ``_api_key_slot`` (from evolution inheritance
    or a previous run) keep it; remaining bots grab the next free slot.
    Surfaces a clear warning if no key was configured anywhere.
    """
    all_slots = ["slot_0", "slot_1", "slot_2", "slot_3"]

    used_slots = {
        bot._api_key_slot for bot in bots
        if getattr(bot, "_api_key_slot", None)
    }
    free_slots = [s for s in all_slots if s not in used_slots]
    for bot in bots:
        if not getattr(bot, "_api_key_slot", None):
            if free_slots:
                bot._api_key_slot = free_slots.pop(0)
            else:
                bot._api_key_slot = all_slots[0]

    if not default_key and not bot_keys:
        logger.warning(
            "No Simmer credentials configured -- bots have no keys and "
            "cannot trade."
        )
        logger.warning("Open the dashboard Settings tab to enter your Simmer API key.")
        return

    for bot in bots:
        key = bot_keys.get(bot._api_key_slot, default_key)
        if key:
            logger.info(f"  {bot.name} -> {bot._api_key_slot} (key: ...{key[-8:]})")
        else:
            logger.warning(
                f"  {bot.name} -> {bot._api_key_slot} "
                "(NO KEY ASSIGNED -- bot will not trade)"
            )


# ----------------------------------------------------------------------
# Secondary bots: maker section + copy-trade bots.
# Run on the same 60s cadence as MarketDiscovery, in on_cycle_complete.
# ----------------------------------------------------------------------

def _create_maker_bots() -> list:
    """Persistent experimental maker bots. NOT part of evolution."""
    maker_bots = [
        LateWindowMakerBot(name="late-window-maker-v1"),
        FeeZoneMakerBot(name="fee-zone-maker-v1"),
    ]
    existing = {b["bot_name"] for b in db.get_active_bots()}
    for bot in maker_bots:
        if bot.name not in existing:
            db.save_bot_config(
                bot.name, bot.strategy_type, bot.generation, bot.strategy_params
            )
            logger.info(f"Registered maker bot: {bot.name} ({bot.strategy_type})")
    return maker_bots


def _create_copy_bots() -> list:
    """Instantiate copy-trade bots from the DB whitelist."""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT address, label, trading_mode FROM copytrading_wallets "
            "WHERE active=1"
        ).fetchall()

    bots = []
    for r in rows:
        mode = "paper"
        try:
            mode = r["trading_mode"] or "paper"
        except (IndexError, KeyError):
            pass
        bot = CopyBot(
            wallet_address=r["address"],
            label=r["label"] or r["address"][:16],
            mode=mode,
            max_size=5.0,
            size_fraction=0.10,
        )
        bots.append(bot)
        logger.info(
            f"Copy bot: [{bot.label}] wallet={r['address'][:16]}... mode={mode}"
        )
    return bots


def _start_wallet_monitors(copy_bots: list) -> None:
    """Attach a real-time WalletMonitor to each copy bot and start it."""
    for bot in copy_bots:
        try:
            monitor = WalletMonitor(bot.wallet, label=bot.label)
            bot.attach_monitor(monitor)
            monitor.start()
        except Exception as e:
            logger.warning(f"WalletMonitor start failed for {bot.label}: {e}")


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
    Called once per discovery cycle (60s cadence) -- cheap because it's
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
    api_key = config.get_credential("simmer_api_key")
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
        for target in maker_targets:
            is_live = (
                live_market_id is not None
                and target.get("id") == live_market_id
            )
            mode = "LIVE" if is_live else "PRE-WINDOW"
            tr = target.get("time_remaining_seconds", 0)
            signals = build_combined_signals(*signal_feeds, market=target)
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

    # --- Copy bots: scan every BTC up/down market the discovery surfaced. ---
    if copy_bots:
        copy_markets_by_token: dict = {}
        for m in all_markets:
            yt = m.get("polymarket_token_id")
            nt = m.get("polymarket_no_token_id")
            if yt:
                copy_markets_by_token[yt] = m
            if nt:
                copy_markets_by_token[nt] = m
        for cb in copy_bots:
            try:
                n = cb.check_and_copy(copy_markets_by_token, api_key)
                if n > 0:
                    logger.info(
                        f"Copy bot [{cb.label}]: mirrored {n} trades this cycle"
                    )
            except Exception as e:
                logger.error(f"Copy bot [{cb.label}] error: {e}")


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
            f"[{maker_bot.name}] market={market_id[:12]}... "
            f"price={market_price:.3f} "
            f"bid={maker_bid:.3f} ask={maker_ask:.3f} mid={maker_mid:.3f} "
            f"edge={edge_bps:.1f}bps lean={maker_side} "
            f"conf={signal.get('confidence', 0.0):.3f}"
        )

        if signal.get("action") == "hold":
            state.mark_traded(key)
            return False

        result = maker_bot.execute(signal, market)
        state.mark_traded(key)
        if result.get("success"):
            maker_logger.info(
                f"[{maker_bot.name}] PAPER {signal['side'].upper()} "
                f"${signal.get('suggested_amount', 0):.2f} "
                f"on {market.get('question', '')[:50]}"
            )
            return True
        else:
            maker_logger.debug(
                f"[{maker_bot.name}] paper execute skipped: "
                f"{result.get('reason')}"
            )
            return False

    except Exception as e:
        maker_logger.error(f"[{maker_bot.name}] Maker section error: {e}")
        state.mark_traded(key)
        return False


# ----------------------------------------------------------------------
# Evolution check on the main coordinator thread
# ----------------------------------------------------------------------

def _evolution_check_loop(bots, state, pos_monitor, trader):
    """Main-thread periodic evolution check.

    Runs every 30s (the trader thread is already at 1s, so we don't need
    to be any faster than this to catch the boundary cleanly).  Body is
    wrapped in try/except so a single bad evolution cycle can't silently
    crash the main coordinator while the four daemon workers continue
    independently without any clear log.
    """
    evolution_interval = config.EVOLUTION_INTERVAL_HOURS * 3600

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
        try:
            if time.time() - last_evolution >= evolution_interval:
                cycle_number += 1
                bots = run_evolution(bots, cycle_number)
                last_evolution = time.time()
                db.set_arena_state("evolution_cycle", str(cycle_number))
                db.set_arena_state("last_evolution_time", str(last_evolution))

                state.reset()
                api_key = config.get_credential("simmer_api_key")
                bot_keys = load_bot_keys()
                assign_bot_slots(bots, bot_keys, api_key)
                trader.set_bots(bots)
                pos_monitor.update_bots(bots)
        except Exception as e:
            logger.error(f"Evolution cycle error (caught): {e}")
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
    api_key = config.get_credential("simmer_api_key")
    bot_keys = load_bot_keys()
    assign_bot_slots(bots, bot_keys, api_key)
    if len(bot_keys) >= config.NUM_BOTS:
        logger.info(
            f"Multi-account mode: {len(bot_keys)} Simmer accounts loaded"
        )
    else:
        logger.info(
            f"Single-account mode: {len(bot_keys)} bot keys found "
            f"(need {config.NUM_BOTS} for independent trading)"
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

    maker_bots = _create_maker_bots()
    copy_bots = _create_copy_bots()
    if copy_bots:
        logger.info(f"Copy bots: {[b.label for b in copy_bots]}")
        _start_wallet_monitors(copy_bots)
    logger.info(
        f"Maker bots (experimental, paper-only): {[b.name for b in maker_bots]}"
    )

    pos_monitor = PositionMonitorThread()
    pos_monitor.update_bots(bots)
    pos_monitor.start()

    discovery = MarketDiscovery(
        on_cycle_complete=_make_secondary_hook(
            maker_bots, copy_bots, signal_feeds, state,
        )
    )
    discovery.start()

    resolver = TradeResolver()
    resolver.start()

    trader = Trader(
        discovery=discovery,
        state=state,
        price_feed=price_feed,
        sentiment_feed=sentiment_feed,
        polymarket_price_feed=pm_price_feed,
    )
    trader.set_bots(bots)
    trader.start()

    # Mark the start of this session so the dashboard's "Current Session"
    # performance row can scope stats to trades placed since this boot.
    # Stored in the same UTC "%Y-%m-%d %H:%M:%S" format as trades.created_at
    # so a plain string comparison (created_at >= session_start) works.
    db.set_arena_state(
        "session_start", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    )

    logger.info(
        f"Arena started with {len(bots)} bots in {config.get_current_mode()} mode"
    )
    logger.info(f"Bots: {[b.name for b in bots]}")
    logger.info(f"Evolution every {config.EVOLUTION_INTERVAL_HOURS}h")

    try:
        _evolution_check_loop(bots, state, pos_monitor, trader)
    except KeyboardInterrupt:
        logger.info("Arena stopped by user")
    finally:
        for w in (trader, resolver, discovery, pos_monitor):
            w.stop()
        time.sleep(0.5)
        logger.info("All workers stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Bot Arena")
    parser.add_argument(
        "--mode", choices=["paper", "live"], default=None,
        help="Trading mode (default: from config)",
    )
    parser.add_argument("--setup", action="store_true", help="Run setup verification first")
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

    if args.setup:
        import setup
        if not setup.main():
            sys.exit(1)

    api_key = config.get_credential("simmer_api_key")
    if not api_key:
        logger.warning("=" * 80)
        logger.warning("NO SIMMER API KEY CONFIGURED")
        logger.warning(
            "The bot arena will start, but bots will NOT trade until "
            "you enter a Simmer API key. Open the dashboard:"
        )
        logger.warning(
            f"    http://localhost:{config.DASHBOARD_PORT}/  "
            "(HTTP-Basic admin / Thor)"
        )
        logger.warning("Use the Settings tab to enter your Simmer API key.")
        logger.warning("=" * 80)
        logger.info("Continuing in monitoring-only mode. Will retry each cycle.")

    bots = create_default_bots()

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

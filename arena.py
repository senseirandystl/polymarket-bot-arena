"""Bot Arena Manager — runs 4 competing bots with 2-hour evolution cycles."""

import argparse
import json
import logging
import sys
import time
import random
import threading
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
from signals.price_feed import get_feed as get_price_feed
from signals.sentiment import get_feed as get_sentiment_feed
from signals.orderflow import get_feed as get_orderflow_feed
from copytrading.tracker import WalletTracker
from copytrading.copier import TradeCopier

import tkinter as tk
from tkinter import filedialog

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_DIR / "arena.log"),
    ]
)
logger = logging.getLogger("arena")

# Market check interval (seconds)
TRADE_INTERVAL = 15    # Discover markets + place trades every 15s (fast market discovery)
RESOLVE_INTERVAL = 60  # Resolve trades + expire stale every 60s (expensive, no need to rush)
FAST_POLL_INTERVAL = 0.5  # Poll market prices for SL/TP exits every 0.5s


def create_default_bots():
    """Create the 4 bots from active DB configs (or defaults for first run)."""
    active = db.get_active_bots()
    if active:
        bot_classes = {
            "momentum": MomentumBot,
            "mean_reversion": MeanRevBot,
            "mean_reversion_sl": MeanRevSLBot,
            "mean_reversion_tp": MeanRevTPBot,
            "sniper": SniperBot,
            "sentiment": SentimentBot,
            "hybrid": HybridBot,
        }
        bots = []
        for cfg in active:
            cls = bot_classes.get(cfg["strategy_type"], MomentumBot)
            params = cfg["params"]
            if isinstance(params, str):
                import json as _j
                params = _j.loads(params)
            bots.append(cls(
                name=cfg["bot_name"],
                params=params,
                generation=cfg["generation"],
                lineage=cfg.get("lineage"),
            ))
        if bots:
            return bots

    # First run fallback
    return [
        MomentumBot(name="momentum-v1", generation=0),
        HybridBot(name="hybrid-v1", generation=0),
        MeanRevSLBot(name="meanrev-sl25-v1", generation=0),
        MeanRevTPBot(name="meanrev-tp2x-v1", generation=0),
    ]


def create_evolved_bot(winner, loser_type, gen_number):
    """Create an evolved bot based on the winner's influence + loser's strategy.

    Uses the loser strategy's DEFAULT params as a base, copies over any
    shared keys from the winner (e.g. lookback_candles, position_size_pct),
    then mutates. This prevents KeyError when winner and loser have
    different param schemas.
    """
    from bots.bot_momentum import DEFAULT_PARAMS as MOMENTUM_DEFAULTS
    from bots.bot_mean_rev import DEFAULT_PARAMS as MEANREV_DEFAULTS
    from bots.bot_hybrid import DEFAULT_PARAMS as HYBRID_DEFAULTS
    from bots.bot_sentiment import DEFAULT_PARAMS as SENTIMENT_DEFAULTS

    from bots.bot_sniper import DEFAULT_PARAMS as SNIPER_DEFAULTS

    bot_classes = {
        "momentum": MomentumBot,
        "mean_reversion": MeanRevBot,
        "mean_reversion_sl": MeanRevSLBot,
        "mean_reversion_tp": MeanRevTPBot,
        "sniper": SniperBot,
        "sentiment": SentimentBot,
        "hybrid": HybridBot,
    }

    default_params_map = {
        "momentum": MOMENTUM_DEFAULTS,
        "mean_reversion": MEANREV_DEFAULTS,
        "mean_reversion_sl": MEANREV_DEFAULTS,
        "mean_reversion_tp": MEANREV_DEFAULTS,
        "sniper": SNIPER_DEFAULTS,
        "sentiment": SENTIMENT_DEFAULTS,
        "hybrid": HYBRID_DEFAULTS,
    }

    # Start with the target strategy's defaults
    base_params = default_params_map.get(loser_type, MOMENTUM_DEFAULTS).copy()

    # Copy shared keys from winner (transfers learned tuning for common params)
    winner_params = winner.export_params()["params"]
    for key in base_params:
        if key in winner_params:
            base_params[key] = winner_params[key]

    # Mutate
    new_params = winner.mutate(base_params)
    name = f"{loser_type}-g{gen_number}-{random.randint(100,999)}"

    cls = bot_classes.get(loser_type, MomentumBot)
    return cls(
        name=name,
        params=new_params,
        generation=gen_number,
        lineage=f"{winner.name} -> {name}",
    )


def _validate_bot(bot):
    """Smoke-test a bot by running make_decision with dummy data.
    Returns True if bot ...(truncated 26086 characters)... 0:
                            logger.debug(f"Skipping expired market (remaining={time_remaining:.0f}s): {m.get('question', '')[:50]}")
                            continue
                        if time_remaining < 90:
                            logger.debug(f"Skipping late-window market (remaining={time_remaining:.0f}s): {m.get('question', '')[:50]}")
                            continue
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Could not parse resolves_at '{resolves_at_str}': {e}")
                        # Still tradeable, just without time context
                        m["time_remaining_seconds"] = None
                        m["window_age_seconds"] = None

                tradeable_markets.append(m)

            if not tradeable_markets:
                logger.debug(f"All {len(five_min_markets)} 5-min markets filtered out (late-window), waiting...")
                time.sleep(TRADE_INTERVAL)
                continue

            five_min_markets = tradeable_markets

            # Gather signals
            price_signals = price_feed.get_signals("btc")
            sent_signals = sentiment_feed.get_signals("btc")

            new_trades = 0
            for market in five_min_markets:
                market_id = market.get("id") or market.get("market_id")
                of_signals = orderflow_feed.get_signals(market_id, api_key)
                combined_signals = {**price_signals, **sent_signals, **of_signals}

                # Each bot trades independently on its own account
                for bot in bots:
                    key = (bot.name, market_id)
                    if key in traded:
                        continue

                    # Enforce concurrent trades
                    current_open = len([t for t in db.get_bot_trades(bot.name) if t['outcome'] is None])
                    if current_open >= config.get_max_concurrent_trades():
                        logger.debug(f"[{bot.name}] Skipping: max concurrent trades reached ({current_open})")
                        continue

                    try:
                        signal = bot.make_decision(market, combined_signals)

                        # Skip if bot sees no edge
                        if signal.get("action") == "skip":
                            traded.add(key)
                            continue

                        result = bot.execute(signal, market)
                        traded.add(key)
                        if result.get("success"):
                            new_trades += 1
                            logger.info(f"[{bot.name}] {signal['side'].upper()} ${signal['suggested_amount']:.2f} (conf={signal['confidence']:.2f}) on {market.get('question', '')[:50]}")
                        else:
                            logger.debug(f"[{bot.name}] Trade failed on {market_id}: {result.get('reason')}")
                    except Exception as e:
                        logger.error(f"[{bot.name}] Error on {market_id}: {e}")
                        traded.add(key)

            if new_trades > 0:
                logger.info(f"Placed {new_trades} new trades this cycle")

            # Position monitor thread polls Simmer every 0.5s for SL/TP
            time.sleep(TRADE_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Arena stopped by user")
            break
        except Exception as e:
            logger.error(f"Arena loop error: {e}")
            time.sleep(10)

    # Individual evolution check (every 2 hours)
    if (datetime.utcnow() - individual_evolution_start) >= timedelta(hours=config.INDIVIDUAL_EVOLUTION_INTERVAL_HOURS):
        logger.info("=== Individual Bot Evolution ===")
        for i, bot in enumerate(bots):
            perf = bot.get_performance(hours=config.INDIVIDUAL_EVOLUTION_INTERVAL_HOURS)
            if perf['total_trades'] < config.MIN_TRADES_FOR_INDIVIDUAL:
                continue
            if perf['win_rate'] < config.MIN_WIN_RATE_FOR_SURVIVAL or perf['total_pnl'] < 0:
                # Evolve: mutate based on learning data
                suggestions = learning.suggest_mutations(bot.name, bot.strategy_params)
                new_params = bot.mutate(suggestions or bot.strategy_params)  # Fallback to random mutate
                new_gen = bot.generation + 1
                new_name = f"{bot.strategy_type}-v{new_gen}-g{random.randint(100,999)}"
                new_bot = bot.__class__(name=new_name, params=new_params, generation=new_gen, lineage=f"{bot.name} -> {new_name} (individual)")
                
                # Replace in list and DB
                bots[i] = new_bot
                db.retire_bot(bot.name)
                db.save_bot_config(new_name, bot.strategy_type, new_gen, new_params, lineage=new_bot.lineage)
                logger.info(f"Evolved {bot.name} to {new_name} (WR={perf['win_rate']:.2f}, PNL={perf['total_pnl']:.2f})")
        
        individual_evolution_start = datetime.utcnow()
    
    # Overall evolution (as existing, but change to 4 hours)
    if (datetime.utcnow() - evolution_start) >= timedelta(hours=config.OVERALL_EVOLUTION_INTERVAL_HOURS):
        # ... (existing run_evolution)


def main():
    parser = argparse.ArgumentParser(description="Polymarket Bot Arena")
    parser.add_argument("--mode", choices=["paper", "live"], default=None,
                        help="Trading mode (default: from config)")
    parser.add_argument("--setup", action="store_true", help="Run setup verification first")
    args = parser.parse_args()

    if args.mode:
        if args.mode == "live":
            confirm = input("You are switching to LIVE trading with real USDC. Type YES to confirm: ")
            if confirm.strip() != "YES":
                print("Cancelled. Staying in paper mode.")
                sys.exit(0)
        config.set_trading_mode(args.mode)
        logger.info(f"Trading mode set to: {args.mode}")

    if args.setup:
        import setup
        if not setup.main():
            sys.exit(1)

    api_key = load_api_key()
    if not api_key:
        print("No Simmer API key found. Run: python3 setup.py")
        sys.exit(1)

    bots = create_default_bots()

    # Save initial bot configs (only if not already saved)
    existing = {b["bot_name"] for b in db.get_active_bots()}
    for bot in bots:
        if bot.name not in existing:
            db.save_bot_config(bot.name, bot.strategy_type, bot.generation, bot.strategy_params)

    # Load per-bot trading modes from DB
    for bot in bots:
        bot.trading_mode = db.get_bot_mode(bot.name)

    # Backfill learning data from old resolved trades that had no trade_features
    backfilled = learning.backfill_from_resolved_trades()
    if backfilled:
        logger.info(f"Backfilled learning from {backfilled} historical trades")

    main_loop(bots, api_key)


def load_api_key():
    key_path = db.get_arena_state("simmer_key_path")
    if key_path:
        try:
            with open(key_path) as f:
                return json.load(f).get("api_key")
        except:
            pass
    
    # Prompt if not set
    root = tk.Tk()
    root.withdraw()  # Hide main window
    key_path = filedialog.askopenfilename(title="Select Simmer API Key File", filetypes=[("JSON", "*.json")])
    if not key_path:
        print("No file selected. Exiting.")
        sys.exit(1)
    
    db.set_arena_state("simmer_key_path", key_path)
    with open(key_path) as f:
        return json.load(f).get("api_key")


if __name__ == "__main__":
    main()

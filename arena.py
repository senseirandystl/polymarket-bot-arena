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
import re

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
    Returns True if bot can trade, False if it crashes."""
    dummy_market = {"current_price": 0.52, "id": "test", "question": "test"}
    dummy_signals = {"prices": [97000, 97050, 97100], "latest": 97100}
    try:
        result = bot.make_decision(dummy_market, dummy_signals)
        return result.get("action") in ("buy", "skip")
    except Exception as e:
        logger.error(f"  VALIDATION FAILED for {bot.name}: {e}")
        return False


def run_evolution(bots, cycle_number):
    """Run evolution cycle — kill bots below WR threshold, mutate from survivors."""
    logger.info(f"=== Evolution Cycle {cycle_number} ===")

    # Gather performance and classify by WR
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

    # Sort by WR (not P&L) for ranking display
    rankings.sort(key=lambda x: x["win_rate"], reverse=True)

    # Classify bots
    immune = []       # <MIN_TRADES resolved trades — not enough data
    above = []        # WR >= MIN_WIN_RATE with enough trades — survive
    below = []        # WR < MIN_WIN_RATE with enough trades — get replaced
    for r in rankings:
        if r["trades"] < config.MIN_TRADES_FOR_JUDGMENT:
            immune.append(r)
        elif r["win_rate"] >= config.MIN_WIN_RATE:
            above.append(r)
        else:
            below.append(r)

    logger.info("Rankings (WR-based):")
    for r in rankings:
        if r in immune:
            status = "IMMUNE"
        elif r in above:
            status = "SURVIVES"
        else:
            status = "REPLACED"
        logger.info(f"  {r['name']}: WR={r['win_rate']:.1%}, P&L=${r['pnl']:.2f}, Trades={r['trades']} [{status}]")

    # Safety net: if ALL would be killed (no immune + no above), keep best 1 by WR
    if not immune and not above and below:
        best = below.pop(0)  # rankings already sorted by WR desc, so first in below is best
        above.append(best)
        logger.info(f"  Safety net: keeping {best['name']} (best WR {best['win_rate']:.1%}) as sole survivor")

    # If nobody needs replacing, early return
    if not below:
        logger.info("  No bots below threshold — skipping evolution")
        for bot in bots:
            bot.reset_daily()
        return bots

    survivor_names = {r["name"] for r in immune + above}
    replaced_names = {r["name"] for r in below}

    new_bots = []
    for bot in bots:
        if bot.name in survivor_names:
            bot.reset_daily()
            new_bots.append(bot)

    # Create replacements from winners
    winners = [b for b in bots if b.name in survivor_names]
    replaced = [b for b in bots if b.name in replaced_names]

    for dead_bot in replaced:
        parent = random.choice(winners)
        evolved = create_evolved_bot(parent, dead_bot.strategy_type, cycle_number)

        # Inherit the dead bot's API key slot so evolved bot uses same Simmer account
        if hasattr(dead_bot, '_api_key_slot'):
            evolved._api_key_slot = dead_bot._api_key_slot
            logger.info(f"  {evolved.name} inherits slot {dead_bot._api_key_slot} from {dead_bot.name}")

        # Validate the new bot can actually trade before committing
        if not _validate_bot(evolved):
            logger.warning(f"  {evolved.name} failed validation, recreating with pure defaults")
            from bots.bot_momentum import DEFAULT_PARAMS as MOMENTUM_DEFAULTS
            from bots.bot_mean_rev import DEFAULT_PARAMS as MEANREV_DEFAULTS
            from bots.bot_hybrid import DEFAULT_PARAMS as HYBRID_DEFAULTS
            from bots.bot_sentiment import DEFAULT_PARAMS as SENTIMENT_DEFAULTS
            from bots.bot_sniper import DEFAULT_PARAMS as SNIPER_DEFAULTS
            fallback_map = {
                "momentum": MOMENTUM_DEFAULTS, "mean_reversion": MEANREV_DEFAULTS,
                "mean_reversion_sl": MEANREV_DEFAULTS, "mean_reversion_tp": MEANREV_DEFAULTS,
                "sniper": SNIPER_DEFAULTS,
                "sentiment": SENTIMENT_DEFAULTS, "hybrid": HYBRID_DEFAULTS,
            }
            bot_classes = {
                "momentum": MomentumBot, "mean_reversion": MeanRevBot,
                "mean_reversion_sl": MeanRevSLBot, "mean_reversion_tp": MeanRevTPBot,
                "sniper": SniperBot,
                "sentiment": SentimentBot, "hybrid": HybridBot,
            }
            cls = bot_classes.get(dead_bot.strategy_type, MomentumBot)
            fallback_params = fallback_map.get(dead_bot.strategy_type, MOMENTUM_DEFAULTS).copy()
            evolved = cls(
                name=evolved.name, params=fallback_params,
                generation=cycle_number, lineage=f"{parent.name} -> {evolved.name} (fallback)",
            )
            if hasattr(dead_bot, '_api_key_slot'):
                evolved._api_key_slot = dead_bot._api_key_slot

        db.retire_bot(dead_bot.name)
        db.save_bot_config(
            evolved.name, evolved.strategy_type, evolved.generation,
            evolved.strategy_params, evolved.lineage
        )

        new_bots.append(evolved)
        logger.info(f"  Created {evolved.name} (from {parent.name}): {json.dumps(evolved.strategy_params)[:200]}")

    # Log evolution event
    db.log_evolution(
        cycle_number,
        list(survivor_names),
        list(replaced_names),
        [b.name for b in new_bots if b.name not in survivor_names],
        rankings,
    )

    # Final validation: confirm all bots have API slots and can trade
    for bot in new_bots:
        slot = getattr(bot, '_api_key_slot', None)
        logger.info(f"  Post-evolution: {bot.name} ({bot.strategy_type}) slot={slot} params_keys={list(bot.strategy_params.keys())}")

    return new_bots


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


def discover_markets(api_key):
    """Find the active BTC 5-min up/down market."""
    import requests
    markets = []
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.get(
            f"{config.SIMMER_BASE_URL}/api/sdk/markets",
            headers=headers,
            params={"status": "active", "limit": 100},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            markets_list = data if isinstance(data, list) else data.get("markets", [])
            for m in markets_list:
                q = m.get("question", "").lower()
                has_btc = "btc" in q or "bitcoin" in q
                has_5min = any(kw in q for kw in config.TARGET_MARKET_KEYWORDS)
                if has_btc and has_5min:
                    markets.append(m)
    except Exception as e:
        logger.error(f"Market discovery error: {e}")
    logger.info(f"Discovered {len(markets)} BTC 5-min markets")
    return markets


def is_5min_market(question):
    """Check if this is a strict 5-minute window market (not 15-min or hourly)."""
    import re
    q = question.lower()
    # Match patterns like "10:00PM-10:05PM" (5-min range)
    range_match = re.search(r'(\d{1,2}):(\d{2})(am|pm)-(\d{1,2}):(\d{2})(am|pm)', q)
    if range_match:
        h1, m1 = int(range_match.group(1)), int(range_match.group(2))
        h2, m2 = int(range_match.group(4)), int(range_match.group(5))
        ap1, ap2 = range_match.group(3), range_match.group(6)
        if ap1 == 'pm' and h1 != 12: h1 += 12
        if ap2 == 'pm' and h2 != 12: h2 += 12
        if ap1 == 'am' and h1 == 12: h1 = 0
        if ap2 == 'am' and h2 == 12: h2 = 0
        t1 = h1 * 60 + m1
        t2 = h2 * 60 + m2
        diff = t2 - t1
        if diff < 0: diff += 24 * 60
        return diff == 5
    return False


def resolve_trades(bots, api_key):
    """Resolve pending trades from Simmer API and update DB + learning."""
    import requests
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(
        f"{config.SIMMER_BASE_URL}/api/sdk/trades",
        headers=headers,
        params={"status": "resolved", "limit": 200},
        timeout=15,
    )
    if resp.status_code != 200:
        logger.error(f"Resolve trades error: {resp.status_code} {resp.text[:200]}")
        return 0

    data = resp.json()
    trades_list = data if isinstance(data, list) else data.get("trades", [])
    resolved_count = 0
    for t in trades_list:
        trade_id = t.get("trade_id") or t.get("id")
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id=? AND outcome IS NULL",
                (trade_id,)
            ).fetchone()
            if not row:
                continue

            bot_name = row["bot_name"]
            side = row["side"]
            amount = row["amount"]
            shares_bought = row["shares_bought"]
            trade_features = row["trade_features"]
            if trade_features:
                features = json.loads(trade_features)
            else:
                features = learning.extract_features_from_reasoning(row["reasoning"])

            outcome = t.get("outcome")
            if outcome == "win":
                pnl = shares_bought - amount
                won = True if side == "yes" else False
            elif outcome == "loss":
                pnl = -amount
                won = False if side == "yes" else True
            else:
                continue  # Not resolved

            conn.execute(
                "UPDATE trades SET outcome=?, pnl=?, resolved_at=datetime('now') WHERE trade_id=?",
                (outcome, pnl, trade_id)
            )

            learning.record_outcome(bot_name, features, side, won)
            resolved_count += 1
            logger.info(f"[{bot_name}] Resolved {trade_id}: {outcome.upper()} PNL=${pnl:.2f}")

    if resolved_count > 0:
        logger.info(f"Resolved {resolved_count} trades")
    return resolved_count


def expire_stale_trades():
    """Expire trades pending >1h (5-min markets resolve in ~10min)."""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, bot_name, market_id FROM trades WHERE outcome IS NULL AND created_at < datetime('now', '-1 hour')"
        ).fetchall()
        expired = 0
        for r in rows:
            conn.execute(
                "UPDATE trades SET outcome='expired', pnl=0, resolved_at=datetime('now') WHERE id=?",
                (r["id"],)
            )
            expired += 1
        if expired > 0:
            logger.info(f"Expired {expired} stale trades")
        return expired


def load_traded_pairs(bots):
    """Load recent (bot, market) pairs from DB to prevent duplicates on restart."""
    traded = set()
    with db.get_conn() as conn:
        for bot in bots:
            rows = conn.execute(
                "SELECT market_id FROM trades WHERE bot_name=? AND created_at > datetime('now', '-2 hour')",
                (bot.name,)
            ).fetchall()
            for r in rows:
                traded.add((bot.name, r["market_id"]))
    logger.info(f"Loaded {len(traded)} recent traded pairs from DB")
    return traded


def assign_api_slots(bots):
    """Assign API key slots to bots from bot_keys.json."""
    try:
        with open(config.SIMMER_BOT_KEYS_PATH) as f:
            bot_keys = json.load(f)
        num_unique = len(set(bot_keys.values()))
        logger.info(f"Loaded {len(bot_keys)} bot slots ({num_unique} unique accounts)")
    except (FileNotFoundError, json.JSONDecodeError):
        bot_keys = {}
        logger.warning("No bot_keys.json — all bots share default API key")

    # Assign slots 0-3 to the 4 bots
    for i, bot in enumerate(bots):
        slot = f"slot_{i}"
        bot._api_key_slot = slot
        if slot in bot_keys:
            bot.api_key = bot_keys[slot]
            logger.info(f"  {bot.name} assigned to {slot} (independent account)")
        else:
            bot.api_key = None  # Fallback to shared


def main_loop(bots, api_key):
    """Main trading loop: discover markets, trade, resolve, evolve."""
    assign_api_slots(bots)

    price_feed = get_price_feed()
    sentiment_feed = get_sentiment_feed()
    orderflow_feed = get_orderflow_feed()

    # Dedup: load recent traded markets from DB
    traded = load_traded_pairs(bots)

    # Evolution tracking
    evolution_start = datetime.utcnow()
    cycle_number = db.get_arena_state("evolution_cycle", 0) + 1
    individual_evolution_start = datetime.utcnow()

    while True:
        try:
            # Discover active BTC 5-min markets
            all_markets = discover_markets(api_key)
            five_min_markets = [m for m in all_markets if is_5min_market(m.get("question", ""))]
            if not five_min_markets:
                logger.debug("No BTC 5-min markets found, waiting...")
                time.sleep(TRADE_INTERVAL)
                continue

            # Filter for tradeable: >90s remaining, <5min age (fresh window)
            tradeable_markets = []
            for m in five_min_markets:
                resolves_at_str = m.get("resolves_at") or m.get("end_time")
                if not resolves_at_str:
                    continue

                try:
                    resolves_at = datetime.fromisoformat(resolves_at_str.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    time_remaining = (resolves_at - now).total_seconds()

                    if time_remaining is not None and time_remaining >= 3600:
                        logger.debug(f"Skipping long-horizon market (remaining={time_remaining:.0f}s > 60min): {m.get('question', '')[:50]}")
                        continue

                    if time_remaining <= 0:
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
        
        # Overall evolution (every 4 hours)
        if (datetime.utcnow() - evolution_start) >= timedelta(hours=config.OVERALL_EVOLUTION_INTERVAL_HOURS):
            bots = run_evolution(bots, cycle_number)
            evolution_start = datetime.utcnow()
            cycle_number += 1
            db.set_arena_state("evolution_cycle", cycle_number - 1)
            assign_api_slots(bots)  # Reassign after evolution


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


if __name__ == "__main__":
    main()

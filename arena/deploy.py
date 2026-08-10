"""Mid-run bot deploy — dashboard queues strategies; arena folds them in.

Dashboard ``POST /api/bots/deploy`` writes strategy types into arena_state
``pending_bot_deploys``. The coordinator loop (~30s) calls
:func:`process_pending_deploys` to instantiate bots, persist configs, and
hot-swap the trader / maker slates without restart.
"""

from __future__ import annotations

import json
import logging
import time

import db

logger = logging.getLogger("arena.deploy")

PENDING_DEPLOY_KEY = "pending_bot_deploys"
LAST_DEPLOY_KEY = "last_bot_deploy"

# Keep in sync with arena.py MAKER_TYPES (makers run on MakerThread cadence).
MAKER_TYPES = frozenset({
    "late_window_maker", "fee_zone_maker", "btc_maker", "true_maker", "copy_trade",
})


def unique_bot_name(preferred: str, taken: set[str]) -> str:
    """Return preferred if free, else preferred-2, preferred-3, …"""
    if preferred not in taken:
        return preferred
    n = 2
    while f"{preferred}-{n}" in taken:
        n += 1
    return f"{preferred}-{n}"


def process_pending_deploys(
    trader_bots: list,
    maker_bots: list,
    trader,
    pos_monitor,
    *,
    maker_types: frozenset | None = None,
) -> tuple[list, list, dict | None]:
    """Drain dashboard deploy queue into the live trader/maker slates.

    Returns (trader_bots, maker_bots, result_or_None).
    """
    maker_types = maker_types or MAKER_TYPES
    raw = db.get_arena_state(PENDING_DEPLOY_KEY)
    if not raw:
        return trader_bots, maker_bots, None
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        db.set_arena_state(PENDING_DEPLOY_KEY, "")
        return trader_bots, maker_bots, None

    items = payload.get("strategies") or payload.get("items") or []
    if not items:
        db.set_arena_state(PENDING_DEPLOY_KEY, "")
        return trader_bots, maker_bots, None

    # Clear queue first so a crash mid-deploy doesn't infinite-retry a bad type.
    db.set_arena_state(PENDING_DEPLOY_KEY, "")

    from arena.startup import instantiate_strategy, strategy_catalog

    catalog = {e["strategy_type"]: e for e in strategy_catalog()}
    active_cfgs = db.get_active_bots()
    active_types = {c["strategy_type"] for c in active_cfgs}
    taken_names = {c["bot_name"] for c in active_cfgs}
    taken_names.update(b.name for b in trader_bots)
    taken_names.update(b.name for b in maker_bots)

    deployed: list[dict] = []
    skipped: list[dict] = []
    trader_changed = False

    for item in items:
        st = item if isinstance(item, str) else (item or {}).get("strategy_type")
        if not st or not isinstance(st, str):
            skipped.append({"strategy_type": st, "reason": "invalid"})
            continue
        st = st.strip()
        if st not in catalog:
            skipped.append({"strategy_type": st, "reason": "unknown_strategy"})
            continue
        if st in active_types:
            skipped.append({"strategy_type": st, "reason": "already_active"})
            continue
        try:
            preferred = catalog[st]["default_name"]
            bot_name = unique_bot_name(preferred, taken_names)
            bot = instantiate_strategy(st, name=bot_name)
            db.save_bot_config(
                bot.name, bot.strategy_type, bot.generation,
                bot.strategy_params, lineage=getattr(bot, "lineage", None),
            )
            try:
                db.set_bot_mode(bot.name, "paper")
            except Exception:
                pass
            bot.trading_mode = "paper"

            if bot.strategy_type in maker_types:
                maker_bots.append(bot)
            else:
                trader_bots.append(bot)
                trader_changed = True

            taken_names.add(bot.name)
            active_types.add(st)
            deployed.append({
                "bot_name": bot.name,
                "strategy_type": bot.strategy_type,
                "role": "maker" if bot.strategy_type in maker_types else "trader",
            })
            logger.info(
                f"Deployed mid-run: {bot.name} ({bot.strategy_type}) "
                f"role={'maker' if bot.strategy_type in maker_types else 'trader'}"
            )
        except Exception as e:
            logger.error(f"Deploy failed for {st}: {e}", exc_info=True)
            skipped.append({"strategy_type": st, "reason": str(e)})

    if trader_changed:
        trader.set_bots(trader_bots)
        pos_monitor.update_bots(trader_bots)

    result = {
        "deployed": deployed,
        "skipped": skipped,
        "ts": time.time(),
        "ok": bool(deployed),
    }
    try:
        db.set_arena_state(LAST_DEPLOY_KEY, json.dumps(result))
    except Exception:
        pass

    if deployed:
        try:
            from arena import portfolio
            portfolio.rebalance(force=True, reason="mid_run_deploy")
        except Exception as pe:
            logger.warning(f"Post-deploy portfolio rebalance failed: {pe}")

    return trader_bots, maker_bots, result

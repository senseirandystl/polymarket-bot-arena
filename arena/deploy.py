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


def _autopsy_lab_skip(spec_id, reason: str, *, source: str = "lab") -> None:
    """Close a lab hyp that the mid-run queue accepted but deploy refused."""
    if not spec_id:
        return
    try:
        from signals.strategy_pipeline.postmortem import write_autopsy
        from signals.strategy_pipeline.store import HypothesisStore
        write_autopsy(
            HypothesisStore(), str(spec_id),
            stage="backtested", reason=str(reason)[:200],
        )
    except Exception:
        logger.warning(
            "%s hyp autopsy after deploy skip failed spec=%s", source, spec_id,
        )


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
        source = "" if isinstance(item, str) else str((item or {}).get("source") or "")
        spec_id = (item or {}).get("spec_id") if isinstance(item, dict) else None
        # Operator catalog click: one live instance per strategy_type.
        # Lab may run two momentums *with different genes* — exact-param
        # clones are still blocked below.
        if st in active_types and source != "lab":
            skipped.append({"strategy_type": st, "reason": "already_active"})
            continue
        try:
            preferred = catalog[st]["default_name"]
            spec_params = {}
            spec_id = None
            if isinstance(item, dict):
                if item.get("name"):
                    preferred = str(item["name"])
                if isinstance(item.get("params"), dict):
                    spec_params = item["params"]
                spec_id = item.get("spec_id")

            if source == "lab":
                from signals.strategy_pipeline.fingerprint import clone_match

                spec_like = {
                    "primitive": st,
                    "params": spec_params,
                    "spec_id": spec_id,
                }
                live_peers = list(trader_bots) + list(maker_bots) + list(active_cfgs)
                clone = clone_match(spec_like, live_peers)
                if clone:
                    skipped.append({
                        "strategy_type": st,
                        "reason": "clone",
                        "clone_of": clone.get("bot_name"),
                    })
                    logger.info(
                        "Lab deploy skipped clone %s of %s",
                        preferred, clone.get("bot_name"),
                    )
                    _autopsy_lab_skip(
                        spec_id,
                        f"clone_of_active:{clone.get('bot_name') or st}",
                        source=source or "lab",
                    )
                    continue

            bot_name = unique_bot_name(preferred, taken_names)
            if source == "lab":
                from signals.strategy_pipeline.compiler import compile_bot

                bot, _spec = compile_bot({
                    "primitive": st,
                    "name": bot_name,
                    "spec_id": spec_id or f"deploy-{st}",
                    "params": spec_params,
                })
            else:
                bot = instantiate_strategy(st, name=bot_name)
                if spec_id:
                    bot.lineage = f"lab:{spec_id}"
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
            if source == "lab":
                _autopsy_lab_skip(
                    spec_id, f"deploy_failed:{e}", source=source or "lab",
                )

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

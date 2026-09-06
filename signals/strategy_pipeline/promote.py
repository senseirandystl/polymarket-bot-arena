"""Promotion helpers: paper queue, ready graduation, live flip."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import db

from signals.strategy_pipeline.compiler import sanitize_spec
from signals.strategy_pipeline.control import cfg, paper_slots
from signals.strategy_pipeline.fingerprint import is_clone
from signals.strategy_pipeline.founders import is_protected_bot
from signals.strategy_pipeline.postmortem import write_autopsy

logger = logging.getLogger("strategy_pipeline.promote")


def promotion_bars() -> tuple[int, int, int]:
    min_trades = int(cfg("STRATEGY_LAB_PROMOTE_MIN_TRADES", 100))
    min_days = int(cfg("STRATEGY_LAB_PROMOTE_MIN_DAYS", 7))
    floor = int(cfg("STRATEGY_LAB_PROMOTE_TRADE_FLOOR", 30))
    return min_trades, min_days, floor


def count_pipeline_paper_active(store) -> int:
    """Count open lab paper hyps + pending lab deploys (roster hard cap)."""
    n = 0
    try:
        n += len(store.open_by_stage("paper"))
    except Exception:
        pass
    raw = db.get_arena_state("pending_bot_deploys") or ""
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {}
    for item in (payload or {}).get("strategies") or []:
        if isinstance(item, dict) and str(item.get("source") or "") == "lab":
            n += 1
    return n


def queue_paper(store, hyp: dict, backtest: dict) -> bool:
    """Queue a paper deploy if slots allow. Returns False if blocked."""
    slots = paper_slots()
    if slots <= 0:
        return False
    if count_pipeline_paper_active(store) >= slots:
        write_autopsy(
            store,
            hyp["spec_id"],
            stage="backtested",
            reason="paper_slots_full",
            evidence={"slots": slots},
        )
        return False

    spec = hyp.get("spec") if isinstance(hyp.get("spec"), dict) else hyp
    spec = sanitize_spec(spec)
    clone = is_clone(spec, store, extra_peers=_pending_deploy_peers())
    if clone:
        write_autopsy(
            store,
            spec["spec_id"],
            stage="backtested",
            reason=f"clone_of_active:{clone.get('bot_name') or clone.get('strategy_type')}",
            evidence={"clone": clone, "primitive": spec.get("primitive")},
        )
        store.log(
            spec["spec_id"], "trader", "paper", "clone_blocked",
            str(clone.get("bot_name") or clone.get("strategy_type") or ""),
        )
        return False

    bot_name = spec["name"]
    raw = db.get_arena_state("pending_bot_deploys") or ""
    try:
        payload = json.loads(raw) if raw else {"strategies": []}
    except json.JSONDecodeError:
        payload = {"strategies": []}
    items = payload.get("strategies") or []
    items.append({
        "strategy_type": spec["primitive"],
        "name": bot_name,
        "source": "lab",
        "spec_id": spec["spec_id"],
        "params": spec.get("params") or {},
    })
    payload["strategies"] = items
    payload["queued_at"] = time.time()
    db.set_arena_state("pending_bot_deploys", json.dumps(payload))
    store.advance(
        spec["spec_id"], "paper", bot_name=bot_name, backtest_summary=backtest
    )
    store.log(spec["spec_id"], "trader", "paper", "queued", bot_name)
    return True


def approve_to_paper(store, spec_id: str) -> dict[str, Any]:
    """Operator: move a passed backtested hyp into paper when slots allow."""
    spec_id = str(spec_id or "").strip()
    hyp = store.get(spec_id)
    if not hyp or hyp.get("status") != "open":
        return {"ok": False, "reason": "not_found"}
    if hyp.get("stage") != "backtested":
        return {"ok": False, "reason": "not_backtested", "stage": hyp.get("stage")}
    if paper_slots() <= 0:
        return {"ok": False, "reason": "paper_slots_zero"}
    bt = hyp.get("backtest_summary") if isinstance(hyp.get("backtest_summary"), dict) else {}
    ok = queue_paper(store, hyp, bt or {"passed": True})
    if not ok:
        return {"ok": False, "reason": "queue_failed"}
    return {"ok": True, "spec_id": spec_id, "stage": "paper"}


def review_paper(store) -> int:
    """Graduate paper hyps to ready (or reject). Auto-live only if configured."""
    min_trades, min_days, floor = promotion_bars()
    auto_live = bool(cfg("STRATEGY_LAB_AUTO_PROMOTE", False))
    promoted = 0
    for hyp in store.open_by_stage("paper"):
        bot_name = hyp.get("bot_name") or (hyp.get("spec") or {}).get("name")
        if not bot_name:
            continue
        if is_protected_bot(bot_name):
            # Never touch founders via pipeline paper review.
            continue
        stats = _paper_stats(bot_name)
        days = stats["days"]
        n = stats["trades"]
        pnl = stats["pnl"]
        ready = n >= min_trades or (days >= min_days and n >= floor)
        summary = {**stats, "ready": ready}
        store.advance(hyp["spec_id"], "paper", paper_summary=summary)
        if not ready:
            continue
        if pnl <= 0:
            write_autopsy(
                store,
                hyp["spec_id"],
                stage="paper",
                reason=f"paper_pnl_{pnl:.4f}_n={n}",
                evidence=summary,
            )
            continue
        store.advance(
            hyp["spec_id"], "ready", bot_name=bot_name, paper_summary=summary,
        )
        store.log(
            hyp["spec_id"], "risk", "ready", "awaiting_promote",
            f"{bot_name} n={n} days={days:.1f} pnl={pnl:.2f}",
        )
        if auto_live:
            out = promote_to_live(store, hyp["spec_id"])
            if out.get("ok"):
                promoted += 1
    if auto_live:
        for hyp in store.open_by_stage("ready"):
            out = promote_to_live(store, hyp["spec_id"])
            if out.get("ok"):
                promoted += 1
    return promoted


def promote_to_live(store, spec_id: str) -> dict[str, Any]:
    """Operator (or auto) promote: ready → live. Never auto unless called."""
    spec_id = str(spec_id or "").strip()
    if not spec_id:
        return {"ok": False, "reason": "missing_spec_id"}
    hyp = store.get(spec_id)
    if not hyp or hyp.get("status") != "open":
        return {"ok": False, "reason": "not_found"}
    if hyp.get("stage") != "ready":
        return {"ok": False, "reason": "not_ready", "stage": hyp.get("stage")}
    bot_name = hyp.get("bot_name") or (hyp.get("spec") or {}).get("name")
    if not bot_name:
        return {"ok": False, "reason": "no_bot"}
    if is_protected_bot(bot_name):
        return {"ok": False, "reason": "protected_founder"}
    err = _flip_to_live(bot_name)
    if err:
        return {"ok": False, "reason": err}
    store.advance(
        spec_id, "live", status="open", bot_name=bot_name,
        live_summary={"promoted_at": time.time(), "source": "lab"},
    )
    store.log(spec_id, "risk", "live", "promoted", bot_name)
    return {"ok": True, "bot_name": bot_name, "spec_id": spec_id}


def _flip_to_live(bot_name: str) -> str | None:
    try:
        active = {c.get("bot_name") for c in (db.get_active_bots() or [])}
    except Exception as e:
        return f"mode_failed:{e}"
    if bot_name not in active:
        return "bot_missing"
    try:
        db.set_bot_mode(bot_name, "live")
    except Exception as e:
        return f"mode_failed:{e}"
    if str(db.get_bot_mode(bot_name) or "") != "live":
        return "mode_failed"
    return None


def _pending_deploy_peers() -> list[dict]:
    raw = db.get_arena_state("pending_bot_deploys") or ""
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return []
    peers = []
    for item in (payload or {}).get("strategies") or []:
        if isinstance(item, str):
            peers.append({"strategy_type": item, "params": {}})
        elif isinstance(item, dict) and item.get("strategy_type"):
            peers.append({
                "strategy_type": item["strategy_type"],
                "params": item.get("params") if isinstance(item.get("params"), dict) else {},
                "bot_name": item.get("name"),
            })
    return peers


def _paper_stats(bot_name: str) -> dict[str, Any]:
    with db.get_conn() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n,
                      COALESCE(SUM(pnl), 0) AS pnl,
                      MIN(created_at) AS first_at,
                      MAX(created_at) AS last_at
               FROM trades
               WHERE bot_name=? AND outcome IS NOT NULL""",
            (bot_name,),
        ).fetchone()
    n = int(row["n"] or 0) if row else 0
    pnl = float(row["pnl"] or 0.0) if row else 0.0
    days = 0.0
    if row and row["first_at"] and row["last_at"]:
        try:
            from datetime import datetime
            fmt = "%Y-%m-%d %H:%M:%S"
            a = datetime.strptime(str(row["first_at"])[:19], fmt)
            b = datetime.strptime(str(row["last_at"])[:19], fmt)
            days = max(0.0, (b - a).total_seconds() / 86400.0)
        except Exception:
            days = 0.0
    return {"trades": n, "pnl": pnl, "days": days}

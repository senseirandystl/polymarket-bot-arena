"""Desk host: run one six-stage tick, or loop it on a worker thread."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import config
import db

from desk import research as research_mod
from desk.compiler import compile_bot, sanitize_spec
from desk.postmortem import write_autopsy
from desk.roles import FloorSnapshot, RoleStatus, get_floor, set_role
from desk.store import HypothesisStore

logger = logging.getLogger("desk.cycle")


def _cfg(name: str, default):
    return getattr(config, name, default)


def promotion_bars() -> tuple[int, int, int]:
    """min_trades OR min_days, with a hard floor on trades."""
    min_trades = int(_cfg("DESK_PROMOTE_MIN_TRADES", 100))
    min_days = int(_cfg("DESK_PROMOTE_MIN_DAYS", 7))
    floor = int(_cfg("DESK_PROMOTE_TRADE_FLOOR", 30))
    return min_trades, min_days, floor


class DeskHost:
    def __init__(self) -> None:
        self.store = HypothesisStore()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_cycle: dict[str, Any] | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="desk-cycle", daemon=True
        )
        self._thread.start()
        logger.info("Desk cycle host started")

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        interval = float(_cfg("DESK_CYCLE_INTERVAL_SEC", 300.0))
        self._stop.wait(5.0)
        while not self._stop.is_set():
            try:
                self.last_cycle = self.tick()
            except Exception:
                logger.exception("desk tick failed")
                set_role("desk_lead", "blocked", "tick exception")
            self._stop.wait(interval)

    def tick(self) -> dict[str, Any]:
        set_role("desk_lead", "working", "running six-stage tick")
        report: dict[str, Any] = {
            "ts": time.time(),
            "proposed": 0,
            "coded": 0,
            "backtested": 0,
            "rejected": 0,
            "papered": 0,
            "promoted_live": 0,
            "autopsied": 0,
        }

        open_research = self.store.open_by_stage("idea", "researched")
        pending_code = [h for h in open_research if h.get("stage") == "researched"]
        max_open = int(_cfg("DESK_MAX_OPEN_SPECS", 8))
        if len(self.store.open_by_stage(
            "idea", "researched", "coded", "backtested", "paper"
        )) < max_open:
            proposed = research_mod.propose(self.store, max_new=int(_cfg("DESK_MAX_NEW_PER_TICK", 2)))
            report["proposed"] = len(proposed)
            pending_code.extend(proposed)

        for hyp in pending_code:
            spec = hyp.get("spec") if isinstance(hyp.get("spec"), dict) else hyp
            try:
                _, spec = compile_bot(spec)
            except Exception as e:
                write_autopsy(self.store, hyp["spec_id"], stage="coded", reason=f"compile_failed:{e}")
                report["rejected"] += 1
                report["autopsied"] += 1
                continue
            self.store.advance(hyp["spec_id"], "coded", spec_update=spec)
            self.store.log(hyp["spec_id"], "coder", "code", "compiled", spec["primitive"])
            report["coded"] += 1
            set_role("coder", "done", spec["name"])

        for hyp in self.store.open_by_stage("coded"):
            result = self._backtest(hyp)
            report["backtested"] += 1
            if not result.get("passed"):
                write_autopsy(
                    self.store,
                    hyp["spec_id"],
                    stage="backtested",
                    reason=result.get("reason") or "backtest_fail",
                    evidence=result,
                )
                report["rejected"] += 1
                report["autopsied"] += 1
            else:
                self.store.advance(
                    hyp["spec_id"], "backtested", backtest_summary=result
                )
                self._queue_paper(hyp, result)
                report["papered"] += 1

        promoted = self._review_paper()
        report["promoted_live"] = promoted
        autopsied = self._harvest_ga_deaths()
        report["autopsied"] += autopsied
        set_role("desk_lead", "idle", "tick complete")
        self.store.log(None, "desk_lead", "orchestrate", "tick", json.dumps(report))
        return report

    def _backtest(self, hyp: dict) -> dict[str, Any]:
        set_role("backtester", "working", hyp.get("name") or hyp["spec_id"])
        spec = hyp.get("spec") if isinstance(hyp.get("spec"), dict) else hyp
        try:
            bot, spec = compile_bot(spec)
        except Exception as e:
            return {"passed": False, "reason": f"compile:{e}"}
        try:
            from evolution.backtest_gate import evaluate_offspring
            result = evaluate_offspring(bot, baseline_bot=None)
            summary = {
                "passed": bool(getattr(result, "passed", False)),
                "reason": getattr(result, "reason", "") or "",
                "child_pnl": getattr(result, "child_pnl", None),
                "baseline_pnl": getattr(result, "baseline_pnl", None),
                "markets": getattr(result, "markets", 0),
                "elapsed_sec": getattr(result, "elapsed_sec", 0.0),
                "detail": getattr(result, "detail", ""),
                "primitive": spec.get("primitive"),
            }
            if summary["reason"] == "data_unavailable":
                summary["passed"] = False
            set_role("backtester", "done", summary["reason"] or "ok")
            return summary
        except Exception as e:
            logger.warning("desk backtest failed: %s", e)
            return {"passed": False, "reason": f"backtest_error:{e}"}

    def _queue_paper(self, hyp: dict, backtest: dict) -> None:
        spec = hyp.get("spec") if isinstance(hyp.get("spec"), dict) else hyp
        spec = sanitize_spec(spec)
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
            "source": "desk",
            "spec_id": spec["spec_id"],
        })
        payload["strategies"] = items
        payload["queued_at"] = time.time()
        db.set_arena_state("pending_bot_deploys", json.dumps(payload))
        self.store.advance(
            spec["spec_id"], "paper", bot_name=bot_name, backtest_summary=backtest
        )
        self.store.log(spec["spec_id"], "trader", "paper", "queued", bot_name)
        set_role("trader", "working", f"paper {bot_name}")

    def _review_paper(self) -> int:
        min_trades, min_days, floor = promotion_bars()
        promoted = 0
        for hyp in self.store.open_by_stage("paper"):
            bot_name = hyp.get("bot_name") or (hyp.get("spec") or {}).get("name")
            if not bot_name:
                continue
            stats = _paper_stats(bot_name)
            days = stats["days"]
            n = stats["trades"]
            pnl = stats["pnl"]
            ready = n >= min_trades or (days >= min_days and n >= floor)
            summary = {**stats, "ready": ready}
            self.store.advance(hyp["spec_id"], "paper", paper_summary=summary)
            if not ready:
                continue
            if pnl <= 0:
                write_autopsy(
                    self.store,
                    hyp["spec_id"],
                    stage="paper",
                    reason=f"paper_pnl_{pnl:.4f}_n={n}",
                    evidence=summary,
                )
                continue
            auto_live = bool(_cfg("DESK_AUTO_LIVE", False))
            if auto_live:
                db.set_bot_mode(bot_name, "live")
                self.store.advance(
                    hyp["spec_id"], "live", status="open",
                    live_summary={"promoted_at": time.time()},
                )
                self.store.log(hyp["spec_id"], "risk", "live", "promoted", bot_name)
                promoted += 1
                set_role("risk", "done", f"live {bot_name}")
            else:
                self.store.log(
                    hyp["spec_id"], "risk", "paper", "cleared_bar",
                    f"{bot_name} n={n} days={days:.1f} pnl={pnl:.2f} awaiting human live toggle",
                )
                set_role("risk", "idle", f"{bot_name} ready for live toggle")
        return promoted

    def _harvest_ga_deaths(self) -> int:
        n = 0
        try:
            events = db.get_evolution_history(limit=5) if hasattr(db, "get_evolution_history") else []
        except Exception:
            events = []
        retired_names: list[str] = []
        for ev in events or []:
            replaced = ev.get("replaced") if isinstance(ev, dict) else None
            if isinstance(replaced, str):
                try:
                    replaced = json.loads(replaced)
                except json.JSONDecodeError:
                    replaced = []
            if isinstance(replaced, list):
                for item in replaced:
                    if isinstance(item, str):
                        retired_names.append(item)
                    elif isinstance(item, dict) and item.get("name"):
                        retired_names.append(item["name"])
        if not retired_names:
            return 0
        open_live = self.store.open_by_stage("live", "paper")
        by_name = {h.get("bot_name"): h for h in open_live if h.get("bot_name")}
        for name in retired_names:
            hyp = by_name.get(name)
            if not hyp:
                continue
            write_autopsy(
                self.store,
                hyp["spec_id"],
                stage="live",
                reason=f"ga_replaced:{name}",
                evidence={"bot_name": name},
            )
            n += 1
        return n

    def snapshot(self) -> FloorSnapshot:
        roles = [RoleStatus(**{
            "role_id": r.role_id, "title": r.title, "job": r.job,
            "state": r.state, "detail": r.detail, "updated_ts": r.updated_ts,
        }) for r in get_floor().values()]
        from desk import llm
        return FloorSnapshot(
            roles=roles,
            hypotheses=self.store.list(limit=30),
            pipeline_counts=self.store.counts(),
            last_cycle=self.last_cycle,
            provider=llm.configured_provider(),
            factory_mode=bool(_cfg("DESK_FACTORY_MODE", False)),
            universe_phase=int(_cfg("CRYPTO_UNIVERSE_PHASE", 1)),
        )


_HOST: DeskHost | None = None


def get_host() -> DeskHost:
    global _HOST
    if _HOST is None:
        _HOST = DeskHost()
    return _HOST


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

"""Lab host: research → compile → backtest → (optional paper) → ready."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from signals.strategy_pipeline import research as research_mod
from signals.strategy_pipeline.backtest import run_strict_backtest
from signals.strategy_pipeline.compiler import compile_bot
from signals.strategy_pipeline.control import cfg, paper_slots
from signals.strategy_pipeline.founders import mark_founders_in_db
from signals.strategy_pipeline.postmortem import write_autopsy
from signals.strategy_pipeline.promote import (
    approve_to_paper,
    promote_to_live,
    queue_paper,
    review_paper,
)
from signals.strategy_pipeline.store import HypothesisStore

logger = logging.getLogger("strategy_pipeline.cycle")


class LabHost:
    def __init__(self) -> None:
        self.store = HypothesisStore()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_cycle: dict[str, Any] | None = None
        try:
            mark_founders_in_db()
        except Exception:
            pass

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="strategy-lab-cycle", daemon=True
        )
        self._thread.start()
        logger.info("Strategy Lab cycle host started")

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        from signals.strategy_pipeline.control import (
            is_paused, try_acquire_tick_lock, release_tick_lock, save_last_cycle,
        )

        interval = float(cfg("STRATEGY_LAB_CYCLE_INTERVAL_SEC", 300.0))
        self._stop.wait(5.0)
        last = 0.0
        while not self._stop.is_set():
            now = time.time()
            due = (not is_paused()) and (last == 0.0 or (now - last) >= interval)
            if due and try_acquire_tick_lock():
                try:
                    self.last_cycle = self.tick()
                    save_last_cycle(self.last_cycle)
                    last = time.time()
                except Exception:
                    logger.exception("lab tick failed")
                    last = time.time()
                finally:
                    release_tick_lock()
            self._stop.wait(1.0)

    def tick(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "ts": time.time(),
            "proposed": 0,
            "coded": 0,
            "backtested": 0,
            "rejected": 0,
            "papered": 0,
            "ready": 0,
            "promoted_live": 0,
            "autopsied": 0,
            "clones_blocked": 0,
            "held_backtested": 0,
            "paper_slots": paper_slots(),
        }

        open_research = self.store.open_by_stage("idea", "researched")
        pending_code = [h for h in open_research if h.get("stage") == "researched"]
        max_open = int(cfg("STRATEGY_LAB_MAX_OPEN", 8))
        if len(self.store.open_by_stage(
            "idea", "researched", "coded", "backtested", "paper", "ready"
        )) < max_open:
            proposed = research_mod.propose(
                self.store, max_new=int(cfg("STRATEGY_LAB_MAX_NEW_PER_TICK", 2)),
            )
            report["proposed"] = len(proposed)
            pending_code.extend(proposed)

        for hyp in pending_code:
            spec = hyp.get("spec") if isinstance(hyp.get("spec"), dict) else hyp
            try:
                _, spec = compile_bot(spec)
            except Exception as e:
                write_autopsy(
                    self.store, hyp["spec_id"], stage="coded",
                    reason=f"compile_failed:{e}",
                )
                report["rejected"] += 1
                report["autopsied"] += 1
                continue
            self.store.advance(hyp["spec_id"], "coded", spec_update=spec)
            self.store.log(
                hyp["spec_id"], "coder", "code", "compiled", spec["primitive"],
            )
            report["coded"] += 1

        for hyp in self.store.open_by_stage("coded"):
            result = run_strict_backtest(hyp)
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
                continue

            self.store.advance(
                hyp["spec_id"], "backtested", backtest_summary=result,
            )
            slots = paper_slots()
            if slots <= 0:
                # Phase-1 default: stop at backtested; no deploy.
                report["held_backtested"] += 1
                self.store.log(
                    hyp["spec_id"], "trader", "backtested", "held_no_slots",
                    "PAPER_SLOTS=0",
                )
                continue

            queued = queue_paper(self.store, hyp, result)
            if queued:
                report["papered"] += 1
            else:
                report["clones_blocked"] += 1
                report["rejected"] += 1
                report["autopsied"] += 1

        promoted = review_paper(self.store)
        report["promoted_live"] = promoted
        report["ready"] = len(self.store.open_by_stage("ready"))
        self.store.log(None, "lab", "orchestrate", "tick", json.dumps(report))
        try:
            from signals.strategy_pipeline.control import save_last_cycle
            save_last_cycle(report)
        except Exception:
            pass
        return report

    def promote_to_live(self, spec_id: str) -> dict[str, Any]:
        return promote_to_live(self.store, spec_id)

    def approve_to_paper(self, spec_id: str) -> dict[str, Any]:
        return approve_to_paper(self.store, spec_id)

    def snapshot(self) -> dict[str, Any]:
        from signals.strategy_pipeline.control import effective

        overlay = effective()
        last = self.last_cycle or overlay.get("last_cycle")
        return {
            "hypotheses": self.store.list(limit=30),
            "pipeline_counts": self.store.counts(),
            "last_cycle": last,
            "provider": overlay.get("llm_provider"),
            "paused": bool(overlay.get("paused")),
            "enabled": bool(overlay.get("enabled")),
            "auto_promote": bool(overlay.get("auto_promote")),
            "paper_slots": int(overlay.get("paper_slots") or 0),
            "founder_locks": overlay.get("founder_locks") or {},
        }


_HOST: LabHost | None = None


def get_host() -> LabHost:
    global _HOST
    if _HOST is None:
        _HOST = LabHost()
    return _HOST

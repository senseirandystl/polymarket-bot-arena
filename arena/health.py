"""Production health checks + restart recommendations.

Aggregates process liveness, log freshness, risk/kill state, paper pool,
DB reachability, and feed staleness into a single dashboard-friendly report.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import config
import db

logger = logging.getLogger("arena.health")


def _check(name: str, ok: bool, *, level: str = "ok",
           message: str = "", recommend: str = "",
           detail: Optional[dict] = None) -> dict:
    return {
        "name": name,
        "ok": bool(ok),
        "level": level if not ok else "ok",  # ok | warn | critical
        "message": message,
        "recommend": recommend,
        "detail": detail or {},
    }


def check_arena_log() -> dict:
    stale_after = float(os.environ.get(
        "ARENA_LOG_STALE_SEC",
        getattr(config, "ARENA_LOG_STALE_SEC", 300),
    ))
    log_path = Path(config.LOG_DIR) / "arena.log"
    age = None
    try:
        age = time.time() - log_path.stat().st_mtime
    except OSError:
        return _check(
            "arena_log", False, level="critical",
            message="arena.log missing — arena may not be running",
            recommend="Start the arena: launchctl load -w ~/Library/LaunchAgents/"
                      "com.polymarket.botarena.plist  (or ./bin/arena)",
            detail={"path": str(log_path)},
        )
    if age > stale_after:
        return _check(
            "arena_log", False, level="critical",
            message=f"arena.log stale ({age:.0f}s > {stale_after:.0f}s)",
            recommend="Restart arena: launchctl kickstart -k "
                      f"gui/$(id -u)/com.polymarket.botarena  "
                      f"(or run {Path(__file__).resolve().parents[1]}/arena_watchdog.sh)",
            detail={"age_sec": round(age, 1), "stale_after": stale_after},
        )
    return _check(
        "arena_log", True,
        message=f"arena.log fresh ({age:.0f}s ago)",
        detail={"age_sec": round(age, 1)},
    )


def check_db() -> dict:
    try:
        with db.get_conn() as conn:
            conn.execute("SELECT 1").fetchone()
            n = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE outcome IS NULL"
            ).fetchone()[0]
        return _check("database", True, message="SQLite reachable",
                      detail={"open_trades": int(n or 0)})
    except Exception as e:
        return _check(
            "database", False, level="critical",
            message=f"DB error: {e}",
            recommend="Check DB_PATH permissions and disk space; restart arena after fix",
        )


def check_kill_switch() -> dict:
    try:
        from arena.risk_engine import is_killed, load_state
        killed = is_killed()
        st = load_state()
        if killed:
            return _check(
                "kill_switch", False, level="critical",
                message=f"KILL SWITCH armed: {st.get('kill_reason') or 'halted'}",
                recommend="Clear via Overview → Risk → CLEAR KILL SWITCH, or "
                          "delete logs/KILL_SWITCH",
                detail={"reason": st.get("kill_reason"), "source": st.get("kill_source")},
            )
        return _check("kill_switch", True, message="Kill switch clear")
    except Exception as e:
        return _check("kill_switch", False, level="warn",
                      message=f"Could not read kill switch: {e}")


def check_risk_status() -> dict:
    try:
        from arena.risk_engine import load_state
        st = load_state()
        port = st.get("portfolio") or {}
        paused_bots = [
            n for n, b in (st.get("bots") or {}).items()
            if (b or {}).get("status") == "paused"
        ]
        if port.get("status") == "paused":
            return _check(
                "risk", False, level="critical",
                message=f"Portfolio risk-paused: {port.get('reason')}",
                recommend="Review Risk Engine limits / daily P&L; resume only after root cause",
                detail={"portfolio": port, "paused_bots": paused_bots},
            )
        if paused_bots:
            return _check(
                "risk", False, level="warn",
                message=f"{len(paused_bots)} bot(s) risk-paused: {', '.join(paused_bots[:5])}",
                recommend="Inspect per-bot drawdown/daily loss on Overview → Risk",
                detail={"paused_bots": paused_bots},
            )
        dd = float(port.get("drawdown") or 0)
        if dd >= 0.20:
            return _check(
                "risk", False, level="warn",
                message=f"Portfolio drawdown elevated ({dd * 100:.0f}%)",
                recommend="Consider lowering Kelly fraction or enabling portfolio allocation",
                detail={"drawdown": dd},
            )
        return _check("risk", True,
                      message=f"Risk OK (port={port.get('status', 'active')}, "
                              f"dd={dd * 100:.0f}%)",
                      detail={"portfolio": port})
    except Exception as e:
        return _check("risk", False, level="warn", message=str(e))


def check_paper_pool() -> dict:
    try:
        avail = float(db.get_paper_available())
        bankroll = float(db.get_paper_bankroll())
        if avail < 5.0:
            return _check(
                "paper_pool", False, level="critical",
                message=f"Paper available only ${avail:.2f}",
                recommend="Settings → Paper Balance → top up (e.g. $200 one-click)",
                detail={"available": avail, "bankroll": bankroll},
            )
        if avail < 25.0:
            return _check(
                "paper_pool", False, level="warn",
                message=f"Paper pool low (${avail:.2f})",
                recommend="Top up paper balance before size floors thrash fills",
                detail={"available": avail, "bankroll": bankroll},
            )
        return _check("paper_pool", True,
                      message=f"Paper available ${avail:.2f}",
                      detail={"available": avail, "bankroll": bankroll})
    except Exception as e:
        return _check("paper_pool", False, level="warn", message=str(e))


def check_session() -> dict:
    try:
        start = db.get_arena_state("session_start")
        if not start:
            return _check(
                "session", False, level="warn",
                message="No session_start — arena may not have booted this DB",
                recommend="Start/restart the arena process",
            )
        return _check("session", True, message=f"Session since {start}",
                      detail={"session_start": start})
    except Exception as e:
        return _check("session", False, level="warn", message=str(e))


def check_price_feed() -> dict:
    """Best-effort: arena_state heartbeat written by arena.alerts.publish_price_feed_status."""
    try:
        raw = db.get_arena_state("price_feed_status")
        if not raw:
            return _check(
                "price_feed", True, level="ok",
                message="No feed heartbeat stored (check arena.log for Binance WS)",
            )
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return _check("price_feed", True, message="Feed status unreadable")
        # Ignore very old heartbeats as stale (arena not publishing)
        ts = float(data.get("ts") or 0)
        age = time.time() - ts if ts else None
        stale = bool(data.get("stale"))
        if age is not None and age > float(getattr(config, "ALERT_FEED_STALE_SEC", 90)) * 2:
            stale = True
        if stale:
            return _check(
                "price_feed", False, level="warn",
                message="Price feed marked stale / unavailable",
                recommend="Restart arena to reconnect Binance WebSocket",
                detail=data,
            )
        return _check("price_feed", True, message="Feed OK", detail=data)
    except Exception as e:
        return _check("price_feed", True, message=f"feed status n/a ({e})")


def check_disk() -> dict:
    try:
        usage = os.statvfs(str(config.DB_PATH.parent))
        free_gb = (usage.f_bavail * usage.f_frsize) / (1024 ** 3)
        if free_gb < 0.5:
            return _check(
                "disk", False, level="critical",
                message=f"Only {free_gb:.2f} GB free",
                recommend="Free disk space; SQLite + logs will fail hard when full",
                detail={"free_gb": round(free_gb, 2)},
            )
        if free_gb < 2.0:
            return _check(
                "disk", False, level="warn",
                message=f"Low disk ({free_gb:.1f} GB free)",
                recommend="Prune logs/ and old DB backups",
                detail={"free_gb": round(free_gb, 2)},
            )
        return _check("disk", True, message=f"{free_gb:.1f} GB free",
                      detail={"free_gb": round(free_gb, 2)})
    except Exception as e:
        return _check("disk", True, message=f"disk check n/a ({e})")


def run_health_checks() -> dict[str, Any]:
    """Full authenticated health report for the dashboard."""
    checks = [
        check_arena_log(),
        check_db(),
        check_kill_switch(),
        check_risk_status(),
        check_paper_pool(),
        check_session(),
        check_price_feed(),
        check_disk(),
    ]
    critical = [c for c in checks if not c["ok"] and c["level"] == "critical"]
    warns = [c for c in checks if not c["ok"] and c["level"] == "warn"]
    if critical:
        overall = "critical"
    elif warns:
        overall = "degraded"
    else:
        overall = "healthy"

    recommendations = []
    for c in checks:
        if c.get("recommend") and not c["ok"]:
            recommendations.append({
                "from": c["name"],
                "level": c["level"],
                "text": c["recommend"],
            })

    # Restart recommendation summary
    restart = None
    log_c = next((c for c in checks if c["name"] == "arena_log"), None)
    if log_c and not log_c["ok"]:
        restart = {
            "needed": True,
            "urgency": "critical",
            "command": "launchctl kickstart -k gui/$(id -u)/com.polymarket.botarena",
            "alt": "./arena_watchdog.sh",
            "reason": log_c["message"],
        }
    elif critical:
        restart = {
            "needed": True,
            "urgency": "high",
            "command": "launchctl kickstart -k gui/$(id -u)/com.polymarket.botarena",
            "reason": critical[0]["message"],
        }
    else:
        restart = {"needed": False, "urgency": "none",
                   "reason": "No automatic restart indicated"}

    return {
        "status": overall,
        "ts": time.time(),
        "checks": checks,
        "recommendations": recommendations,
        "restart": restart,
        "counts": {
            "ok": sum(1 for c in checks if c["ok"]),
            "warn": len(warns),
            "critical": len(critical),
        },
    }


def maybe_alert_on_health(prev_status: Optional[str] = None) -> Optional[dict]:
    """Run checks; alert if overall status worsens to degraded/critical."""
    report = run_health_checks()
    status = report["status"]
    try:
        last = db.get_arena_state("health_last_status")
        db.set_arena_state("health_last_status", status)
        db.set_arena_state("health_last_report",
                          json.dumps({
                              "status": status,
                              "ts": report["ts"],
                              "counts": report["counts"],
                              "restart": report["restart"],
                          }, default=str))
    except Exception:
        last = prev_status

    if status in ("degraded", "critical") and last != status:
        try:
            from arena.alerts import alert_health
            parts = [f"{c['name']}: {c['message']}"
                     for c in report["checks"] if not c["ok"]]
            alert_health(
                f"Health {status}",
                "\n".join(parts[:8]),
                level="critical" if status == "critical" else "warn",
            )
        except Exception as e:
            logger.debug("health alert failed: %s", e)
    return report

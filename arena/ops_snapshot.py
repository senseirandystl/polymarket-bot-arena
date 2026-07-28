"""Compact ops snapshot for the dashboard command-center strip.

Aggregates regime, risk, portfolio allocation, recent signal contributions,
and health into one payload so the UI can paint a single coherent view.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import db

# Contribution tokens from blend.log_str: P=0.55[drift=+0.12 mom=-0.03 ...]
_CONTRIB_RE = re.compile(r"P=[\d.]+\[([^\]]*)\]")
_PAIR_RE = re.compile(r"([a-zA-Z_]+)=([+-]?[\d.]+)")


def recent_signal_contributions(hours: float = 6.0, limit: int = 200) -> dict[str, Any]:
    """Mean absolute + signed contribution per lane from recent trade reasoning."""
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=float(hours))
    ).strftime("%Y-%m-%d %H:%M:%S")
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT reasoning, side, outcome, bot_name FROM trades
               WHERE created_at >= ? AND reasoning IS NOT NULL
               ORDER BY created_at DESC LIMIT ?""",
            (cutoff, int(limit)),
        ).fetchall()
    sums: dict[str, float] = defaultdict(float)
    abs_sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    n_parsed = 0
    for r in rows:
        text = r["reasoning"] or ""
        m = _CONTRIB_RE.search(text)
        if not m:
            continue
        n_parsed += 1
        for lane, val in _PAIR_RE.findall(m.group(1)):
            v = float(val)
            sums[lane] += v
            abs_sums[lane] += abs(v)
            counts[lane] += 1
    lanes = []
    for lane, n in sorted(counts.items(), key=lambda kv: -abs_sums[kv[0]]):
        lanes.append({
            "lane": lane,
            "n": n,
            "mean": round(sums[lane] / n, 4),
            "mean_abs": round(abs_sums[lane] / n, 4),
        })
    return {
        "hours": hours,
        "trades_scanned": len(rows),
        "trades_with_blend": n_parsed,
        "lanes": lanes,
    }


def ops_snapshot() -> dict[str, Any]:
    """One-shot payload for Overview command center."""
    out: dict[str, Any] = {"ts": time.time()}

    # Regime
    try:
        from signals.regime_detector import get_detector
        st = get_detector().status()
        cur = st.get("current") or {}
        out["regime"] = {
            "id": cur.get("regime_id") or cur.get("label") or "unknown",
            "legacy": cur.get("legacy") or cur.get("regime"),
            "confidence": cur.get("confidence"),
            "meta_bucket": cur.get("meta_bucket"),
            "features": cur.get("features") or {},
        }
    except Exception as e:
        out["regime"] = {"id": "unknown", "error": str(e)}

    # Risk
    try:
        from arena.risk_engine import dashboard_snapshot
        risk = dashboard_snapshot(limit_events=5)
        port = risk.get("portfolio") or {}
        out["risk"] = {
            "enabled": risk.get("enabled"),
            "killed": risk.get("killed"),
            "kill_reason": risk.get("kill_reason"),
            "portfolio_status": port.get("status"),
            "portfolio_dd": port.get("drawdown"),
            "portfolio_daily_pnl": port.get("daily_pnl"),
            "portfolio_var": port.get("var_1d"),
            "paused_bots": [
                n for n, b in (risk.get("bots") or {}).items()
                if (b or {}).get("status") == "paused"
            ],
            "events": risk.get("events") or [],
        }
    except Exception as e:
        out["risk"] = {"error": str(e)}

    # Portfolio allocation
    try:
        from arena.portfolio import load_state
        p = load_state()
        weights = p.get("weights") or {}
        top = sorted(weights.items(), key=lambda kv: -kv[1])[:6]
        out["allocation"] = {
            "enabled": p.get("enabled"),
            "method": p.get("method"),
            "n_active": p.get("n_active"),
            "last_rebalance_at": p.get("last_rebalance_at"),
            "rebalance_reason": p.get("rebalance_reason"),
            "top_weights": [{"bot": k, "weight": v} for k, v in top],
        }
    except Exception as e:
        out["allocation"] = {"error": str(e)}

    # Signal contributions
    try:
        out["signals"] = recent_signal_contributions(hours=6.0)
    except Exception as e:
        out["signals"] = {"error": str(e), "lanes": []}

    # Health (lightweight — full report on /api/health)
    try:
        from arena.health import check_arena_log, check_kill_switch
        log_c = check_arena_log()
        kill_c = check_kill_switch()
        overall = "healthy"
        if not log_c["ok"] or not kill_c["ok"]:
            overall = "critical" if (
                log_c.get("level") == "critical"
                or kill_c.get("level") == "critical"
            ) else "degraded"
        out["health"] = {
            "status": overall,
            "arena_log": log_c,
            "kill_switch": kill_c,
        }
    except Exception as e:
        out["health"] = {"status": "unknown", "error": str(e)}

    # Evolution / GA last cycle
    try:
        cycle = db.get_arena_state("evolution_cycle")
        last_t = db.get_arena_state("last_evolution_time")
        trigger = db.get_arena_state("last_evolution_trigger")
        out["evolution"] = {
            "cycle": int(cycle) if cycle else 0,
            "last_evolution_time": float(last_t) if last_t else None,
            "last_trigger": trigger,
        }
    except Exception as e:
        out["evolution"] = {"error": str(e)}

    # Quick bankroll / kelly for ops strip
    try:
        out["sizing"] = {
            "paper_available": db.get_paper_available(),
            "paper_bankroll": db.get_paper_bankroll(),
            "kelly_fraction": db.get_kelly_fraction(),
        }
    except Exception as e:
        out["sizing"] = {"error": str(e)}

    # Live BTC (and ETH) from arena-written price_feed_status — no WS in dashboard.
    try:
        import json as _json
        raw = db.get_arena_state("price_feed_status")
        pf = _json.loads(raw) if raw else {}
        syms = (pf or {}).get("symbols") or {}
        btc = syms.get("btc") or {}
        eth = syms.get("eth") or {}
        out["prices"] = {
            "btc": btc.get("latest"),
            "btc_stale": bool(btc.get("stale") or pf.get("stale")),
            "btc_age_sec": btc.get("age_sec"),
            "eth": eth.get("latest"),
            "eth_stale": bool(eth.get("stale")),
            "ts": pf.get("ts"),
        }
    except Exception as e:
        out["prices"] = {"error": str(e)}

    return out

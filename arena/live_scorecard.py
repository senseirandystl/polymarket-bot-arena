"""Live unique-market scorecard for Signal Lab.

Judges lanes and gates on ONE row per (strategy, market) — the buy if the
bot traded, otherwise the last skip — so 25 ticks in the same window cannot
outvote a single fill. Net edge is per-share after the crypto taker fee.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import config
import db
import polymarket_fills

logger = logging.getLogger("arena.live_scorecard")

_LANES = ("drift", "mom", "strat", "fut", "tech", "xasset")
_DEADBAND = 0.05


def unique_market_rows(conn, *, hours: float | None = None) -> list[dict]:
    """One resolved decision per (strategy_type, market_id)."""
    where = "market_up IS NOT NULL"
    params: list = []
    if hours is not None and hours > 0:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=float(hours)))
        where += " AND created_at >= ?"
        params.append(cutoff.strftime("%Y-%m-%d %H:%M:%S"))
    sql = f"""
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY COALESCE(strategy_type, ''), market_id
                       ORDER BY CASE WHEN action='buy' THEN 0 ELSE 1 END,
                                id DESC
                   ) AS rn
            FROM decision_events
            WHERE {where}
        )
        SELECT * FROM ranked WHERE rn = 1
    """
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _lane_stats(rows: list[dict], lane: str, deadband: float) -> dict:
    n = correct = 0
    edge_sum = 0.0
    n_edge = 0
    for r in rows:
        reading = r.get(lane)
        if reading is None:
            continue
        try:
            reading = float(reading)
        except (TypeError, ValueError):
            continue
        if abs(reading) < deadband:
            continue
        market_up = bool(r.get("market_up"))
        pred_up = reading > 0
        ok = pred_up == market_up
        n += 1
        correct += int(ok)
        entry = r.get("entry_price")
        try:
            entry = float(entry) if entry is not None else None
        except (TypeError, ValueError):
            entry = None
        if entry is None:
            continue
        side = r.get("side")
        if (pred_up and side == "yes") or ((not pred_up) and side == "no"):
            cost = entry
        else:
            cost = max(0.01, min(0.99, 1.0 - entry))
        fee = polymarket_fills.taker_fee(1.0, cost)
        edge_sum += (1.0 - cost - fee) if ok else (-cost - fee)
        n_edge += 1
    return {
        "markets": n,
        "accuracy": (correct / n) if n else None,
        "net_edge": (edge_sum / n_edge) if n_edge else None,
        "n_priced": n_edge,
    }


def _price_band(entry) -> str:
    """Split gate hyp so 50–58¢ skips are not drowned by 90¢ locks."""
    try:
        px = float(entry) if entry is not None else None
    except (TypeError, ValueError):
        px = None
    if px is None:
        return "unknown"
    lo = float(getattr(config, "GATE_TUNE_MIDBAND_LO", 0.50))
    hi = float(getattr(config, "GATE_TUNE_MIDBAND_HI", 0.58))
    cheap = float(getattr(config, "GATE_TUNE_CHEAP_MAX", 0.62))
    if lo <= px <= hi:
        return "mid_50_58"
    if px <= cheap:
        return "cheap_other"
    return "expensive"


def _summarize_gate_group(grp: list) -> dict:
    n = len(grp)
    wins = [g for g in grp if g.get("would_win") is not None]
    hyps = [float(g["hyp_pnl"]) for g in grp if g.get("hyp_pnl") is not None]
    entries = [
        float(g["entry_price"]) for g in grp if g.get("entry_price") is not None
    ]
    wr = None
    if wins:
        wr = sum(int(g["would_win"]) for g in wins) / len(wins)
    return {
        "markets": n,
        "ticks": n,
        "wr": wr,
        "avg_hyp_pnl": (sum(hyps) / len(hyps)) if hyps else None,
        "avg_entry": (sum(entries) / len(entries)) if entries else None,
        "n_hyp": len(hyps),
    }


def _gate_stats(rows: list[dict]) -> dict:
    buckets: dict[str, list] = {}
    for r in rows:
        if (r.get("action") or "skip") == "buy":
            key = "buy"
        else:
            key = r.get("skip_reason") or "skip"
        buckets.setdefault(key, []).append(r)
    out = {}
    for key, grp in buckets.items():
        rec = _summarize_gate_group(grp)
        bands: dict[str, list] = {}
        for g in grp:
            bands.setdefault(_price_band(g.get("entry_price")), []).append(g)
        rec["by_band"] = {
            b: _summarize_gate_group(gg) for b, gg in bands.items()
        }
        out[key] = rec
    return out


def _fill_stats(conn, *, hours: float | None = None) -> dict:
    where = "outcome IN ('win','loss','exit_tp','exit_sl')"
    params: list = []
    if hours is not None and hours > 0:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=float(hours)))
        where += " AND created_at >= ?"
        params.append(cutoff.strftime("%Y-%m-%d %H:%M:%S"))
    rows = conn.execute(
        f"""SELECT bot_name, COUNT(*) n,
                   SUM(CASE WHEN outcome IN ('win','exit_tp') THEN 1 ELSE 0 END) wins,
                   SUM(pnl) pnl, AVG(entry_price) avg_entry,
                   SUM(COALESCE(fee,0)) fees
            FROM trades WHERE {where}
            GROUP BY bot_name""",
        params,
    ).fetchall()
    by_bot = {}
    for r in rows:
        n = int(r["n"] or 0)
        by_bot[r["bot_name"]] = {
            "n": n,
            "wr": (int(r["wins"] or 0) / n) if n else None,
            "pnl": float(r["pnl"] or 0.0),
            "avg_entry": float(r["avg_entry"]) if r["avg_entry"] is not None else None,
            "fees": float(r["fees"] or 0.0),
        }
    return by_bot


def _exchange_side_stats(conn, *, hours: float | None = None) -> dict:
    """Resolved fill WR/P&L by (exchange, side) for Kalshi NO shadowing."""
    where = "outcome IN ('win','loss','exit_tp','exit_sl')"
    params: list = []
    if hours is not None and hours > 0:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=float(hours)))
        where += " AND created_at >= ?"
        params.append(cutoff.strftime("%Y-%m-%d %H:%M:%S"))
    rows = conn.execute(
        f"""SELECT CASE WHEN market_id LIKE 'kalshi:%' THEN 'kalshi'
                        ELSE 'polymarket' END AS ex,
                   LOWER(side) AS side,
                   COUNT(*) n,
                   SUM(CASE WHEN outcome IN ('win','exit_tp') THEN 1 ELSE 0 END) wins,
                   SUM(pnl) pnl, AVG(entry_price) avg_entry
            FROM trades WHERE {where}
            GROUP BY 1, 2""",
        params,
    ).fetchall()
    out: dict = {}
    for r in rows:
        n = int(r["n"] or 0)
        key = f"{r['ex']}_{r['side']}"
        out[key] = {
            "exchange": r["ex"],
            "side": r["side"],
            "n": n,
            "wr": (int(r["wins"] or 0) / n) if n else None,
            "pnl": float(r["pnl"] or 0.0),
            "avg_entry": float(r["avg_entry"]) if r["avg_entry"] is not None else None,
        }
    return out


def build_live_scorecard(*, hours: float | None = 72.0,
                         conn=None) -> dict:
    """Return the unique-market lane/gate scorecard (and persist it)."""
    deadband = float(getattr(config, "LANE_MONITOR_DEADBAND", _DEADBAND))
    own = conn is None
    if own:
        ctx = db.get_conn()
        conn = ctx.__enter__()
    try:
        rows = unique_market_rows(conn, hours=hours)
        raw_n = conn.execute(
            "SELECT COUNT(*) c FROM decision_events WHERE market_up IS NOT NULL"
        ).fetchone()["c"]
        lanes = {ln: _lane_stats(rows, ln, deadband) for ln in _LANES}
        gates = _gate_stats(rows)
        fills = _fill_stats(conn, hours=hours)
        exchange_side = _exchange_side_stats(conn, hours=hours)
        mkts = {r["market_id"] for r in rows}
        # Always show the raw skip mix so a wipe with unresolved rows
        # is not a blank Signal Lab.
        skip_where = "1=1"
        skip_params: list = []
        if hours is not None and hours > 0:
            from datetime import datetime, timedelta, timezone
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=float(hours)))
            skip_where = "created_at >= ?"
            skip_params.append(cutoff.strftime("%Y-%m-%d %H:%M:%S"))
        raw_skips = {}
        for r in conn.execute(
            f"""SELECT COALESCE(skip_reason, action) k, COUNT(*) n
                FROM decision_events WHERE {skip_where}
                GROUP BY 1 ORDER BY n DESC""",
            skip_params,
        ):
            raw_skips[r["k"]] = int(r["n"])
        n_total = conn.execute(
            f"SELECT COUNT(*) c FROM decision_events WHERE {skip_where}",
            skip_params,
        ).fetchone()["c"]
        report = {
            "lanes": lanes,
            "gates": gates,
            "fills": fills,
            "exchange_side": exchange_side,
            "raw_skips": raw_skips,
            "meta": {
                "hours": hours,
                "unique_markets": len(mkts),
                "unique_rows": len(rows),
                "resolved_ticks": int(raw_n or 0),
                "n_decisions": int(n_total or 0),
                "ts": time.time(),
            },
        }
    finally:
        if own:
            ctx.__exit__(None, None, None)
    try:
        db.set_arena_state("live_scorecard", json.dumps(report))
    except Exception as e:
        logger.debug("live_scorecard persist failed: %s", e)
    return report


def maybe_refresh(force: bool = False) -> Optional[dict]:
    """Cadence-gated refresh for the evolution loop."""
    iv = float(getattr(config, "LIVE_SCORECARD_INTERVAL_SEC", 300) or 300)
    try:
        raw = db.get_arena_state("live_scorecard")
        prev = json.loads(raw) if raw else {}
    except Exception:
        prev = {}
    ts = float((prev.get("meta") or {}).get("ts") or 0.0)
    if not force and (time.time() - ts) < iv:
        return None
    try:
        return build_live_scorecard(
            hours=float(getattr(config, "LIVE_SCORECARD_HOURS", 72) or 72),
        )
    except Exception as e:
        logger.warning("live scorecard failed: %s", e)
        return None

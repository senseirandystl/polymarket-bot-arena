"""Counterfactual decision log — evaluate signals beyond placed trades.

Hot path
--------
``enqueue(...)`` is O(1): append to a bounded deque + optional throttle.
A daemon flusher batch-INSERTs every ``DECISION_LOG_FLUSH_SEC``. Never blocks
the 1s trader on SQLite (except a full queue drop of oldest events).

Offline
-------
``resolve_pending(market_outcomes)`` stamps ``market_up`` / ``would_win`` /
``hyp_pnl`` once Polymarket resolves a market.

``rollup()`` aggregates resolved decisions for:
  * core-lane accuracy per strategy (incl. skips that carried lane reads)
  * candidate-lane shadow accuracy + net edge
  * skip-reason counterfactual WR
  * optional feed into ``bot_learning`` feature counts

Wired into core_lane_tuner / lane_promoter when ``DECISION_LEARN_FROM_ALL``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from typing import Any, Optional

import config
import db
import polymarket_fills

logger = logging.getLogger("arena.decision_log")

_queue: deque = deque()
_queue_lock = threading.Lock()
_throttle: dict[tuple, float] = {}  # (bot, market) -> last_ts for non-buy
_throttle_lock = threading.Lock()
_flusher_started = False
_flusher_lock = threading.Lock()
_last_rollup = 0.0

_CAND_RE = re.compile(
    r"cand\(fut=([+-][\d.]+) tech=([+-][\d.]+) xa=([+-][\d.]+)"
    r"(?: lag=([+-][\d.]+))?(?: ms=([+-][\d.]+))?(?: fd=([+-][\d.]+))?\)")
_CAND_LANE_GROUP = {
    "fut": 1, "tech": 2, "xasset": 3,
    "lag": 4, "ms_mom": 5, "flow_decay": 6,
}
_LANE_RE = {
    "drift": re.compile(r"drift=([+-][\d.]+)"),
    "mom": re.compile(r"(?<![a-z])mom=([+-][\d.]+)"),
    "strat": re.compile(r"strat=([+-][\d.]+)"),
    "model": re.compile(r"model=([+-][\d.]+)"),
}


def _start_flusher() -> None:
    global _flusher_started
    with _flusher_lock:
        if _flusher_started:
            return
        t = threading.Thread(target=_flush_loop, name="decision-log-flush",
                             daemon=True)
        t.start()
        _flusher_started = True


def _flush_loop() -> None:
    interval = float(getattr(config, "DECISION_LOG_FLUSH_SEC", 2.0))
    while True:
        time.sleep(max(0.5, interval))
        try:
            flush()
        except Exception as e:
            logger.debug("decision_log flush error: %s", e)


def classify_skip_reason(reasoning: str | None) -> Optional[str]:
    """Match trader.py skip buckets for consistent telemetry."""
    why = (reasoning or "").lower()
    if not why:
        return "skip"
    if "dead-zone" in why or "dead zone" in why:
        return "dead_zone"
    if "extreme-drift" in why or "extreme drift" in why:
        return "extreme_drift"
    if "macro-release" in why or "macro" in why:
        return "macro"
    if "consensus" in why:
        return "consensus"
    if "high-price" in why:
        return "high_price"
    if "model lean" in why or "weak lean" in why:
        return "weak_lean"
    if "no edge" in why:
        return "no_edge"
    if "book inconsistency" in why:
        return "book"
    if "exposure" in why:
        return "exposure_cap"
    if "learned_skip" in why or "learned skip" in why:
        return "learned_skip"
    if "ask gap" in why:
        return "ask_quality"
    return "skip"


def _f(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _extract_lanes(signal: dict) -> dict[str, Optional[float]]:
    """Prefer structured signal fields; fall back to reasoning tokens."""
    sigs = signal.get("signals") or {}
    out = {
        "drift": _f(sigs.get("drift")),
        "mom": _f(sigs.get("mom")),
        "strat": _f(sigs.get("strat")),
        "fut": _f(sigs.get("fut") if "fut" in sigs else sigs.get("fut_taker")),
        "tech": _f(sigs.get("tech") if "tech" in sigs else sigs.get("tech_mtf")),
        "xasset": _f(sigs.get("xasset") if "xasset" in sigs else sigs.get("xa")),
        "model_prob": _f(sigs.get("model_prob")),
        "trust_eff": _f(sigs.get("trust_eff")),
    }
    text = signal.get("reasoning") or ""
    if out["drift"] is None:
        m = _LANE_RE["drift"].search(text)
        if m:
            out["drift"] = float(m.group(1))
    if out["mom"] is None:
        m = _LANE_RE["mom"].search(text)
        if m:
            out["mom"] = float(m.group(1))
    if out["strat"] is None:
        m = _LANE_RE["strat"].search(text)
        if m:
            out["strat"] = float(m.group(1))
    if out["model_prob"] is None:
        m = _LANE_RE["model"].search(text)
        if m:
            out["model_prob"] = float(m.group(1))
    if out["fut"] is None or out["tech"] is None or out["xasset"] is None:
        m = _CAND_RE.search(text)
        if m:
            if out["fut"] is None:
                out["fut"] = float(m.group(1))
            if out["tech"] is None:
                out["tech"] = float(m.group(2))
            if out["xasset"] is None:
                out["xasset"] = float(m.group(3))
    return out


def enqueue(
    *,
    bot_name: str,
    strategy_type: str | None,
    market_id: str,
    signal: dict,
    trade_id: int | str | None = None,
    force: bool = False,
) -> bool:
    """Queue one decision for batch insert. Returns True if accepted.

    Hot-path safe: no SQLite, no network. Throttles non-buy actions.
    """
    if not getattr(config, "DECISION_LOG_ENABLED", True):
        return False
    if not market_id or not bot_name:
        return False

    action = (signal.get("action") or "skip").lower()
    if action not in ("buy", "skip", "hold"):
        action = "skip"

    # Throttle non-buy re-evals of the same (bot, market).
    if action != "buy" and not force:
        key = (bot_name, market_id)
        min_iv = float(getattr(config, "DECISION_LOG_MIN_INTERVAL_SEC", 20.0))
        now = time.time()
        with _throttle_lock:
            last = _throttle.get(key, 0.0)
            if (now - last) < min_iv:
                return False
            _throttle[key] = now
            # Bound throttle map growth (markets rotate every 5m).
            if len(_throttle) > 4000:
                cutoff = now - 600
                dead = [k for k, ts in _throttle.items() if ts < cutoff]
                for k in dead:
                    _throttle.pop(k, None)

    lanes = _extract_lanes(signal)
    side = signal.get("side")
    if side not in ("yes", "no"):
        side = None
    skip_reason = None
    if action != "buy":
        skip_reason = classify_skip_reason(signal.get("reasoning"))

    feats = signal.get("features")
    if feats is not None and not isinstance(feats, str):
        try:
            feats = json.dumps(feats)
        except (TypeError, ValueError):
            feats = None

    regime = None
    if isinstance(signal.get("signals"), dict):
        regime = signal["signals"].get("regime")
    if not regime and signal.get("reasoning"):
        m = re.search(r"reg=([a-z0-9_]+)", signal["reasoning"] or "")
        if m:
            regime = m.group(1)

    row = {
        "bot_name": bot_name,
        "strategy_type": strategy_type,
        "market_id": market_id,
        "action": action,
        "side": side,
        "skip_reason": skip_reason,
        "edge": _f(signal.get("edge")),
        "confidence": _f(signal.get("confidence")),
        "entry_price": _f(signal.get("entry_price")),
        "drift": lanes["drift"],
        "mom": lanes["mom"],
        "strat": lanes["strat"],
        "fut": lanes["fut"],
        "tech": lanes["tech"],
        "xasset": lanes["xasset"],
        "model_prob": lanes["model_prob"],
        "trust_eff": lanes["trust_eff"],
        "regime": regime if isinstance(regime, str) else None,
        "features": feats,
        "trade_id": int(trade_id) if trade_id not in (None, "") else None,
    }

    qmax = int(getattr(config, "DECISION_LOG_QUEUE_MAX", 8000))
    with _queue_lock:
        if len(_queue) >= qmax:
            try:
                _queue.popleft()
            except IndexError:
                pass
        _queue.append(row)
    _start_flusher()
    return True


def flush() -> int:
    """Drain the queue into SQLite. Returns rows inserted."""
    batch: list[dict] = []
    with _queue_lock:
        while _queue:
            batch.append(_queue.popleft())
    if not batch:
        return 0
    with db.get_conn() as conn:
        conn.executemany(
            """INSERT INTO decision_events (
                   bot_name, strategy_type, market_id, action, side, skip_reason,
                   edge, confidence, entry_price,
                   drift, mom, strat, fut, tech, xasset,
                   model_prob, trust_eff, regime, features, trade_id
               ) VALUES (
                   :bot_name, :strategy_type, :market_id, :action, :side,
                   :skip_reason, :edge, :confidence, :entry_price,
                   :drift, :mom, :strat, :fut, :tech, :xasset,
                   :model_prob, :trust_eff, :regime, :features, :trade_id
               )""",
            batch,
        )
    return len(batch)


def _hyp_pnl(side: str, market_up: bool, entry_price: float | None) -> float | None:
    """Per-share hypothetical PnL if we had bought ``side`` at entry.

    Hold-to-resolution: direction alone is not enough — cost must be known.
    When ``entry_price`` is missing, return None rather than assuming 0.50
    (that default inflated extreme_drift skip CF to ~+47¢ overnight while
    features said price_very_high).
    """
    if entry_price is None:
        return None
    cost = max(0.01, min(0.99, float(entry_price)))
    fee = polymarket_fills.fee_per_share(
        cost,
        is_maker=(getattr(config, "ORDER_STYLE", "limit") == "limit"
                  and getattr(config, "LIMIT_PRICE_MODE", "passive_mid")
                  in ("passive_mid", "join_bid")),
    )
    won = (side == "yes" and market_up) or (side == "no" and not market_up)
    return (1.0 - cost - fee) if won else (-cost - fee)


def resolve_pending(market_outcomes: dict[str, bool]) -> int:
    """Stamp unresolved decision_events from a condition_id → market_up map.

    ``market_outcomes`` values: True = UP won, False = DOWN won.
    """
    if not market_outcomes:
        return 0
    n = 0
    # Collect learning updates AFTER the write txn commits — nested
    # get_conn() inside an open write transaction deadlocks SQLite.
    learn_jobs: list[tuple] = []
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT id, market_id, side, entry_price, features, bot_name,
                      action
               FROM decision_events WHERE market_up IS NULL
               AND market_id IN ({})""".format(
                ",".join("?" * len(market_outcomes))
            ),
            tuple(market_outcomes.keys()),
        ).fetchall()
        for r in rows:
            mid = r["market_id"]
            if mid not in market_outcomes:
                continue
            market_up = bool(market_outcomes[mid])
            side = r["side"]
            would_win = None
            hyp = None
            if side in ("yes", "no"):
                would_win = int(
                    (side == "yes" and market_up)
                    or (side == "no" and not market_up)
                )
                hyp = _hyp_pnl(side, market_up, r["entry_price"])
            conn.execute(
                """UPDATE decision_events
                   SET market_up=?, would_win=?, hyp_pnl=?,
                       resolved_at=datetime('now')
                   WHERE id=?""",
                (1 if market_up else 0, would_win, hyp, r["id"]),
            )
            n += 1
            # Feature learning for SKIPS only — buys already update
            # bot_learning via the trade resolver (avoid double-count).
            if (r["action"] != "buy" and side in ("yes", "no")
                    and would_win is not None and r["features"]):
                learn_jobs.append(
                    (r["bot_name"], r["features"], side, bool(would_win)))
    for bot_name, feats_raw, side, won in learn_jobs:
        try:
            feats = json.loads(feats_raw) if isinstance(feats_raw, str) else feats_raw
            if feats:
                import learning as _learning
                _learning.record_outcome(bot_name, feats, side, won)
        except Exception:
            pass
    if n:
        logger.info("Resolved %d decision_events against market outcomes", n)
    return n


def resolve_from_resolution_map(resolved: dict) -> int:
    """Adapter for polymarket_markets.recent_resolutions() shape."""
    # resolved: condition_id -> True/False (UP/DOWN)
    clean = {k: bool(v) for k, v in (resolved or {}).items() if k}
    return resolve_pending(clean)


# ---------------------------------------------------------------------------
# Attribution helpers (for core tuner / promoter)
# ---------------------------------------------------------------------------

def core_lane_attribution(conn, deadband: float = 0.05,
                          *, strategy_type: str | None = None) -> dict:
    """{strategy_type: {lane: {n, accuracy}}} from resolved decision_events.

    Uses ALL sided-or-lane-bearing decisions, not just placed trades.
    """
    lanes = ("drift", "mom", "strat")
    q = """SELECT strategy_type, market_up, drift, mom, strat
           FROM decision_events
           WHERE market_up IS NOT NULL
             AND (drift IS NOT NULL OR mom IS NOT NULL OR strat IS NOT NULL)"""
    params: list = []
    if strategy_type:
        q += " AND strategy_type=?"
        params.append(strategy_type)
    rows = conn.execute(q, params).fetchall()
    agg: dict = {}
    for r in rows:
        st = r["strategy_type"] or "unknown"
        market_up = bool(r["market_up"])
        for lane in lanes:
            reading = r[lane]
            if reading is None:
                continue
            reading = float(reading)
            if abs(reading) < deadband:
                continue
            cell = agg.setdefault(st, {}).setdefault(lane, {"n": 0, "correct": 0})
            cell["n"] += 1
            cell["correct"] += int((reading > 0) == market_up)
    out: dict = {}
    for st, ld in agg.items():
        out[st] = {}
        for lane, c in ld.items():
            out[st][lane] = {
                "n": c["n"],
                "accuracy": (c["correct"] / c["n"]) if c["n"] else None,
            }
    return out


def candidate_lane_attribution(conn, lane: str, deadband: float = 0.05,
                               *, since: str | None = None) -> dict:
    """Shadow accuracy + net edge for fut/tech/xasset from decision_events."""
    col = {"fut": "fut", "tech": "tech", "xasset": "xasset"}.get(lane)
    if not col:
        return {"n": 0, "accuracy": None, "net_edge": None}
    q = f"""SELECT market_up, side, entry_price, {col} AS reading
            FROM decision_events
            WHERE market_up IS NOT NULL AND {col} IS NOT NULL"""
    params: list = []
    if since:
        q += " AND created_at >= ?"
        params.append(since)
    rows = conn.execute(q, params).fetchall()
    n = correct = 0
    edge_sum = 0.0
    for r in rows:
        reading = float(r["reading"])
        if abs(reading) < deadband:
            continue
        market_up = bool(r["market_up"])
        pred_up = reading > 0
        ok = pred_up == market_up
        n += 1
        correct += int(ok)
        # Hyp cost of following the lane sign
        entry = r["entry_price"]
        try:
            entry = float(entry) if entry is not None else 0.5
        except (TypeError, ValueError):
            entry = 0.5
        side = r["side"]
        if (pred_up and side == "yes") or ((not pred_up) and side == "no"):
            cost = entry
        else:
            cost = max(0.01, min(0.99, 1.0 - entry))
        fee = polymarket_fills.taker_fee(1.0, cost)
        edge_sum += ((1.0 - cost - fee) if ok else (-cost - fee))
    return {
        "n": n,
        "accuracy": (correct / n) if n else None,
        "net_edge": (edge_sum / n) if n else None,
    }


def resolved_count(conn) -> int:
    row = conn.execute(
        "SELECT COUNT(*) c FROM decision_events WHERE market_up IS NOT NULL"
    ).fetchone()
    return int(row["c"] if row else 0)


def rollup() -> dict[str, Any]:
    """Aggregate resolved decisions; persist to arena_state ``decision_rollup``."""
    deadband = float(getattr(config, "LANE_MONITOR_DEADBAND", 0.05))
    with db.get_conn() as conn:
        n_total = conn.execute(
            "SELECT COUNT(*) c FROM decision_events").fetchone()["c"]
        n_res = resolved_count(conn)
        core = core_lane_attribution(conn, deadband)
        candidates = {
            lane: candidate_lane_attribution(conn, lane, deadband)
            for lane in ("fut", "tech", "xasset")
        }
        # Skip-reason counterfactual: among skips with a side, what WR?
        skip_cf = {}
        for r in conn.execute(
            """SELECT skip_reason,
                      COUNT(*) n,
                      SUM(CASE WHEN would_win=1 THEN 1 ELSE 0 END) wins,
                      AVG(hyp_pnl) avg_hyp
               FROM decision_events
               WHERE market_up IS NOT NULL AND action='skip' AND side IS NOT NULL
               GROUP BY skip_reason"""
        ).fetchall():
            n = int(r["n"] or 0)
            wins = int(r["wins"] or 0)
            skip_cf[r["skip_reason"] or "skip"] = {
                "n": n,
                "wr": (wins / n) if n else None,
                "avg_hyp_pnl": (round(float(r["avg_hyp"]), 4)
                                if r["avg_hyp"] is not None else None),
            }
        # Action mix
        by_action = {}
        for r in conn.execute(
            """SELECT action, COUNT(*) n FROM decision_events GROUP BY action"""
        ).fetchall():
            by_action[r["action"]] = int(r["n"])

        # Per-strategy buy vs skip hyp (resolved)
        by_strat = {}
        for r in conn.execute(
            """SELECT strategy_type, action,
                      COUNT(*) n,
                      AVG(CASE WHEN would_win IS NOT NULL
                               THEN would_win END) wr,
                      AVG(hyp_pnl) avg_hyp
               FROM decision_events
               WHERE market_up IS NOT NULL AND side IS NOT NULL
               GROUP BY strategy_type, action"""
        ).fetchall():
            st = r["strategy_type"] or "unknown"
            by_strat.setdefault(st, {})[r["action"]] = {
                "n": int(r["n"]),
                "wr": (round(float(r["wr"]), 3) if r["wr"] is not None else None),
                "avg_hyp_pnl": (round(float(r["avg_hyp"]), 4)
                                if r["avg_hyp"] is not None else None),
            }

    report = {
        "n_total": n_total,
        "n_resolved": n_res,
        "by_action": by_action,
        "core_lanes": core,
        "candidate_lanes": candidates,
        "skip_counterfactual": skip_cf,
        "by_strategy": by_strat,
        "ts": time.time(),
    }
    try:
        db.set_arena_state("decision_rollup", json.dumps(report))
    except Exception as e:
        logger.debug("decision_rollup persist failed: %s", e)
    return report


def maybe_rollup(force: bool = False) -> Optional[dict]:
    """Cadence-gated rollup for the evolution loop."""
    global _last_rollup
    iv = float(getattr(config, "DECISION_ROLLUP_INTERVAL_SEC", 900))
    now = time.time()
    if not force and (now - _last_rollup) < iv:
        return None
    _last_rollup = now
    try:
        flush()
        return rollup()
    except Exception as e:
        logger.warning("decision rollup failed: %s", e)
        return None


def should_use_decision_attribution(conn=None) -> bool:
    """True when enough resolved decisions exist to replace trade-only paths."""
    if not getattr(config, "DECISION_LEARN_FROM_ALL", True):
        return False
    min_n = int(getattr(config, "DECISION_LEARN_MIN_RESOLVED", 30))
    if conn is not None:
        return resolved_count(conn) >= min_n
    with db.get_conn() as c:
        return resolved_count(c) >= min_n

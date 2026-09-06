"""Compact ops snapshot for the dashboard command-center strip.

Aggregates regime, risk, portfolio allocation, lane health, and health into
one payload so the UI can paint a single coherent view.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import db

# Contribution tokens from blend.log_str: P=0.55[drift=+0.12 mom=-0.03 ...]
# (kept for recent_signal_contributions legacy helper / tests)
_CONTRIB_RE = re.compile(r"P=[\d.]+\[([^\]]*)\]")
_PAIR_RE = re.compile(r"([a-zA-Z_]+)=([+-]?[\d.]+)")
# Core lane raw reads as logged by base_bot.make_decision (same tokens as
# arena/core_lane_tuner.py — keep format in lockstep).
_CORE_LANE_RE = {
    "drift": re.compile(r"\bdrift=([+-][\d.]+)"),
    "mom": re.compile(r"\bmom=([+-][\d.]+)"),
    "strat": re.compile(r"\bstrat=([+-][\d.]+)"),
}
# Shadow candidate reads (pre kill-switch). Tokens:
# cand(fut=.. tech=.. xa=.. lag=.. ms=.. fd=..)
_CAND_RE = re.compile(
    r"cand\(fut=([+-][\d.]+) tech=([+-][\d.]+) xa=([+-][\d.]+)"
    r"(?: lag=([+-][\d.]+))?(?: ms=([+-][\d.]+))?(?: fd=([+-][\d.]+))?\)")
_CAND_LANES = (
    (1, "fut"), (2, "tech"), (3, "xasset"),
    (4, "lag"), (5, "ms_mom"), (6, "flow_decay"),
)
# Display order: live core first, then candidates.
_LANE_ORDER = (
    "drift", "mom", "strat",
    "fut", "tech", "xasset", "lag", "ms_mom", "flow_decay",
)
_CORE_LANES = frozenset({"drift", "mom", "strat"})


def _row_get(r: Any, key: str, default=None):
    """sqlite3.Row-safe getter (supports dicts in tests)."""
    try:
        if hasattr(r, "keys") and key in r.keys():
            return r[key]
    except Exception:
        pass
    if isinstance(r, dict):
        return r.get(key, default)
    try:
        return r[key]
    except (KeyError, IndexError, TypeError):
        return default


def _parse_lane_readings(text: str) -> dict[str, float]:
    """Extract core + candidate lane readings from a trade reasoning string."""
    out: dict[str, float] = {}
    if not text:
        return out
    for lane, rx in _CORE_LANE_RE.items():
        m = rx.search(text)
        if m:
            try:
                out[lane] = float(m.group(1))
            except (TypeError, ValueError):
                pass
    cm = _CAND_RE.search(text)
    if cm:
        for group, lane in _CAND_LANES:
            try:
                raw = cm.group(group)
            except IndexError:
                raw = None
            if raw is None:
                continue
            try:
                out[lane] = float(raw)
            except (TypeError, ValueError):
                pass
    return out


def _empty_side_cell() -> dict[str, Any]:
    return {
        "n": 0, "correct": 0, "cost_sum": 0.0, "edge_sum": 0.0,
        "entry_sum": 0.0, "entry_n": 0,
    }


def _finalize_side(cell: dict[str, Any]) -> dict[str, Any]:
    n = int(cell["n"] or 0)
    if n <= 0:
        return {"n": 0, "wr": None, "be_gap": None, "net_cents": None}
    wr = float(cell["correct"]) / n
    avg_cost = float(cell["cost_sum"]) / n
    return {
        "n": n,
        "wr": round(wr, 4),
        "be_gap": round(wr - avg_cost, 4),
        "net_cents": round(100.0 * float(cell["edge_sum"]) / n, 2),
    }


def _lane_live_status(lane: str, overrides: Optional[dict] = None) -> str:
    """live = carries decision weight; shadow = monitor-only / kill-switched."""
    if lane in _CORE_LANES:
        return "live"
    ov = (overrides or {}).get(lane) or {}
    if ov.get("enabled") is True:
        return "live"
    return "shadow"


def _accumulate_reading(
    store: dict[str, dict[str, Any]],
    lane: str,
    reading: float,
    *,
    market_up: bool,
    side: str,
    entry: float,
    mode: str,
    deadband: float,
) -> None:
    """Update per-lane lean-side stats for one resolved observation."""
    if abs(reading) < deadband:
        return
    pred_up = reading > 0
    side_l = (side or "").lower()
    traded_with = (
        (pred_up and side_l == "yes") or ((not pred_up) and side_l == "no")
    )
    if mode == "trade" and not traded_with:
        return

    # Cost of following the lane (follow) or actual entry (trade-with).
    if mode == "trade" or traded_with:
        cost = max(0.01, min(0.99, float(entry)))
    else:
        # Opposite side of the filled trade ≈ 1 − entry (same proxy as
        # lane_monitor shadow net edge).
        cost = max(0.01, min(0.99, 1.0 - float(entry)))

    try:
        import polymarket_fills
        fee = float(polymarket_fills.taker_fee(1.0, cost))
    except Exception:
        fee = 0.0

    ok = pred_up == market_up
    edge = (1.0 - cost - fee) if ok else (-cost - fee)

    lean = "up" if pred_up else "down"
    if lane not in store:
        store[lane] = {"up": _empty_side_cell(), "down": _empty_side_cell()}
    cell = store[lane][lean]
    cell["n"] += 1
    cell["correct"] += int(ok)
    cell["cost_sum"] += cost
    cell["edge_sum"] += edge
    if mode == "trade" or traded_with:
        cell["entry_sum"] += float(entry)
        cell["entry_n"] += 1


def lane_health_matrix(
    hours: float = 12.0,
    limit: int = 500,
    deadband: Optional[float] = None,
    min_n: int = 5,
) -> dict[str, Any]:
    """Per-lane sign health split by lean side (UP / DOWN).

    Default metric family matches lane_monitor / promoter: when
    ``|reading| ≥ deadband``, does the sign match market direction
    (UP iff YES won or NO lost)?

    Two modes are always returned:
      * **follow** — every resolved trade with a lane read (signal honesty)
      * **trade** — only when the bot bought the side the lane leaned toward
        (did we extract it?)

    Each side cell: ``n``, ``wr``, ``be_gap`` (WR − avg cost of that lean),
    ``net_cents`` (per-share EV after taker fee, in cents).
    """
    try:
        import config as _cfg
        db_dead = float(getattr(_cfg, "LANE_MONITOR_DEADBAND", 0.05))
    except Exception:
        db_dead = 0.05
    deadband = float(deadband if deadband is not None else db_dead)

    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=float(hours))
    ).strftime("%Y-%m-%d %H:%M:%S")
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT reasoning, side, outcome, entry_price, bot_name
               FROM trades
               WHERE created_at >= ?
                 AND outcome IN ('win', 'loss')
                 AND reasoning IS NOT NULL
               ORDER BY created_at DESC LIMIT ?""",
            (cutoff, int(limit)),
        ).fetchall()

    overrides: dict = {}
    try:
        overrides = db.get_lane_overrides() or {}
    except Exception:
        overrides = {}

    follow_b: dict[str, dict] = {}
    trade_b: dict[str, dict] = {}

    n_with_lanes = 0
    for r in rows:
        text = _row_get(r, "reasoning") or ""
        readings = _parse_lane_readings(text)
        if not readings:
            continue
        n_with_lanes += 1
        side = (_row_get(r, "side") or "").lower()
        outcome = (_row_get(r, "outcome") or "").lower()
        if side not in ("yes", "no") or outcome not in ("win", "loss"):
            continue
        market_up = (side == "yes") == (outcome == "win")
        try:
            entry = float(_row_get(r, "entry_price") or 0.5)
        except (TypeError, ValueError):
            entry = 0.5
        entry = max(0.01, min(0.99, entry))

        for lane, reading in readings.items():
            _accumulate_reading(
                follow_b, lane, reading,
                market_up=market_up, side=side, entry=entry,
                mode="follow", deadband=deadband,
            )
            _accumulate_reading(
                trade_b, lane, reading,
                market_up=market_up, side=side, entry=entry,
                mode="trade", deadband=deadband,
            )

    def _build_lanes(store: dict) -> list[dict[str, Any]]:
        rows_out: list[dict[str, Any]] = []
        seen = set(store.keys())
        ordered = [l for l in _LANE_ORDER if l in seen] + sorted(
            seen - set(_LANE_ORDER)
        )
        for lane in ordered:
            up = _finalize_side(store[lane]["up"])
            down = _finalize_side(store[lane]["down"])
            n = up["n"] + down["n"]
            if n <= 0:
                continue
            correct = 0
            edge_sum = 0.0
            cost_sum = 0.0
            for lean in ("up", "down"):
                c = store[lane][lean]
                correct += int(c["correct"])
                edge_sum += float(c["edge_sum"])
                cost_sum += float(c["cost_sum"])
            wr = correct / n
            avg_cost = cost_sum / n
            rows_out.append({
                "lane": lane,
                "status": _lane_live_status(lane, overrides),
                "n": n,
                "wr": round(wr, 4),
                "be_gap": round(wr - avg_cost, 4),
                "net_cents": round(100.0 * edge_sum / n, 2),
                "up": up,
                "down": down,
            })
        order_idx = {name: i for i, name in enumerate(_LANE_ORDER)}
        rows_out.sort(
            key=lambda x: (
                0 if x["status"] == "live" else 1,
                order_idx.get(x["lane"], 99),
                -int(x.get("n") or 0),
            )
        )
        return rows_out

    return {
        "kind": "lane_health",
        "hours": hours,
        "deadband": deadband,
        "min_n": int(min_n),
        "default_mode": "follow",
        "trades_scanned": len(rows),
        "trades_with_lanes": n_with_lanes,
        "modes": {
            "follow": {
                "label": "Follow accuracy",
                "hint": "Sign of lane vs market outcome when |lane| ≥ deadband",
                "lanes": _build_lanes(follow_b),
            },
            "trade": {
                "label": "When we traded with it",
                "hint": "Only trades where bot side matched the lane lean",
                "lanes": _build_lanes(trade_b),
            },
        },
    }


def recent_signal_contributions(hours: float = 6.0, limit: int = 200) -> dict[str, Any]:
    """Legacy mean-contribution bars (tests / any old clients).

    Overview now uses :func:`lane_health_matrix`. This helper remains so
    existing unit tests and any external consumer keep working.
    """
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
    shadow_sums: dict[str, float] = defaultdict(float)
    shadow_abs: dict[str, float] = defaultdict(float)
    shadow_counts: dict[str, int] = defaultdict(int)
    n_parsed = 0
    n_cand = 0
    for r in rows:
        text = _row_get(r, "reasoning") or ""
        m = _CONTRIB_RE.search(text)
        if m:
            n_parsed += 1
            for lane, val in _PAIR_RE.findall(m.group(1)):
                v = float(val)
                sums[lane] += v
                abs_sums[lane] += abs(v)
                counts[lane] += 1
        cm = _CAND_RE.search(text)
        if cm:
            n_cand += 1
            for group, lane in _CAND_LANES:
                try:
                    raw = cm.group(group)
                except IndexError:
                    raw = None
                if raw is None:
                    continue
                v = float(raw)
                shadow_sums[lane] += v
                shadow_abs[lane] += abs(v)
                shadow_counts[lane] += 1
    lanes = []
    for lane, n in sorted(counts.items(), key=lambda kv: -abs_sums[kv[0]]):
        lanes.append({
            "lane": lane,
            "n": n,
            "mean": round(sums[lane] / n, 4),
            "mean_abs": round(abs_sums[lane] / n, 4),
            "source": "blend",
        })
    for lane, n in sorted(shadow_counts.items(),
                          key=lambda kv: -shadow_abs[kv[0]]):
        if lane in counts:
            continue
        lanes.append({
            "lane": lane,
            "n": n,
            "mean": round(shadow_sums[lane] / n, 4),
            "mean_abs": round(shadow_abs[lane] / n, 4),
            "source": "shadow",
        })
    return {
        "hours": hours,
        "trades_scanned": len(rows),
        "trades_with_blend": n_parsed,
        "trades_with_cand": n_cand,
        "lanes": lanes,
    }


def ops_snapshot() -> dict[str, Any]:
    """One-shot payload for Overview command center."""
    out: dict[str, Any] = {"ts": time.time()}

    # Regime (+ relative calibration / adapt policy for Overview)
    try:
        from signals.regime_detector import get_detector
        st = get_detector().status()
        cur = st.get("current") or {}
        feats = cur.get("features") or {}
        out["regime"] = {
            "id": cur.get("regime_id") or cur.get("label") or "unknown",
            "legacy": cur.get("legacy") or cur.get("regime"),
            "confidence": cur.get("confidence"),
            "held_sec": cur.get("held_sec"),
            "actionable": cur.get("actionable"),
            "meta_bucket": cur.get("meta_bucket"),
            "features": feats,
            "vol_abs": feats.get("vol_abs", feats.get("vol")),
            "vol_rel": feats.get("vol_rel"),
            "direction": feats.get("direction"),
            "chop": feats.get("chop"),
        }
        try:
            from signals.regime_calibration import get_calibrator
            out["regime"]["calibration"] = get_calibrator().status()
        except Exception:
            pass
        try:
            from arena.regime_adapt import snapshot as _ra_snap
            out["regime"]["adapt"] = _ra_snap()
        except Exception:
            pass
        try:
            from arena.regime_continuous import status as _rc_st
            out["regime"]["continuous"] = _rc_st()
        except Exception:
            pass
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
        # Full weight vector (desc) so the Overview bar/list can sum to 100%.
        # Previously truncated to 6 and left a visible gap when n_active > 6.
        ranked = sorted(weights.items(), key=lambda kv: -float(kv[1] or 0))
        out["allocation"] = {
            "enabled": p.get("enabled"),
            "method": p.get("method"),
            "n_active": p.get("n_active") or len(ranked),
            "last_rebalance_at": p.get("last_rebalance_at"),
            "rebalance_reason": p.get("rebalance_reason"),
            "top_weights": [
                {"bot": k, "weight": float(v or 0)} for k, v in ranked
            ],
        }
    except Exception as e:
        out["allocation"] = {"error": str(e)}

    # Lane health matrix (replaces mean-contribution bars on Overview)
    try:
        out["signals"] = lane_health_matrix(hours=12.0)
    except Exception as e:
        out["signals"] = {
            "kind": "lane_health", "error": str(e),
            "modes": {"follow": {"lanes": []}, "trade": {"lanes": []}},
        }

    # Health — same overall status as /api/health (not a log-only subset).
    # The ops ribbon "Health" chip must match the Health card + hero ticker.
    try:
        from arena.health import run_health_checks
        report = run_health_checks()
        checks = {c.get("name"): c for c in (report.get("checks") or [])
                  if isinstance(c, dict)}
        out["health"] = {
            "status": report.get("status") or "unknown",
            "counts": report.get("counts"),
            "arena_log": checks.get("arena_log") or {},
            "kill_switch": checks.get("kill_switch") or {},
            "restart": report.get("restart"),
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

    # Live BTC (and ETH) from arena-written price_feed_status (dashboard
    # also has a browser RTDS socket; this is the SQLite fallback path).
    try:
        import json as _json
        raw = db.get_arena_state("price_feed_status")
        pf = _json.loads(raw) if raw else {}
        syms = (pf or {}).get("symbols") or {}
        btc = syms.get("btc") or {}
        eth = syms.get("eth") or {}
        # Display BTC = TWAP when available (Polymarket Current Price parity).
        btc_display = btc.get("display_price") or btc.get("twap") or btc.get("latest")
        btc_stale = bool(
            btc.get("twap_stale") if btc.get("twap") else btc.get("stale")
        ) or bool(pf.get("stale"))
        out["prices"] = {
            "btc": btc_display,
            "btc_twap": btc.get("twap"),
            "btc_spot": btc.get("latest"),
            "btc_display_source": btc.get("display_source") or (
                "twap" if btc.get("twap") else "spot"
            ),
            "btc_stale": btc_stale,
            "btc_age_sec": btc.get("twap_age_sec") or btc.get("age_sec"),
            "eth": eth.get("latest"),
            "eth_stale": bool(eth.get("stale")),
            "ts": pf.get("ts"),
        }
    except Exception as e:
        out["prices"] = {"error": str(e)}

    # Paper gate profile (Pass A) — Overview chip + /api/ops consumers
    try:
        import config as _cfg
        if hasattr(_cfg, "paper_gate_snapshot"):
            out["paper_gates"] = _cfg.paper_gate_snapshot()
        else:
            out["paper_gates"] = {
                "profile": getattr(_cfg, "PAPER_GATE_PROFILE", "off"),
                "active": bool(getattr(_cfg, "paper_gates_active", lambda: False)()),
                "overrides": {},
            }
        out["paper_gate_profile"] = out["paper_gates"].get("profile")
        out["paper_gates_active"] = bool(out["paper_gates"].get("active"))
    except Exception as e:
        out["paper_gates"] = {"active": False, "profile": "off", "error": str(e)}
        out["paper_gate_profile"] = "off"
        out["paper_gates_active"] = False

    return out

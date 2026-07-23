"""Live lane monitor — the DEMOTION half of the lane-promotion pipeline.

The harness (tools/validate_signals.py) promotes candidate lanes on
backfilled data; this module demotes them on live ground truth. Its reason
to exist: the 2026-07-19 run approved the tech lane at a harness-measured
74-80% follow-WR, and live it scored 51.7% direction-accuracy over 209
trades. Harness numbers carry adverse-selection and stale-mid optimism the
live tape doesn't — so an approved lane must keep re-earning its weight.

Every directional trade logs the raw candidate-lane reads in its reasoning
string (``cand(fut=+0.96 tech=+0.25 xa=+0.33)``, written by
base_bot.make_decision *pre kill-switch*, so readings exist whether or not
the lane carries weight). For each ENABLED lane override, the monitor:

1. collects resolved trades placed after the lane's ``approved_at``,
2. scores the lane's sign against the actual market direction
   (UP iff a YES trade won or a NO trade lost), ignoring readings inside
   the deadband,
3. once ``LANE_MONITOR_MIN_TRADES`` readings accumulate, auto-disables the
   lane via db.disable_lane_override() if accuracy is below
   ``LANE_MONITOR_MIN_ACCURACY`` — the same safety hatch as the dashboard
   Disable button, picked up by the hot path within HOTPATH_CACHE_TTL_SEC.

The latest per-lane report is persisted to arena_state key 'lane_monitor'
so the dashboard Signal Lab can display live accuracy next to the
harness metrics.
"""

import json
import logging
import re

import config
import db

logger = logging.getLogger("arena.lane_monitor")

# Raw candidate reads as logged by base_bot.make_decision. The reasoning
# token for xasset is "xa" — map it back to the lane name.
_CAND_RE = re.compile(
    r"cand\(fut=([+-][\d.]+) tech=([+-][\d.]+) xa=([+-][\d.]+)\)")
_LANE_GROUP = {"fut": 1, "tech": 2, "xasset": 3}


def _lane_accuracy(rows, lane: str, deadband: float) -> dict:
    """Sign-vs-outcome accuracy of one lane over resolved trade rows.

    A row counts when the lane's logged reading is outside the deadband;
    "correct" means the reading's sign matched the market's actual
    direction (UP iff a YES trade won or a NO trade lost).
    """
    group = _LANE_GROUP[lane]
    n = correct = 0
    for r in rows:
        m = _CAND_RE.search(r["reasoning"] or "")
        if not m:
            continue
        reading = float(m.group(group))
        if abs(reading) < deadband:
            continue
        market_up = (r["side"] == "yes") == (r["outcome"] == "win")
        n += 1
        correct += ((reading > 0) == market_up)
    return {"n": n, "accuracy": (correct / n) if n else None}


def check_lanes() -> dict:
    """Score every enabled lane override live; disable the ones that fail.

    Returns the per-lane report (also persisted to arena_state) so callers
    and tests can inspect the outcome. Safe to call on any cadence — it
    only reads resolved trades and flips overrides.
    """
    overrides = db.get_lane_overrides()
    enabled = {k: v for k, v in overrides.items() if v.get("enabled")}
    report: dict = {}
    if not enabled:
        db.set_arena_state("lane_monitor", json.dumps(report))
        return report

    min_trades = getattr(config, "LANE_MONITOR_MIN_TRADES", 50)
    min_acc = getattr(config, "LANE_MONITOR_MIN_ACCURACY", 0.53)
    deadband = getattr(config, "LANE_MONITOR_DEADBAND", 0.05)

    with db.get_conn() as conn:
        for lane, ov in enabled.items():
            if lane not in _LANE_GROUP:
                continue  # future lanes without a logged reasoning token
            rows = conn.execute(
                """SELECT side, outcome, reasoning FROM trades
                   WHERE outcome IN ('win', 'loss') AND created_at >= ?
                     AND reasoning LIKE '%cand(%'""",
                (ov.get("approved_at") or "1970-01-01",),
            ).fetchall()
            stats = _lane_accuracy(rows, lane, deadband)
            verdict = "collecting"
            if stats["n"] >= min_trades:
                if stats["accuracy"] >= min_acc:
                    verdict = "healthy"
                else:
                    verdict = "disabled"
            report[lane] = {**stats, "verdict": verdict,
                            "min_trades": min_trades, "min_accuracy": min_acc}

    # Disabling writes arena_state — do it outside the read connection.
    for lane, r in report.items():
        if r["verdict"] == "disabled":
            db.disable_lane_override(lane)
            logger.warning(
                f"Lane '{lane}' AUTO-DISABLED: live accuracy "
                f"{r['accuracy']:.1%} over {r['n']} resolved trades "
                f"(bar {r['min_accuracy']:.0%} after {r['min_trades']})"
            )
        else:
            acc = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "n/a"
            logger.info(
                f"Lane '{lane}': live accuracy {acc} "
                f"over {r['n']} trades [{r['verdict']}]"
            )

    db.set_arena_state("lane_monitor", json.dumps(report))
    return report

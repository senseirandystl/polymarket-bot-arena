"""Live-gated auto-approve — the PROMOTION half of the closed loop.

The pipeline has three actors that must agree before a candidate lane carries
live weight:

1. The harness (tools/validate_signals.py) NOMINATES a lane by filing a pending
   ``lane_proposals`` row — offline, backfilled, and optimistic (stale-mid +
   adverse-selection inflate its numbers; it approved tech at 74-80% follow-WR
   that scored 51.7% live).
2. This promoter JUDGES that nomination against LIVE ground truth. Every
   directional trade logs the raw candidate-lane reads in its reasoning
   ``cand(fut=.. tech=.. xa=..)`` string *pre kill-switch*, so a shadow lane at
   weight 0 still accumulates a live sign-vs-outcome record. The promoter scores
   that record; a lane is only auto-approved once it clears a LIVE bar
   (``AUTO_APPROVE_MIN_ACCURACY`` over ``AUTO_APPROVE_MIN_TRADES`` readings) that
   is deliberately stricter than the demotion floor the monitor uses to keep it
   alive — hysteresis, so a borderline lane can't flap.
3. arena/lane_monitor.py DEMOTES an approved lane the moment its live accuracy
   decays below the survival floor.

The operator toggle (arena_state 'auto_approve_lanes', dashboard Signal Lab)
decides whether step 2 actually flips the proposal or merely annotates it with
the live evidence and waits for a human click. Either way the evidence attached
to every proposal is LIVE, never the harness number alone.

This module writes only lane_proposals / arena_state — never trade tables. Safe
to call on any cadence; it piggybacks the evolution loop next to the monitor.
"""

import logging

import config
import db
# Reuse the monitor's battle-tested reasoning parser so promotion and demotion
# score a lane identically — one definition of "the lane was right".
from arena.lane_monitor import _CAND_RE, _LANE_GROUP, _lane_accuracy

logger = logging.getLogger("arena.lane_promoter")


def _shadow_accuracy(conn, lane: str, deadband: float) -> dict:
    """Live sign-vs-outcome accuracy of a (still-shadow) candidate lane.

    Unlike the monitor — which scores only trades placed *after* a lane went
    live — a pending lane has no approval time, so we score every resolved
    directional trade that carries a cand(...) read. That is the lane's live
    predictiveness measured with zero live weight.
    """
    rows = conn.execute(
        """SELECT side, outcome, reasoning FROM trades
           WHERE outcome IN ('win', 'loss') AND reasoning LIKE '%cand(%'"""
    ).fetchall()
    return _lane_accuracy(rows, lane, deadband)


def check_proposals() -> dict:
    """Annotate every pending proposal with live evidence; auto-approve the ones
    that clear the live bar when the toggle is on.

    Returns a per-proposal report {lane: {...}} for callers/tests/dashboard.
    """
    pending = [p for p in db.get_lane_proposals(status="pending")]
    report = {}
    if not pending:
        return report

    auto_on = db.get_auto_approve_lanes()
    min_trades = getattr(config, "AUTO_APPROVE_MIN_TRADES", 60)
    min_acc = getattr(config, "AUTO_APPROVE_MIN_ACCURACY", 0.55)
    max_active = getattr(config, "AUTO_APPROVE_MAX_ACTIVE", 3)
    deadband = getattr(config, "LANE_MONITOR_DEADBAND", 0.05)

    # Count CANDIDATE lanes already live so we never blow past the concentration
    # cap. Core-lane overrides (drift/mom/strat, written by the core-lane tuner)
    # share the same store but are NOT candidate lanes — they must not count
    # against the candidate cap or a tuned core lane would block promotion.
    active = sum(1 for k, v in db.get_lane_overrides().items()
                 if v.get("enabled") and k in _LANE_GROUP)

    with db.get_conn() as conn:
        for p in pending:
            lane = p["lane"]
            if lane not in _LANE_GROUP:
                # No logged reasoning token yet — cannot verify live, leave it
                # for a human (or a future reasoning-string extension).
                report[lane] = {"verdict": "unverifiable", "proposal_id": p["id"]}
                continue
            stats = _shadow_accuracy(conn, lane, deadband)
            acc = stats["accuracy"]
            clears = (stats["n"] >= min_trades and acc is not None
                      and acc >= min_acc)
            verdict = "collecting"
            if stats["n"] >= min_trades:
                verdict = "clears_bar" if clears else "below_bar"
            report[lane] = {
                **stats, "verdict": verdict, "proposal_id": p["id"],
                "min_trades": min_trades, "min_accuracy": min_acc,
                "auto_approve": auto_on,
            }

    # Persist live evidence onto each proposal (outside the read connection) so
    # the dashboard shows live numbers next to the harness metrics regardless of
    # the toggle.
    for lane, r in report.items():
        pid = r.get("proposal_id")
        if pid is not None and "accuracy" in r:
            db.annotate_lane_proposal(pid, {
                "n": r["n"], "accuracy": r["accuracy"],
                "min_trades": r["min_trades"], "min_accuracy": r["min_accuracy"],
            })

    if not auto_on:
        for lane, r in report.items():
            if r.get("verdict") == "clears_bar":
                logger.info(
                    f"Lane '{lane}' clears the live bar "
                    f"({r['accuracy']:.1%}/{r['n']}) but auto-approve is OFF — "
                    f"awaiting human decision in Signal Lab"
                )
        return report

    # Auto-approve pass — bounded: one lane per call, respect the active cap.
    for lane, r in report.items():
        if r.get("verdict") != "clears_bar":
            continue
        if active >= max_active:
            logger.info(
                f"Lane '{lane}' clears the live bar but the active-lane cap "
                f"({max_active}) is full — not promoting"
            )
            continue
        try:
            db.decide_lane_proposal(r["proposal_id"], "approve")
        except ValueError as e:
            logger.warning(f"Auto-approve of '{lane}' skipped: {e}")
            continue
        active += 1
        r["verdict"] = "auto_approved"
        logger.warning(
            f"Lane '{lane}' AUTO-APPROVED: live accuracy {r['accuracy']:.1%} "
            f"over {r['n']} shadow reads (bar {min_acc:.0%} after {min_trades})"
        )
        break  # one promotion per cycle — let the monitor watch it before more

    return report

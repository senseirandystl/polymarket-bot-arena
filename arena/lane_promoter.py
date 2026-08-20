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

    Prefers decision_events (buys + throttled skips) so promotion uses *all*
    evaluations, not only filled trades. Falls back to trade cand(...) parse.
    """
    try:
        from arena.decision_log import (
            candidate_lane_attribution, should_use_decision_attribution,
        )
        if should_use_decision_attribution(conn):
            stats = candidate_lane_attribution(conn, lane, deadband)
            if (stats.get("n") or 0) > 0:
                return stats
        # Even when below the global floor, merge decision samples if any.
        dec = candidate_lane_attribution(conn, lane, deadband)
    except Exception:
        dec = {"n": 0, "accuracy": None, "net_edge": None}

    rows = conn.execute(
        """SELECT side, outcome, reasoning, entry_price FROM trades
           WHERE outcome IN ('win', 'loss') AND reasoning LIKE '%cand(%'"""
    ).fetchall()
    trade_stats = _lane_accuracy(rows, lane, deadband)
    # Prefer the larger sample; decision_events typically win once logging runs.
    if (dec.get("n") or 0) >= (trade_stats.get("n") or 0) and (dec.get("n") or 0) > 0:
        return dec
    return trade_stats


def check_proposals() -> dict:
    """Annotate every pending proposal with live evidence; auto-approve the ones
    that clear the live bar when the toggle is on.

    Returns a per-proposal report {lane: {...}} for callers/tests/dashboard.
    """
    pending = [p for p in db.get_lane_proposals(status="pending")]
    report: dict = {}
    if not pending:
        return report

    auto_on = db.get_auto_approve_lanes()
    min_trades = getattr(config, "AUTO_APPROVE_MIN_TRADES", 60)
    min_acc = getattr(config, "AUTO_APPROVE_MIN_ACCURACY", 0.55)
    min_net = getattr(config, "AUTO_APPROVE_MIN_NET_EDGE", 0.005)
    min_fills = int(getattr(config, "AUTO_APPROVE_MIN_FILLS", 15) or 15)
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
            net = stats.get("net_edge")
            # Fill-level bar: harness + tick shadow cannot promote alone.
            fill = {"n": 0, "accuracy": None, "net_edge": None}
            try:
                trows = conn.execute(
                    """SELECT side, outcome, reasoning, entry_price FROM trades
                       WHERE outcome IN ('win', 'loss') AND reasoning LIKE '%cand(%'"""
                ).fetchall()
                fill = _lane_accuracy(trows, lane, deadband)
            except Exception:
                pass
            fill_n = int(fill.get("n") or 0)
            fill_net = fill.get("net_edge")
            fill_ok = (
                fill_n >= min_fills
                and fill_net is not None
                and float(fill_net) >= float(min_net)
            )
            shadow_ok = (
                stats["n"] >= min_trades
                and acc is not None and acc >= min_acc
                and net is not None and net >= min_net
            )
            clears = shadow_ok and fill_ok
            if stats["n"] < min_trades or fill_n < min_fills:
                verdict = "collecting"
            else:
                verdict = "clears_bar" if clears else "below_bar"
            report[lane] = {
                **stats, "verdict": verdict, "proposal_id": p["id"],
                "min_trades": min_trades, "min_accuracy": min_acc,
                "min_net_edge": min_net,
                "min_fills": min_fills,
                "fill_n": fill_n,
                "fill_net_edge": fill_net,
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
                "net_edge": r.get("net_edge"),
                "min_trades": r["min_trades"], "min_accuracy": r["min_accuracy"],
                "min_net_edge": r.get("min_net_edge"),
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
        try:
            from arena.alerts import alert_lane_change
            alert_lane_change(
                "approve", lane,
                accuracy=r.get("accuracy"), n=r.get("n"),
                detail={"min_accuracy": min_acc, "min_trades": min_trades},
            )
        except Exception:
            pass
        break  # one promotion per cycle — let the monitor watch it before more

    return report

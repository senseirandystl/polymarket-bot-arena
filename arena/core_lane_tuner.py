"""Core-lane auto-tuner — the loop's core half (drift/mom/strat per strategy).

The candidate-lane promoter (arena/lane_promoter.py) tunes fut/tech/xasset,
which feed a few bots at ~0.10 weight. This module tunes the lanes that decide
100% of every directional trade — drift, mom, strat — PER strategy, on that
strategy's own live attribution.

Every directional trade logs its lane READINGS in the reasoning string
(``... drift=+0.03 mom=+0.43 ... strat=+0.60 ...``). Joined to the placing
bot's strategy_type (bot_configs), that is a per-(strategy, lane) record of
whether the lane's SIGN predicted the market direction (UP iff a YES trade won
or a NO trade lost). A lane that is predictive for a strategy earns a small
weight nudge UP; one that is anti-predictive earns a nudge DOWN.

Because these lanes drive the whole decision (unlike a 0.10 candidate lane), the
tuner is deliberately timid and heavily bounded:

* a real per-(strategy, lane) sample floor (``CORE_TUNE_MIN_TRADES``),
* one capped step per lane per cycle (``CORE_TUNE_STEP``),
* hysteresis — up only above ``CORE_TUNE_HIGH_ACC``, down only below
  ``CORE_TUNE_LOW_ACC``; the band between holds steady,
* a per-lane BAND around the hand-set class default (``CORE_TUNE_BAND``) so no
  lane can run away or collapse — drift, the one validated lane, can never be
  tuned to zero, and a strategy's identity weight stays recognizable,
* a COMPLETE per-strategy profile is written for each tuned lane, because a
  core-lane override zeroes any strategy it omits (candidate lanes default to 0,
  core lanes emphatically do not).

Gated by the SAME toggle as the promoter (``db.get_auto_approve_lanes()``):
OFF => compute and persist the suggested weights to arena_state 'core_lane_tuner'
for a human to read, but never apply. Writes only lane_overrides / arena_state,
never trade tables. Safe on any cadence; hosted by the evolution loop.
"""

import json
import logging
import re
from datetime import datetime, timezone

import config
import db
from bots.base_bot import BaseBot

logger = logging.getLogger("arena.core_lane_tuner")

CORE_LANES = ("drift", "mom", "strat")

# Lane readings as logged by base_bot.make_decision's reasoning string.
_LANE_RE = {
    "drift": re.compile(r"drift=([+-][\d.]+)"),
    "mom": re.compile(r"mom=([+-][\d.]+)"),
    "strat": re.compile(r"strat=([+-][\d.]+)"),
}


def _strategy_map(conn) -> dict:
    """bot_name -> strategy_type for every bot ever configured (trades outlive
    active bots, so we read all rows, not just active)."""
    return {r["bot_name"]: r["strategy_type"]
            for r in conn.execute(
                "SELECT bot_name, strategy_type FROM bot_configs")}


def compute_core_attribution(conn, deadband: float) -> dict:
    """{strategy_type: {lane: {n, accuracy}}} from resolved directional trades."""
    smap = _strategy_map(conn)
    rows = conn.execute(
        """SELECT bot_name, side, outcome, reasoning FROM trades
           WHERE outcome IN ('win', 'loss') AND reasoning LIKE 'fair=%'"""
    ).fetchall()
    agg: dict = {}
    for r in rows:
        strat = smap.get(r["bot_name"])
        if strat is None:
            continue
        market_up = (r["side"] == "yes") == (r["outcome"] == "win")
        text = r["reasoning"] or ""
        for lane, rx in _LANE_RE.items():
            m = rx.search(text)
            if not m:
                continue
            reading = float(m.group(1))
            if abs(reading) < deadband:
                continue
            cell = agg.setdefault(strat, {}).setdefault(lane, {"n": 0, "correct": 0})
            cell["n"] += 1
            cell["correct"] += int((reading > 0) == market_up)
    # finalize accuracy
    out: dict = {}
    for strat, lanes in agg.items():
        out[strat] = {}
        for lane, c in lanes.items():
            out[strat][lane] = {"n": c["n"],
                                "accuracy": c["correct"] / c["n"] if c["n"] else None}
    return out


def _effective_weight(overrides: dict, lane: str, strat: str, default: float) -> float:
    """Current live weight of ``lane`` for ``strat`` — the override value if the
    lane is already tuned, else the class default."""
    ov = overrides.get(lane)
    if ov and ov.get("enabled"):
        return float(ov.get("profile", {}).get(strat, default))
    return default


def tune() -> dict:
    """Score core lanes per strategy; nudge weights when the toggle is on.

    Returns a per-lane report (also persisted to arena_state 'core_lane_tuner')
    describing, for every strategy with enough data, the measured accuracy and
    the current/suggested weight — so the dashboard can show the tuning whether
    or not it was applied.
    """
    if not getattr(config, "CORE_TUNE_ENABLED", True):
        return {}

    apply = db.get_auto_approve_lanes()
    min_trades = getattr(config, "CORE_TUNE_MIN_TRADES", 40)
    high_acc = getattr(config, "CORE_TUNE_HIGH_ACC", 0.56)
    low_acc = getattr(config, "CORE_TUNE_LOW_ACC", 0.48)
    step = getattr(config, "CORE_TUNE_STEP", 0.05)
    band = getattr(config, "CORE_TUNE_BAND", 0.20)
    wmax = getattr(config, "CORE_TUNE_WEIGHT_MAX", 0.90)
    wmin = getattr(config, "CORE_TUNE_WEIGHT_MIN", 0.0)
    deadband = getattr(config, "LANE_MONITOR_DEADBAND", 0.05)

    profiles = BaseBot.STRATEGY_SIGNAL_PROFILE
    strategies = list(profiles.keys())

    with db.get_conn() as conn:
        attribution = compute_core_attribution(conn, deadband)

    overrides = db.get_lane_overrides()
    report: dict = {"applied": apply, "lanes": {}}
    new_overrides = dict(overrides)
    dirty = False
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for lane in CORE_LANES:
        lane_report = {}
        # Seed a COMPLETE profile from current effective weights so we never
        # zero a strategy the override omits.
        profile: dict = {}
        changed = False
        for strat in strategies:
            default = float(profiles.get(strat, BaseBot.DEFAULT_SIGNAL_PROFILE)
                            .get(lane, 0.0))
            cur = _effective_weight(overrides, lane, strat, default)
            new_w = cur
            st = attribution.get(strat, {}).get(lane)
            action = "hold"
            if st and st["n"] >= min_trades and st["accuracy"] is not None:
                lo = max(wmin, default - band)
                hi = min(wmax, default + band)
                if st["accuracy"] >= high_acc and cur < hi:
                    new_w = round(min(hi, cur + step), 3)
                    action = "up"
                elif st["accuracy"] <= low_acc and cur > lo:
                    new_w = round(max(lo, cur - step), 3)
                    action = "down"
                if new_w != cur:
                    changed = True
                lane_report[strat] = {
                    "n": st["n"], "accuracy": round(st["accuracy"], 3),
                    "current": cur, "suggested": new_w, "action": action,
                    "default": default,
                }
            profile[strat] = new_w
        report["lanes"][lane] = lane_report
        if apply and changed:
            new_overrides[lane] = {
                "enabled": True, "profile": profile, "core": True,
                "tuned_at": stamp,
            }
            dirty = True
            for strat, r in lane_report.items():
                if r["action"] != "hold":
                    logger.info(
                        f"Core-lane tune: {strat}.{lane} {r['current']}->"
                        f"{r['suggested']} (acc {r['accuracy']:.1%}/{r['n']})"
                    )

    if apply and dirty:
        db.set_arena_state("lane_overrides", json.dumps(new_overrides))

    db.set_arena_state("core_lane_tuner", json.dumps(report))
    return report

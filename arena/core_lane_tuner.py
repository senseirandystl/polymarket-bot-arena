"""Core-lane auto-tuner — signal weights from live attribution.

Tunes:
  * **Core** lanes (drift/mom/strat) per strategy — drive every directional trade
  * **Enabled live overrides** (candidate lanes with ``enabled: true`` in
    ``lane_overrides``) so approved fut/tech/xasset/… get the same closed loop
    as core once they carry weight (2026-08-07)

Every directional trade logs lane READINGS in reasoning
(``drift=… mom=… strat=… cand(fut=… tech=… xa=…)``). Joined to
strategy_type, that is per-(strategy, lane) sign-vs-outcome accuracy.

Core bounds (timid): sample floor, one step/cycle, HIGH/LOW hysteresis, band
around class default, complete profile (omit = 0).

Candidate bounds: band around approved starter, may DOWN to 0, merge-only
profile (never mark ``core: true``), lower weight ceiling.

Gated by ``db.get_auto_core_tune()`` (falls back to auto_approve_lanes).
Hosted by the evolution loop (``CORE_TUNE_INTERVAL_SEC``).
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

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
# Shadow candidate token (same as lane_monitor / ops_snapshot).
_CAND_RE = re.compile(
    r"cand\(fut=([+-][\d.]+) tech=([+-][\d.]+) xa=([+-][\d.]+)"
    r"(?: lag=([+-][\d.]+))?(?: ms=([+-][\d.]+))?(?: fd=([+-][\d.]+))?\)")
_CAND_GROUPS = {
    "fut": 1, "tech": 2, "xasset": 3,
    "lag": 4, "ms_mom": 5, "flow_decay": 6,
}


def live_tune_lanes(overrides: Optional[dict] = None) -> tuple[str, ...]:
    """Core lanes + enabled non-core overrides (live candidates)."""
    extra: list[str] = []
    for lane, ov in (overrides or {}).items():
        if lane in CORE_LANES:
            continue
        if ov and ov.get("enabled") is True:
            extra.append(str(lane))
    return CORE_LANES + tuple(sorted(extra))


def _is_core_lane(lane: str) -> bool:
    return lane in CORE_LANES


def _strategy_map(conn) -> dict:
    """bot_name -> strategy_type for every bot ever configured (trades outlive
    active bots, so we read all rows, not just active)."""
    return {r["bot_name"]: r["strategy_type"]
            for r in conn.execute(
                "SELECT bot_name, strategy_type FROM bot_configs")}


def _parse_readings(text: str, lanes: Iterable[str]) -> dict[str, float]:
    """Extract requested lane readings from a reasoning string."""
    out: dict[str, float] = {}
    if not text:
        return out
    want = set(lanes)
    for lane, rx in _LANE_RE.items():
        if lane not in want:
            continue
        m = rx.search(text)
        if m:
            try:
                out[lane] = float(m.group(1))
            except (TypeError, ValueError):
                pass
    cand_want = want & set(_CAND_GROUPS)
    if cand_want:
        cm = _CAND_RE.search(text)
        if cm:
            for lane in cand_want:
                g = _CAND_GROUPS[lane]
                try:
                    raw = cm.group(g)
                except IndexError:
                    raw = None
                if raw is None:
                    continue
                try:
                    out[lane] = float(raw)
                except (TypeError, ValueError):
                    pass
    return out


def _finalize_agg(agg: dict) -> dict:
    out: dict = {}
    for strat, lanes in agg.items():
        out[strat] = {}
        for lane, c in lanes.items():
            n = int(c["n"] or 0)
            n_ev = int(c.get("n_ev") or 0)
            sum_pnl = float(c.get("sum_pnl") or 0.0)
            out[strat][lane] = {
                "n": n,
                "accuracy": (c["correct"] / n) if n else None,
                "n_ev": n_ev,
                "sum_pnl": round(sum_pnl, 4),
                "mean_ev": (sum_pnl / n_ev) if n_ev else None,
            }
    return out


def compute_core_attribution(conn, deadband: float, *,
                             cell_filter: tuple | None = None,
                             regime_id: str | None = None,
                             lanes: Iterable[str] | None = None) -> dict:
    """{strategy_type: {lane: {n, accuracy}}} from resolved decisions/trades.

    Prefers ``decision_events`` for core lanes when mass is high. Always
    merges trade-reasoning parse for candidate lanes (and as fallback).
    ``lanes`` defaults to CORE_LANES only (backward-compatible).
    """
    lane_list = tuple(lanes) if lanes is not None else CORE_LANES
    core_need = [l for l in lane_list if l in CORE_LANES]
    cand_need = [l for l in lane_list if l not in CORE_LANES]
    out: dict = {}

    # Prefer decision_events for core (global, no cell/regime filter).
    if core_need and cell_filter is None and not regime_id:
        try:
            from arena.decision_log import (
                should_use_decision_attribution, core_lane_attribution,
            )
            if should_use_decision_attribution(conn):
                attr = core_lane_attribution(conn, deadband)
                if attr:
                    for st, ld in attr.items():
                        for lane, st_lane in ld.items():
                            if lane in core_need:
                                out.setdefault(st, {})[lane] = st_lane
        except Exception:
            pass

    # Per-regime core from decision_events
    if core_need and regime_id and not out:
        try:
            attr = _attribution_by_regime(conn, deadband, regime_id,
                                         lanes=core_need)
            if attr:
                out = attr
        except Exception as e:
            logger.debug("regime attribution failed: %s", e)

    # Trade reasoning parse: fill missing core + all candidates
    need_trade = bool(cand_need) or not out
    if need_trade or any(
        not out.get(st, {}).get(lane)
        for st in out
        for lane in core_need
    ) or not out:
        smap = _strategy_map(conn)
        rows = conn.execute(
            """SELECT bot_name, side, outcome, pnl, reasoning, context,
                      trade_features
               FROM trades
               WHERE outcome IN ('win', 'loss') AND reasoning LIKE 'fair=%'"""
        ).fetchall()
        agg: dict = {}
        for r in rows:
            strat = smap.get(r["bot_name"])
            if strat is None:
                continue
            if regime_id:
                feats = r["trade_features"] or ""
                tag = f"regime:{regime_id}"
                if tag not in str(feats):
                    continue
            if cell_filter is not None:
                ctx_raw = r["context"]
                if not ctx_raw:
                    continue
                try:
                    from signals.context import context_cell
                    if context_cell(json.loads(ctx_raw)) != cell_filter:
                        continue
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
            market_up = (r["side"] == "yes") == (r["outcome"] == "win")
            side_yes = (r["side"] == "yes")
            try:
                trade_pnl = float(r["pnl"] or 0.0)
            except (TypeError, ValueError):
                trade_pnl = 0.0
            readings = _parse_readings(r["reasoning"] or "", lane_list)
            for lane, reading in readings.items():
                if abs(reading) < deadband:
                    continue
                cell = agg.setdefault(strat, {}).setdefault(
                    lane, {"n": 0, "correct": 0, "n_ev": 0, "sum_pnl": 0.0})
                cell["n"] += 1
                cell["correct"] += int((reading > 0) == market_up)
                # Lane EV: P&L when the trade followed this lane's sign
                if (reading > 0) == side_yes:
                    cell["n_ev"] += 1
                    cell["sum_pnl"] += trade_pnl
        trade_out = _finalize_agg(agg)
        for st, ld in trade_out.items():
            for lane, st_lane in ld.items():
                prev = out.setdefault(st, {}).get(lane)
                if prev is None or (prev.get("n") or 0) < (st_lane.get("n") or 0):
                    out.setdefault(st, {})[lane] = st_lane

    # Merge decision_events core on top when trade path was primary
    if core_need and not regime_id:
        try:
            from arena.decision_log import core_lane_attribution, resolved_count
            if resolved_count(conn) > 0:
                dec = core_lane_attribution(conn, deadband)
                for st, ld in dec.items():
                    for lane, st_lane in ld.items():
                        if lane not in core_need:
                            continue
                        prev = out.setdefault(st, {}).get(lane)
                        if prev is None or (prev.get("n") or 0) < (st_lane.get("n") or 0):
                            out.setdefault(st, {})[lane] = st_lane
        except Exception:
            pass
    return out


def _attribution_by_regime(conn, deadband: float, regime_id: str,
                           lanes: Iterable[str] | None = None) -> dict:
    """{strategy: {lane: {n, accuracy}}} filtered to one detector regime."""
    lane_list = tuple(lanes) if lanes is not None else CORE_LANES
    # decision_events has core columns; candidates still come from trades.
    cols = [l for l in lane_list if l in ("drift", "mom", "strat")]
    if not cols:
        return {}
    col_sql = ", ".join(cols)
    rows = conn.execute(
        f"""SELECT strategy_type, market_up, {col_sql}, regime
            FROM decision_events
            WHERE market_up IS NOT NULL AND regime = ?
              AND action IN ('buy', 'skip')""",
        (regime_id,),
    ).fetchall()
    agg: dict = {}
    for r in rows:
        strat = r["strategy_type"]
        if not strat:
            continue
        market_up = bool(r["market_up"])
        for lane in cols:
            raw = r[lane]
            if raw is None:
                continue
            reading = float(raw)
            if abs(reading) < deadband:
                continue
            cell = agg.setdefault(strat, {}).setdefault(lane, {"n": 0, "correct": 0})
            cell["n"] += 1
            cell["correct"] += int((reading > 0) == market_up)
    return _finalize_agg(agg)


def _effective_weight(overrides: dict, lane: str, strat: str, default: float) -> float:
    """Current live weight of ``lane`` for ``strat`` — the override value if the
    lane is already tuned, else the class default."""
    ov = overrides.get(lane)
    if ov and ov.get("enabled"):
        return float(ov.get("profile", {}).get(strat, default))
    return default


def _strategy_regime_pnl(regime_id: Optional[str]) -> dict:
    """{strategy: {n, wins, pnl, wr}} for live detector regime (cached)."""
    if not regime_id:
        return {}
    try:
        from arena.regime_stats import snapshot
        by = (snapshot().get("by_strategy") or {}).get(regime_id) or {}
        return {k: dict(v) for k, v in by.items()}
    except Exception:
        return {}


def _strategy_global_pnl(hours: float = 48.0) -> dict:
    """{strategy: {n, pnl}} across all regimes for always-on P&L gate."""
    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                """SELECT bc.strategy_type AS st,
                          COUNT(*) AS n,
                          COALESCE(SUM(t.pnl), 0) AS pnl
                   FROM trades t
                   JOIN bot_configs bc ON bc.bot_name = t.bot_name
                   WHERE t.outcome IN ('win', 'loss', 'exit_tp', 'exit_sl')
                     AND datetime(t.resolved_at) >= datetime('now', ?)
                   GROUP BY bc.strategy_type""",
                (f"-{float(hours)} hours",),
            ).fetchall()
        return {
            r["st"]: {"n": int(r["n"] or 0), "pnl": float(r["pnl"] or 0.0)}
            for r in rows if r["st"]
        }
    except Exception:
        return {}


def _seed_or_default(regime: Optional[str], strat: str, lane: str,
                     default: float) -> float:
    try:
        from arena.regime_profiles import seed_weight
        sw = seed_weight(regime, strat, lane)
        if sw is not None:
            return float(sw)
    except Exception:
        pass
    return float(default)


def _scorecard_net_by_strategy(hours: float | None = None) -> dict | None:
    """Unique-market net edge per (strategy, lane) from the live scorecard."""
    try:
        from arena.live_scorecard import unique_market_rows, _lane_stats
        h = hours if hours is not None else float(
            getattr(config, "LIVE_SCORECARD_HOURS", 72) or 72
        )
        with db.get_conn() as conn:
            rows = unique_market_rows(conn, hours=h)
        by_st: dict[str, list] = {}
        for r in rows:
            st = str(r.get("strategy_type") or "")
            if st:
                by_st.setdefault(st, []).append(r)
        out: dict = {}
        lanes = CORE_LANES + ("xasset", "fut", "tech", "ms_mom")
        cheap_max = float(getattr(config, "CORE_TUNE_SCORECARD_MAX_ENTRY", 0.62))
        for st, grp in by_st.items():
            cheap = []
            for r in grp:
                try:
                    e = r.get("entry_price")
                    e = float(e) if e is not None else None
                except (TypeError, ValueError):
                    e = None
                if e is None or e <= cheap_max:
                    cheap.append(r)
            for lane in lanes:
                stats = _lane_stats(cheap, lane, 0.05)
                if (stats.get("n_priced") or 0) or (stats.get("markets") or 0):
                    out.setdefault(st, {})[lane] = stats
        return out
    except Exception:
        logger.exception("scorecard net overlay failed")
        return None


def tune() -> dict:
    """Score core lanes per strategy; nudge weights when the toggle is on.

    Returns a per-lane report (also persisted to arena_state 'core_lane_tuner')
    describing, for every strategy with enough data, the measured accuracy and
    the current/suggested weight — so the dashboard can show the tuning whether
    or not it was applied.
    """
    if not getattr(config, "CORE_TUNE_ENABLED", True):
        return {}

    # Separate toggle from lane auto-approve (2026-08 audit).
    try:
        apply = db.get_auto_core_tune()
    except Exception:
        apply = db.get_auto_approve_lanes()
    min_trades = getattr(config, "CORE_TUNE_MIN_TRADES", 40)
    min_trades_reg = int(getattr(config, "CORE_TUNE_MIN_TRADES_REGIME", 40))
    high_acc = getattr(config, "CORE_TUNE_HIGH_ACC", 0.56)
    low_acc = getattr(config, "CORE_TUNE_LOW_ACC", 0.48)
    step = getattr(config, "CORE_TUNE_STEP", 0.05)
    band = getattr(config, "CORE_TUNE_BAND", 0.20)
    wmax = getattr(config, "CORE_TUNE_WEIGHT_MAX", 0.90)
    wmin = getattr(config, "CORE_TUNE_WEIGHT_MIN", 0.0)
    deadband = getattr(config, "LANE_MONITOR_DEADBAND", 0.05)
    try:
        from arena.regime_settings import get_bool as _reg_bool
        profile_adapt = bool(_reg_bool("profile_adapt"))
    except Exception:
        profile_adapt = bool(getattr(config, "REGIME_PROFILE_ADAPT_ENABLED", True))

    profiles = BaseBot.STRATEGY_SIGNAL_PROFILE
    strategies = list(profiles.keys())
    overrides = db.get_lane_overrides() or {}
    tune_lanes = live_tune_lanes(overrides)
    cand_wmax = float(getattr(config, "CANDIDATE_TUNE_WEIGHT_MAX", 0.35))
    cand_band = float(getattr(config, "CANDIDATE_TUNE_BAND", 0.25))
    cand_min = int(getattr(config, "CANDIDATE_TUNE_MIN_TRADES", 30))

    # Live detector regime for per-regime profile tuning
    live_regime = None
    try:
        from signals.regime_detector import get_detector
        cur = get_detector().status().get("current") or {}
        live_regime = cur.get("regime_id")
        if live_regime in (None, "unknown") or not cur.get("actionable", False):
            live_regime = None
    except Exception:
        live_regime = None

    # Regime-conditioning (Layer 3): when the toggle is on and the current
    # regime is known, tune each lane on ITS attribution within that regime.
    # Off / no current_cell -> None -> global attribution, byte-for-byte as before.
    cell_filter = None
    try:
        if db.get_regime_conditioning():
            cur = db.get_regime_map().get("current_cell")
            cell_filter = tuple(cur) if cur else None
    except Exception:
        cell_filter = None

    regime_local = False
    with db.get_conn() as conn:
        # Prefer detector-regime attribution when profile adapt is on
        if profile_adapt and live_regime:
            attribution = compute_core_attribution(
                conn, deadband, regime_id=live_regime, lanes=tune_lanes)
            min_trades_use = min_trades_reg
            regime_local = True
        else:
            attribution = compute_core_attribution(
                conn, deadband, cell_filter=cell_filter, lanes=tune_lanes)
            min_trades_use = min_trades
        # Audit fix: fine-grained context cells often starve every
        # (strategy, lane) below CORE_TUNE_MIN_TRADES, leaving lanes:{} empty
        # while applied=true (soak 2026-07-27). Fall back to GLOBAL attribution
        # when the cell has no strategy with a full sample on any core lane.
        # IMPORTANT: global fallback must NOT write by_regime clones (soak
        # 2026-08-06: identical by_regime killed profile seeds).
        fallback = None
        enough = any(
            (st.get("n") or 0) >= min_trades_use
            for lanes_d in attribution.values()
            for st in lanes_d.values()
        )
        if not enough and (cell_filter is not None or live_regime):
            attribution = compute_core_attribution(
                conn, deadband, cell_filter=None, regime_id=None,
                lanes=tune_lanes)
            min_trades_use = min_trades
            fallback = "global_insufficient_regime_samples"
            regime_local = False
            logger.info(
                "Core-lane tuner: regime/cell starved samples — "
                "using global attribution (no by_regime write)",
            )

    # Live P&L per strategy — always-on gate (regime-local preferred, else global).
    strat_pnl = _strategy_regime_pnl(live_regime) if regime_local else {}
    if not strat_pnl:
        strat_pnl = _strategy_global_pnl(48.0)
    pnl_min_n = int(getattr(config, "CORE_TUNE_PNL_MIN_TRADES", 15))
    # Regime-local cells are thin (overnight: n=3–8 trades while attribution
    # n is large). Use a lower bar so red $ still blocks accuracy-driven UP.
    if regime_local:
        pnl_min_n = min(
            pnl_min_n,
            int(getattr(config, "CORE_TUNE_PNL_MIN_TRADES_REGIME", 5)),
        )
    ev_primary = bool(getattr(config, "CORE_TUNE_EV_PRIMARY", True))
    ev_min_n = int(getattr(config, "CORE_TUNE_EV_MIN_TRADES", 20))
    ev_up_min = float(getattr(config, "CORE_TUNE_EV_UP_MIN", 0.0))
    ev_down_max = float(getattr(config, "CORE_TUNE_EV_DOWN_MAX", -0.05))
    scorecard_net = _scorecard_net_by_strategy()
    scorecard_unavailable = scorecard_net is None
    if scorecard_unavailable:
        scorecard_net = {}
    sc_min = int(getattr(config, "CORE_TUNE_SCORECARD_MIN", 20))
    sc_block = float(getattr(config, "CORE_TUNE_SCORECARD_DOWN_MAX", 0.0))
    sc_force = float(getattr(config, "CORE_TUNE_SCORECARD_FORCE_DOWN", -0.005))

    report: dict = {"applied": apply,
                    "cell_filter": list(cell_filter) if cell_filter else None,
                    "live_regime": live_regime,
                    "fallback": fallback,
                    "regime_local": regime_local,
                    "tune_lanes": list(tune_lanes),
                    "lanes": {}}
    new_overrides = dict(overrides)
    dirty = False
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for lane in tune_lanes:
        is_core = _is_core_lane(lane)
        lane_min = min_trades_use if is_core else min(min_trades_use, cand_min)
        lane_band = band if is_core else cand_band
        lane_wmax = wmax if is_core else min(wmax, cand_wmax)
        lane_report = {}
        # Core: complete profile. Candidate: merge only strategies we touch.
        profile: dict = {}
        if not is_core:
            profile = dict((overrides.get(lane) or {}).get("profile") or {})
        by_regime_lane: dict = dict(
            (overrides.get(lane) or {}).get("by_regime") or {}
        )
        by_regime_meta: dict = dict(
            (overrides.get(lane) or {}).get("by_regime_meta") or {}
        )
        # Start reg_profile from seed (not elevated global) so first local
        # write is chop-aware rather than a clone of global mom=0.50.
        if live_regime and regime_local and profile_adapt and is_core:
            if live_regime in by_regime_lane and by_regime_lane[live_regime]:
                reg_profile: dict = dict(by_regime_lane[live_regime])
            else:
                reg_profile = {
                    s: _seed_or_default(
                        live_regime, s, lane,
                        float(profiles.get(s, BaseBot.DEFAULT_SIGNAL_PROFILE)
                              .get(lane, 0.0)),
                    )
                    for s in strategies
                }
        elif live_regime and regime_local and profile_adapt and not is_core:
            reg_profile = dict(by_regime_lane.get(live_regime) or profile)
        else:
            reg_profile = {}
        changed = False
        reg_changed = False
        for strat in strategies:
            default = float(profiles.get(strat, BaseBot.DEFAULT_SIGNAL_PROFILE)
                            .get(lane, 0.0))
            # Candidate default = approved starter (current override) or 0
            if not is_core:
                default = float(
                    (overrides.get(lane) or {}).get("profile", {}).get(strat, 0.0)
                    or 0.0
                )
                # Prefer original approved weight as anchor if stored
                approved = (overrides.get(lane) or {}).get("approved_weight")
                if approved is not None and strat in (
                        (overrides.get(lane) or {}).get("profile") or {}):
                    try:
                        # single scalar approved_weight is rare; per-strat profile is source
                        pass
                    except Exception:
                        pass
            seed_w = (
                _seed_or_default(live_regime, strat, lane, default)
                if is_core else default
            )
            cur = _effective_weight(overrides, lane, strat, default)
            if regime_local and live_regime and strat in reg_profile:
                cur = float(reg_profile[strat])
            new_w = cur
            st = attribution.get(strat, {}).get(lane)
            action = "hold"
            pnl_cell = strat_pnl.get(strat) or {}
            pnl_n = int(pnl_cell.get("n") or 0)
            pnl_val = float(pnl_cell.get("pnl") or 0.0)
            # Candidates with zero weight and no samples: skip (don't bloat report)
            if not is_core and cur <= 0 and not (st and st.get("n")):
                continue
            if st and st["n"] >= lane_min and st["accuracy"] is not None:
                # Band around seed when regime-local, else class default
                anchor = seed_w if (regime_local and is_core) else default
                mean_ev = st.get("mean_ev")
                n_ev = int(st.get("n_ev") or 0)
                try:
                    mean_ev_f = float(mean_ev) if mean_ev is not None else None
                except (TypeError, ValueError):
                    mean_ev_f = None
                # Soften drift floor when EV is red (priced-in predictive)
                red_ev = (
                    mean_ev_f is not None
                    and n_ev >= ev_min_n
                    and mean_ev_f <= ev_down_max
                )
                if is_core:
                    lo = max(wmin, anchor - lane_band)
                    hi = min(lane_wmax, max(anchor, default) + lane_band)
                    if lane == "drift":
                        floor = float(getattr(
                            config, "CORE_TUNE_DRIFT_FLOOR", 0.10))
                        if red_ev or (pnl_n >= pnl_min_n and pnl_val < 0):
                            floor = float(getattr(
                                config, "CORE_TUNE_DRIFT_FLOOR_WHEN_RED", 0.05))
                        lo = max(lo, floor)
                else:
                    # Candidates may go to 0
                    lo = 0.0
                    hi = min(lane_wmax, max(anchor, 0.05) + lane_band)
                acc = float(st["accuracy"])
                revert_below = float(
                    getattr(config, "CORE_TUNE_REVERT_BELOW_ACC", high_acc)
                )
                # Always-on P&L gate (not only regime-local)
                pnl_blocks_up = (
                    pnl_n >= pnl_min_n
                    and pnl_val < 0
                    and bool(getattr(config, "CORE_TUNE_PNL_GATE", True))
                )
                ev_blocks_up = (
                    ev_primary
                    and mean_ev_f is not None
                    and n_ev >= ev_min_n
                    and mean_ev_f < ev_up_min
                )
                ev_forces_down = (
                    ev_primary
                    and mean_ev_f is not None
                    and n_ev >= ev_min_n
                    and mean_ev_f <= ev_down_max
                    and cur > lo
                )
                sc = (scorecard_net.get(strat) or {}).get(lane) or {}
                sc_n = int(sc.get("n_priced") or 0)
                try:
                    sc_net = float(sc["net_edge"]) if sc.get("net_edge") is not None else None
                except (TypeError, ValueError, KeyError):
                    sc_net = None
                if scorecard_unavailable:
                    # Missing judge must not allow accuracy-led UP.
                    ev_blocks_up = True
                elif sc_n >= sc_min and sc_net is not None:
                    if sc_net <= sc_block:
                        ev_blocks_up = True
                    if sc_net <= sc_force and cur > lo:
                        ev_forces_down = True
                if ev_forces_down:
                    new_w = round(max(lo, cur - step), 3)
                    action = "ev_down"
                elif (
                    acc >= high_acc
                    and cur < hi
                    and not pnl_blocks_up
                    and not ev_blocks_up
                    and (
                        not ev_primary
                        or mean_ev_f is None
                        or n_ev < ev_min_n
                        or mean_ev_f >= ev_up_min
                    )
                ):
                    new_w = round(min(hi, cur + step), 3)
                    action = "up"
                elif acc >= high_acc and (pnl_blocks_up or ev_blocks_up):
                    target = min(cur, max(lo, anchor))
                    if cur > target + 1e-9:
                        new_w = round(max(lo, cur - step), 3)
                        action = "pnl_revert" if pnl_blocks_up else "ev_revert"
                    else:
                        # Timeout UP disabled when hours <= 0 (default)
                        timeout_h = float(getattr(
                            config, "CORE_TUNE_PNL_GATE_TIMEOUT_HOURS", 0.0))
                        if timeout_h <= 0:
                            action = "hold_pnl_gate"
                        else:
                            reg_meta_cell = by_regime_meta.get(
                                live_regime or "_global", {}).get(strat, {})
                            last_gate = reg_meta_cell.get("pnl_gate_since")
                            try:
                                last_gate = (
                                    float(last_gate) if last_gate else None
                                )
                            except (TypeError, ValueError):
                                last_gate = None
                            key_reg = live_regime or "_global"
                            if last_gate is None:
                                by_regime_meta.setdefault(
                                    key_reg, {}).setdefault(
                                        strat, {})["pnl_gate_since"] = time.time()
                                action = "hold_pnl_gate"
                            elif (time.time() - last_gate) > timeout_h * 3600:
                                # Only re-test if EV no longer deep red
                                if not ev_forces_down:
                                    new_w = round(
                                        min(hi, cur + step * 0.5), 3)
                                    action = "pnl_gate_timeout_up"
                                else:
                                    action = "hold_pnl_gate"
                                by_regime_meta.setdefault(
                                    key_reg, {}).setdefault(
                                        strat, {})["pnl_gate_since"] = time.time()
                            else:
                                action = "hold_pnl_gate"
                elif acc <= low_acc and cur > lo:
                    new_w = round(max(lo, cur - step), 3)
                    action = "down"
                elif (acc < revert_below and cur > anchor + 1e-9
                      and cur > lo):
                    new_w = round(max(anchor, lo, cur - step), 3)
                    action = "revert"
                if new_w != cur:
                    changed = True
                    if live_regime and profile_adapt and regime_local:
                        reg_changed = True
                lane_report[strat] = {
                    "n": st["n"], "accuracy": round(st["accuracy"], 3),
                    "current": cur, "suggested": new_w, "action": action,
                    "default": default,
                    "seed": seed_w if regime_local else None,
                    "regime": live_regime,
                    "regime_pnl": round(pnl_val, 2) if pnl_n else None,
                    "regime_pnl_n": pnl_n or None,
                    "mean_ev": (round(mean_ev_f, 4)
                                if mean_ev_f is not None else None),
                    "n_ev": n_ev,
                    "kind": "core" if is_core else "candidate",
                }
            elif st and st["n"]:
                lane_report[strat] = {
                    "n": st["n"],
                    "accuracy": (round(st["accuracy"], 3)
                                 if st.get("accuracy") is not None else None),
                    "current": cur, "suggested": cur, "action": "collecting",
                    "default": default,
                    "regime": live_regime,
                    "kind": "core" if is_core else "candidate",
                }
            if is_core or strat in lane_report or strat in profile:
                profile[strat] = new_w
            if live_regime and profile_adapt and regime_local:
                reg_profile[strat] = new_w
        report["lanes"][lane] = lane_report
        if apply and changed:
            if is_core:
                entry = {
                    "enabled": True, "profile": profile, "core": True,
                    "tuned_at": stamp,
                }
            else:
                # Merge-only candidate profile; never set core:true
                prev = dict(overrides.get(lane) or {})
                prev_profile = dict(prev.get("profile") or {})
                prev_profile.update(profile)
                entry = dict(prev)
                entry.update({
                    "enabled": True,
                    "profile": prev_profile,
                    "tuned_at": stamp,
                })
                entry.pop("core", None)
            # Only write by_regime when attribution was regime-local (earned)
            if live_regime and profile_adapt and reg_changed and regime_local:
                by_regime_lane[live_regime] = reg_profile
                entry["by_regime"] = by_regime_lane
                reg_meta = dict(by_regime_meta.get(live_regime) or {})
                for strat, r in lane_report.items():
                    if r.get("action") in (
                        "hold", "collecting", "hold_pnl_gate"
                    ) and strat not in reg_meta:
                        if (r.get("n") or 0) < lane_min:
                            continue
                    cell_meta = dict(reg_meta.get(strat) or {})
                    if not isinstance(cell_meta, dict):
                        cell_meta = {}
                    cell_meta[lane] = {
                        "earned": True,
                        "n": int(r.get("n") or 0),
                        "accuracy": r.get("accuracy"),
                        "pnl": r.get("regime_pnl"),
                        "tuned_at": stamp,
                    }
                    reg_meta[strat] = cell_meta
                by_regime_meta[live_regime] = reg_meta
                entry["by_regime_meta"] = by_regime_meta
            else:
                prev = overrides.get(lane) or {}
                if prev.get("by_regime"):
                    entry["by_regime"] = prev.get("by_regime")
                # Always persist by_regime_meta — pnl_gate_since must
                # survive hold cycles (audit: local mutations were silently
                # discarded when reg_changed was False).
                entry["by_regime_meta"] = by_regime_meta
            new_overrides[lane] = entry
            dirty = True
            for strat, r in lane_report.items():
                if r["action"] not in ("hold", "collecting", "hold_pnl_gate"):
                    logger.info(
                        f"Lane tune: {strat}.{lane} {r['current']}->"
                        f"{r['suggested']} (acc {r.get('accuracy')}/"
                        f"{r['n']}"
                        f"{' reg='+str(live_regime) if live_regime else ''}"
                        f" action={r['action']}"
                        f" kind={'core' if is_core else 'cand'})"
                    )

    # Renormalize core profile weights per strategy so Σw(core) ≈ 1.
    # Only when a complete core profile exists for that strategy (all CORE_LANES
    # present); partial writes must not scale a single lane to 1.0.
    if (
        apply
        and dirty
        and bool(getattr(config, "CORE_TUNE_NORMALIZE_PROFILE", True))
    ):
        core_lanes = list(CORE_LANES)
        strats = set()
        for lane in core_lanes:
            prof = (new_overrides.get(lane) or {}).get("profile") or {}
            strats.update(prof.keys())
        for strat in strats:
            vals = []
            complete = True
            for lane in core_lanes:
                entry = new_overrides.get(lane) or {}
                prof = entry.get("profile") or {}
                if strat not in prof:
                    complete = False
                    break
                vals.append(float(prof.get(strat) or 0.0))
            if not complete:
                continue
            total = sum(vals)
            if total <= 1e-9:
                continue
            scale = 1.0 / total
            for lane, v in zip(core_lanes, vals):
                entry = new_overrides.get(lane)
                if not entry or "profile" not in entry:
                    continue
                entry["profile"][strat] = round(v * scale, 3)

    if apply and dirty:
        db.set_arena_state("lane_overrides", json.dumps(new_overrides))
        # Notify operators of applied weight shifts (large moves only).
        try:
            changes = []
            for lane, lane_report in (report.get("lanes") or {}).items():
                for strat, r in (lane_report or {}).items():
                    if r.get("action") in (
                        "up", "down", "ev_down", "pnl_revert", "ev_revert"
                    ) and r.get("current") != r.get("suggested"):
                        changes.append({
                            "lane": lane,
                            "strategy": strat,
                            "from": r.get("current"),
                            "to": r.get("suggested"),
                            "accuracy": r.get("accuracy"),
                            "action": r.get("action"),
                        })
            if changes:
                from arena.alerts import alert_core_lane_tune
                alert_core_lane_tune(changes)
        except Exception:
            pass

    # Continuous residual online update (Phase 5) — best-effort from attribution
    try:
        from arena.regime_settings import get_bool as _reg_bool
        _cont = bool(_reg_bool("continuous_blend"))
    except Exception:
        _cont = bool(getattr(config, "REGIME_CONTINUOUS_BLEND", False))
    if _cont:
        try:
            from arena.regime_continuous import observe, persist
            from signals.regime_detector import get_detector
            feats = (get_detector().status().get("current") or {}).get("features") or {}
            for strat, lanes in attribution.items():
                for lane, st in lanes.items():
                    if lane not in CORE_LANES or not st or not st.get("n"):
                        continue
                    acc = st.get("accuracy")
                    if acc is None:
                        continue
                    # Proxy: accuracy as P(correct); emit synthetic observe bursts
                    n = min(int(st["n"]), 20)
                    n_ok = int(round(acc * n))
                    for _ in range(n_ok):
                        observe(lane, strat, feats, correct=True, reading_sign=1.0)
                    for _ in range(n - n_ok):
                        observe(lane, strat, feats, correct=False, reading_sign=1.0)
            persist()
        except Exception:
            pass

    db.set_arena_state("core_lane_tuner", json.dumps(report))
    return report

"""Data-driven skip / go / size rules mined from decision_events.

Learning loop (not hard-coded regime gates):

1. **Mine** resolved decision_events into context cells
   ``(regime, price_band, drift_band, side[, strategy_type])``.
2. **Continuous size/edge mults** from cell WR (not only binary skip/go).
3. **Hard SKIP** only at the toxic tail; **GO** eases edge when cells print
   or when skips miss winners (counterfactual).
4. **Walk-forward OOS**: train on older events, promote only if the hold-out
   fold still supports the rule (lane-promoter-style hysteresis).
5. **Auto per-strategy cells** once sample mass is large enough (no manual
   ``LEARNED_RULES_PER_STRATEGY=True`` required when auto mode is on).
6. **Skip-reason bandit**: if a static guard's skips would often have won,
   auto-soften that guard's threshold; reverse when counterfactual fades.
7. **Apply** on the hot path via :func:`evaluate` + :func:`skip_softening`
   (fail-open).

State: arena_state ``learned_trade_rules``. Hosted by the evolution loop
after ``decision_log.maybe_rollup``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import config
import db

logger = logging.getLogger("arena.learned_rules")

STATE_KEY = "learned_trade_rules"

_PRICE_BANDS = (
    ("cheap", 0.0, 0.42),
    ("mid", 0.42, 0.58),
    ("high", 0.58, 1.01),
)
_DRIFT_BANDS = (
    ("flat", 0.0, 0.10),
    ("weak", 0.10, 0.20),
    ("moderate", 0.20, 0.30),
    ("strong", 0.30, 2.0),
)

# Static skip_reason keys we may soften (decision_log.classify_skip_reason).
_SOFTENABLE_SKIPS = frozenset({
    "dead_zone", "no_edge", "weak_lean", "consensus", "high_price",
    "extreme_drift", "book",
})


def price_band(price: float | None) -> str:
    p = float(price or 0.5)
    for name, lo, hi in _PRICE_BANDS:
        if lo <= p < hi:
            return name
    return "mid"


def drift_band(drift: float | None) -> str:
    d = abs(float(drift or 0.0))
    for name, lo, hi in _DRIFT_BANDS:
        if lo <= d < hi:
            return name
    return "strong"


def cell_key(
    *,
    regime: str | None,
    price: float | None,
    drift: float | None,
    side: str | None,
    strategy_type: str | None = None,
    per_strategy: bool = False,
) -> str:
    reg = (regime or "unknown").strip() or "unknown"
    pb = price_band(price)
    dbnd = drift_band(drift)
    sd = (side or "any").lower()
    if per_strategy and strategy_type:
        return f"{reg}|{pb}|{dbnd}|{sd}|{strategy_type}"
    return f"{reg}|{pb}|{dbnd}|{sd}"


def parse_cell(key: str) -> dict[str, str | None]:
    parts = (key or "").split("|")
    return {
        "regime": parts[0] if len(parts) > 0 else "unknown",
        "price_band": parts[1] if len(parts) > 1 else "mid",
        "drift_band": parts[2] if len(parts) > 2 else "flat",
        "side": parts[3] if len(parts) > 3 else "any",
        "strategy_type": parts[4] if len(parts) > 4 else None,
    }


# ---------------------------------------------------------------------------
# Continuous mult mapping (process 1)
# ---------------------------------------------------------------------------

def continuous_size_mult(wr: float | None, n: int) -> float:
    """Map cell buy-WR → size mult in [SIZE_MIN, SIZE_MAX].

    Neutral 1.0 until ``min_n`` samples. Linear in WR between bad and good
    bars (same spirit as regime_adapt).
    """
    min_n = int(getattr(config, "LEARNED_RULES_MIN_N", 25))
    if wr is None or n < min_n:
        return 1.0
    bad = float(getattr(config, "LEARNED_RULES_CONT_BAD_WR", 0.45))
    good = float(getattr(config, "LEARNED_RULES_CONT_GOOD_WR", 0.60))
    lo = float(getattr(config, "LEARNED_RULES_CONT_SIZE_MIN", 0.40))
    hi = float(getattr(config, "LEARNED_RULES_CONT_SIZE_MAX", 1.25))
    if good <= bad + 1e-9:
        return 1.0
    t = (float(wr) - bad) / (good - bad)
    t = max(0.0, min(1.0, t))
    return round(lo + t * (hi - lo), 4)


def continuous_edge_mult(wr: float | None, n: int) -> float:
    """Map cell WR → min_edge multiplier (>1 tighter, <1 softer)."""
    min_n = int(getattr(config, "LEARNED_RULES_MIN_N", 25))
    if wr is None or n < min_n:
        return 1.0
    bad = float(getattr(config, "LEARNED_RULES_CONT_BAD_WR", 0.45))
    good = float(getattr(config, "LEARNED_RULES_CONT_GOOD_WR", 0.60))
    # At bad WR → tighten edge (1.25); at good → ease (0.80)
    tight = float(getattr(config, "LEARNED_RULES_CONT_EDGE_TIGHT", 1.25))
    soft = float(getattr(config, "LEARNED_RULES_CONT_EDGE_SOFT", 0.80))
    if good <= bad + 1e-9:
        return 1.0
    t = (float(wr) - bad) / (good - bad)
    t = max(0.0, min(1.0, t))
    return round(tight + t * (soft - tight), 4)


# ---------------------------------------------------------------------------
# Per-strategy auto (process 2)
# ---------------------------------------------------------------------------

def resolve_per_strategy(conn=None) -> tuple[bool, str]:
    """Whether cells should include strategy_type.

    Returns ``(enabled, reason)``.

    * ``LEARNED_RULES_PER_STRATEGY`` True → always on (manual force).
    * False + auto off → always off.
    * False/auto + ``LEARNED_RULES_PER_STRATEGY_AUTO`` → on when sample mass
      clears min resolved + min rich per-strategy cells.
    """
    forced = getattr(config, "LEARNED_RULES_PER_STRATEGY", False)
    if forced is True or forced == "true" or forced == 1:
        return True, "forced_on"
    auto = bool(getattr(config, "LEARNED_RULES_PER_STRATEGY_AUTO", True))
    if not auto:
        return False, "auto_off"

    min_res = int(getattr(config, "LEARNED_RULES_PER_STRATEGY_MIN_RESOLVED", 200))
    min_cells = int(getattr(config, "LEARNED_RULES_PER_STRATEGY_MIN_CELLS", 8))
    min_n = int(getattr(config, "LEARNED_RULES_MIN_N", 25))

    def _check(c) -> tuple[bool, str]:
        n_res = c.execute(
            "SELECT COUNT(*) n FROM decision_events WHERE market_up IS NOT NULL"
        ).fetchone()["n"]
        if int(n_res or 0) < min_res:
            return False, f"resolved={n_res}<{min_res}"
        # Count strategy-tagged cells with enough buys (using per-strat keys)
        rows = c.execute(
            """SELECT strategy_type, regime, side,
                      CASE
                        WHEN entry_price < 0.42 THEN 'cheap'
                        WHEN entry_price < 0.58 THEN 'mid'
                        ELSE 'high'
                      END AS pb,
                      CASE
                        WHEN ABS(COALESCE(drift,0)) < 0.10 THEN 'flat'
                        WHEN ABS(COALESCE(drift,0)) < 0.20 THEN 'weak'
                        WHEN ABS(COALESCE(drift,0)) < 0.30 THEN 'moderate'
                        ELSE 'strong'
                      END AS db,
                      SUM(CASE WHEN action='buy' THEN 1 ELSE 0 END) bn
               FROM decision_events
               WHERE market_up IS NOT NULL AND side IS NOT NULL
                 AND strategy_type IS NOT NULL AND strategy_type != ''
               GROUP BY 1,2,3,4,5
               HAVING bn >= ?""",
            (min_n,),
        ).fetchall()
        n_rich = len(rows)
        if n_rich < min_cells:
            return False, f"rich_strat_cells={n_rich}<{min_cells}"
        return True, f"auto_on resolved={n_res} rich_cells={n_rich}"

    try:
        if conn is not None:
            return _check(conn)
        with db.get_conn() as c:
            return _check(c)
    except Exception as e:
        logger.debug("per_strategy resolve failed: %s", e)
        return False, f"error:{e}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_state() -> dict[str, Any]:
    try:
        raw = db.get_arena_state(STATE_KEY)
        if not raw:
            return _empty_state()
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return _empty_state()
        data.setdefault("rules", [])
        data.setdefault("cells", {})
        data.setdefault("skip_soften", {})
        data.setdefault("per_strategy", False)
        data.setdefault("per_strategy_reason", "")
        data.setdefault("oos", {})
        return data
    except Exception as e:
        logger.debug("learned_rules load failed: %s", e)
        return _empty_state()


def _empty_state() -> dict[str, Any]:
    return {
        "rules": [],
        "cells": {},
        "skip_soften": {},
        "per_strategy": False,
        "per_strategy_reason": "",
        "oos": {},
        "updated_at": None,
        "n_rules": 0,
        "n_cells": 0,
    }


def save_state(state: dict[str, Any]) -> None:
    try:
        db.set_arena_state(STATE_KEY, json.dumps(state))
    except Exception as e:
        logger.warning("learned_rules save failed: %s", e)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_cells(
    conn,
    *,
    per_strategy: bool,
    created_before: str | None = None,
    created_on_or_after: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build per-cell stats; optional chronological filters for OOS folds."""
    clauses = ["market_up IS NOT NULL", "side IS NOT NULL"]
    params: list[Any] = []
    if created_before:
        clauses.append("created_at < ?")
        params.append(created_before)
    if created_on_or_after:
        clauses.append("created_at >= ?")
        params.append(created_on_or_after)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"""SELECT action, side, regime, entry_price, drift, strategy_type,
                   would_win, hyp_pnl
            FROM decision_events
            WHERE {where}""",
        tuple(params),
    ).fetchall()

    cells: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = cell_key(
            regime=r["regime"],
            price=r["entry_price"],
            drift=r["drift"],
            side=r["side"],
            strategy_type=r["strategy_type"],
            per_strategy=per_strategy,
        )
        c = cells.setdefault(key, {
            "buy_n": 0, "buy_wins": 0, "buy_hyp_sum": 0.0,
            "skip_n": 0, "skip_would_win": 0, "skip_hyp_sum": 0.0,
        })
        action = (r["action"] or "").lower()
        ww = r["would_win"]
        hyp = float(r["hyp_pnl"]) if r["hyp_pnl"] is not None else 0.0
        if action == "buy":
            c["buy_n"] += 1
            if ww:
                c["buy_wins"] += 1
            c["buy_hyp_sum"] += hyp
        elif action == "skip":
            c["skip_n"] += 1
            if ww:
                c["skip_would_win"] += 1
            c["skip_hyp_sum"] += hyp

    for c in cells.values():
        bn, sn = c["buy_n"], c["skip_n"]
        c["buy_wr"] = (c["buy_wins"] / bn) if bn else None
        c["buy_avg_hyp"] = (c["buy_hyp_sum"] / bn) if bn else None
        c["skip_cf_wr"] = (c["skip_would_win"] / sn) if sn else None
        c["skip_avg_hyp"] = (c["skip_hyp_sum"] / sn) if sn else None
    return cells


def _oos_cutoff(conn) -> str | None:
    """Chronological cutoff for train/test split (quantile of created_at)."""
    frac = float(getattr(config, "LEARNED_RULES_OOS_TRAIN_FRAC", 0.70))
    frac = max(0.5, min(0.9, frac))
    row = conn.execute(
        """SELECT created_at FROM decision_events
           WHERE market_up IS NOT NULL
           ORDER BY created_at ASC"""
    ).fetchall()
    if len(row) < int(getattr(config, "LEARNED_RULES_OOS_MIN_EVENTS", 40)):
        return None
    idx = max(0, min(len(row) - 1, int(len(row) * frac) - 1))
    return row[idx]["created_at"]


def _passes_oos(
    key: str,
    rule_type: str,
    test_cells: dict[str, dict[str, Any]],
    *,
    min_n: int,
    skip_wr: float,
    go_wr: float,
    demote_skip_wr: float,
    demote_go_wr: float,
) -> tuple[bool, str]:
    """OOS fold must not reverse the train signal (hysteresis)."""
    if not test_cells:
        return True, "oos_skipped_thin"
    tc = test_cells.get(key)
    if not tc:
        # No OOS samples in this cell — require train-only only if allowed
        if bool(getattr(config, "LEARNED_RULES_OOS_REQUIRE_TEST_CELL", False)):
            return False, "oos_no_test_cell"
        return True, "oos_no_test_cell_soft"
    bn = int(tc.get("buy_n") or 0)
    sn = int(tc.get("skip_n") or 0)
    buy_wr = tc.get("buy_wr")
    skip_cf = tc.get("skip_cf_wr")
    oos_min = max(5, min_n // 3)

    if rule_type == "skip":
        if bn < oos_min:
            return True, f"oos_skip_thin_n={bn}"
        if buy_wr is not None and buy_wr >= demote_skip_wr:
            return False, f"oos_skip_wr={buy_wr:.3f}>={demote_skip_wr}"
        # Prefer still weak; allow mid if still below go_wr
        if buy_wr is not None and buy_wr > skip_wr + 0.08:
            return False, f"oos_skip_wr_recovered={buy_wr:.3f}"
        return True, f"oos_skip_ok wr={buy_wr}"

    if rule_type == "go":
        if bn >= oos_min and buy_wr is not None:
            if buy_wr < demote_go_wr:
                return False, f"oos_go_wr={buy_wr:.3f}<{demote_go_wr}"
            return True, f"oos_go_ok wr={buy_wr}"
        if sn >= oos_min and skip_cf is not None:
            if skip_cf < go_wr - 0.05:
                return False, f"oos_go_skip_cf={skip_cf:.3f}"
            return True, f"oos_go_skip_cf_ok={skip_cf:.3f}"
        return True, "oos_go_thin_soft"

    if rule_type == "continuous":
        # Continuous always allowed if train had mass; OOS only drops extreme
        return True, "oos_cont"

    return True, "oos_unknown"


# ---------------------------------------------------------------------------
# Skip-reason bandit (process 3)
# ---------------------------------------------------------------------------

def _mine_skip_soften(conn) -> dict[str, dict[str, Any]]:
    """If skips of reason R often would have won, soften guard R."""
    min_n = int(getattr(config, "LEARNED_RULES_SKIP_BANDIT_MIN_N", 30))
    high_cf = float(getattr(config, "LEARNED_RULES_SKIP_BANDIT_HIGH_CF", 0.58))
    low_cf = float(getattr(config, "LEARNED_RULES_SKIP_BANDIT_LOW_CF", 0.48))
    max_soften = float(getattr(config, "LEARNED_RULES_SKIP_BANDIT_MAX_SOFTEN", 0.25))

    rows = conn.execute(
        """SELECT skip_reason,
                  COUNT(*) n,
                  SUM(CASE WHEN would_win=1 THEN 1 ELSE 0 END) wins,
                  AVG(hyp_pnl) avg_hyp
           FROM decision_events
           WHERE market_up IS NOT NULL AND action='skip'
             AND skip_reason IS NOT NULL AND side IS NOT NULL
           GROUP BY skip_reason"""
    ).fetchall()

    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        reason = (r["skip_reason"] or "").strip()
        if reason not in _SOFTENABLE_SKIPS:
            continue
        n = int(r["n"] or 0)
        if n < min_n:
            continue
        wins = int(r["wins"] or 0)
        cf_wr = wins / n if n else 0.0
        avg_hyp = float(r["avg_hyp"]) if r["avg_hyp"] is not None else 0.0
        # High CF WR + positive hyp → we incorrectly stood down → soften
        if cf_wr >= high_cf and avg_hyp > 0:
            # Scale soften 0..max by how far above high_cf
            t = min(1.0, (cf_wr - high_cf) / max(1e-6, 0.20))
            soften = round(t * max_soften, 4)
            out[reason] = {
                "soften": soften,
                "n": n,
                "cf_wr": round(cf_wr, 4),
                "avg_hyp": round(avg_hyp, 4),
                "direction": "ease",
                "reason": f"skip_cf_wr={cf_wr:.3f} n={n} (missed winners)",
            }
        elif cf_wr <= low_cf and avg_hyp < 0:
            # Skips were correct — optional mild tighten (negative soften)
            t = min(1.0, (low_cf - cf_wr) / max(1e-6, 0.15))
            tighten = round(t * max_soften * 0.5, 4)
            out[reason] = {
                "soften": -tighten,
                "n": n,
                "cf_wr": round(cf_wr, 4),
                "avg_hyp": round(avg_hyp, 4),
                "direction": "tighten",
                "reason": f"skip_cf_wr={cf_wr:.3f} n={n} (correct skips)",
            }
    return out


# ---------------------------------------------------------------------------
# Mine + promote
# ---------------------------------------------------------------------------

def mine_and_update() -> dict[str, Any]:
    """Recompute cells, OOS-validate rules, skip bandit, persist."""
    if not getattr(config, "LEARNED_RULES_ENABLED", True):
        return load_state()

    min_n = int(getattr(config, "LEARNED_RULES_MIN_N", 25))
    skip_wr = float(getattr(config, "LEARNED_RULES_SKIP_WR_MAX", 0.47))
    skip_hyp = float(getattr(config, "LEARNED_RULES_SKIP_HYP_MAX", -0.005))
    go_wr = float(getattr(config, "LEARNED_RULES_GO_WR_MIN", 0.58))
    go_hyp = float(getattr(config, "LEARNED_RULES_GO_HYP_MIN", 0.01))
    go_from_skip_wr = float(getattr(config, "LEARNED_RULES_MISSED_WR_MIN", 0.60))
    demote_skip_wr = float(getattr(config, "LEARNED_RULES_DEMOTE_SKIP_WR", 0.53))
    demote_go_wr = float(getattr(config, "LEARNED_RULES_DEMOTE_GO_WR", 0.50))
    max_rules = int(getattr(config, "LEARNED_RULES_MAX", 40))
    oos_enabled = bool(getattr(config, "LEARNED_RULES_OOS_ENABLED", True))
    cont_enabled = bool(getattr(config, "LEARNED_RULES_CONTINUOUS", True))

    with db.get_conn() as conn:
        per_strat, per_reason = resolve_per_strategy(conn)
        cutoff = _oos_cutoff(conn) if oos_enabled else None
        if cutoff:
            train_cells = _aggregate_cells(
                conn, per_strategy=per_strat, created_before=cutoff,
            )
            test_cells = _aggregate_cells(
                conn, per_strategy=per_strat, created_on_or_after=cutoff,
            )
            all_cells = _aggregate_cells(conn, per_strategy=per_strat)
        else:
            train_cells = _aggregate_cells(conn, per_strategy=per_strat)
            test_cells = {}
            all_cells = train_cells
        skip_soften = _mine_skip_soften(conn)

    state = load_state()
    existing = {r["cell"]: r for r in (state.get("rules") or []) if r.get("cell")}
    new_rules: list[dict] = []
    oos_stats = {
        "enabled": oos_enabled,
        "cutoff": cutoff,
        "promoted": 0,
        "rejected_oos": 0,
        "train_cells": len(train_cells),
        "test_cells": len(test_cells),
    }

    for key, c in train_cells.items():
        prev = existing.get(key)
        rule_type = None
        effect: dict[str, float] = {}
        reason = ""

        bn = int(c["buy_n"])
        sn = int(c["skip_n"])
        buy_wr = c["buy_wr"]
        buy_hyp = c["buy_avg_hyp"]
        skip_cf = c["skip_cf_wr"]

        # Continuous mults always computed when mass allows
        size_c = continuous_size_mult(buy_wr, bn)
        edge_c = continuous_edge_mult(buy_wr, bn)

        # Hard SKIP: toxic tail
        if (
            bn >= min_n
            and buy_wr is not None
            and buy_wr <= skip_wr
            and buy_hyp is not None
            and buy_hyp <= skip_hyp
        ):
            rule_type = "skip"
            effect = {"size_mult": size_c, "edge_mult": edge_c}
            reason = f"buy_wr={buy_wr:.3f} hyp={buy_hyp:.4f} n={bn}"
        # GO: good buys (high-price needs stricter hyp bar — fee-aware BE)
        elif (
            bn >= min_n
            and buy_wr is not None
            and buy_wr >= go_wr
            and buy_hyp is not None
            and buy_hyp >= go_hyp
        ):
            match = parse_cell(key)
            pb = (match.get("price_band") or "mid")
            hi_hyp = float(getattr(config, "LEARNED_RULES_GO_HIGH_MIN_HYP", 0.02))
            if pb == "high" and buy_hyp < hi_hyp:
                rule_type = None  # not enough fee-aware edge at high prices
            else:
                rule_type = "go"
                effect = {
                    "size_mult": max(
                        size_c,
                        float(getattr(config, "LEARNED_RULES_GO_SIZE_MULT", 1.15)),
                    ),
                    "edge_mult": min(
                        edge_c,
                        float(getattr(config, "LEARNED_RULES_GO_EDGE_MULT", 0.85)),
                    ),
                }
                reason = f"buy_wr={buy_wr:.3f} hyp={buy_hyp:.4f} n={bn}"
        # GO from missed winners
        elif (
            sn >= min_n
            and skip_cf is not None
            and skip_cf >= go_from_skip_wr
            and (c["skip_avg_hyp"] or 0) > 0
        ):
            # Ban pure skip-counterfactual GO on high-price cells (2026-08:
            # eased expensive favorites that printed negative live BE).
            match = parse_cell(key)
            pb = (match.get("price_band") or "mid")
            ban_high = bool(
                getattr(config, "LEARNED_RULES_BAN_GO_HIGH_FROM_SKIP", True))
            if ban_high and pb == "high":
                # Only allow if real buys also clear a fee-aware bar.
                hi_n = int(getattr(config, "LEARNED_RULES_GO_HIGH_MIN_BUY_N", 15))
                hi_hyp = float(getattr(config, "LEARNED_RULES_GO_HIGH_MIN_HYP", 0.02))
                if not (
                    bn >= hi_n
                    and buy_wr is not None
                    and buy_wr >= go_wr
                    and buy_hyp is not None
                    and buy_hyp >= hi_hyp
                ):
                    rule_type = None
                    reason = ""
                else:
                    rule_type = "go"
                    effect = {
                        "size_mult": max(size_c, 1.0),
                        "edge_mult": min(
                            edge_c,
                            float(getattr(
                                config, "LEARNED_RULES_MISSED_EDGE_MULT", 0.80)),
                        ),
                    }
                    reason = (
                        f"skip_cf_wr={skip_cf:.3f}+buy_wr={buy_wr:.3f} "
                        f"n_buy={bn} (high band with buy evidence)"
                    )
            else:
                rule_type = "go"
                effect = {
                    "size_mult": max(size_c, 1.0),
                    "edge_mult": min(
                        edge_c,
                        float(getattr(
                            config, "LEARNED_RULES_MISSED_EDGE_MULT", 0.80)),
                    ),
                }
                reason = f"skip_cf_wr={skip_cf:.3f} n_skip={sn} (missed winners)"
        # Continuous-only soft rule (middle of the WR spectrum)
        elif cont_enabled and bn >= min_n and buy_wr is not None:
            if abs(size_c - 1.0) >= 0.05 or abs(edge_c - 1.0) >= 0.05:
                rule_type = "continuous"
                effect = {"size_mult": size_c, "edge_mult": edge_c}
                reason = f"cont size={size_c:.3f} edge={edge_c:.3f} wr={buy_wr:.3f} n={bn}"

        # Demotion of prior hard rules
        if prev and not rule_type:
            pt = prev.get("type")
            if pt == "skip" and bn >= min_n and buy_wr is not None and buy_wr >= demote_skip_wr:
                continue
            if pt == "go" and bn >= min_n and buy_wr is not None and buy_wr < demote_go_wr:
                continue
            if pt in ("skip", "go", "continuous"):
                if bn < min_n and sn < min_n:
                    new_rules.append(prev)
                continue

        if not rule_type:
            continue

        # Walk-forward OOS gate
        ok, oos_detail = _passes_oos(
            key, rule_type, test_cells,
            min_n=min_n, skip_wr=skip_wr, go_wr=go_wr,
            demote_skip_wr=demote_skip_wr, demote_go_wr=demote_go_wr,
        )
        if not ok:
            oos_stats["rejected_oos"] += 1
            logger.debug("learned_rules OOS reject %s %s: %s", rule_type, key, oos_detail)
            continue
        oos_stats["promoted"] += 1

        # Live stats prefer full-window all_cells for dashboard honesty
        live = all_cells.get(key) or c
        new_rules.append({
            "cell": key,
            "type": rule_type,
            "match": parse_cell(key),
            "effect": effect,
            "stats": {
                "buy_n": int(live["buy_n"]),
                "buy_wr": live["buy_wr"],
                "buy_avg_hyp": live["buy_avg_hyp"],
                "skip_n": int(live["skip_n"]),
                "skip_cf_wr": live["skip_cf_wr"],
                "train_buy_n": bn,
                "train_buy_wr": buy_wr,
            },
            "reason": reason,
            "oos": oos_detail,
            "updated_at": time.time(),
        })

    def _score(r: dict) -> float:
        st = r.get("stats") or {}
        n = float(st.get("buy_n") or st.get("skip_n") or 0)
        if r.get("type") == "skip":
            wr = st.get("buy_wr")
            return n * (0.5 - (wr if wr is not None else 0.5))
        if r.get("type") == "continuous":
            wr = st.get("buy_wr") or 0.5
            return n * abs(wr - 0.5) * 0.5  # lower priority than hard rules
        wr = st.get("buy_wr") or st.get("skip_cf_wr") or 0.5
        return n * (wr - 0.5)

    new_rules.sort(key=_score, reverse=True)
    # Prefer hard skip/go over continuous when capping
    hard = [r for r in new_rules if r.get("type") in ("skip", "go")]
    cont = [r for r in new_rules if r.get("type") == "continuous"]
    max_hard = max_rules
    max_cont = int(getattr(config, "LEARNED_RULES_MAX_CONTINUOUS", 30))
    new_rules = hard[:max_hard] + cont[:max_cont]

    state = {
        "rules": new_rules,
        "cells": {
            k: {
                "buy_n": v["buy_n"],
                "buy_wr": v["buy_wr"],
                "buy_avg_hyp": v["buy_avg_hyp"],
                "skip_n": v["skip_n"],
                "skip_cf_wr": v["skip_cf_wr"],
                "size_mult": continuous_size_mult(v["buy_wr"], int(v["buy_n"])),
                "edge_mult": continuous_edge_mult(v["buy_wr"], int(v["buy_n"])),
            }
            for k, v in list(all_cells.items())[:300]
        },
        "skip_soften": skip_soften,
        "per_strategy": per_strat,
        "per_strategy_reason": per_reason,
        "oos": oos_stats,
        "updated_at": time.time(),
        "n_rules": len(new_rules),
        "n_cells": len(all_cells),
        "n_skip": sum(1 for r in new_rules if r.get("type") == "skip"),
        "n_go": sum(1 for r in new_rules if r.get("type") == "go"),
        "n_continuous": sum(1 for r in new_rules if r.get("type") == "continuous"),
    }
    save_state(state)
    global _eval_cache, _soften_cache
    _eval_cache = (0.0, [])
    _soften_cache = (0.0, {})
    logger.info(
        "learned_rules: %d rules (%d skip/%d go/%d cont) from %d cells "
        "per_strat=%s (%s) oos_reject=%d soften=%d",
        len(new_rules), state["n_skip"], state["n_go"], state["n_continuous"],
        len(all_cells), per_strat, per_reason,
        oos_stats["rejected_oos"], len(skip_soften),
    )
    return state


# ---------------------------------------------------------------------------
# Hot-path evaluate
# ---------------------------------------------------------------------------

_eval_cache: tuple[float, list[dict]] = (0.0, [])
_soften_cache: tuple[float, dict] = (0.0, {})
_mode_cache: tuple[float, bool] = (0.0, False)


def _rules_cached() -> list[dict]:
    global _eval_cache
    now = time.time()
    ttl = float(getattr(config, "LEARNED_RULES_CACHE_SEC", 30.0))
    if (now - _eval_cache[0]) < ttl:
        return _eval_cache[1]
    st = load_state()
    rules = list(st.get("rules") or [])
    _eval_cache = (now, rules)
    return rules


def _per_strat_cached() -> bool:
    global _mode_cache
    now = time.time()
    ttl = float(getattr(config, "LEARNED_RULES_CACHE_SEC", 30.0))
    if (now - _mode_cache[0]) < ttl:
        return _mode_cache[1]
    st = load_state()
    # Prefer last mine result; fall back to config force
    ps = bool(st.get("per_strategy"))
    if getattr(config, "LEARNED_RULES_PER_STRATEGY", False) is True:
        ps = True
    _mode_cache = (now, ps)
    return ps


def _cell_matches(
    rule: dict,
    *,
    regime: str | None,
    side_price: float | None,
    drift: float | None,
    side: str | None,
    strategy_type: str | None,
    per_strat: bool,
) -> bool:
    m = rule.get("match") or parse_cell(rule.get("cell") or "")
    if m.get("regime") != (regime or "unknown"):
        return False
    if m.get("price_band") != price_band(side_price):
        return False
    if m.get("drift_band") != drift_band(drift):
        return False
    ms = m.get("side")
    if ms and ms not in ("any", side):
        return False
    if per_strat and m.get("strategy_type") and m["strategy_type"] != strategy_type:
        return False
    return True


def evaluate(
    *,
    regime: str | None,
    side_price: float | None,
    drift: float | None,
    side: str | None,
    strategy_type: str | None = None,
) -> dict[str, Any]:
    """Hot-path rule match (skip / go / continuous mults). Fail-open."""
    default = {
        "action": "allow",
        "size_mult": 1.0,
        "edge_mult": 1.0,
        "rule": None,
        "reason": "",
        "type": None,
    }
    if not getattr(config, "LEARNED_RULES_ENABLED", True):
        return default
    try:
        per_strat = _per_strat_cached()
        keys = [
            cell_key(
                regime=regime, price=side_price, drift=drift, side=side,
                strategy_type=strategy_type, per_strategy=per_strat,
            ),
            cell_key(
                regime=regime, price=side_price, drift=drift, side="any",
                strategy_type=strategy_type, per_strategy=per_strat,
            ),
        ]
        # Prefer hard skip, then go, then continuous (rules list already scored)
        matched: dict | None = None
        for rule in _rules_cached():
            ck = rule.get("cell")
            if ck not in keys and not _cell_matches(
                rule, regime=regime, side_price=side_price, drift=drift,
                side=side, strategy_type=strategy_type, per_strat=per_strat,
            ):
                continue
            rtype = rule.get("type")
            if rtype == "skip":
                return {
                    "action": "skip",
                    "size_mult": 1.0,
                    "edge_mult": 1.0,
                    "rule": rule,
                    "type": "skip",
                    "reason": f"learned_skip:{ck} ({rule.get('reason', '')})",
                }
            if matched is None or (
                rtype == "go" and matched.get("type") == "continuous"
            ):
                matched = rule
                matched = {**rule, "_ck": ck}
                if rtype == "go":
                    break  # go beats continuous; skip already returned

        if matched:
            eff = matched.get("effect") or {}
            rtype = matched.get("type")
            return {
                "action": "allow",
                "size_mult": float(eff.get("size_mult") or 1.0),
                "edge_mult": float(eff.get("edge_mult") or 1.0),
                "rule": matched,
                "type": rtype,
                "reason": f"learned_{rtype}:{matched.get('_ck') or matched.get('cell')}",
            }
        return default
    except Exception as e:
        logger.debug("learned_rules evaluate failed: %s", e)
        return default


def skip_softening(skip_reason: str | None) -> dict[str, Any]:
    """Hot-path softening for a static skip_reason (bandit). Fail-open neutral."""
    neutral = {"soften": 0.0, "factor": 1.0, "direction": None, "detail": None}
    if not getattr(config, "LEARNED_RULES_ENABLED", True):
        return neutral
    if not getattr(config, "LEARNED_RULES_SKIP_BANDIT_ENABLED", True):
        return neutral
    reason = (skip_reason or "").strip()
    if not reason:
        return neutral
    try:
        global _soften_cache
        now = time.time()
        ttl = float(getattr(config, "LEARNED_RULES_CACHE_SEC", 30.0))
        if (now - _soften_cache[0]) >= ttl:
            st = load_state()
            _soften_cache = (now, dict(st.get("skip_soften") or {}))
        entry = (_soften_cache[1] or {}).get(reason)
        if not entry:
            return neutral
        soften = float(entry.get("soften") or 0.0)
        # factor: 1 - soften for "ease" (lower barriers); 1 + |soften| tighten
        if soften >= 0:
            factor = max(0.5, 1.0 - soften)
        else:
            factor = min(1.5, 1.0 - soften)  # soften negative → >1
        return {
            "soften": soften,
            "factor": round(factor, 4),
            "direction": entry.get("direction"),
            "detail": entry,
        }
    except Exception as e:
        logger.debug("skip_softening failed: %s", e)
        return neutral


def snapshot() -> dict[str, Any]:
    """Dashboard / ops view (full enough for Signal Lab card)."""
    st = load_state()
    rules = list(st.get("rules") or [])
    return {
        "enabled": bool(getattr(config, "LEARNED_RULES_ENABLED", True)),
        "n_rules": len(rules),
        "n_skip": st.get("n_skip") or sum(1 for r in rules if r.get("type") == "skip"),
        "n_go": st.get("n_go") or sum(1 for r in rules if r.get("type") == "go"),
        "n_continuous": st.get("n_continuous") or sum(
            1 for r in rules if r.get("type") == "continuous"
        ),
        "n_cells": st.get("n_cells"),
        "updated_at": st.get("updated_at"),
        "per_strategy": st.get("per_strategy"),
        "per_strategy_reason": st.get("per_strategy_reason"),
        "oos": st.get("oos") or {},
        "skip_soften": st.get("skip_soften") or {},
        "rules": rules[:40],
        "config": {
            "min_n": getattr(config, "LEARNED_RULES_MIN_N", 25),
            "continuous": getattr(config, "LEARNED_RULES_CONTINUOUS", True),
            "oos_enabled": getattr(config, "LEARNED_RULES_OOS_ENABLED", True),
            "per_strategy_auto": getattr(
                config, "LEARNED_RULES_PER_STRATEGY_AUTO", True
            ),
            "skip_bandit": getattr(
                config, "LEARNED_RULES_SKIP_BANDIT_ENABLED", True
            ),
        },
    }

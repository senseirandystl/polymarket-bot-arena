"""Signal Lab combination + foundational-rule explorer.

Judges *combinations* of already-logged lanes on unique-market rows
(one row per strategy × window) after the crypto taker fee. This is how
we find edge that is not "follow live drift" — e.g. mom+tech when drift
is flat — without loosening dual-gate / lean floor / sweeper certainty.

Hot path: :func:`try_confirm` lets an *earned* cheap combo stand in for
a weak model lean, and may bypass dual-gate *only* when the combo does
not depend on drift. Dead-zone, high-price, consensus, and fee math are
untouched.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import config
import db
import polymarket_fills

logger = logging.getLogger("arena.combo_explorer")

_LANES = ("drift", "mom", "strat", "tech", "xasset")
_PAIR_COMBOS: tuple[tuple[str, ...], ...] = (
    ("drift", "mom"),
    ("drift", "tech"),
    ("drift", "xasset"),
    ("mom", "tech"),
    ("mom", "xasset"),
    ("tech", "xasset"),
    ("drift", "mom", "tech"),
    ("mom", "tech", "xasset"),
    ("drift", "tech", "xasset"),
)
_STATE_KEY = "combo_explorer"
_CACHE: tuple[float, dict] = (0.0, {})
_CACHE_TTL = 5.0


def _deadband() -> float:
    return float(getattr(config, "COMBO_DEADBAND", 0.05) or 0.05)


def _max_entry() -> float:
    return float(getattr(config, "COMBO_MAX_ENTRY", 0.62) or 0.62)


def _f(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _readings(row: dict) -> dict[str, Optional[float]]:
    return {ln: _f(row.get(ln)) for ln in _LANES}


def _agree(readings: dict, lanes: tuple[str, ...] | list[str],
           deadband: float) -> Optional[int]:
    """+1 if every named lane leans the same way past the deadband."""
    signs: list[int] = []
    for ln in lanes:
        v = readings.get(ln)
        if v is None or abs(v) < deadband:
            return None
        signs.append(1 if v > 0 else -1)
    if not signs or any(s != signs[0] for s in signs):
        return None
    return signs[0]


def _priced_edge(row: dict, pred_up: bool) -> Optional[float]:
    entry = _f(row.get("entry_price"))
    if entry is None:
        return None
    market_up = bool(row.get("market_up"))
    side = row.get("side")
    if (pred_up and side == "yes") or ((not pred_up) and side == "no"):
        cost = entry
    else:
        cost = max(0.01, min(0.99, 1.0 - entry))
    if cost > _max_entry():
        return None
    fee = polymarket_fills.taker_fee(1.0, cost)
    ok = pred_up == market_up
    return (1.0 - cost - fee) if ok else (-cost - fee)


def _score(rows: list[dict], pred_fn, *, cheap_only: bool = False) -> dict:
    n = correct = 0
    edge_sum = 0.0
    n_edge = 0
    by_reg: dict[str, list] = {}
    max_e = _max_entry()
    for r in rows:
        pred = pred_fn(r)
        if pred is None:
            continue
        entry = _f(r.get("entry_price"))
        cheap = entry is not None and entry <= max_e
        ev = _priced_edge(r, pred)
        if cheap_only and ev is None:
            continue
        n += 1
        ok = pred == bool(r.get("market_up"))
        correct += int(ok)
        if ev is not None:
            edge_sum += ev
            n_edge += 1
        reg = (r.get("regime") or "unknown").strip() or "unknown"
        by_reg.setdefault(reg, []).append(ok)
    acc = (correct / n) if n else None
    net = (edge_sum / n_edge) if n_edge else None
    return {
        "markets": n,
        "accuracy": acc,
        "net_edge": net,
        "n_priced": n_edge,
        "by_regime": {
            k: {"markets": len(v), "accuracy": (sum(v) / len(v)) if v else None}
            for k, v in by_reg.items()
        },
    }


def _verdict(stats: dict, *, cheap_n: int) -> str:
    min_n = int(getattr(config, "COMBO_MIN_MARKETS", 20) or 20)
    min_acc = float(getattr(config, "COMBO_MIN_ACCURACY", 0.55) or 0.55)
    min_ev = float(getattr(config, "COMBO_MIN_NET_EDGE", 0.0) or 0.0)
    n_priced = int(stats.get("n_priced") or 0)
    if cheap_n < min_n or n_priced < min_n:
        return "collecting"
    acc = stats.get("accuracy")
    ev = stats.get("net_edge")
    if acc is None or acc < min_acc:
        return "below_bar"
    if ev is None or ev < min_ev:
        return "no_edge"
    return "earned"


def build_combo_report(rows: list[dict]) -> dict:
    """Score pairwise combos + named foundational rules on unique-market rows."""
    dbn = _deadband()
    combos: dict[str, dict] = {}
    earned: list[dict] = []

    def _pack(name: str, lanes: tuple[str, ...] | list[str],
              stats_all: dict, stats_cheap: dict) -> dict:
        cheap_n = int(stats_cheap.get("markets") or 0)
        verdict = _verdict(stats_cheap, cheap_n=cheap_n)
        bypass = ("drift" not in lanes) and len(lanes) >= 2
        rec = {
            "lanes": list(lanes),
            "markets": stats_all.get("markets") or 0,
            "accuracy": stats_all.get("accuracy"),
            "net_edge": stats_all.get("net_edge"),
            "n_priced": stats_all.get("n_priced") or 0,
            "cheap_markets": cheap_n,
            "cheap_accuracy": stats_cheap.get("accuracy"),
            "cheap_net_edge": stats_cheap.get("net_edge"),
            "by_regime": stats_cheap.get("by_regime") or {},
            "verdict": verdict,
            "bypass_dual_gate": bypass,
        }
        apply_ok = (
            name != "mom_tech_midband"
            and (bypass or name in ("drift_flat_confirm", "agree2_cheap"))
        )
        if verdict == "earned" and apply_ok:
            earned.append({
                "name": name,
                "lanes": list(lanes),
                "bypass_dual_gate": bypass,
                "accuracy": rec["cheap_accuracy"],
                "net_edge": rec["cheap_net_edge"],
            })
        return rec

    for lanes in _PAIR_COMBOS:
        name = "+".join(lanes)

        def _pred(row, _lanes=lanes):
            s = _agree(_readings(row), _lanes, dbn)
            if s is None:
                return None
            return s > 0

        combos[name] = _pack(
            name, lanes,
            _score(rows, _pred, cheap_only=False),
            _score(rows, _pred, cheap_only=True),
        )

    def _drift_lag(row):
        d = _f(row.get("drift"))
        if d is None or abs(d) < 0.15:
            return None
        return d > 0

    def _drift_flat(row):
        d = _f(row.get("drift"))
        if d is None or abs(d) >= 0.10:
            return None
        s = _agree(_readings(row), ("mom", "tech", "xasset"), dbn)
        if s is None:
            # 2-of-3 is enough when one candidate is missing/flat
            for pair in (("mom", "tech"), ("mom", "xasset"), ("tech", "xasset")):
                s = _agree(_readings(row), pair, dbn)
                if s is not None:
                    break
        if s is None:
            return None
        return s > 0

    def _mom_tech_midband(row):
        """Shadow-only: mom+tech agreement on unique 50–58¢ fills."""
        entry = _f(row.get("entry_price"))
        lo = float(getattr(config, "COMBO_MIDBAND_LO", 0.50))
        hi = float(getattr(config, "COMBO_MIDBAND_HI", 0.58))
        if entry is None or entry < lo or entry > hi:
            return None
        s = _agree(_readings(row), ("mom", "tech"), dbn)
        if s is None:
            return None
        return s > 0

    def _agree2(row):
        rds = _readings(row)
        signed = []
        for ln in ("drift", "mom", "tech", "xasset"):
            v = rds.get(ln)
            if v is not None and abs(v) >= dbn:
                signed.append(1 if v > 0 else -1)
        if len(signed) < 2:
            return None
        ups = signed.count(1)
        dns = signed.count(-1)
        if ups >= 2 and ups > dns:
            return True
        if dns >= 2 and dns > ups:
            return False
        return None

    rules = {}
    for name, lanes, fn in (
        ("drift_lag", ("drift",), _drift_lag),
        ("drift_flat_confirm", ("mom", "tech", "xasset"), _drift_flat),
        ("agree2_cheap", ("drift", "mom", "tech", "xasset"), _agree2),
        ("mom_tech_midband", ("mom", "tech"), _mom_tech_midband),
    ):
        rules[name] = _pack(
            name, lanes,
            _score(rows, fn, cheap_only=False),
            _score(rows, fn, cheap_only=True),
        )

    # Prefer non-drift earned combos for the hot-path confirm.
    earned.sort(
        key=lambda e: (
            0 if e.get("bypass_dual_gate") else 1,
            -(e.get("net_edge") or 0.0),
            -(e.get("accuracy") or 0.0),
        )
    )
    return {
        "combos": combos,
        "rules": rules,
        "earned": earned,
        "meta": {
            "rows": len(rows),
            "n_earned": len(earned),
            "deadband": dbn,
            "max_entry": _max_entry(),
            "ts": time.time(),
        },
    }


def informed_market_rows(conn, *, hours: float | None = None) -> list[dict]:
    """One row per (strategy, market), preferring a tick that actually
    stamped lane reads. The last skip of a window is often a bare
    'waiting' / generic skip with NULLs — useless for combo scoring.
    """
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
                       PARTITION BY market_id
                       ORDER BY CASE WHEN drift IS NOT NULL
                                       OR mom IS NOT NULL
                                       OR tech IS NOT NULL
                                       OR xasset IS NOT NULL
                                     THEN 0 ELSE 1 END,
                                CASE WHEN action='buy' THEN 0 ELSE 1 END,
                                id DESC
                   ) AS rn
            FROM decision_events
            WHERE {where}
        )
        SELECT * FROM ranked WHERE rn = 1
    """
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def build_and_persist(*, hours: float | None = None) -> dict:
    hours = hours if hours is not None else float(
        getattr(config, "COMBO_EXPLORE_HOURS", 72) or 72
    )
    with db.get_conn() as conn:
        rows = informed_market_rows(conn, hours=hours)
    report = build_combo_report(rows)
    report["meta"]["hours"] = hours
    try:
        db.set_arena_state(_STATE_KEY, json.dumps(report))
    except Exception as e:
        logger.debug("combo_explorer persist failed: %s", e)
    return report


def load_report() -> dict:
    import time as _t
    global _CACHE
    now = _t.time()
    ts, cached = _CACHE
    if (now - ts) < _CACHE_TTL and cached:
        return cached
    try:
        raw = db.get_arena_state(_STATE_KEY)
        data = json.loads(raw) if raw else {}
        data = data if isinstance(data, dict) else {}
    except Exception:
        data = {}
    _CACHE = (now, data)
    return data


def maybe_refresh(force: bool = False) -> Optional[dict]:
    if not getattr(config, "COMBO_EXPLORE_ENABLED", True):
        return None
    iv = float(getattr(config, "COMBO_EXPLORE_INTERVAL_SEC", 300) or 300)
    prev = load_report()
    ts = float((prev.get("meta") or {}).get("ts") or 0.0)
    if not force and (time.time() - ts) < iv:
        return None
    try:
        report = build_and_persist()
        global _CACHE
        _CACHE = (time.time(), report)
        return report
    except Exception as e:
        logger.warning("combo explorer failed: %s", e)
        return None


def try_confirm(
    signals: dict,
    *,
    yes_mid: float | None,
    no_mid: float | None,
    yes_ask: float | None = None,
    no_ask: float | None = None,
    apply: bool | None = None,
) -> Optional[dict]:
    """Return an earned combo thesis, or None.

    Never fires on expensive mids (``COMBO_MAX_ENTRY``). Apply is gated by
    ``COMBO_CONFIRM_APPLY`` (default off) *and* an earned cheap sample.
    """
    if apply is None:
        apply = bool(getattr(config, "COMBO_CONFIRM_APPLY", False))
    if not apply:
        return None
    report = load_report()
    earned = list(report.get("earned") or [])
    if not earned:
        return None
    readings = {
        "drift": _f(signals.get("drift") if "drift" in signals
                    else signals.get("btc_drift")),
        "mom": _f(signals.get("mom") if "mom" in signals
                  else signals.get("btc_momentum")),
        "strat": _f(signals.get("strat")),
        "tech": _f(signals.get("tech") if "tech" in signals
                   else (signals.get("tech_mtf"))),
        "xasset": _f(signals.get("xasset")),
    }
    dbn = _deadband()
    max_e = _max_entry()
    for rec in earned:
        lanes = rec.get("lanes") or []
        if len(lanes) < int(getattr(config, "COMBO_CONFIRM_MIN_LANES", 2) or 2):
            # Named single-lane rules (drift_lag) do not confirm on their own.
            if rec.get("name") not in ("drift_flat_confirm", "agree2_cheap"):
                continue
        sign = _agree(readings, lanes, dbn)
        if rec.get("name") == "drift_flat_confirm":
            d = readings.get("drift")
            if d is None or abs(d) >= 0.10:
                continue
            sign = None
            for pair in (("mom", "tech"), ("mom", "xasset"), ("tech", "xasset"),
                         ("mom", "tech", "xasset")):
                sign = _agree(readings, pair, dbn)
                if sign is not None:
                    lanes = list(pair)
                    break
        if rec.get("name") == "agree2_cheap":
            signed = []
            for ln in ("drift", "mom", "tech", "xasset"):
                v = readings.get(ln)
                if v is not None and abs(v) >= dbn:
                    signed.append((ln, 1 if v > 0 else -1, v))
            ups = [s for s in signed if s[1] > 0]
            dns = [s for s in signed if s[1] < 0]
            if len(ups) >= 2 and len(ups) > len(dns):
                sign = 1
                lanes = [s[0] for s in ups]
            elif len(dns) >= 2 and len(dns) > len(ups):
                sign = -1
                lanes = [s[0] for s in dns]
            else:
                sign = None
        if sign is None:
            continue
        px = (yes_ask if sign > 0 else no_ask)
        if px is None:
            px = yes_mid if sign > 0 else no_mid
        if px is None or float(px) > max_e:
            continue
        vals = [abs(readings[ln]) for ln in lanes
                if readings.get(ln) is not None]
        if not vals:
            continue
        strength = min(vals)
        lean = 0.5 * strength
        p_model = 0.5 + (lean if sign > 0 else -lean)
        return {
            "name": rec.get("name") or "+".join(lanes),
            "lanes": list(lanes),
            "side": "yes" if sign > 0 else "no",
            "strength": strength,
            "lean": lean,
            "p_model": p_model,
            "bypass_dual_gate": bool(rec.get("bypass_dual_gate")),
            "accuracy": rec.get("accuracy"),
            "net_edge": rec.get("net_edge"),
        }
    return None

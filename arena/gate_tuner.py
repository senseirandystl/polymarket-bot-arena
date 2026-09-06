"""Suggest / apply bounded nudges to the live trade GATES.

Gates (dual-gate, dead-zone, NO-side) now decide more P&L than candidate
lanes. This tuner reads the unique-market scorecard and loosens a knob only
when skipped markets would have been *cheap and profitable* — never when
the counterfactual is a 93¢ favorite (that's the sweeper's book).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import config
import db

logger = logging.getLogger("arena.gate_tuner")

# skip_reason → config knob, direction, and "cheap enough to be directional"
_GATE_KNOBS = {
    "dead_zone": {
        "key": "DEAD_ZONE_DRIFT_MIN",
        "step": 0.02,
        "lo_mult": 0.5,
        "hi_mult": 1.5,
        "max_entry": 0.62,
    },
    "drift_dual_gate": {
        "key": "DRIFT_MIN_ABS_Z",
        "step": 0.03,
        "lo_mult": 0.6,
        "hi_mult": 1.4,
        "max_entry": 0.62,
    },
    "no_side_gate": {
        "key": "NO_SIDE_MAX_MID",
        "step": 0.02,
        "lo_mult": 0.85,
        "hi_mult": 1.15,
        "max_entry": 0.62,
        "loosen_increases": True,
    },
}

_STATE_KEY = "gate_overrides"
_REPORT_KEY = "gate_tuner"
_OV_CACHE: tuple[float, dict] = (0.0, {})
_OV_TTL = 5.0


def _load_scorecard(hours: float | None = None) -> dict:
    hours = hours if hours is not None else float(
        getattr(config, "LIVE_SCORECARD_HOURS", 72) or 72
    )
    from arena.live_scorecard import build_live_scorecard
    return build_live_scorecard(hours=hours)


def load_overrides() -> dict:
    import time
    global _OV_CACHE
    now = time.time()
    ts, cached = _OV_CACHE
    if (now - ts) < _OV_TTL:
        return cached
    try:
        raw = db.get_arena_state(_STATE_KEY)
        data = json.loads(raw) if raw else {}
        data = data if isinstance(data, dict) else {}
    except Exception:
        data = {}
    _OV_CACHE = (now, data)
    return data


def gate_float(name: str, default: float) -> float:
    """Hot-path read: paper profile → DB override → config default.

    Order: start from ``default`` (usually config.*), apply paper-only
    ``effective_float`` when active, then any persisted gate-tuner override.
    """
    if name == "_applied":
        return float(default)
    base = float(default)
    try:
        base = float(config.effective_float(name, base))
    except Exception:
        pass
    ov = load_overrides()
    val = ov.get(name)
    if val is None:
        return base
    try:
        return float(val)
    except (TypeError, ValueError):
        return base


def _band(default: float, spec: dict) -> tuple[float, float]:
    lo = float(default) * float(spec.get("lo_mult", 0.5))
    hi = float(default) * float(spec.get("hi_mult", 1.5))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def suggest(*, apply: bool = False, hours: float | None = None) -> dict:
    """Compute suggestions; optionally persist overrides (bounded)."""
    min_m = int(getattr(config, "GATE_TUNE_MIN_MARKETS", 30) or 30)
    loosen_wr = float(getattr(config, "GATE_TUNE_LOOSEN_WR", 0.58) or 0.58)
    loosen_edge = float(getattr(config, "GATE_TUNE_LOOSEN_EDGE", 0.02) or 0.02)
    card = _load_scorecard(hours)
    gates = card.get("gates") or {}
    suggestions: dict = {}
    current = load_overrides()
    new_ov = dict(current)

    for reason, spec in _GATE_KNOBS.items():
        g = gates.get(reason) or {}
        key = spec["key"]
        default = float(getattr(config, key))
        lo, hi = _band(default, spec)
        cur = float(current.get(key, default))
        # Dual-gate: judge the 50–58¢ band, not the 90¢ lock blend.
        src = g
        band_tag = ""
        if reason == "drift_dual_gate":
            bands = g.get("by_band") or {}
            mid = bands.get("mid_50_58") or {}
            if int(mid.get("n_hyp") or 0) >= min_m:
                src = mid
                band_tag = "mid_50_58 "
        n = int(src.get("markets") or g.get("markets") or 0)
        n_hyp = int(src.get("n_hyp") or 0)
        wr = src.get("wr")
        hyp = src.get("avg_hyp_pnl")
        entry = src.get("avg_entry")
        action = "hold"
        suggested = cur
        why = "collecting" if n_hyp < min_m else "no_edge_evidence"
        max_entry = float(spec.get("max_entry", 0.62))
        cheap = entry is not None and float(entry) <= max_entry
        hyp_ok = hyp is not None and float(hyp) >= loosen_edge
        wr_ok = wr is not None and float(wr) >= loosen_wr
        if n_hyp >= min_m and hyp_ok and wr_ok and cheap:
            step = float(spec["step"])
            if spec.get("loosen_increases"):
                suggested = min(hi, cur + step)
            else:
                suggested = max(lo, cur - step)
            if abs(suggested - cur) > 1e-12:
                action = "loosen"
                why = (
                    f"{band_tag}n={n} wr={wr:.2f} hyp={float(hyp):+.3f} "
                    f"entry={float(entry):.2f}"
                )
        elif n_hyp >= min_m and not cheap and wr_ok:
            why = f"expensive_entry={entry}"
        suggestions[key] = {
            "gate": reason,
            "action": action,
            "current": cur,
            "suggested": suggested,
            "default": default,
            "why": why,
            "markets": n,
            "n_hyp": n_hyp,
            "wr": wr,
            "avg_hyp_pnl": hyp,
            "avg_entry": entry,
        }
        if apply and action == "loosen":
            # Mid-band hyp must not write a global Z floor (sniper 15bp
            # underdogs share DRIFT_MIN_ABS_Z). Suggest-only for band evidence.
            if band_tag:
                suggestions[key]["why"] = (suggestions[key]["why"]
                                           + " (suggest-only: midband, not global)")
            else:
                cooldown = float(getattr(config, "GATE_TUNE_APPLY_COOLDOWN_SEC", 86400) or 0)
                last_ts = float((current.get("_applied") or {}).get(key) or 0)
                import time as _t
                if cooldown > 0 and (_t.time() - last_ts) < cooldown:
                    suggestions[key]["action"] = "hold"
                    suggestions[key]["why"] = "cooldown"
                else:
                    new_ov[key] = round(suggested, 4)
                    applied_at = dict(new_ov.get("_applied") or {})
                    applied_at[key] = _t.time()
                    new_ov["_applied"] = applied_at

    applied = False
    if apply and {k: v for k, v in new_ov.items() if k != "_applied"} != {
        k: v for k, v in current.items() if k != "_applied"
    }:
        db.set_arena_state(_STATE_KEY, json.dumps(new_ov))
        global _OV_CACHE
        import time as _t
        _OV_CACHE = (_t.time(), new_ov)
        applied = True
        logger.info("Gate tuner applied overrides: %s", new_ov)

    report = {
        "applied": applied,
        "suggestions": suggestions,
        "overrides": new_ov if apply else current,
        "ts": card.get("meta", {}).get("ts"),
    }
    try:
        db.set_arena_state(_REPORT_KEY, json.dumps(report))
    except Exception:
        pass
    return report


def maybe_tune() -> Optional[dict]:
    if not getattr(config, "GATE_TUNE_ENABLED", True):
        return None
    # Never piggyback lane auto-approve — that toggle is for candidate lanes,
    # not for loosening dual-gate / NO-side / dead-zone.
    apply = bool(getattr(config, "GATE_TUNE_APPLY", False))
    return suggest(apply=apply)

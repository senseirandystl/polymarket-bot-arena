"""Regime-adaptive sizing from live ``regime_performance``.

Reads the detector's rolling per-regime win-rate / P&L and returns a size
multiplier in [REGIME_ADAPT_SIZE_MIN, REGIME_ADAPT_SIZE_MAX]. Bad regimes
(e.g. low_vol_trend sub-50% WR) automatically get smaller bets; strong
regimes get a mild boost. No hard-coded skip list — the data drives it.

Hot-path cached (REGIME_ADAPT_CACHE_SEC). Fail-open to 1.0 on any error.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import config
import db

logger = logging.getLogger("arena.regime_adapt")

_cache: tuple[float, dict[str, float]] = (0.0, {})


def _load_perf() -> dict[str, Any]:
    raw = db.get_arena_state("regime_performance")
    if not raw:
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _mult_from_stats(n: int, wins: int, pnl: float) -> float:
    min_n = int(getattr(config, "REGIME_ADAPT_MIN_TRADES", 15))
    if n < min_n:
        return 1.0
    wr = wins / n if n else 0.5
    bad = float(getattr(config, "REGIME_ADAPT_BAD_WR", 0.48))
    good = float(getattr(config, "REGIME_ADAPT_GOOD_WR", 0.62))
    lo = float(getattr(config, "REGIME_ADAPT_SIZE_MIN", 0.35))
    hi = float(getattr(config, "REGIME_ADAPT_SIZE_MAX", 1.15))
    # Linear map WR in [bad, good] → [lo, hi]; clamp outside.
    if good <= bad + 1e-9:
        return 1.0
    t = (wr - bad) / (good - bad)
    t = max(0.0, min(1.0, t))
    mult = lo + t * (hi - lo)
    # Extra damp when net P&L is negative even if WR is middling.
    if pnl < 0 and wr < good:
        mult = min(mult, (lo + 1.0) / 2.0)
    return round(mult, 3)


def _rebuild() -> dict[str, float]:
    perf = _load_perf()
    out: dict[str, float] = {}
    for label, st in perf.items():
        if not isinstance(st, dict):
            continue
        n = int(st.get("n") or 0)
        wins = int(st.get("wins") or 0)
        pnl = float(st.get("pnl") or 0.0)
        out[str(label)] = _mult_from_stats(n, wins, pnl)
    return out


def size_multiplier(regime_label: Optional[str]) -> float:
    """Size mult for the current regime label (1.0 if unknown / disabled)."""
    if not getattr(config, "REGIME_ADAPT_ENABLED", True):
        return 1.0
    if not regime_label or regime_label in ("unknown", ""):
        return 1.0
    global _cache
    now = time.time()
    ttl = float(getattr(config, "REGIME_ADAPT_CACHE_SEC", 30.0))
    if (now - _cache[0]) >= ttl:
        try:
            _cache = (now, _rebuild())
        except Exception as e:
            logger.debug("regime_adapt rebuild failed: %s", e)
            return 1.0
    return float(_cache[1].get(str(regime_label), 1.0))


def snapshot() -> dict[str, Any]:
    """Dashboard / soak-report view of current multipliers."""
    try:
        table = _rebuild()
        perf = _load_perf()
        rows = []
        for label, mult in sorted(table.items()):
            st = perf.get(label) or {}
            n = int(st.get("n") or 0)
            wins = int(st.get("wins") or 0)
            rows.append({
                "regime": label,
                "n": n,
                "wr": (wins / n) if n else None,
                "pnl": round(float(st.get("pnl") or 0.0), 2),
                "size_mult": mult,
            })
        return {"enabled": bool(getattr(config, "REGIME_ADAPT_ENABLED", True)),
                "regimes": rows}
    except Exception as e:
        return {"enabled": False, "error": str(e), "regimes": []}

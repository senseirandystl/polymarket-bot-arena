"""Paper-eval decision taxes: high-vol favorite + mom-vs-drift fight.

Pure functions so BaseBot and sniper share the same bars. Not a skip —
they raise min_edge (and optionally cut size) so 5¢ “edges” at 56¢ die.
"""

from __future__ import annotations

from typing import Optional

import config


def high_vol_favorite_mult(
    *,
    strategy_type: str | None,
    regime: str | None,
    side_mid: float | None,
) -> tuple[float, float]:
    """(edge_mult, size_mult) when high_vol_trend ∧ chosen-side mid ≥ 0.52."""
    if not bool(getattr(config, "HIGH_VOL_FAVORITE_ENABLED", True)):
        return 1.0, 1.0
    regimes = tuple(getattr(config, "HIGH_VOL_FAVORITE_REGIMES",
                            ("high_vol_trend",)) or ())
    if str(regime or "") not in regimes:
        return 1.0, 1.0
    strats = tuple(getattr(config, "HIGH_VOL_FAVORITE_STRATEGIES",
                           ("momentum", "sniper", "hybrid")) or ())
    if str(strategy_type or "").lower() not in {s.lower() for s in strats}:
        return 1.0, 1.0
    try:
        mid = float(side_mid)
    except (TypeError, ValueError):
        return 1.0, 1.0
    bar = float(getattr(config, "HIGH_VOL_FAVORITE_MID", 0.52))
    if mid < bar:
        return 1.0, 1.0
    e = float(getattr(config, "HIGH_VOL_FAVORITE_EDGE_MULT", 1.50) or 1.0)
    s = float(getattr(config, "HIGH_VOL_FAVORITE_SIZE_MULT", 0.60) or 1.0)
    return max(1.0, e), min(1.0, max(0.1, s))


def mom_drift_fight_mult(
    *,
    mom: float | None,
    drift: float | None,
    regime: str | None = None,
) -> float:
    """Extra min_edge mult when 1m mom fights a non-trivial drift."""
    if not bool(getattr(config, "MOM_DRIFT_FIGHT_ENABLED", True)):
        return 1.0
    try:
        m = float(mom)
        d = float(drift)
    except (TypeError, ValueError):
        return 1.0
    veto = float(getattr(config, "DRIFT_VETO_MIN", 0.05) or 0.05)
    need = float(getattr(config, "MOM_DRIFT_FIGHT_MOM_ABS", 0.50) or 0.50)
    if abs(d) < veto or abs(m) < need:
        return 1.0
    if (m > 0) == (d > 0):
        return 1.0
    hv = tuple(getattr(config, "HIGH_VOL_FAVORITE_REGIMES",
                       ("high_vol_trend",)) or ())
    if str(regime or "") in hv:
        return float(getattr(
            config, "MOM_DRIFT_FIGHT_HIGH_VOL_EDGE_MULT", 1.75) or 1.75)
    return float(getattr(config, "MOM_DRIFT_FIGHT_EDGE_MULT", 1.35) or 1.35)

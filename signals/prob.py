"""Single live P(win) for directional edge, sniper, sweeper, and overlay.

Tanh(z) is a bounded lane score, not a probability. Edge math must call
``live_side_prob`` / ``directional_net_edge`` — never ``0.5 + 0.5·tanh``.
"""

from __future__ import annotations

import math
from typing import Optional

import polymarket_fills


def _phi_from_z(z: float) -> float:
    if not math.isfinite(z):
        return 0.5
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def phi_yes_from_signals(
    signals: Optional[dict] = None,
    *,
    signed_lane: Optional[float] = None,
) -> float:
    """Φ(z) in [0, 1] from ``btc_implied_yes`` → ``btc_drift_z`` → artanh(lane).

    Missing / non-finite → 0.5. ``signed_lane`` is YES-frame ``btc_drift``.
    """
    sig = signals or {}
    try:
        iy = sig.get("btc_implied_yes")
        if iy is not None:
            iy = float(iy)
            if math.isfinite(iy):
                return max(0.0, min(1.0, iy))
    except (TypeError, ValueError):
        pass
    try:
        z = sig.get("btc_drift_z")
        if z is not None:
            z = float(z)
            if math.isfinite(z):
                return max(0.0, min(1.0, _phi_from_z(z)))
    except (TypeError, ValueError):
        pass
    if signed_lane is not None:
        try:
            t = float(signed_lane)
            if math.isfinite(t):
                t = max(-0.999999, min(0.999999, t))
                return max(0.0, min(1.0, _phi_from_z(math.atanh(t))))
        except (TypeError, ValueError):
            pass
    return 0.5


def _empirical_yes(
    signals: dict,
    strategy_type: Optional[str],
) -> Optional[float]:
    """Overlay lookup. Missing / thin / not promoted → None (use Φ)."""
    if not strategy_type:
        return None
    try:
        from arena.empirical_prob import lookup_yes
    except Exception:
        return None
    try:
        return lookup_yes(signals, strategy_type)
    except Exception:
        return None


def live_side_prob(
    *,
    side: str,
    signals: Optional[dict] = None,
    strategy_type: Optional[str] = None,
    signed_lane: Optional[float] = None,
) -> tuple[float, str]:
    """Return ``(p_side, source)`` with source in ``empirical`` / ``phi`` / ``half``.

    Empirical only when the overlay has promoted this strategy×regime and the
    cell is thick enough; otherwise Φ. Never tanh-as-probability.
    """
    sig = dict(signals or {})
    yes = None
    source = "half"
    emp = _empirical_yes(sig, strategy_type)
    if emp is not None:
        try:
            emp_f = float(emp)
        except (TypeError, ValueError):
            emp_f = float("nan")
        if math.isfinite(emp_f):
            yes = max(0.0, min(1.0, emp_f))
            source = "empirical"
    if yes is None:
        yes = phi_yes_from_signals(sig, signed_lane=signed_lane)
        has_phi = (
            sig.get("btc_implied_yes") is not None
            or sig.get("btc_drift_z") is not None
            or signed_lane is not None
        )
        source = "phi" if has_phi else "half"
    if str(side).lower() == "no":
        return 1.0 - yes, source
    return yes, source


def directional_net_edge(p_side: float, ask: float) -> float:
    """``p_side − ask − taker fee/share``. No trust multiplier."""
    try:
        p = float(p_side)
        a = float(ask)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(p) or not math.isfinite(a):
        return 0.0
    fee = polymarket_fills.fee_per_share(a, is_maker=False)
    return p - a - float(fee)

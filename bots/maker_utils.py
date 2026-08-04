"""Shared helpers for zone/maker bots (fee-zone, late-window, btc-maker).

Paper and live currently fill as *takers* against warm asks. Decision edges
must therefore use a consistent mid (crowd view) and ask (executable cost).
A large mid/ask gap is almost always a book-field bug or crossed book — not
a free gift of edge.
"""

from __future__ import annotations

from typing import Optional, Tuple

import config


def mid_ask_gap_ok(
    side_mid: float,
    side_exec: float,
    *,
    max_gap: float | None = None,
    allow_ask_below_mid: float = 0.02,
) -> Tuple[bool, str]:
    """Return (ok, reason) for mid vs executable-ask consistency.

    Rejects:
      * missing / non-finite prices
      * |mid − ask| above ``max_gap`` (config.MAKER_MAX_MID_ASK_GAP)
      * ask substantially *below* mid (impossible on a sane one-sided book
        for the same token — the classic phantom cheap-fill path)
    """
    try:
        mid = float(side_mid)
        ask = float(side_exec)
    except (TypeError, ValueError):
        return False, "mid/ask non-numeric"
    if not (0.0 < mid < 1.0) or not (0.0 < ask < 1.0):
        return False, f"mid/ask out of (0,1): mid={mid:.3f} ask={ask:.3f}"
    gap = abs(mid - ask)
    lim = float(max_gap if max_gap is not None
                else getattr(config, "MAKER_MAX_MID_ASK_GAP", 0.08))
    if gap > lim + 1e-12:
        return False, f"mid/ask gap {gap:.3f} > {lim:.2f} (mid={mid:.2f} ask={ask:.2f})"
    if ask < mid - float(allow_ask_below_mid):
        return False, (
            f"ask {ask:.2f} << mid {mid:.2f} "
            f"(crossed/stale book — refuse fantasy fill)"
        )
    return True, ""


def resolve_side_exec(
    market: dict,
    side: str,
    side_mid: float,
) -> Tuple[Optional[float], str]:
    """Pick executable ask for ``side``; fall back to mid only if ask missing."""
    if side == "yes":
        ask = market.get("yes_ask")
    else:
        ask = market.get("no_ask")
    if ask is None:
        return float(side_mid), "ask_missing_used_mid"
    try:
        return float(ask), "ask"
    except (TypeError, ValueError):
        return float(side_mid), "ask_bad_used_mid"


def maker_kelly_amount(
    edge: float,
    price: float,
    bankroll_slice: float,
    *,
    size_pct_cap: float,
    inv_headroom: float,
    kelly_fraction: float | None = None,
) -> float:
    """Conservative Kelly USD for maker taker-fills, capped by inventory + pct.

    Uses the same concave edge calibration as directional bots when available.
    """
    price = max(float(price), 0.01)
    try:
        from bots.edge_calibration import calibrated_sizing_edge
        se = calibrated_sizing_edge(float(edge))
    except Exception:
        cap = float(getattr(config, "KELLY_EDGE_CAP", 0.10))
        se = min(max(0.0, float(edge)), cap)
    kf = float(kelly_fraction if kelly_fraction is not None
               else getattr(config, "KELLY_FRACTION", 0.25))
    f_star = se / max(1.0 - price, 0.05)
    kelly_usd = f_star * kf * max(0.0, float(bankroll_slice))
    pct_cap = max(0.0, float(size_pct_cap)) * float(config.get_max_position())
    return max(0.0, min(kelly_usd, pct_cap, float(inv_headroom)))

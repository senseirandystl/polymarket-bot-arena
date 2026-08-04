"""Edge → sizing confidence calibration (confidence-inversion fix).

Live soak (2026-07-31 → 2026-08-02): confidence = EDGE_TO_CONFIDENCE × edge
made *larger* edges look more confident, but those trades had the worst WR
(conf ≥ 0.50: 43% WR / −$5; conf < 0.20: 68% WR / +$167). Kelly already
capped sizing edge, but everything at/above the cap still max-sized, and
makers/legacy paths treated high conf as quality.

This module:

1. **Concave sizing edge** — full credit for modest edges; diminishing
   returns (and a hard cap) for outsized model–market disagreement.
2. **Quality confidence** — a [0,1] score that rises with *historically
   good* structure (moderate edge, strong drift, lagging mid), not raw
   edge magnitude. Used for logging / min_confidence only; sizing never
   multiplies by confidence.
"""

from __future__ import annotations

import math
from typing import Optional

import config


def calibrated_sizing_edge(
    raw_edge: float,
    *,
    full_credit: float | None = None,
    hard_cap: float | None = None,
    taper_scale: float | None = None,
) -> float:
    """Map raw fee-adjusted edge → edge used for Kelly sizing.

    * ``e ≤ full_credit`` → pass through (honest small edges size normally)
    * ``full_credit < e ≤ hard_cap`` → concave taper (tanh)
    * ``e > hard_cap`` → hard_cap

    So a 15¢ "edge" no longer gets the same full Kelly as a well-calibrated
    4¢ edge after the old flat KELLY_EDGE_CAP alone.
    """
    e = max(0.0, float(raw_edge))
    e0 = float(full_credit if full_credit is not None
               else getattr(config, "EDGE_CALIB_FULL_CREDIT", 0.04))
    cap = float(hard_cap if hard_cap is not None
                else getattr(config, "KELLY_EDGE_CAP", 0.08))
    scale = float(taper_scale if taper_scale is not None
                  else getattr(config, "EDGE_CALIB_TAPER_SCALE", 0.06))
    if e <= e0:
        return e
    # Remaining room to the hard cap, approached asymptotically via tanh.
    room = max(0.0, cap - e0)
    if room <= 1e-12 or scale <= 1e-12:
        return min(e, cap)
    over = e - e0
    tapered = e0 + room * math.tanh(over / scale)
    return min(tapered, cap)


def quality_confidence(
    *,
    edge: float,
    abs_drift: float,
    side_mid: float,
    side: str = "yes",
    regime_label: Optional[str] = None,
) -> float:
    """Structure-based confidence in [0, 1] — NOT proportional to |edge|.

    Peaks when:
      * edge is moderate (not huge disagreement)
      * |drift| is meaningful (validated fundamental)
      * mid still lags a coin-flip-to-mild-favorite band (BE-friendly)
    Decays when edge is extreme (stale-input regime) or drift is flat.
    """
    e = max(0.0, float(edge))
    d = max(0.0, float(abs_drift))
    mid = float(side_mid)

    # Edge quality: full credit near EDGE_CALIB_FULL_CREDIT, decay past it.
    e0 = float(getattr(config, "EDGE_CALIB_FULL_CREDIT", 0.04))
    if e <= 1e-9:
        edge_q = 0.0
    elif e <= e0:
        edge_q = 0.55 + 0.35 * (e / max(e0, 1e-6))
    else:
        # Diminishing: 15¢ edge is *less* confident than 4¢.
        over = (e - e0) / max(e0, 1e-6)
        edge_q = max(0.20, 0.90 - 0.35 * math.tanh(over))

    # Drift quality: saturate around FLOW_ONLY_DRIFT_FULL_TRUST.
    d_full = float(getattr(config, "FLOW_ONLY_DRIFT_FULL_TRUST", 0.25))
    drift_q = min(1.0, d / max(d_full, 1e-6))

    # Price-band quality: 0.45–0.62 mids historically print; deep longshots
    # and expensive favorites less so (after fees).
    if 0.45 <= mid <= 0.62:
        mid_q = 1.0
    elif 0.38 <= mid < 0.45 or 0.62 < mid <= 0.70:
        mid_q = 0.70
    elif mid < 0.38:
        mid_q = 0.40  # underdog / consensus fight
    else:
        mid_q = 0.45  # expensive

    # Mild regime prior (data-driven damp applied elsewhere via regime_adapt).
    reg = (regime_label or "").lower()
    reg_q = 0.85 if reg == "low_vol_trend" else 1.0

    conf = (0.40 * edge_q + 0.35 * drift_q + 0.25 * mid_q) * reg_q
    return max(0.0, min(0.95, conf))

"""Smooth scoring curves shared by signals and strategies.

Hard thresholds make decisions brittle: a value of 0.199 and 0.201 should not
produce opposite behavior. These helpers map raw inputs onto smooth, bounded
scores so lane values and confidence functions degrade gracefully near their
boundaries. All outputs are plain floats, safe on the 1s hot path.

NOTE: these curves are for *scoring* (lane values, confidence, zone
membership). Validated hard SAFETY gates (consensus/high-price guards, drift
veto, book-sum gate, model-lean floor) intentionally stay hard — they encode
measured live-loss boundaries, not scores (see BUG_HISTORY #24-#28).
"""

import math


def soft_saturate(x: float, scale: float) -> float:
    """tanh-based saturation: maps x onto (-1, 1), ~linear below ``scale``.

    Drop-in replacement for ``max(-1, min(1, x / scale))``: identical slope
    near zero but approaches +/-1 smoothly instead of clipping, so values just
    past the calibration point don't all collapse to exactly 1.0 (the
    "sign(tape)" saturation disease of the old pm/cvd lanes).
    """
    if scale <= 0:
        return 0.0
    return math.tanh(x / scale)


def sigmoid(x: float, center: float = 0.0, steepness: float = 1.0) -> float:
    """Logistic curve in (0, 1), 0.5 at ``center``."""
    z = steepness * (x - center)
    # Guard exp overflow for extreme inputs.
    if z > 35:
        return 1.0
    if z < -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def gaussian_zone(x: float, center: float, width: float) -> float:
    """Smooth zone membership in (0, 1]: 1.0 at ``center``, ~0.6 at 1 width.

    Replaces hard price-zone brackets (``lo <= x <= hi``) for *confidence
    scaling*: a price 1c outside a preferred zone gets slightly less
    conviction, not a cliff.
    """
    if width <= 0:
        return 1.0 if x == center else 0.0
    z = (x - center) / width
    return math.exp(-0.5 * z * z)


def smooth_ramp(x: float, start: float, end: float) -> float:
    """Smoothstep from 0 (x<=start) to 1 (x>=end); C1-continuous between."""
    if end == start:
        return 1.0 if x >= end else 0.0
    t = (x - start) / (end - start)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

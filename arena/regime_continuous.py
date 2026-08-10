"""Continuous regime residual: w = clip(w0 + B·F, lo, hi).

Small ridge-style residual so weights move smoothly as relative features
drift without requiring a discrete label flip. Disabled until sample mass
exists (REGIME_CONTINUOUS_BLEND + min samples).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

import config
import db

logger = logging.getLogger("arena.regime_continuous")

STATE_KEY = "regime_continuous_B"
# Feature order for the residual vector
FEATURE_ORDER = ("vol_c", "dir_c", "chop_c", "flow_c")
CORE_LANES = ("drift", "mom", "strat")

_lock = threading.Lock()
_B: dict[str, dict[str, list[float]]] = {}  # lane -> strategy -> coeffs
_n_obs = 0
_loaded = False


def _empty_coeffs() -> list[float]:
    return [0.0] * len(FEATURE_ORDER)


def _ensure_loaded() -> None:
    global _loaded, _B, _n_obs
    if _loaded:
        return
    _loaded = True
    try:
        raw = db.get_arena_state(STATE_KEY)
        if not raw:
            return
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return
        _n_obs = int(data.get("n_obs") or 0)
        raw_b = data.get("B") or {}
        for lane, by_st in raw_b.items():
            if not isinstance(by_st, dict):
                continue
            _B.setdefault(lane, {})
            for st, coeffs in by_st.items():
                if isinstance(coeffs, list) and len(coeffs) == len(FEATURE_ORDER):
                    _B[lane][st] = [float(x) for x in coeffs]
    except Exception as e:
        logger.debug("regime_continuous load failed: %s", e)


def persist() -> None:
    try:
        with _lock:
            payload = {
                "B": {lane: dict(st) for lane, st in _B.items()},
                "n_obs": _n_obs,
                "updated_at": time.time(),
            }
        db.set_arena_state(STATE_KEY, json.dumps(payload))
    except Exception as e:
        logger.debug("regime_continuous persist failed: %s", e)


def feature_vector(features: Optional[dict]) -> list[float]:
    """Centered relative features in ~[-0.5, 0.5]."""
    f = features or {}
    vol = float(f.get("vol_rel", f.get("vol", 0.5)) or 0.5)
    direction = float(f.get("direction", f.get("trend", 0.5)) or 0.5)
    chop = float(f.get("chop", 0.5) or 0.5)
    flow = float(f.get("flow", 0.3) or 0.3)
    return [vol - 0.5, direction - 0.5, chop - 0.5, flow - 0.5]


def residual(
    lane: str,
    strategy_type: str,
    features: Optional[dict],
) -> float:
    """Signed weight delta for this lane (capped)."""
    if not _blend_enabled():
        return 0.0
    _ensure_loaded()
    min_n = int(getattr(config, "REGIME_CONTINUOUS_MIN_SAMPLES", 200))
    with _lock:
        if _n_obs < min_n:
            return 0.0
        coeffs = (_B.get(lane) or {}).get(strategy_type)
        if not coeffs:
            return 0.0
        c = list(coeffs)
    vec = feature_vector(features)
    raw = sum(c[i] * vec[i] for i in range(len(FEATURE_ORDER)))
    cap = float(getattr(config, "REGIME_CONTINUOUS_MAX_DELTA", 0.08))
    return max(-cap, min(cap, raw))


def _blend_enabled() -> bool:
    try:
        from arena.regime_settings import get_bool
        return bool(get_bool("continuous_blend"))
    except Exception:
        return bool(getattr(config, "REGIME_CONTINUOUS_BLEND", False))


def apply_residuals(
    weights: dict[str, float],
    strategy_type: str,
    features: Optional[dict],
) -> dict[str, float]:
    """Return new weights dict with continuous residual applied to core lanes."""
    if not _blend_enabled():
        return weights
    out = dict(weights)
    for lane in CORE_LANES:
        if lane not in out:
            continue
        d = residual(lane, strategy_type, features)
        if abs(d) < 1e-12:
            continue
        w = float(out[lane]) + d
        # Drift floor: never fully zero a positive drift weight via residual
        if lane == "drift" and float(weights.get("drift") or 0) > 0:
            w = max(0.05, w)
        out[lane] = max(0.0, min(0.95, w))
    return out


def observe(
    lane: str,
    strategy_type: str,
    features: Optional[dict],
    *,
    correct: bool,
    reading_sign: float,
) -> None:
    """Tiny online update: push residual toward features when lane sign was right.

    ``reading_sign`` is the signed lane reading; only updates when |reading| > 0.
    """
    if not _blend_enabled():
        return
    if strategy_type is None or lane not in CORE_LANES:
        return
    if abs(float(reading_sign or 0.0)) < 1e-9:
        return
    _ensure_loaded()
    vec = feature_vector(features)
    eta = float(getattr(config, "REGIME_CONTINUOUS_ETA", 0.002))
    # Target: correct → reinforce sign*features; wrong → oppose
    sign = 1.0 if correct else -1.0
    global _n_obs
    with _lock:
        by_st = _B.setdefault(lane, {})
        coeffs = list(by_st.get(strategy_type) or _empty_coeffs())
        for i in range(len(FEATURE_ORDER)):
            coeffs[i] += eta * sign * vec[i]
            # L2 soft shrink
            coeffs[i] *= 0.999
            coeffs[i] = max(-0.5, min(0.5, coeffs[i]))
        by_st[strategy_type] = coeffs
        _n_obs += 1
        n = _n_obs
    if n % 50 == 0:
        persist()


def status() -> dict[str, Any]:
    _ensure_loaded()
    with _lock:
        return {
            "enabled": _blend_enabled(),
            "n_obs": _n_obs,
            "min_samples": int(getattr(config, "REGIME_CONTINUOUS_MIN_SAMPLES", 200)),
            "lanes": list(_B.keys()),
        }


def reset_for_tests() -> None:
    global _B, _n_obs, _loaded
    with _lock:
        _B = {}
        _n_obs = 0
        _loaded = True

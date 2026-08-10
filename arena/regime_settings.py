"""Dashboard-editable regime control flags (arena_state + config defaults).

Boolean flags and the adapt primary mode are stored in arena_state so the
Settings tab can flip them without restarting. Hot-path readers use a short
TTL cache (same idea as bot_mode / kelly caches).
"""

from __future__ import annotations

import time
from typing import Any, Optional

import config
import db

# name -> (arena_state key, config attr, default)
_BOOL_FLAGS: dict[str, tuple[str, str, bool]] = {
    "continuous_blend": (
        "regime_continuous_blend", "REGIME_CONTINUOUS_BLEND", False,
    ),
    "use_relative": (
        "regime_use_relative", "REGIME_USE_RELATIVE", True,
    ),
    "profile_adapt": (
        "regime_profile_adapt", "REGIME_PROFILE_ADAPT_ENABLED", True,
    ),
    "profile_seeds": (
        "regime_profile_seeds", "REGIME_PROFILE_SEEDS_ENABLED", True,
    ),
    "hard_skip": (
        "regime_hard_skip", "REGIME_HARD_SKIP_ENABLED", False,
    ),
    "adapt_enabled": (
        "regime_adapt_enabled", "REGIME_ADAPT_ENABLED", True,
    ),
    "freq_target": (
        "regime_freq_target", "REGIME_FREQ_TARGET_ENABLED", False,
    ),
    # Alias for the existing regime_conditioning key (portfolio + map tilt)
    "conditioning": (
        "regime_conditioning", "REGIME_CONDITIONING_ENABLED", True,
    ),
    # Strategy×regime style-skip (toxic strategy stands down in that regime)
    "style_skip": (
        "regime_style_skip", "REGIME_STYLE_SKIP_ENABLED", True,
    ),
}

_PRIMARY_KEY = "regime_adapt_primary"
_CACHE_TTL = 3.0
_cache: dict[str, tuple[float, Any]] = {}


def _cache_get(key: str):
    hit = _cache.get(key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1], True
    return None, False


def _cache_set(key: str, val: Any) -> None:
    _cache[key] = (time.time(), val)


def invalidate_cache(name: Optional[str] = None) -> None:
    if name is None:
        _cache.clear()
        return
    _cache.pop(name, None)
    if name in _BOOL_FLAGS:
        _cache.pop(_BOOL_FLAGS[name][0], None)


def get_bool(name: str) -> bool:
    """Read a regime bool setting (cached). Unknown name → False."""
    if name not in _BOOL_FLAGS:
        return False
    state_key, cfg_attr, default = _BOOL_FLAGS[name]
    cached, ok = _cache_get(state_key)
    if ok:
        return bool(cached)
    # conditioning reuses existing db helper for compatibility
    if name == "conditioning":
        try:
            val = bool(db.get_regime_conditioning())
        except Exception:
            val = bool(getattr(config, cfg_attr, default))
        _cache_set(state_key, val)
        return val
    raw = db.get_arena_state(state_key)
    if raw is None:
        val = bool(getattr(config, cfg_attr, default))
    else:
        val = str(raw) in ("1", "true", "True", "yes", "on")
    _cache_set(state_key, val)
    return val


def set_bool(name: str, enabled: bool) -> bool:
    """Persist a regime bool setting; returns the value stored."""
    if name not in _BOOL_FLAGS:
        raise ValueError(f"unknown regime setting: {name}")
    state_key, _, _ = _BOOL_FLAGS[name]
    if name == "conditioning":
        db.set_regime_conditioning(bool(enabled))
    else:
        db.set_arena_state(state_key, "1" if enabled else "0")
    invalidate_cache(name)
    _cache_set(state_key, bool(enabled))
    return bool(enabled)


def get_adapt_primary() -> str:
    """``style`` (default) or ``throttle`` (legacy)."""
    cached, ok = _cache_get(_PRIMARY_KEY)
    if ok and cached in ("style", "throttle"):
        return str(cached)
    raw = db.get_arena_state(_PRIMARY_KEY)
    if raw is None:
        val = str(getattr(config, "REGIME_ADAPT_PRIMARY", "style") or "style")
    else:
        val = str(raw).strip().lower()
    if val not in ("style", "throttle"):
        val = "style"
    _cache_set(_PRIMARY_KEY, val)
    return val


def set_adapt_primary(mode: str) -> str:
    mode = str(mode or "style").strip().lower()
    if mode not in ("style", "throttle"):
        raise ValueError("adapt_primary must be 'style' or 'throttle'")
    db.set_arena_state(_PRIMARY_KEY, mode)
    invalidate_cache(_PRIMARY_KEY)
    _cache_set(_PRIMARY_KEY, mode)
    return mode


def snapshot() -> dict[str, Any]:
    """Full payload for Settings / API."""
    flags = {name: get_bool(name) for name in _BOOL_FLAGS}
    return {
        "flags": flags,
        "adapt_primary": get_adapt_primary(),
        "defaults": {
            name: bool(getattr(config, cfg, default))
            for name, (_, cfg, default) in _BOOL_FLAGS.items()
        },
        "labels": {
            "continuous_blend": "Continuous residual blend (w₀ + B·F)",
            "use_relative": "Relative vol/trend calibration",
            "profile_adapt": "Per-regime profile tuner (by_regime writes)",
            "profile_seeds": "Per-regime profile seeds",
            "hard_skip": "Hard directional skip (toxic regimes)",
            "adapt_enabled": "Regime adapt engine master",
            "freq_target": "Soft frequency target (edge ease)",
            "conditioning": "Regime map capital / tuner conditioning",
            "style_skip": "Strategy×regime style-skip (data-driven)",
        },
        "blurb": {
            "continuous_blend":
                "When ON, core lane weights get a small continuous residual "
                "from relative features (capped). Needs sample mass; off by default.",
            "use_relative":
                "When ON, high/low vol means high/low for recent BTC (percentiles), "
                "not a fixed absolute threshold.",
            "profile_adapt":
                "When ON, the core-lane tuner may write per-regime weights "
                "under lane_overrides.by_regime.",
            "profile_seeds":
                "When ON, hand-set regime×strategy seeds apply until overrides exist.",
            "hard_skip":
                "When ON, toxic regimes can hard-block directionals (emergency bar). "
                "Default OFF under adapt-not-throttle.",
            "adapt_enabled":
                "Master switch for regime_adapt adjustments (size/edge/style).",
            "freq_target":
                "Optional soft ease of min_edge when fill rate is low (experimental).",
            "conditioning":
                "Portfolio allocator + map tilt toward bots that work in the "
                "current validated regime cell.",
            "style_skip":
                "When ON, a strategy that is live-toxic in a regime (WR/P&L bar) "
                "stands down only for that strategy — other bots keep trading. "
                "Clears with hysteresis when the cell recovers.",
            "adapt_primary":
                "style = reweight lanes/capital; throttle = legacy size cuts + "
                "higher bars in weak regimes.",
        },
    }

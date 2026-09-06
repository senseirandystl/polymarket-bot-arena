"""Strategy Lab control plane (SQLite arena_state key strategy_lab_control)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import db

from signals.strategy_pipeline.founders import ensure_founder_locks

STATE_KEY = "strategy_lab_control"
LOCK_TTL_SEC = 180.0

_SETTING_MAP: dict[str, tuple[str, Callable[[Any], Any], Any]] = {
    "STRATEGY_LAB_AUTO_PROMOTE": ("auto_promote", lambda v: bool(v), False),
    "STRATEGY_LAB_LLM_PROVIDER": (
        "llm_provider", lambda v: str(v or "none").strip().lower(), "none",
    ),
    "STRATEGY_LAB_PAPER_SLOTS": ("paper_slots", lambda v: max(0, int(v)), 0),
}

_LLM_ALLOWED = frozenset({"none", "ollama", "grok", "xai"})


def _blank() -> dict[str, Any]:
    return {
        "paused": False,
        "last_cycle": None,
        "updated_ts": 0.0,
        "tick_lock_ts": 0.0,
        "tick_lock_pid": None,
        "auto_promote": None,
        "llm_provider": None,
        "paper_slots": None,
        "founder_locks": None,
    }


def load_control() -> dict[str, Any]:
    out = _blank()
    raw = db.get_arena_state(STATE_KEY)
    if not raw:
        out = ensure_founder_locks(out)
        return out
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        out = ensure_founder_locks(out)
        return out
    if not isinstance(data, dict):
        out = ensure_founder_locks(out)
        return out
    out.update({k: data[k] for k in out if k in data})
    out["paused"] = bool(out.get("paused"))
    out = ensure_founder_locks(out)
    return out


def _save(data: dict[str, Any]) -> dict[str, Any]:
    data = ensure_founder_locks(dict(data))
    data["updated_ts"] = time.time()
    db.set_arena_state(STATE_KEY, json.dumps(data))
    return data


def set_paused(paused: bool) -> dict[str, Any]:
    data = load_control()
    data["paused"] = bool(paused)
    return _save(data)


def is_paused() -> bool:
    return bool(load_control().get("paused"))


def update_settings(patch: dict[str, Any]) -> dict[str, Any]:
    data = load_control()
    if "paused" in patch:
        data["paused"] = bool(patch["paused"])
    if "auto_promote" in patch:
        data["auto_promote"] = bool(patch["auto_promote"])
    if "llm_provider" in patch:
        val = str(patch["llm_provider"] or "none").strip().lower()
        if val == "xai":
            val = "grok"
        if val in _LLM_ALLOWED - {"xai"} or val == "grok":
            data["llm_provider"] = val
    if "paper_slots" in patch:
        try:
            data["paper_slots"] = max(0, int(patch["paper_slots"]))
        except (TypeError, ValueError):
            pass
    if "founder_locks" in patch and isinstance(patch["founder_locks"], dict):
        data["founder_locks"] = patch["founder_locks"]
    return _save(data)


def cfg(name: str, default):
    """Read a lab knob: operator overlay first, then config, then default."""
    import config

    mapping = _SETTING_MAP.get(name)
    if mapping:
        key, caster, fallback = mapping
        data = load_control()
        raw = data.get(key)
        if raw is not None and raw != "":
            try:
                return caster(raw)
            except (TypeError, ValueError):
                pass
        default = fallback if default is None else default
    return getattr(config, name, default)


def effective() -> dict[str, Any]:
    data = load_control()
    llm = cfg("STRATEGY_LAB_LLM_PROVIDER", "none")
    llm = str(llm or "none").strip().lower()
    if llm == "xai":
        llm = "grok"
    if llm not in ("none", "ollama", "grok"):
        llm = "none"
    return {
        "paused": bool(data.get("paused")),
        "auto_promote": bool(cfg("STRATEGY_LAB_AUTO_PROMOTE", False)),
        "llm_provider": str(llm),
        "paper_slots": int(cfg("STRATEGY_LAB_PAPER_SLOTS", 0) or 0),
        "founder_locks": data.get("founder_locks") or {},
        "last_cycle": data.get("last_cycle"),
        "updated_ts": float(data.get("updated_ts") or 0.0),
        "enabled": bool(getattr(__import__("config"), "STRATEGY_LAB_ENABLED", True)),
    }


def save_last_cycle(report: dict[str, Any] | None) -> dict[str, Any]:
    data = load_control()
    data["last_cycle"] = report
    return _save(data)


def try_acquire_tick_lock(*, ttl: float = LOCK_TTL_SEC) -> bool:
    data = load_control()
    now = time.time()
    ts = float(data.get("tick_lock_ts") or 0.0)
    if ts and (now - ts) < ttl:
        return False
    data["tick_lock_ts"] = now
    data["tick_lock_pid"] = os.getpid()
    _save(data)
    return True


def release_tick_lock() -> None:
    data = load_control()
    data["tick_lock_ts"] = 0.0
    data["tick_lock_pid"] = None
    _save(data)


def paper_slots() -> int:
    return max(0, int(cfg("STRATEGY_LAB_PAPER_SLOTS", 0) or 0))

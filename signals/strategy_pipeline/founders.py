"""Founder lock: sniper-v1, arbitrage-v1, sweeper-v1 must not be retired by Lab/GA."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("strategy_pipeline.founders")

FOUNDER_BOTS: tuple[str, ...] = ("sniper-v1", "arbitrage-v1", "sweeper-v1")
FOUNDER_TYPES: frozenset[str] = frozenset({"sniper", "arbitrage", "sweeper"})


def default_founder_locks() -> dict[str, Any]:
    return {name: {"protected": True, "reason": "founder"} for name in FOUNDER_BOTS}


def ensure_founder_locks(control_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge default founder locks into a control dict (or arena_state overlay).

    Defaults protect founders. Operator may unlock via founder_locks[name].protected=False;
    that override is preserved across reloads.
    """
    data = dict(control_data or {})
    locks = data.get("founder_locks")
    if not isinstance(locks, dict):
        locks = {}
    for name, meta in default_founder_locks().items():
        existing = locks.get(name)
        if not isinstance(existing, dict):
            locks[name] = dict(meta)
        else:
            merged = {**meta, **existing}
            if "protected" in existing:
                merged["protected"] = bool(existing["protected"])
            else:
                merged["protected"] = True
            locks[name] = merged
    data["founder_locks"] = locks
    return data


def is_protected_bot(bot_name: str, *, control_data: dict | None = None) -> bool:
    name = str(bot_name or "").strip()
    if not name:
        return False
    data = control_data
    if data is None:
        try:
            from signals.strategy_pipeline.control import load_control
            data = load_control()
        except Exception:
            data = {}
    locks = (data or {}).get("founder_locks") if isinstance(data, dict) else None
    if isinstance(locks, dict) and name in locks:
        meta = locks.get(name)
        if isinstance(meta, dict):
            return bool(meta.get("protected"))
        if meta is True:
            return True
        if meta is False:
            return False
    if name in FOUNDER_BOTS:
        return True
    try:
        import db
        with db.get_conn() as conn:
            try:
                row = conn.execute(
                    "SELECT protected FROM bot_configs WHERE bot_name=? AND active=1",
                    (name,),
                ).fetchone()
            except Exception:
                row = None
        if row is not None:
            try:
                return int(row["protected"] or 0) == 1
            except (TypeError, ValueError, KeyError):
                pass
    except Exception:
        pass
    return False


def mark_founders_in_db() -> None:
    """Best-effort: add protected column and mark founder bots."""
    try:
        import db
        with db.get_conn() as conn:
            try:
                conn.execute(
                    "ALTER TABLE bot_configs ADD COLUMN protected INTEGER DEFAULT 0"
                )
            except Exception:
                pass
            for name in FOUNDER_BOTS:
                try:
                    conn.execute(
                        "UPDATE bot_configs SET protected=1 WHERE bot_name=? AND active=1",
                        (name,),
                    )
                except Exception:
                    pass
    except Exception as e:
        logger.debug("mark_founders_in_db failed: %s", e)


def ga_may_retire(bot_name: str) -> bool:
    """Return False when GA must not retire this bot (founder lock)."""
    try:
        from signals.strategy_pipeline.control import load_control
        data = load_control()
    except Exception:
        data = {}
    return not is_protected_bot(bot_name, control_data=data)

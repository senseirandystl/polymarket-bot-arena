"""Empirical P(win) overlay — shadow now, auto-promote later.

``lookup_yes`` is the hot-path hook used by ``signals.prob.live_side_prob``.
Until a cell is promoted it returns None (caller uses Φ).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import db

logger = logging.getLogger("arena.empirical_prob")

STATE_KEY = "empirical_prob"


def load_state() -> dict:
    try:
        raw = db.get_arena_state(STATE_KEY)
        if isinstance(raw, dict):
            return raw
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {"promoted": {}, "cells": {}}


def lookup_yes(signals: dict, strategy_type: str) -> Optional[float]:
    """Return empirical P(YES) when this strategy×regime is promoted."""
    st = load_state()
    promoted = st.get("promoted") or {}
    if not promoted:
        return None
    label = ""
    try:
        label = str(
            (signals.get("market_regime") or {}).get("label")
            or signals.get("regime")
            or ""
        )
    except Exception:
        label = ""
    key = f"{strategy_type}|{label}"
    if not promoted.get(key) and not promoted.get(strategy_type):
        return None
    cells = st.get("cells") or {}
    # Thin overlay: no cell fit yet → caller keeps Φ.
    if not cells:
        return None
    return None

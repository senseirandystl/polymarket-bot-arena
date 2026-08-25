"""Kalshi adapter — discovery / book / resolve entry points."""

from __future__ import annotations

import kalshi_markets
from exchanges import KALSHI, exchange_enabled


def discover_live(limit: int | None = None) -> list[dict]:
    if not exchange_enabled(KALSHI):
        return []
    return kalshi_markets.discover_live(limit=limit or 12)


def recent_resolutions() -> dict:
    """ticker (namespaced id) → True if Up/Yes."""
    try:
        return kalshi_markets.recent_resolutions()
    except Exception:
        return {}

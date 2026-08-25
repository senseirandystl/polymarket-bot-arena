"""Polymarket adapter — wraps existing Gamma/CLOB helpers."""

from __future__ import annotations

import config
import polymarket_markets
from exchanges import POLYMARKET, stamp_exchange

WINDOW_SEC = int(getattr(config, "MARKET_WINDOW_SEC", 300) or 300)
SETTLEMENT = "chainlink_twap"


def stamp(market: dict) -> dict:
    return stamp_exchange(
        market, POLYMARKET, window_sec=WINDOW_SEC, settlement=SETTLEMENT,
    )


def discover_live(limit: int | None = None) -> list[dict]:
    return [stamp(m) for m in (polymarket_markets.discover_markets(limit) or [])]


def get_book(token_id: str | None, timeout: float | None = None) -> dict:
    return polymarket_markets.get_order_book(token_id, timeout=timeout)


def recent_resolutions() -> dict:
    return dict(polymarket_markets.recent_resolutions() or {})

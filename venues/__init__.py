"""Execution venues — a clean split between paper and live trading.

Two engines, one per venue, each owning ALL order placement for its venue so
the code paths never intermix:

  * ``paper``  → :class:`venues.paper.PaperEngine`  — local simulated fills
    (priced from the real market), unlimited, optionally mirrored to Simmer.
  * ``live``   → :class:`venues.live.LiveEngine`    — Polymarket CLOB orders.

``base_bot.execute()`` selects an engine via :func:`get_engine` and delegates
placement to it. Market *discovery* is shared (the Simmer fast-5m feed already
carries Polymarket token ids), but pricing/fills/settlement are per-venue.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TradeResult:
    """Outcome of an order-placement attempt.

    ``trade_id`` is the internal ``trades`` row id (as a string) on success, so
    callers can correlate. ``fill_source`` records how it filled:
    ``'local_sim'`` | ``'simmer'`` | ``'polymarket'``.
    """

    success: bool
    reason: Optional[str] = None
    trade_id: Optional[str] = None
    fill_source: Optional[str] = None
    shares: float = 0.0
    entry_price: float = 0.0


def get_engine(mode: str):
    """Return the execution engine singleton for ``mode`` ('paper' | 'live')."""
    if mode == "live":
        from venues.live import LiveEngine
        return LiveEngine.instance()
    from venues.paper import PaperEngine
    return PaperEngine.instance()

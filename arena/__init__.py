"""Threaded runtime modules split out from the old monolithic arena.py.

Each module owns one concern (discovery, trader, resolver, position monitor)
and runs in its own daemon thread.  Root ``arena.py`` is now a thin
coordinator that starts the threads and runs the evolution cycle on its
main thread.

Public API:

    from arena import (
        MarketDiscovery,   # arena/discovery.py     -- 60s tick
        Trader,            # arena/trader.py        -- 1s tick (lean)
        TradeResolver,     # arena/resolver.py      -- 60s tick
        PositionMonitorThread,  # arena/position_monitor.py -- 0.5s SL/TP
        SharedArenaState,  # arena/state.py         -- thread-safe dedup
        build_combined_signals,  # arena/signals.py
    )
"""

from arena.discovery import MarketDiscovery
from arena.trader import Trader
from arena.resolver import TradeResolver
from arena.position_monitor import PositionMonitorThread
from arena.state import SharedArenaState
from arena.signals import build_combined_signals

__all__ = [
    "MarketDiscovery",
    "Trader",
    "TradeResolver",
    "PositionMonitorThread",
    "SharedArenaState",
    "build_combined_signals",
]

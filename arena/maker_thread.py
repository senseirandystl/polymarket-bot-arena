"""Maker section as its own thread (decoupled from discovery).

Previously makers ran inside MarketDiscovery's on_cycle_complete hook, so
long paper fills delayed Gamma scans. This thread wakes on discovery
cadence (~20s) and reuses the same secondary-bot logic without blocking
window selection.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import config

logger = logging.getLogger("arena.maker")


class MakerThread(threading.Thread):
    """Run maker + copy secondary bots on DISCOVERY_INTERVAL_SEC cadence."""

    def __init__(
        self,
        discovery,
        tick_fn: Callable,
        interval: float | None = None,
    ) -> None:
        super().__init__(daemon=True, name="maker-section")
        self._stop_event = threading.Event()
        self._discovery = discovery
        self._tick_fn = tick_fn
        self._interval = float(
            interval if interval is not None
            else getattr(config, "DISCOVERY_INTERVAL_SEC", 20)
        )

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info(f"Maker thread started (interval={self._interval}s)")
        # Small initial delay so discovery can prime markets.
        self._stop_event.wait(min(2.0, self._interval))
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self._tick_fn(self._discovery)
            except Exception as e:
                logger.error(f"Maker tick error: {e}", exc_info=True)
            elapsed = time.perf_counter() - t0
            remain = max(0.0, self._interval - elapsed)
            self._stop_event.wait(remain)
        logger.info("Maker thread stopped")

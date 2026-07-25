"""Module-private shared state for the threaded arena runtime.

The ``SharedArenaState`` object holds the ``(bot_name, market_id)`` dedup
set that prevents double-trading in the same 5-min window.  It lives here
(in ``arena/state.py``) rather than on the root ``arena.py`` module so the
worker threads can import the state class directly — without also pulling
in the entire bot lifecycle and copy-trade setup machinery that
``arena.py`` collects on its import.
"""

import time
import threading


class SharedArenaState:
    """Thread-safe dedup state for taker / maker / copy workers.

    A (worker, market_id) pair forms the dedup key.  Once a worker has
    either evaluated a market or attempted to trade on it, we don't
    revisit the pair until the next window (when a NEW ``market_id``
    appears).  Trade exceptions also add the key so a poisoned pair
    can't busy-loop the worker.
    """

    def __init__(self) -> None:
        # ``_lock`` is a regular Lock because the operations are quick and
        # we never hold it across an I/O call.  An RLock would still be safe
        # but isn't needed.
        self._lock = threading.Lock()
        self.traded: set = set()
        # Skip is a first-class outcome (the research: the best bots skip far
        # more than they trade). Tally skip reasons so runs are explainable —
        # why the arena sat flat, not just what it traded.
        self.skip_counts: dict = {}
        # (bot, market_id) -> unix time until which execute is suppressed after
        # a slippage reject (config.SLIPPAGE_RETRY_COOLDOWN_SEC).
        self._slippage_until: dict = {}

    def is_traded(self, key: tuple) -> bool:
        with self._lock:
            return key in self.traded

    def mark_traded(self, key: tuple) -> None:
        with self._lock:
            self.traded.add(key)
            # A real fill ends any pending slippage cooldown for this pair.
            self._slippage_until.pop(key, None)

    def is_slippage_cooling(self, key: tuple, now: float | None = None) -> bool:
        """True while this (bot, market) is in post-slippage backoff."""
        now = time.time() if now is None else now
        with self._lock:
            until = self._slippage_until.get(key)
            if until is None:
                return False
            if now >= until:
                del self._slippage_until[key]
                return False
            return True

    def mark_slippage_reject(
        self, key: tuple, cooldown_sec: float, now: float | None = None
    ) -> None:
        """Start/refresh the slippage backoff for this (bot, market)."""
        now = time.time() if now is None else now
        cooldown_sec = max(0.0, float(cooldown_sec))
        with self._lock:
            self._slippage_until[key] = now + cooldown_sec

    def note_skip(self, reason: str) -> None:
        """Record a skip by coarse reason (e.g. 'session', 'no_edge', 'no_book')."""
        with self._lock:
            self.skip_counts[reason] = self.skip_counts.get(reason, 0) + 1

    def skip_snapshot(self) -> dict:
        """A copy of the skip-reason tally for reporting."""
        with self._lock:
            return dict(self.skip_counts)

    def load_from_db(self, conn) -> int:
        """Rehydrate the dedup set from the trades table (1h lookback).

        Returns the number of (bot, market) keys loaded.  Called once at
        startup so restarts don't double-trade a market that was already
        traded before the previous process exited.
        """
        rows = conn.execute(
            "SELECT bot_name, market_id FROM trades "
            "WHERE created_at >= datetime('now', '-1 hours')"
        ).fetchall()
        with self._lock:
            for r in rows:
                self.traded.add((r["bot_name"], r["market_id"]))
            return len(self.traded)

    def reset(self) -> None:
        """Clear the set.  Called by the coordinator after each evolution
        cycle so surviving + new bots can re-evaluate the next window
        immediately."""
        with self._lock:
            self.traded.clear()
            self._slippage_until.clear()

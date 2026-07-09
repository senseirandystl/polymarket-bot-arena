"""Module-private shared state for the threaded arena runtime.

The ``SharedArenaState`` object holds the ``(bot_name, market_id)`` dedup
set that prevents double-trading in the same 5-min window.  It lives here
(in ``arena/state.py``) rather than on the root ``arena.py`` module so the
worker threads can import the state class directly — without also pulling
in the entire bot lifecycle and copy-trade setup machinery that
``arena.py`` collects on its import.
"""

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

    def is_traded(self, key: tuple) -> bool:
        with self._lock:
            return key in self.traded

    def mark_traded(self, key: tuple) -> None:
        with self._lock:
            self.traded.add(key)

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

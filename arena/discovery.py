"""Market discovery + per-market orderflow cache.

Runs as a daemon thread that ticks once every ``config.DISCOVERY_INTERVAL_SEC``
seconds.  It owns ALL the non-trade-evaluating HTTPS traffic that the old
``arena.py`` used to make on every 15s iteration:

  - ``/api/sdk/markets?tags=fast-5m``  -- upcoming windows (SDK)
  - ``/api/markets``                   -- currently-live windows (public)
  - ``/api/sdk/context/{id}``          -- orderflow probability + 24h volume
                                          (refreshed only for the live
                                          market, on a per-cycle TTL)

The ``Trader`` thread reads snapshots via ``current_market_snapshot``
(takes a *deep copy* under the lock so the caller can use the dict
freely without coordinating further) and ``all_markets_snapshot``.
``next_market_snapshot`` is a returning-``None`` stub for back-compat.

Secondary bots (the maker section + copy-trade bots) read from
``maker_target_markets_snapshot`` so they can begin quoting on the
next-imminent market in the gap between windows (≤ MAKER_UPCOMING_WINDOW_SEC,
default 20 min).  This deliberately diverges from the Trader's "swap
only on actual rollover" policy -- the Trader reads
``current_market_snapshot`` and self-quiesces on rollover, while the
maker quoting economy benefits from a pre-window warm-up.

A user-supplied ``on_cycle_complete`` hook fires after every successful
scan so the secondary bots can run on the same data the trader is using.
"""

import copy
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import config
import polymarket_markets
from signals import orderflow_signals
from arena.market_utils import (
    compute_time_remaining_seconds,
    is_5min_market,
    select_current_market,
)

logger = logging.getLogger("arena.discovery")


class MarketDiscovery(threading.Thread):
    """Background scanner for Polymarket BTC 5-min up/down markets."""

    def __init__(
        self,
        on_cycle_complete: Optional[Callable[["MarketDiscovery"], None]] = None,
    ) -> None:
        super().__init__(daemon=True, name="market-discovery")
        self._stop_event = threading.Event()
        self._on_cycle_complete = on_cycle_complete

        # All snapshot state is mutated under _lock.  Snapshot helpers
        # return *deep-copies* so callers don't have to coordinate on
        # the lock further.
        self._lock = threading.Lock()
        self._current_market: Optional[dict] = None
        self._maker_fallback_market: Optional[dict] = None
        # NOTE: we deliberately do NOT pre-publish `next_market`.  Per the
        # user's "swap only on actual rollover" policy, the Trader loop
        # reads only ``current_market_snapshot`` and self-quiesces when
        # ``time_remaining_seconds`` drops below 1.  The next scan will
        # pick up the new window and re-publish ``current_market``
        # naturally without a speculative hop.
        self._markets_cache: List[dict] = []
        self._orderflow_cache: Dict[str, dict] = {}
        self._orderflow_cache_ts: Dict[str, float] = {}
        self._last_scan_ts: float = 0.0

    # ----------------------------------------------------------------------
    # Thread lifecycle
    # ----------------------------------------------------------------------

    def run(self) -> None:
        logger.info(
            f"Market discovery started (interval={config.DISCOVERY_INTERVAL_SEC}s)"
        )
        # First tick fires immediately on start so the trader sees a
        # current_market on its first iteration rather than waiting 20s.
        while not self._stop_event.is_set():
            try:
                self._do_scan()
                if self._on_cycle_complete is not None:
                    try:
                        self._on_cycle_complete(self)
                    except Exception as e:
                        logger.error(f"on_cycle_complete hook error: {e}")
            except Exception as e:
                logger.error(f"Discovery scan error: {e}")
            self._stop_event.wait(config.DISCOVERY_INTERVAL_SEC)
        logger.info("Market discovery stopped")

    def stop(self) -> None:
        self._stop_event.set()

    # ----------------------------------------------------------------------
    # Public snapshot API -- every call is a deep-copy so the caller can
    # use the dict freely without re-locking.
    # ----------------------------------------------------------------------

    def current_market_snapshot(self) -> Optional[dict]:
        with self._lock:
            return (
                copy.deepcopy(self._current_market)
                if self._current_market else None
            )

    def next_market_snapshot(self) -> Optional[dict]:
        """Returns ``None`` -- Kept for back-compat. The Trader honours the
        user's "swap only on actual rollover" policy and only reads
        ``current_market_snapshot``. Discovery no longer pre-publishes a
        speculative next market because nothing consumes it."""
        return None

    def all_markets_snapshot(self) -> List[dict]:
        with self._lock:
            return copy.deepcopy(self._markets_cache)

    def maker_target_markets_snapshot(self) -> List[dict]:
        """Markets the secondary-bot tick should quote on.

        Prefers the live ``current_market``; if no market currently
        contains the wall clock, falls back to the soonest non-expired
        market resolving within ``MAKER_UPCOMING_WINDOW_SEC`` (default
        1200s = 20 min) so the maker can begin quoting bid/ask before
        its window opens.  This restores the behavior the old
        monolithic ``arena.py`` had pre-refactor.

        Deliberately does **not** influence the Trader: the Trader
        honours the "swap only on actual rollover" policy and only
        reads ``current_market_snapshot``.  The maker economy is
        different (it quotes both sides, wants warm-up time before
        resolution) so the secondary section gets separate targeting
        logic.

        Returns a list (in practice 0 or 1 element, kept as a list
        so the caller doesn't have to special-case empty input).
        """
        with self._lock:
            cur = self._current_market
            fallback = self._maker_fallback_market

        if cur is not None and cur.get("time_remaining_seconds", 0) >= 1:
            return [copy.deepcopy(cur)]
        if fallback is not None:
            return [copy.deepcopy(fallback)]
        return []

    @property
    def last_scan_ts(self) -> float:
        with self._lock:
            return self._last_scan_ts

    # ----------------------------------------------------------------------
    # Scan / classify pipeline
    # ----------------------------------------------------------------------

    def _do_scan(self) -> None:
        markets = polymarket_markets.discover_markets()
        if not markets:
            return

        now_utc = datetime.now(timezone.utc)
        # Decorate every market with time-remaining / window-age so
        # downstream code doesn't need to re-parse resolves_at.
        for m in markets:
            tr = compute_time_remaining_seconds(m, now_utc)
            m["time_remaining_seconds"] = tr
            m["window_age_seconds"] = max(0, 300 - tr)

        # Bots trade ONLY 5-minute windows -- drop any 15-min (or other
        # non-5-min) BTC up/down markets before anything downstream can
        # see them.  This gates the trader, the maker fallback AND the
        # all_markets snapshot, so a 15-min window can never surface.
        five_min = [
            m for m in markets if is_5min_market(m.get("question", "") or "")
        ]
        non_expired = [
            m for m in five_min if m.get("time_remaining_seconds", 0) > 0
        ]

        # Pick the live market by its REAL resolves_at timestamp
        # (0 < time_remaining <= 300), never by ET time-of-day -- so a
        # future-dated window whose clock time straddles "now" is never
        # chosen.  Per the user's policy we do NOT pre-pick a speculative
        # next_market; nothing leaks across the rollover until the next
        # scan sees the new window for real.
        current = select_current_market(non_expired, now_utc)

        # Maker fallback target: if no market currently contains the
        # wall clock, the maker section quotes the soonest non-expired
        # market resolving within MAKER_UPCOMING_WINDOW_SEC so it can
        # begin posting bid/ask during the pre-window ramp.  Restores
        # the behavior the old monolithic ``arena.py`` had pre-refactor.
        maker_fallback: Optional[dict] = None
        if current is None:
            fallback_pool = sorted(
                (
                    m for m in non_expired
                    if 0 < m.get("time_remaining_seconds", 0)
                       <= config.MAKER_UPCOMING_WINDOW_SEC
                ),
                key=lambda m: m.get("time_remaining_seconds", 999),
            )
            if fallback_pool:
                maker_fallback = fallback_pool[0]

        # Refresh orderflow for the current market AND for the maker
        # fallback (when present).  When both exist they are guaranteed
        # distinct markets so we issue two calls.  Total cost: 1-2
        # HTTPS calls per 20s cycle -- the fallback call only fires
        # in the no-current-market gap, so the hot path stays at one.
        #
        # NOTE: keep these calls LOCK-FREE; _fetch_orderflow_for_market
        # re-acquires self._lock for its cache writes.  Wrapping this
        # block in `with self._lock:` would deadlock.
        if current is not None:
            self._refresh_market_data(current)
        if maker_fallback is not None and maker_fallback is not current:
            self._refresh_market_data(maker_fallback)

        prev_id = (self._current_market or {}).get("id")
        with self._lock:
            self._markets_cache = non_expired
            self._current_market = current
            self._maker_fallback_market = maker_fallback
            self._last_scan_ts = time.time()

        # Only announce (INFO) when the live window actually rolls over;
        # otherwise stay at DEBUG so the log isn't flooded every cycle.
        cur_id = (current or {}).get("id")
        msg = (
            f"Discovery: {len(non_expired)} unexpired windows, "
            f"current={current.get('question', '')[:38] if current else 'none'}"
        )
        if cur_id != prev_id:
            logger.info(msg)
        else:
            logger.debug(msg)

    def _refresh_market_data(self, m: dict) -> None:
        """Set fresh price + orderflow on a selected market from the CLOB book.

        ``current_price`` becomes the live Up-token mid, and ``orderflow`` is
        populated so the signal stack (which reads ``current_probability`` and
        ``volume_24h``) has data. Best-effort — leaves the fields untouched if
        the book is unavailable.
        """
        polymarket_markets.refresh_price(m)  # sets m["current_price"] from CLOB
        # Order-book imbalance on the Up/YES token — one extra book call per
        # discovery cycle (~20s), off the trader hot path. obi > 0 = bid-heavy
        # (upward/YES pressure). Best-effort: 0.0 when the book is unavailable.
        obi = 0.0
        up_tok = m.get("polymarket_token_id")
        if up_tok:
            book = polymarket_markets.get_order_book(up_tok)
            obi = orderflow_signals.order_book_imbalance(book)
        m["orderflow"] = {
            "current_probability": m.get("current_price") or 0.5,
            "volume_24h": m.get("volume_24h", 0) or 0,
            "obi": obi,
            "warnings": [],
        }

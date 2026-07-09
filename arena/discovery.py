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

import requests

import config
from arena.market_utils import (
    compute_time_remaining_seconds,
    is_btc_updown,
    window_contains_now,
)

logger = logging.getLogger("arena.discovery")


class MarketDiscovery(threading.Thread):
    """Background scanner for Simmer BTC 5-min markets."""

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
        # current_market on its first iteration rather than waiting 60s.
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
        api_key = config.get_credential("simmer_api_key")
        if not api_key:
            logger.debug("Skipping discovery: no Simmer API key")
            return

        markets = self._fetch_markets(api_key)
        if not markets:
            return

        now_utc = datetime.now(timezone.utc)
        # Decorate every market with time-remaining / window-age so
        # downstream code doesn't need to re-parse resolves_at.
        for m in markets:
            tr = compute_time_remaining_seconds(m, now_utc)
            m["time_remaining_seconds"] = tr
            m["window_age_seconds"] = max(0, 300 - tr)

        non_expired = [
            m for m in markets if m.get("time_remaining_seconds", 0) > 0
        ]

        # Pick the live market.  Prefer window_contains_now; fall back
        # to the soonest-resolving market whose remaining lifetime is
        # in (0, 300].  We deliberately stop here -- per the user's
        # policy we do NOT pre-pick a speculative next_market so
        # nothing can leak across the rollover until the next scan
        # sees the new window for real.
        candidates = sorted(
            [
                m for m in non_expired
                if m.get("time_remaining_seconds", 999) <= 300
                or window_contains_now(m.get("question", ""), now_utc)
            ],
            key=lambda m: m.get("time_remaining_seconds", 999),
        )
        current = candidates[0] if candidates else None

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
        # HTTPS calls per 60s cycle -- the fallback call only fires
        # in the no-current-market gap, so the hot path stays at one.
        #
        # NOTE: keep these calls LOCK-FREE; _fetch_orderflow_for_market
        # re-acquires self._lock for its cache writes.  Wrapping this
        # block in `with self._lock:` would deadlock.
        if current is not None:
            self._fetch_orderflow_for_market(api_key, current, time.time())
        if maker_fallback is not None and maker_fallback is not current:
            self._fetch_orderflow_for_market(
                api_key, maker_fallback, time.time()
            )

        with self._lock:
            self._markets_cache = non_expired
            self._current_market = current
            self._maker_fallback_market = maker_fallback
            self._last_scan_ts = time.time()

        logger.info(
            f"Discovery: {len(markets)} BTC candidates, "
            f"{len(non_expired)} unexpired, "
            f"current={'yes' if current else 'no'}, "
            f"maker_fallback={'yes' if maker_fallback else 'no'}"
        )

    def _fetch_markets(self, api_key: str) -> List[dict]:
        """Scan both Simmer SDK + public endpoints, dedupe by id."""
        seen: Dict[str, dict] = {}

        def _scan_page(page, source: str):
            for m in page:
                mid = m.get("id") or m.get("market_id", "unknown")
                if mid in seen:
                    continue
                if is_btc_updown(m):
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(
                            f"  CANDIDATE [{source}]: {mid[:12]}... | "
                            f"{m.get('question')} | resolves_at={m.get('resolves_at')}"
                        )
                    seen[mid] = m

        # --- Source 1: SDK upcoming markets ---
        try:
            resp = requests.get(
                f"{config.SIMMER_BASE_URL}/api/sdk/markets",
                headers={"Authorization": f"Bearer {api_key}"},
                params={"limit": 50, "tags": "fast-5m"},
                timeout=20,
            )
            if resp.status_code == 200:
                data = resp.json()
                page = data if isinstance(data, list) else data.get("markets", [])
                _scan_page(page, "upcoming")
        except Exception as e:
            logger.warning(f"SDK /api/sdk/markets call failed: {e}")

        # --- Source 2: public endpoint for currently-live markets ---
        try:
            resp = requests.get(
                f"{config.SIMMER_BASE_URL}/api/markets",
                headers={"Authorization": f"Bearer {api_key}"},
                params={"limit": 20},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                page = data if isinstance(data, list) else data.get("markets", [])
                _scan_page(page, "live")
        except Exception as e:
            logger.warning(f"Public /api/markets call failed: {e}")

        return list(seen.values())

    def _fetch_orderflow_for_market(self, api_key: str, m: dict, now: float) -> None:
        """Fetch (or reuse cached) ``/api/sdk/context/{id}`` data for one market.

        Cached values stay valid for ``config.ORDERFLOW_CACHE_SECONDS`` so
        the trader (which runs at 1Hz) can read the same snapshot without
        triggering a network call per tick.
        """
        mid = m.get("id") or m.get("market_id", "")
        if not mid:
            return

        ts = self._orderflow_cache_ts.get(mid, 0.0)
        if now - ts < config.ORDERFLOW_CACHE_SECONDS:
            with self._lock:
                cached = self._orderflow_cache.get(mid)
            if cached is not None:
                m["orderflow"] = cached
            return

        try:
            resp = requests.get(
                f"{config.SIMMER_BASE_URL}/api/sdk/context/{mid}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                ctx = resp.json()
                of = {
                    "current_probability": ctx.get("current_probability", 0.5),
                    "volume_24h": ctx.get("volume_24h", 0),
                    "time_to_resolution": ctx.get("time_to_resolution_seconds", 0),
                    "warnings": ctx.get("warnings", []),
                }
                with self._lock:
                    self._orderflow_cache[mid] = of
                    self._orderflow_cache_ts[mid] = now
                m["orderflow"] = of
            else:
                logger.debug(f"orderflow HTTP {resp.status_code} for {mid[:12]}...")
        except Exception as e:
            logger.debug(f"orderflow fetch error for {mid[:12]}...: {e}")

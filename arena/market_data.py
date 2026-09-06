"""Market-data warmer — the single owner of all per-market network reads.

Before this thread existed, the per-market reads were scattered and slow:
  * ``Trader`` fetched the YES ``/midpoint`` on every 1s tick (blocking, on the
    hot path — one slow CLOB response stalled *every* bot for that tick).
  * ``build_combined_signals`` fetched CVD + PM momentum (cached ~20s).
  * The arbitrage bot fetched YES and NO books itself.
  * OBI was computed once per 20s discovery cycle.

The warmer consolidates all of that into ONE background thread that, every
``config.MARKET_DATA_INTERVAL_SEC`` (default 1s), refreshes the live market's:

    YES book, NO book, YES price, NO price, OBI, CVD, PM momentum

into a shared, lock-protected :class:`MarketDataStore`. Consumers (the trader,
the arbitrage bot, ``build_combined_signals``) then read *warm* in-memory data
with **zero network on their hot path**, and every trading-decision input stays
<=1s fresh — the freshness the arb bot needs and directional bots benefit from.

Producer/consumer split: the warmer is the sole producer; the module-level
store (``store()``) is the shared state everyone reads. Bots don't need a
reference to the thread — they just read the store.

2026-08 streamlining:
  * Parallel YES/NO book fetches with short timeouts
  * CVD/PM throttled while kill-switched (deadline-based cycle sleep)
  * Fail soft: keep last snapshot fields when a fetch fails
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Optional

import config
import polymarket_markets
from signals import clean_tick, orderflow_signals
from signals.strike import get_strike_registry

logger = logging.getLogger("arena.market_data")

# Shared pool for parallel book GETs (warmer only). Sized >2 so a straggler
# pair from a slow CLOB cycle cannot starve the next tick's submits (the old
# as_completed(timeout=) path raised TimeoutError mid-cycle, abandoned the
# two in-flight futures, and left max_workers=2 fully occupied → cascade of
# "N (of 2) futures unfinished" every second until the hung GETs returned).
_book_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="warm-book")

# Rate-limit book-timeout warnings (one line per cluster, not one per tick).
_last_book_timeout_log = 0.0


class MarketDataStore:
    """Thread-safe latest-snapshot-per-market cache (producer: the warmer)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snap: dict[str, dict] = {}

    def put(self, market_id: str, data: dict) -> None:
        with self._lock:
            self._snap[market_id] = data

    def get(self, market_id: Optional[str]) -> Optional[dict]:
        """Return a shallow copy of the latest snapshot, or ``None``.

        Books inside are treated as read-only by all consumers, so a shallow
        copy is enough to keep the caller from mutating the stored dict.
        """
        if not market_id:
            return None
        with self._lock:
            snap = self._snap.get(market_id)
            return dict(snap) if snap is not None else None

    def prune(self, keep_market_id: Optional[str] = None,
              keep_ids: Optional[set] = None) -> None:
        """Drop snapshots not in the live window set."""
        keep = set(keep_ids or ())
        if keep_market_id:
            keep.add(keep_market_id)
        with self._lock:
            self._snap = {k: v for k, v in self._snap.items() if k in keep}


_store = MarketDataStore()


def store() -> MarketDataStore:
    """Shared market-data store read by the trader and the arbitrage bot."""
    return _store


def warm_age_sec(warm: Optional[dict], now: float | None = None) -> float | None:
    """Seconds since the warm snapshot was written, or None if unknown."""
    if not warm:
        return None
    ts = warm.get("ts")
    if ts is None:
        return None
    try:
        return max(0.0, float(now if now is not None else time.time()) - float(ts))
    except (TypeError, ValueError):
        return None


def is_warm_fresh(warm: Optional[dict], max_age: float | None = None,
                  now: float | None = None) -> bool:
    """True when warm has books/prices and is younger than max_age."""
    if not warm:
        return False
    age = warm_age_sec(warm, now=now)
    if age is None:
        return False
    limit = float(
        max_age if max_age is not None
        else getattr(config, "WARM_MAX_AGE_SEC", 3.0)
    )
    if age > limit:
        return False
    # Need at least one usable price to trade.
    return warm.get("yes_price") is not None or warm.get("no_price") is not None


def lay_warm_onto_market(market: dict, warm: Optional[dict]) -> None:
    """Lay warm prices/books onto a market snapshot (mutates ``market``).

    Used by the trader and maker section so decisions and paper fills share
    the same book snapshot: ``yes_ask``/``no_ask`` for edge + expected price,
    and ``yes_book``/``no_book`` passed into ``venues.paper.place(book=...)``
    so the fill does not re-fetch a drifted CLOB book (slippage path A).
    """
    if not warm:
        return
    if warm.get("yes_price") is not None:
        market["current_price"] = warm["yes_price"]
    if warm.get("no_price") is not None:
        market["no_price"] = warm["no_price"]
    # Full books first so asks and side-book fields stay consistent.
    yes_book = warm.get("yes_book")
    no_book = warm.get("no_book")
    if yes_book is not None:
        market["yes_book"] = yes_book
    if no_book is not None:
        market["no_book"] = no_book
    for ask_key, book in (("yes_ask", yes_book or {}),
                          ("no_ask", no_book or {})):
        if book.get("valid") and book.get("best_ask") is not None:
            market[ask_key] = book["best_ask"]
        elif book.get("valid") and book.get("asks"):
            # Tests / partial books may omit best_ask; derive from top ask.
            market[ask_key] = book["asks"][0][0]
    market["orderflow"] = {
        **(market.get("orderflow") or {}),
        "obi": warm.get("obi", 0.0),
    }
    # Microstructure context for spread tax / shadow lanes.
    if warm.get("micro_spread") is not None:
        market["micro_spread"] = warm["micro_spread"]
    if warm.get("micro_spread_score") is not None:
        market["micro_spread_score"] = warm["micro_spread_score"]


def side_book(market: dict, side: str) -> Optional[dict]:
    """Return the warm side book if present and valid, else ``None``."""
    if side == "yes":
        book = market.get("yes_book")
    elif side == "no":
        book = market.get("no_book")
    else:
        return None
    if isinstance(book, dict) and book.get("valid"):
        return book
    return None


def _lane_live(lane: str) -> bool:
    """True when a kill-switched lane has a live override (needs fresh feed)."""
    try:
        from bots.base_bot import _lane_overrides
        ov = _lane_overrides().get(lane) or {}
        return bool(ov.get("enabled"))
    except Exception:
        return False


def _flow_needs_refresh() -> bool:
    """CVD/PM are kill-switched unless a related override is live."""
    if float(getattr(config, "SIGNAL_WEIGHT_CVD", 0.0) or 0.0) > 0:
        return True
    if float(getattr(config, "SIGNAL_WEIGHT_PM", 0.0) or 0.0) > 0:
        return True
    if float(getattr(config, "SIGNAL_WEIGHT_FLOW_DECAY", 0.0) or 0.0) > 0:
        return True
    return _lane_live("cvd") or _lane_live("pm") or _lane_live("flow_decay")


class MarketDataWarmer(threading.Thread):
    """Refresh the live market's book/price/orderflow data every ~1s."""

    def __init__(self, discovery, cvd_feed, pm_feed, interval: float | None = None) -> None:
        super().__init__(daemon=True, name="market-data-warmer")
        self._stop_event = threading.Event()
        self._discovery = discovery
        self._cvd_feed = cvd_feed
        self._pm_feed = pm_feed
        self._interval = interval or config.MARKET_DATA_INTERVAL_SEC
        self._last_flow_ts = 0.0
        self._last_cvd = 0.0
        self._last_pm: dict = {"momentum": 0.0, "prices": []}
        self._last_cycle_ms = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info(f"Market-data warmer started (interval={self._interval}s)")
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self._warm_once()
            except Exception as e:
                logger.error(f"Market-data warm error: {e}")
            elapsed = time.perf_counter() - t0
            self._last_cycle_ms = elapsed * 1000.0
            # Deadline-based sleep: stay near 1 Hz under light load.
            remain = max(0.0, float(self._interval) - elapsed)
            self._stop_event.wait(remain)
        logger.info("Market-data warmer stopped")

    def _fetch_books_parallel(self, yes_tok: str, no_tok: str) -> tuple[dict, dict]:
        """Fetch YES+NO books in parallel; never raise on partial timeout.

        Uses ``wait`` (not ``as_completed``) so a slow/hung CLOB response does
        not raise ``TimeoutError: N (of 2) futures unfinished`` out of the
        warmer cycle. Incomplete sides return ``valid=False`` and the caller
        fail-softs to the previous snapshot.
        """
        global _last_book_timeout_log
        timeout = float(getattr(config, "BOOK_FETCH_TIMEOUT_SEC", 2.0))
        # Bound total wait slightly above the per-request timeout so both
        # parallel GETs can finish under load without abandoning early.
        wait_timeout = timeout + 0.75
        futs = {
            _book_pool.submit(polymarket_markets.get_order_book, yes_tok,
                              timeout=timeout): "yes",
            _book_pool.submit(polymarket_markets.get_order_book, no_tok,
                              timeout=timeout): "no",
        }
        yes_book: dict = {"valid": False}
        no_book: dict = {"valid": False}
        done, not_done = wait(list(futs.keys()), timeout=wait_timeout)
        unfinished = len(not_done)
        if unfinished:
            now = time.time()
            # Log at most once per 15s — during a blip this used to print every
            # 1s cycle and drown the real signal.
            if now - _last_book_timeout_log >= 15.0:
                _last_book_timeout_log = now
                logger.warning(
                    "Book fetch partial/timeout: %d of %d sides unfinished "
                    "(fail-soft to last snapshot; CLOB lag or pool pressure)",
                    unfinished, len(futs),
                )
            # Best-effort cancel of not-yet-started work; running GETs will
            # finish on the extra pool workers without blocking this cycle.
            for fut in not_done:
                fut.cancel()
        for fut in done:
            side = futs[fut]
            try:
                book = fut.result(timeout=0)
            except Exception as e:
                logger.debug(f"parallel book {side} failed: {e}")
                book = {"valid": False}
            if not isinstance(book, dict):
                book = {"valid": False}
            if side == "yes":
                yes_book = book
            else:
                no_book = book
        return yes_book, no_book

    def _warm_once(self) -> None:
        markets = {}
        if hasattr(self._discovery, "current_markets_snapshot"):
            try:
                markets = self._discovery.current_markets_snapshot() or {}
            except Exception:
                markets = {}
        if not markets:
            market = self._discovery.current_market_snapshot()
            if market:
                from exchanges import exchange_of
                markets = {exchange_of(market): market}
        if not markets:
            return
        keep_ids: set[str] = set()
        for _ex, market in markets.items():
            mid = self._warm_one_market(market)
            if mid:
                keep_ids.add(mid)
        if keep_ids:
            _store.prune(keep_ids=keep_ids)

    def _warm_kalshi(self, market: dict) -> Optional[str]:
        market_id = market.get("id") or market.get("ticker")
        if not market_id:
            return None
        prev = _store.get(market_id) or {}
        ticker = market.get("ticker") or market.get("native_id") or market_id
        try:
            import kalshi_markets
            books = kalshi_markets.get_order_book(ticker)
        except Exception as e:
            logger.debug("Kalshi warm book failed: %s", e)
            books = {"valid": False}
        yes_book = (books or {}).get("yes") or {"valid": False}
        no_book = (books or {}).get("no") or {"valid": False}
        if not yes_book.get("valid") and (prev.get("yes_book") or {}).get("valid"):
            yes_book = prev["yes_book"]
        if not no_book.get("valid") and (prev.get("no_book") or {}).get("valid"):
            no_book = prev["no_book"]
        books_fresh = bool(yes_book.get("valid") and no_book.get("valid"))
        yes_price = yes_book.get("best_bid")
        ya = yes_book.get("best_ask")
        if yes_price is not None and ya is not None:
            yes_price = round((yes_price + ya) / 2.0, 4)
        elif ya is not None:
            yes_price = ya
        no_price = no_book.get("best_bid")
        na = no_book.get("best_ask")
        if no_price is not None and na is not None:
            no_price = round((no_price + na) / 2.0, 4)
        elif na is not None:
            no_price = na
        if yes_price is None:
            yes_price = prev.get("yes_price")
        if no_price is None:
            no_price = prev.get("no_price")
        strike = market.get("floor_strike") or market.get("btc_strike") or prev.get("strike")
        _store.put(market_id, {
            "market_id": market_id,
            "exchange": "kalshi",
            "yes_price": yes_price,
            "no_price": no_price,
            "yes_book": yes_book,
            "no_book": no_book,
            "obi": 0.0,
            "cvd": 0.0,
            "pm_momentum": 0.0,
            "pm_prices": [],
            "strike": strike,
            "ts": (time.time() if books_fresh
                   else float(prev.get("ts") or time.time())),
            "warm_cycle_ms": self._last_cycle_ms,
        })
        return market_id

    def _warm_one_market(self, market: dict) -> Optional[str]:
        from exchanges import exchange_of, KALSHI
        if exchange_of(market) == KALSHI:
            return self._warm_kalshi(market)
        market_id = market.get("id") or market.get("market_id")
        yes_tok = market.get("polymarket_token_id")
        no_tok = market.get("polymarket_no_token_id")
        if not market_id or not yes_tok or not no_tok:
            return None

        prev = _store.get(market_id) or {}

        # Parallel book GETs — critical path for decision freshness.
        yes_book, no_book = self._fetch_books_parallel(yes_tok, no_tok)
        yes_fresh = bool(yes_book.get("valid"))
        no_fresh = bool(no_book.get("valid"))
        # Fail soft: keep previous valid book if this fetch failed.
        if not yes_book.get("valid") and (prev.get("yes_book") or {}).get("valid"):
            yes_book = prev["yes_book"]
        if not no_book.get("valid") and (prev.get("no_book") or {}).get("valid"):
            no_book = prev["no_book"]
        books_fresh = yes_fresh and no_fresh

        yes_mid = polymarket_markets.midpoint(yes_book) if yes_book.get("valid") else None
        no_mid = polymarket_markets.midpoint(no_book) if no_book.get("valid") else None
        yes_price = clean_tick.clean_price(yes_tok, yes_mid)
        no_price = clean_tick.clean_price(no_tok, no_mid)
        if yes_price is None and prev.get("yes_price") is not None:
            yes_price = prev["yes_price"]
        if no_price is None and prev.get("no_price") is not None:
            no_price = prev["no_price"]

        obi = orderflow_signals.order_book_imbalance(yes_book) if yes_book.get("valid") else float(
            prev.get("obi", 0.0) or 0.0)

        # Microstructure context (pure, local).
        micro_spread = 0.0
        micro_spread_score = 0.5
        try:
            from signals import microstructure
            micro = microstructure.compute(yes_book, no_book)
            micro_spread = float(micro.get("micro_spread") or 0.0)
            micro_spread_score = float(micro.get("micro_spread_score") or 0.5)
        except Exception as e:
            logger.warning("swallowed exception: %s", e)

        # CVD + PM: throttle while kill-switched.
        now = time.time()
        slow = float(getattr(config, "SIGNAL_SLOW_REFRESH_SEC", 10.0))
        need_flow = _flow_needs_refresh() or (now - self._last_flow_ts) >= slow
        cond = market.get("condition_id") or market_id
        flow_decay = float(prev.get("flow_cvd_decay", 0.0) or 0.0)
        flow_whale = float(prev.get("flow_whale", 0.0) or 0.0)
        if need_flow:
            try:
                cvd = self._cvd_feed.get_cvd(cond) if cond else 0.0
            except Exception:
                cvd = self._last_cvd
            try:
                pm = self._pm_feed.get_momentum(yes_tok) if yes_tok else {}
            except Exception:
                pm = self._last_pm
            self._last_cvd = float(cvd or 0.0)
            self._last_pm = pm if isinstance(pm, dict) else {"momentum": 0.0, "prices": []}
            self._last_flow_ts = now
            # Decayed/whale CVD from the same tape cache when available.
            try:
                from signals import flow as flow_mod
                trades = (
                    self._cvd_feed.last_trades(cond)
                    if cond and hasattr(self._cvd_feed, "last_trades")
                    else []
                )
                if trades:
                    fc = flow_mod.compute(trades, now)
                    flow_decay = float(fc.get("flow_cvd_decay") or 0.0)
                    flow_whale = float(fc.get("flow_whale") or 0.0)
            except Exception as e:
                logger.warning("swallowed exception: %s", e)
        else:
            cvd = self._last_cvd if self._last_flow_ts else float(prev.get("cvd", 0.0) or 0.0)
            pm = self._last_pm if self._last_flow_ts else {
                "momentum": float(prev.get("pm_momentum", 0.0) or 0.0),
                "prices": prev.get("pm_prices") or [],
            }

        strike = get_strike_registry().get_strike(
            market_id, market.get("event_start_time"))
        if strike is None:
            strike = prev.get("strike")

        _store.put(market_id, {
            "market_id": market_id,
            "exchange": "polymarket",
            "yes_price": yes_price,
            "no_price": no_price,
            "yes_book": yes_book,
            "no_book": no_book,
            "obi": obi,
            "cvd": float(cvd or 0.0),
            "pm_momentum": float(pm.get("momentum", 0.0) or 0.0),
            "pm_prices": pm.get("prices", []) or [],
            "flow_cvd_decay": flow_decay,
            "flow_whale": flow_whale,
            "micro_spread": micro_spread,
            "micro_spread_score": micro_spread_score,
            "strike": strike,
            "ts": (time.time() if books_fresh
                   else float(prev.get("ts") or time.time())),
            "warm_cycle_ms": self._last_cycle_ms,
        })
        return market_id

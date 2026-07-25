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
"""

import copy
import logging
import threading
import time
from typing import Optional

import config
import polymarket_markets
from signals import clean_tick, orderflow_signals
from signals.strike import get_strike_registry

logger = logging.getLogger("arena.market_data")


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

    def prune(self, keep_market_id: Optional[str]) -> None:
        """Drop every snapshot except the live market (keeps the map tiny)."""
        with self._lock:
            self._snap = {
                k: v for k, v in self._snap.items() if k == keep_market_id
            }


_store = MarketDataStore()


def store() -> MarketDataStore:
    """Shared market-data store read by the trader and the arbitrage bot."""
    return _store


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


class MarketDataWarmer(threading.Thread):
    """Refresh the live market's book/price/orderflow data every ~1s."""

    def __init__(self, discovery, cvd_feed, pm_feed, interval: float | None = None) -> None:
        super().__init__(daemon=True, name="market-data-warmer")
        self._stop_event = threading.Event()
        self._discovery = discovery
        self._cvd_feed = cvd_feed
        self._pm_feed = pm_feed
        self._interval = interval or config.MARKET_DATA_INTERVAL_SEC

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info(f"Market-data warmer started (interval={self._interval}s)")
        while not self._stop_event.is_set():
            try:
                self._warm_once()
            except Exception as e:
                logger.error(f"Market-data warm error: {e}")
            self._stop_event.wait(self._interval)
        logger.info("Market-data warmer stopped")

    def _warm_once(self) -> None:
        market = self._discovery.current_market_snapshot()
        if not market:
            return
        market_id = market.get("id") or market.get("market_id")
        yes_tok = market.get("polymarket_token_id")
        no_tok = market.get("polymarket_no_token_id")
        if not market_id or not yes_tok or not no_tok:
            return

        # One book per side — the source of price, depth and OBI.
        yes_book = polymarket_markets.get_order_book(yes_tok)
        no_book = polymarket_markets.get_order_book(no_tok)

        yes_mid = polymarket_markets.midpoint(yes_book) if yes_book.get("valid") else None
        no_mid = polymarket_markets.midpoint(no_book) if no_book.get("valid") else None
        # Clean-tick guard (reject implausible single-tick jumps / bad data),
        # keyed per token so YES and NO are independent.
        yes_price = clean_tick.clean_price(yes_tok, yes_mid)
        no_price = clean_tick.clean_price(no_tok, no_mid)

        obi = orderflow_signals.order_book_imbalance(yes_book) if yes_book.get("valid") else 0.0

        # CVD (trade tape) + PM in-market momentum. Feeds coalesce within their
        # sub-second TTL; here they refresh essentially every cycle.
        cond = market.get("condition_id") or market_id
        cvd = self._cvd_feed.get_cvd(cond) if cond else 0.0
        pm = self._pm_feed.get_momentum(yes_tok)

        # Accurate strike (Binance open @ eventStartTime), fetched once per market
        # off the hot path and cached in the registry. None until available.
        strike = get_strike_registry().get_strike(
            market_id, market.get("event_start_time"))

        _store.put(market_id, {
            "market_id": market_id,
            "yes_price": yes_price,
            "no_price": no_price,
            "yes_book": yes_book,
            "no_book": no_book,
            "obi": obi,
            "cvd": cvd,
            "pm_momentum": float(pm.get("momentum", 0.0) or 0.0),
            "pm_prices": pm.get("prices", []) or [],
            "strike": strike,
            "ts": time.time(),
        })
        _store.prune(keep_market_id=market_id)

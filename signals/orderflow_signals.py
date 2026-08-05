"""Order-flow signals: Order Book Imbalance (OBI) and Cumulative Volume Delta (CVD).

These are the two order-flow reads the profitable-bot research repeatedly points
to (0x_Punisher, Jun 2026): unlike RSI/MACD/EMA — which are all just past price
transformed and smoothed — OBI and CVD describe *pressure that has not fully hit
the price yet*.

  * **OBI** — ratio of resting bid depth to ask depth near the top of the book.
    Heavy bids + thin asks means the next aggressive seller gets absorbed and
    price lifts. Computed here from the *Up/YES* token's normalized CLOB book,
    so ``obi > 0`` = upward (YES) pressure.

  * **CVD** — running total of market buys minus market sells. Price can sit flat
    while CVD climbs (silent accumulation). Computed from the market's recent
    executed trades (Polymarket data-api), signed into *Up-space*:
    buying Up or selling Down is bullish (+), selling Up or buying Down is
    bearish (−).

Both return a bounded float in ``[-1, 1]``. ``make_decision`` scales them into
the same signal band as the other secondary lanes before weighting.
"""

import logging
import threading
import time
from typing import Optional

import http_client

import config

logger = logging.getLogger(__name__)

TRADES_URL = "https://data-api.polymarket.com/trades"
# Coalescing-guard TTL only: the 1s market-data warmer is effectively the sole
# caller and refreshes every cycle, so keep this just under the warm interval
# (falls back to 20s if the config knob is absent).
CVD_CACHE_TTL = getattr(config, "SIGNAL_CACHE_TTL_SEC", 20)
CVD_TRADE_LIMIT = 100   # recent trades to sum per market


def order_book_imbalance(book: dict, levels: int = 3) -> float:
    """Order-book imbalance over the top ``levels`` of a normalized book.

    ``book`` is the dict returned by :func:`polymarket_markets.get_order_book`
    (``bids``/``asks`` are ``(price, size)`` lists, best-first). Returns
    ``(bid_vol - ask_vol) / (bid_vol + ask_vol)`` in ``[-1, 1]``:

      * ``> 0`` — more resting bid depth → upward pressure on this token (YES)
      * ``< 0`` — more resting ask depth → downward pressure

    Returns ``0.0`` when the book is invalid or has no depth (no signal).
    """
    if not book or not book.get("valid"):
        return 0.0
    bid_vol = sum(size for _, size in (book.get("bids") or [])[:levels])
    ask_vol = sum(size for _, size in (book.get("asks") or [])[:levels])
    total = bid_vol + ask_vol
    if total <= 0:
        return 0.0
    return max(-1.0, min(1.0, (bid_vol - ask_vol) / total))


def cvd_from_trades(trades: list) -> float:
    """Cumulative volume delta over recent trades, signed into Up-space.

    Each trade dict has ``side`` ('BUY'/'SELL'), ``outcome`` ('Up'/'Down') and
    ``size``. A trade is bullish-on-Up when buying Up or selling Down; bearish
    when selling Up or buying Down. Returns ``net / max(total_volume, floor)``
    in ``[-1, 1]`` (``0.0`` when there are no trades).

    Volume floor (BUG #27): plain ``net/total`` saturated at ±1.0 on any thin
    one-sided tape (3 prints → "maximum conviction"), degrading the lane to
    sign(recent tape). The floor (``config.CVD_VOLUME_FLOOR`` shares) makes
    magnitude mean something: a 30-share one-sided tape reads ~0.15, a
    1500-share one still reads ~1.0.
    """
    net = 0.0
    total = 0.0
    for t in trades or []:
        try:
            size = float(t.get("size", 0) or 0)
        except (TypeError, ValueError):
            continue
        if size <= 0:
            continue
        is_up = str(t.get("outcome", "")).lower() == "up"
        is_buy = str(t.get("side", "")).upper() == "BUY"
        # BUY Up / SELL Down => +size ; SELL Up / BUY Down => -size
        net += size if (is_up == is_buy) else -size
        total += size
    if total <= 0:
        return 0.0
    floor = getattr(config, "CVD_VOLUME_FLOOR", 200.0)
    return max(-1.0, min(1.0, net / max(total, floor)))


class CvdFeed:
    """Per-market CVD, cached ~20s to keep the trader hot path network-free.

    Keyed by ``condition_id``. Brand-new windows return ``0.0`` until trades
    exist. Network failures return the last cached value (or ``0.0``).
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}   # cond -> {ts, cvd, trades}
        self._lock = threading.Lock()

    def _fetch(self, condition_id: str) -> list:
        try:
            resp = http_client.get(
                TRADES_URL,
                params={"market": condition_id, "limit": CVD_TRADE_LIMIT},
                timeout=8,
            )
            if resp.status_code == 200:
                return resp.json() or []
        except Exception as e:
            logger.debug(f"CVD trades fetch error ({str(condition_id)[:12]}…): {e}")
        return []

    def get_cvd(self, condition_id: str) -> float:
        """Return the cached/fresh CVD for a market in ``[-1, 1]``."""
        if not condition_id:
            return 0.0
        now = time.time()
        with self._lock:
            cached = self._cache.get(condition_id)
            if cached and (now - cached["ts"]) < CVD_CACHE_TTL:
                return cached["cvd"]

        trades = self._fetch(condition_id)
        if not trades:
            # Keep last value on a transient miss; 0.0 for a genuinely empty book.
            with self._lock:
                cached = self._cache.get(condition_id)
                return cached["cvd"] if cached else 0.0

        cvd = cvd_from_trades(trades)
        with self._lock:
            self._cache[condition_id] = {"ts": now, "cvd": cvd, "trades": trades}
        return cvd

    def last_trades(self, condition_id: str) -> list:
        """Most recently fetched tape for flow features (empty if never fetched)."""
        if not condition_id:
            return []
        with self._lock:
            cached = self._cache.get(condition_id) or {}
            return list(cached.get("trades") or [])

    def clear(self, condition_id: str | None = None) -> None:
        with self._lock:
            if condition_id:
                self._cache.pop(condition_id, None)
            else:
                self._cache.clear()


_feed: Optional[CvdFeed] = None


def get_cvd_feed() -> CvdFeed:
    global _feed
    if _feed is None:
        _feed = CvdFeed()
    return _feed

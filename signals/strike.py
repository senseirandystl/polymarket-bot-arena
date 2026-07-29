"""BTC 'price to beat' (strike) registry + drift signal.

Polymarket BTC 5-min Up/Down markets resolve UP iff Chainlink BTC/USD at window
CLOSE is ≥ Chainlink BTC/USD at window OPEN — the "price to beat" (strike).
The resolution source is the Chainlink BTC/USD data stream
(https://data.chain.link/streams/btc-usd), **not** Binance or any spot venue.

**Accuracy matters (BUG #23 + 2026-07-29 fix).** Early code used a mid-window
"first sighting" snapshot (inverted drift, blew the account). The next cut used
Binance BTCUSDT 1m OPEN at ``eventStartTime`` as a proxy — still wrong by ~$60–80
(~0.1%) vs Polymarket's displayed Price to Beat because of Binance↔Chainlink
basis. Live edge is measured against the **official** open:

  ``GET https://polymarket.com/api/crypto/crypto-price
       ?symbol=BTC&eventStartTime=…&variant=fiveminute&endDate=…``
  → ``{openPrice, closePrice, …}``

That is the same endpoint Polymarket's UI uses (React Query key
``["crypto-prices","price","BTC", start, "fiveminute", end]``). Live trading
never falls back to Binance (a wrong cached strike is worse than ``drift=0``
for one cycle). Offline harnesses may still reconstruct from Binance klines for
relative ranking only.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import http_client

import config

logger = logging.getLogger(__name__)

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
# Same path Polymarket's frontend hits for Price to Beat (openPrice).
POLYMARKET_CRYPTO_PRICE = "https://polymarket.com/api/crypto/crypto-price"
_WINDOW_SEC = int(getattr(config, "MARKET_WINDOW_SEC", 300) or 300)


def _parse_event_start(event_start_iso: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(event_start_iso.replace("Z", "+00:00"))
    except Exception:
        return None


def _end_iso_from_start(event_start_iso: str) -> Optional[str]:
    """Window end = start + MARKET_WINDOW_SEC (5m for the default series)."""
    start = _parse_event_start(event_start_iso)
    if start is None:
        return None
    end = start + timedelta(seconds=_WINDOW_SEC)
    # Polymarket's query uses trailing Z, not +00:00
    return end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_polymarket_open_price(event_start_iso: str) -> Optional[float]:
    """Official Polymarket Price to Beat (Chainlink open at eventStartTime).

    Returns None when the open is not yet published (pre-window / incomplete)
    or on transport/API errors — callers must not invent a strike.
    """
    end_iso = _end_iso_from_start(event_start_iso)
    if not end_iso:
        return None
    # Normalize start to the same Z form the UI sends.
    start = _parse_event_start(event_start_iso)
    if start is None:
        return None
    start_iso = start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        r = http_client.get(
            POLYMARKET_CRYPTO_PRICE,
            params={
                "symbol": "BTC",
                "eventStartTime": start_iso,
                "variant": "fiveminute",
                "endDate": end_iso,
            },
            timeout=8,
            headers={
                "Accept": "application/json",
                "User-Agent": "polymarket-bot-arena/1.0",
            },
        )
        if r.status_code != 200:
            logger.debug(
                "polymarket openPrice HTTP %s for %s: %s",
                r.status_code, start_iso, (r.text or "")[:160],
            )
            return None
        data = r.json() or {}
        op = data.get("openPrice")
        if op is None:
            return None
        val = float(op)
        return val if val > 0 else None
    except Exception as e:
        logger.debug("polymarket openPrice fetch failed for %s: %s", event_start_iso, e)
        return None


def _fetch_binance_open_at(event_start_iso: str) -> Optional[float]:
    """Binance BTCUSDT 1m OPEN at window open — approximate fallback only."""
    try:
        ts = int(datetime.fromisoformat(
            event_start_iso.replace("Z", "+00:00")).timestamp() * 1000)
        r = http_client.get(BINANCE_KLINES, params={
            "symbol": "BTCUSDT", "interval": "1m", "startTime": ts, "limit": 1,
        }, timeout=6)
        if r.status_code == 200:
            k = r.json()
            if k:
                return float(k[0][1])          # kline open
    except Exception:
        pass
    return None


def _fetch_chainlink_feed_latch(event_start_iso: str) -> Optional[float]:
    """Latch strike from the live Chainlink tick buffer (same oracle as RTDS).

    Only used when the REST openPrice endpoint is unavailable. Never uses
    Binance — that would re-introduce the ~0.1% basis bug.
    """
    start = _parse_event_start(event_start_iso)
    if start is None:
        return None
    try:
        from signals.price_feed import get_feed
        feed = get_feed()
        px = feed.price_at(start.timestamp(), tol_sec=5.0)
        if px and px > 0:
            logger.info(
                "Strike latched from Chainlink RTDS buffer at %s → %.4f",
                event_start_iso, px,
            )
            return float(px)
    except Exception as e:
        logger.debug("Chainlink feed latch failed for %s: %s", event_start_iso, e)
    return None


def _fetch_open_at(event_start_iso: str) -> Optional[float]:
    """Strike for a window open — Chainlink only (never Binance).

    Order:
      1. Polymarket REST ``openPrice`` (exact UI Price to Beat)
      2. Live Chainlink RTDS tick latch at ``eventStartTime``

    If both miss, return None (drift stays 0) and retry next cycle.
    ``_fetch_binance_open_at`` remains for offline harnesses / diagnostics only.
    """
    official = _fetch_polymarket_open_price(event_start_iso)
    if official and official > 0:
        return official
    latched = _fetch_chainlink_feed_latch(event_start_iso)
    if latched and latched > 0:
        return latched
    return None


# How often a provisional (RTDS-latch) strike may re-query openPrice for upgrade.
_PROVISIONAL_REFRESH_SEC = 5.0
# Only accept a Chainlink latch when the tick is within this many seconds of open.
_LATCH_MAX_SKEW_SEC = 2.0


def _fetch_chainlink_feed_latch_strict(event_start_iso: str) -> Optional[float]:
    """Latch only if we have a tick near the true open (not a mid-window sample)."""
    start = _parse_event_start(event_start_iso)
    if start is None:
        return None
    try:
        from signals.price_feed import get_feed
        feed = get_feed()
        px = feed.price_at(start.timestamp(), tol_sec=_LATCH_MAX_SKEW_SEC)
        if px and px > 0:
            logger.info(
                "Strike latched from Chainlink RTDS at %s → %.4f (provisional)",
                event_start_iso, px,
            )
            return float(px)
    except Exception as e:
        logger.debug("Chainlink feed latch failed for %s: %s", event_start_iso, e)
    return None


class StrikeRegistry:
    """Per-market strike with official-vs-provisional caching.

    Production path:
      * Prefer Polymarket REST ``openPrice`` (matches the website PTB) and cache
        permanently as ``source=openPrice``.
      * If REST is rate-limited / not yet published, optionally latch a Chainlink
        RTDS tick near ``eventStartTime`` as ``source=latch`` (provisional).
      * Provisional entries are re-checked every few seconds so a late openPrice
        always wins — prevents a wrong first-sight latch from sticking for the
        whole window (market-turnover PTB bug).

    ``fetcher`` is injectable for tests (returns a float; stored as openPrice).
    """

    def __init__(self, fetcher: Callable[[str], Optional[float]] | None = None) -> None:
        self._lock = threading.Lock()
        self._strikes: dict[str, dict] = {}    # market_id -> {strike, ts, source}
        # None → production multi-source path; set → test double
        self._fetch = fetcher

    def get_strike(self, market_id: Optional[str],
                   event_start_time: Optional[str]) -> Optional[float]:
        """Accurate strike for a market. Official openPrice is sticky; latch is not."""
        if not market_id or not event_start_time:
            return None

        # Test double: simple fetch-and-cache (legacy behaviour)
        if self._fetch is not None:
            with self._lock:
                rec = self._strikes.get(market_id)
                if rec is not None:
                    return rec["strike"]
            strike = self._fetch(event_start_time)
            if not strike or strike <= 0:
                return None
            self._store(market_id, float(strike), "test")
            return float(strike)

        now = time.time()
        with self._lock:
            rec = self._strikes.get(market_id)
            if rec is not None and rec.get("source") == "openPrice":
                return rec["strike"]
            # Hold provisional briefly to avoid hammering the REST API
            if (
                rec is not None
                and rec.get("source") == "latch"
                and (now - float(rec.get("ts") or 0)) < _PROVISIONAL_REFRESH_SEC
            ):
                return rec["strike"]
            provisional = rec["strike"] if rec is not None else None

        # Always try official openPrice first (and to upgrade a provisional)
        official = _fetch_polymarket_open_price(event_start_time)
        if official and official > 0:
            self._store(market_id, float(official), "openPrice")
            return float(official)

        if provisional is not None:
            # Refresh timestamp so we don't spin-retry every tick
            self._store(market_id, float(provisional), "latch")
            return float(provisional)

        latched = _fetch_chainlink_feed_latch_strict(event_start_time)
        if latched and latched > 0:
            self._store(market_id, float(latched), "latch")
            return float(latched)
        return None

    def _store(self, market_id: str, strike: float, source: str) -> None:
        now = time.time()
        with self._lock:
            self._strikes[market_id] = {
                "strike": float(strike), "ts": now, "source": source,
            }
            if len(self._strikes) > 64:
                cutoff = now - 3600
                self._strikes = {
                    k: v for k, v in self._strikes.items() if v["ts"] >= cutoff
                }

    def get_source(self, market_id: str) -> Optional[str]:
        with self._lock:
            rec = self._strikes.get(market_id)
            return rec.get("source") if rec else None


def drift_signal(strike_price: Optional[float], btc_now: float,
                 time_remaining: Optional[float]) -> float:
    """Bounded, time-scaled BTC drift-from-strike signal in ``[-1, 1]``.

    ``>0`` = BTC above the strike (YES/Up favored), ``<0`` = below (NO/Down).
    Magnitude scales with the drift as a fraction of typical remaining-window
    volatility, so the same drift reads stronger as expiry nears.

    Note: production feeds both ``strike_price`` (openPrice / Chainlink latch)
    and ``btc_now`` (Chainlink RTDS via ``signals.price_feed``) from the same
    oracle family so drift is true moneyness vs Polymarket resolution.
    """
    if not strike_price or strike_price <= 0 or not btc_now or btc_now <= 0:
        return 0.0
    drift_pct = (btc_now - strike_price) / strike_price
    if drift_pct == 0.0:
        return 0.0
    window = config.MARKET_WINDOW_SEC
    tr = max(float(time_remaining if time_remaining is not None else window), 10.0)
    sigma_remaining = config.DRIFT_VOL_SCALE * math.sqrt(min(tr, window) / window)
    if sigma_remaining <= 0:
        return 0.0
    z = drift_pct / sigma_remaining
    return math.tanh(z)


_registry: Optional[StrikeRegistry] = None


def get_strike_registry() -> StrikeRegistry:
    global _registry
    if _registry is None:
        _registry = StrikeRegistry()
    return _registry

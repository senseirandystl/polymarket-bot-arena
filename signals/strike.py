"""BTC 'price to beat' (strike) registry + drift signal.

Polymarket BTC 5-min Up/Down markets resolve UP iff the Chainlink **TWAP**
at window CLOSE is ≥ the Chainlink **TWAP** at window OPEN — the "price to
beat" (strike). Effective 2026-08-07 00:00 UTC both open and settlement use
the TWAP feed (``TWAP_WINDOW_SEC`` lookback — **60s** for 5-minute markets).
Spec: https://docs.polymarket.com/market-data/chainlink-twap

The resolution source is Chainlink TWAP (relayed via Polymarket RTDS
``crypto_prices_twap_sixty`` for the 60s lookback), **not** Binance or a
single spot snapshot.

**Accuracy matters (BUG #23 + 2026-07-29 fix + TWAP 2026-08).** Early code used
a mid-window "first sighting" snapshot (inverted drift, blew the account).
Later cuts used Binance or spot Chainlink as proxies. Live edge is measured
against the **official** open:

  ``GET https://polymarket.com/api/crypto/crypto-price
       ?symbol=BTC&eventStartTime=…&variant=fiveminute&endDate=…``
  → ``{openPrice, closePrice, …}``

After the 2026-08-07 TWAP cutover, the Price to Beat is the Chainlink
**TWAP at window open** (same RTDS feed as live BTC; 60s lookback for 5m).
Live soak showed the REST ``openPrice`` field can diverge ~$2–3 from the
Polymarket UI; we therefore prefer RTDS ``twap_at(eventStartTime)`` as sticky
``twap_open`` and use REST only as a fallback when the open tick was missed.
Never Binance live. Offline harnesses may still reconstruct from Binance
klines for relative ranking only.
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


# Backoff after REST 429 so the 1s warmer does not hammer the PTB endpoint.
_OPENPRICE_BACKOFF_UNTIL = 0.0
_OPENPRICE_BACKOFF_LOCK = threading.Lock()
_OPENPRICE_VARIANT = "fiveminute"  # window-specific; other variants return sticky garbage


def _fetch_polymarket_open_price(event_start_iso: str) -> Optional[float]:
    """Official Polymarket Price to Beat (Chainlink TWAP open at eventStartTime).

    Returns None when the open is not yet published (pre-window / incomplete),
    on 429 rate-limit (with process-wide backoff), or on transport/API errors —
    callers must not invent a strike.

    Uses variant ``fiveminute`` only. Other spellings (``fiveMinute``, bare
    params) return a sticky non-window open (~same number across markets) that
    does **not** match UI / settlement.
    """
    global _OPENPRICE_BACKOFF_UNTIL
    end_iso = _end_iso_from_start(event_start_iso)
    if not end_iso:
        return None
    # Normalize start to the same Z form the UI sends.
    start = _parse_event_start(event_start_iso)
    if start is None:
        return None
    start_iso = start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    now = time.time()
    with _OPENPRICE_BACKOFF_LOCK:
        if now < _OPENPRICE_BACKOFF_UNTIL:
            return None

    try:
        r = http_client.get(
            POLYMARKET_CRYPTO_PRICE,
            params={
                "symbol": "BTC",
                "eventStartTime": start_iso,
                "variant": _OPENPRICE_VARIANT,
                "endDate": end_iso,
            },
            timeout=8,
            headers={
                "Accept": "application/json",
                "User-Agent": "polymarket-bot-arena/1.0",
                "Referer": "https://polymarket.com/",
            },
        )
        if r.status_code == 429:
            # Cool down so warmer/dashboard don't stampede the endpoint.
            with _OPENPRICE_BACKOFF_LOCK:
                _OPENPRICE_BACKOFF_UNTIL = time.time() + 8.0
            logger.warning(
                "polymarket openPrice rate-limited for %s — backoff 8s",
                start_iso,
            )
            return None
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
        if val <= 0:
            return None
        logger.info(
            "Strike REST openPrice for %s → %.4f incomplete=%s "
            "(fallback; prefer RTDS TWAP-at-open when available)",
            start_iso, val, data.get("incomplete"),
        )
        return val
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


# Sticky TWAP-at-open is the Price to Beat under 2026-08-07+ resolution.
# REST /api/crypto/crypto-price openPrice can diverge ~$2–3 from the UI/TWAP
# feed (live soak 2026-08-07: REST 64248.01 vs Polymarket UI / RTDS 64250.59).
# Only accept a Chainlink sample within this many seconds of eventStartTime.
_LATCH_MAX_SKEW_SEC = 2.0
# How long after open we still try to latch TWAP from the ring buffer.
# Buffer holds ~2h of ticks; allow recovery for most of the 5m window.
_TWAP_OPEN_LATCH_UNTIL_SEC = 280.0
# How often non-sticky sources re-check for a better TWAP-open sample.
_PROVISIONAL_REFRESH_SEC = 3.0
# Max age of REST/spot provisional before drop (prefer drift=0).
_PROVISIONAL_MAX_AGE_SEC = 60.0
# Sticky sources that match Polymarket UI PTB (never overwrite with REST).
_STICKY_TWAP_SOURCES = frozenset({"twap_open"})
# Weaker sources that may be upgraded to twap_open when buffer has open tick.
_UPGRADABLE_SOURCES = frozenset({"openPrice", "latch", "spot_latch", "rest"})


def _twap_open_from_feed(
    event_start_iso: str,
    *,
    tol_sec: float = _LATCH_MAX_SKEW_SEC,
    allow_late: bool = True,
) -> Optional[float]:
    """Chainlink RTDS TWAP observation nearest to window open.

    This is the Price to Beat Polymarket shows and settles against (TWAP at
    open). ``allow_late=True`` still requires the *tick* to be within
    ``tol_sec`` of open, but permits reading the ring buffer long after open
    (as long as we captured the open sample earlier in the window).
    """
    start = _parse_event_start(event_start_iso)
    if start is None:
        return None
    open_ts = start.timestamp()
    if not allow_late and (time.time() - open_ts) > 15.0:
        return None
    # Past end of window: still OK if buffer retains the open tick.
    if (time.time() - open_ts) > _TWAP_OPEN_LATCH_UNTIL_SEC:
        return None
    try:
        from signals.price_feed import get_feed
        feed = get_feed()
        if not hasattr(feed, "twap_at"):
            return None
        px = feed.twap_at(open_ts, tol_sec=tol_sec)
        if px and px > 0:
            return float(px)
    except Exception as e:
        logger.debug("TWAP open latch failed for %s: %s", event_start_iso, e)
    return None


def _spot_open_from_feed(event_start_iso: str) -> Optional[float]:
    """Last-resort spot Chainlink sample near open (not UI PTB under TWAP)."""
    start = _parse_event_start(event_start_iso)
    if start is None:
        return None
    open_ts = start.timestamp()
    if time.time() - open_ts > 15.0:
        return None
    try:
        from signals.price_feed import get_feed
        feed = get_feed()
        px = feed.price_at(open_ts, tol_sec=_LATCH_MAX_SKEW_SEC)
        if px and px > 0:
            return float(px)
    except Exception as e:
        logger.debug("Spot open latch failed for %s: %s", event_start_iso, e)
    return None


def _fetch_chainlink_feed_latch(event_start_iso: str) -> Optional[float]:
    """Latch strike from live RTDS buffers (TWAP preferred, spot fallback)."""
    twap = _twap_open_from_feed(event_start_iso, allow_late=True)
    if twap and twap > 0:
        logger.info(
            "Strike latched from Chainlink TWAP RTDS at %s → %.4f",
            event_start_iso, twap,
        )
        return twap
    spot = _spot_open_from_feed(event_start_iso)
    if spot and spot > 0:
        logger.info(
            "Strike latched from Chainlink spot RTDS at %s → %.4f",
            event_start_iso, spot,
        )
        return spot
    return None


def _fetch_chainlink_feed_latch_strict(event_start_iso: str) -> Optional[float]:
    """TWAP-at-open only (strict tick skew). Preferred PTB source."""
    px = _twap_open_from_feed(event_start_iso, allow_late=True)
    if px and px > 0:
        logger.info(
            "Strike TWAP-at-open (UI PTB) for %s → %.4f",
            event_start_iso, px,
        )
        return float(px)
    return None


def _fetch_open_at(event_start_iso: str) -> Optional[float]:
    """Strike for a window open — TWAP-at-open first (never Binance).

    Order:
      1. Chainlink RTDS TWAP sample at ``eventStartTime`` (matches Polymarket UI)
      2. REST ``openPrice`` fallback (can diverge ~$2–3 from UI — last resort)
      3. Spot Chainlink latch

    If all miss, return None (drift stays 0) and retry next cycle.
    """
    twap = _twap_open_from_feed(event_start_iso, allow_late=True)
    if twap and twap > 0:
        return twap
    official = _fetch_polymarket_open_price(event_start_iso)
    if official and official > 0:
        return official
    spot = _spot_open_from_feed(event_start_iso)
    if spot and spot > 0:
        return spot
    return None


_STRIKE_STATE_KEY = "strike_cache"
_STRIKE_STATE_MAX = 32  # markets retained across restarts


def _persist_strike(market_id: str, event_start: str, strike: float, source: str) -> None:
    """Persist sticky TWAP-open PTB so mid-window restarts keep the UI value."""
    if source not in _STICKY_TWAP_SOURCES:
        return
    try:
        import json as _json
        import db as _db
        raw = _db.get_arena_state(_STRIKE_STATE_KEY)
        cache = _json.loads(raw) if raw else {}
        if not isinstance(cache, dict):
            cache = {}
        cache[str(market_id)] = {
            "strike": float(strike),
            "source": source,
            "event_start": event_start,
            "ts": time.time(),
        }
        # Drop oldest if oversized
        if len(cache) > _STRIKE_STATE_MAX:
            ordered = sorted(cache.items(), key=lambda kv: float(kv[1].get("ts") or 0))
            for k, _ in ordered[: max(0, len(cache) - _STRIKE_STATE_MAX)]:
                cache.pop(k, None)
        _db.set_arena_state(_STRIKE_STATE_KEY, _json.dumps(cache))
    except Exception as e:
        logger.debug("strike persist failed: %s", e)


def _load_persisted_strike(
    market_id: str, event_start: str,
) -> Optional[tuple[float, str]]:
    """Load persisted TWAP-open for this market+open if present."""
    try:
        import json as _json
        import db as _db
        raw = _db.get_arena_state(_STRIKE_STATE_KEY)
        if not raw:
            return None
        cache = _json.loads(raw)
        rec = (cache or {}).get(str(market_id))
        if not isinstance(rec, dict):
            return None
        if str(rec.get("event_start") or "") != str(event_start):
            return None
        if rec.get("source") not in _STICKY_TWAP_SOURCES:
            return None
        val = float(rec.get("strike") or 0)
        if val <= 0:
            return None
        return val, str(rec.get("source"))
    except Exception:
        return None


class StrikeRegistry:
    """Per-market Price to Beat with TWAP-open priority.

    Production path (2026-08-07+ TWAP resolution):
      1. **Sticky ``twap_open``** — RTDS TWAP observation at
         ``eventStartTime``. Matches Polymarket UI PTB and settlement open.
         Persisted to arena_state so mid-window restarts keep the value.
      2. **REST ``openPrice``** — fallback only when the open tick was missed
         (restart mid-window / RTDS gap). Can diverge from UI (~$2–3 observed);
         always upgrade to ``twap_open`` if the buffer later has the open tick.
      3. **Spot latch** — last resort; never preferred under TWAP resolution.

    Mid-window samples are rejected (tick must be within ``_LATCH_MAX_SKEW_SEC``
    of open). ``fetcher`` is injectable for tests.
    """

    def __init__(self, fetcher: Callable[[str], Optional[float]] | None = None) -> None:
        self._lock = threading.Lock()
        self._strikes: dict[str, dict] = {}    # market_id -> {strike, ts, source}
        # None → production multi-source path; set → test double
        self._fetch = fetcher

    def get_strike(self, market_id: Optional[str],
                   event_start_time: Optional[str]) -> Optional[float]:
        """Price to Beat. Sticky TWAP-at-open preferred over REST openPrice."""
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
            src = rec.get("source") if rec else None
            # Authoritative sticky: RTDS TWAP at open (Polymarket UI PTB)
            if rec is not None and src in _STICKY_TWAP_SOURCES:
                return rec["strike"]
            current = rec["strike"] if rec is not None else None
            current_src = src
            current_age = None
            last_ts = float(rec.get("ts") or 0) if rec else 0.0
            if rec is not None:
                current_age = now - float(rec.get("first_ts") or rec.get("ts") or 0)

        # 0) Hydrate from DB if we latched TWAP-open earlier this window
        #    (survives mid-window arena restart when ring buffer is empty).
        if current is None or current_src not in _STICKY_TWAP_SOURCES:
            persisted = _load_persisted_strike(market_id, event_start_time)
            if persisted:
                p_strike, p_src = persisted
                logger.info(
                    "Strike restored from arena_state TWAP-open for %s → %.4f",
                    market_id, p_strike,
                )
                self._store(
                    market_id, float(p_strike), p_src,
                    event_start=event_start_time,
                )
                return float(p_strike)

        # 1) Always try TWAP-at-open from ring buffer (even if REST already set).
        #    This fixes the live bug: REST openPrice stuck $2.58 below UI PTB.
        #    Never early-return on openPrice — always attempt this upgrade.
        twap_open = _twap_open_from_feed(event_start_time, allow_late=True)
        if twap_open and twap_open > 0:
            if current is not None and current_src not in _STICKY_TWAP_SOURCES:
                delta = abs(float(current) - float(twap_open))
                if delta > 0.50:
                    logger.warning(
                        "Strike corrected to TWAP-at-open=%.4f over %s=%.4f "
                        "for %s (Δ=$%.2f) — REST/spot was wrong vs Polymarket UI",
                        twap_open, current_src, current, market_id, delta,
                    )
                else:
                    logger.info(
                        "Strike upgraded to TWAP-at-open=%.4f (was %s=%.4f) for %s",
                        twap_open, current_src, current, market_id,
                    )
            elif current is None:
                logger.info(
                    "Strike TWAP-at-open (UI PTB) for %s → %.4f",
                    event_start_time, twap_open,
                )
            self._store(
                market_id, float(twap_open), "twap_open",
                event_start=event_start_time,
            )
            return float(twap_open)

        # 2) Already have REST/spot provisional — serve it; throttle REST re-fetch
        if current is not None and current_src in _UPGRADABLE_SOURCES:
            if (
                current_age is not None
                and current_age > _PROVISIONAL_MAX_AGE_SEC
                and current_src not in ("openPrice",)
            ):
                logger.warning(
                    "Dropping provisional strike %.4f (%s) for %s after %.0fs "
                    "without TWAP-at-open — drift stays 0 until open tick found",
                    current, current_src, market_id, current_age,
                )
                with self._lock:
                    self._strikes.pop(market_id, None)
            else:
                # Keep serving; only re-hit REST after refresh interval
                if (now - last_ts) < _PROVISIONAL_REFRESH_SEC:
                    return float(current)
                # fall through to re-try REST/spot after refresh window

        # 3) REST openPrice fallback (may diverge from UI — never beats twap_open)
        rest = _fetch_polymarket_open_price(event_start_time)
        if rest and rest > 0:
            if current is None or current_src != "openPrice":
                logger.info(
                    "Strike REST openPrice fallback for %s → %.4f "
                    "(will upgrade to TWAP-at-open if buffer has open tick)",
                    event_start_time, rest,
                )
            self._store(market_id, float(rest), "openPrice")
            return float(rest)

        # 4) Spot latch last resort (near open only)
        spot = _spot_open_from_feed(event_start_time)
        if spot and spot > 0:
            logger.info(
                "Strike spot latch (last resort) for %s → %.4f",
                event_start_time, spot,
            )
            self._store(market_id, float(spot), "spot_latch")
            return float(spot)

        # Keep last known weak value if REST failed this cycle
        if current is not None:
            self._store(
                market_id, float(current), str(current_src or "latch"),
                keep_first=True,
            )
            return float(current)
        return None

    def _store(
        self,
        market_id: str,
        strike: float,
        source: str,
        *,
        keep_first: bool = False,
        event_start: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            prev = self._strikes.get(market_id)
            first_ts = now
            if keep_first and prev is not None:
                first_ts = float(prev.get("first_ts") or prev.get("ts") or now)
            elif prev is not None and prev.get("source") == source:
                first_ts = float(prev.get("first_ts") or prev.get("ts") or now)
            est = event_start or (prev.get("event_start") if prev else None)
            self._strikes[market_id] = {
                "strike": float(strike),
                "ts": now,
                "first_ts": first_ts,
                "source": source,
                "event_start": est,
            }
            if len(self._strikes) > 64:
                cutoff = now - 3600
                self._strikes = {
                    k: v for k, v in self._strikes.items() if v["ts"] >= cutoff
                }
        if source in _STICKY_TWAP_SOURCES and est:
            _persist_strike(market_id, str(est), float(strike), source)

    def get_source(self, market_id: str) -> Optional[str]:
        with self._lock:
            rec = self._strikes.get(market_id)
            return rec.get("source") if rec else None


def drift_signal(strike_price: Optional[float], btc_now: float,
                 time_remaining: Optional[float],
                 vol_scale: Optional[float] = None) -> float:
    """Bounded, time-scaled BTC drift-from-strike signal in ``[-1, 1]``.

    ``>0`` = BTC above the strike (YES/Up favored), ``<0`` = below (NO/Down).
    Magnitude scales with the drift as a fraction of remaining-window
    volatility, so the same drift reads stronger as expiry nears.

    Full-window vol scale (``vol_scale``) is **adaptive by default** from
    TWAP (preferred) or spot 1m tape via ``signals.drift_scale``.

    **Time-scale floor (2026-08-07):** ``σ_rem`` never uses effective remaining
    time below ``DRIFT_TIME_SCALE_MIN_SEC`` (default 60s), so last-minute TWAP
    noise cannot explode z into fake "strong" drift.
    """
    if not strike_price or strike_price <= 0 or not btc_now or btc_now <= 0:
        return 0.0
    drift_pct = (btc_now - strike_price) / strike_price
    if drift_pct == 0.0:
        return 0.0
    window = float(getattr(config, "MARKET_WINDOW_SEC", 300) or 300)
    tr_raw = float(time_remaining if time_remaining is not None else window)
    tr_raw = max(tr_raw, 10.0)
    # Floor the *time factor* so late-window noise cannot dominate.
    t_floor = float(getattr(config, "DRIFT_TIME_SCALE_MIN_SEC", 60.0) or 60.0)
    t_floor = max(10.0, min(window, t_floor))
    tr_eff = max(tr_raw, t_floor)
    try:
        from signals.drift_scale import resolve_vol_scale
        scale = resolve_vol_scale(vol_scale)
    except Exception:
        scale = float(vol_scale if vol_scale is not None
                      else getattr(config, "DRIFT_VOL_SCALE", 0.0022) or 0.0022)
    if scale <= 0:
        scale = float(getattr(config, "DRIFT_VOL_SCALE", 0.0022) or 0.0022)
    sigma_remaining = scale * math.sqrt(min(tr_eff, window) / window)
    if sigma_remaining <= 0:
        return 0.0
    z = drift_pct / sigma_remaining
    return math.tanh(z)


def drift_pct(strike_price: Optional[float], btc_now: float) -> float:
    """Raw moneyness (fraction): (btc − strike) / strike. 0 if unknown."""
    if not strike_price or strike_price <= 0 or not btc_now or btc_now <= 0:
        return 0.0
    return (float(btc_now) - float(strike_price)) / float(strike_price)


_registry: Optional[StrikeRegistry] = None


def get_strike_registry() -> StrikeRegistry:
    global _registry
    if _registry is None:
        _registry = StrikeRegistry()
    return _registry

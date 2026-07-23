"""BTC 'price to beat' (strike) registry + drift signal.

Polymarket BTC 5-min Up/Down markets resolve UP iff BTC's price at window CLOSE
exceeds its price at window OPEN — the "price to beat" (strike), taken from
Chainlink BTC/USD at the market's ``eventStartTime``. The distance of the
current BTC price from that strike is the market's true moneyness and the
strongest fundamental for these markets.

**Accuracy matters (BUG #23).** The strike is fetched as the Binance BTCUSDT 1m
OPEN at ``eventStartTime`` — the exact window open — NOT a mid-window "first
sighting" snapshot. The first-sighting bug anchored the strike at a local
mid-window price, which inverted the drift sign and blew up the account. Offline
validation on 300 resolved markets (``tools/validate_signals.py``) shows
drift-from-accurate-strike is ~76% predictive (86% near expiry), symmetric and
regime-agnostic. Binance vs Chainlink basis is ~0.005% — immaterial for direction.

The signal is regime-agnostic (favors YES above the strike, NO below;
self-correcting) and time-scaled (more decisive as expiry nears).
"""

import math
import threading
import time
from datetime import datetime
from typing import Callable, Optional

import http_client

import config

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


def _fetch_open_at(event_start_iso: str) -> Optional[float]:
    """Binance BTCUSDT 1m OPEN at the window open time = the strike (None on error)."""
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


class StrikeRegistry:
    """Per-market strike, fetched once (accurately) then cached.

    ``fetcher`` is injectable for tests; production uses the Binance open at the
    window's ``eventStartTime``.
    """

    def __init__(self, fetcher: Callable[[str], Optional[float]] = _fetch_open_at) -> None:
        self._lock = threading.Lock()
        self._strikes: dict[str, dict] = {}    # market_id -> {strike, ts}
        self._fetch = fetcher

    def get_strike(self, market_id: Optional[str],
                   event_start_time: Optional[str]) -> Optional[float]:
        """Accurate strike for a market (cached). Fetches on first request."""
        if not market_id or not event_start_time:
            return None
        with self._lock:
            rec = self._strikes.get(market_id)
            if rec is not None:
                return rec["strike"]
        strike = self._fetch(event_start_time)
        if not strike or strike <= 0:
            return None                        # unavailable -> drift stays 0 (no guess)
        now = time.time()
        with self._lock:
            self._strikes[market_id] = {"strike": float(strike), "ts": now}
            if len(self._strikes) > 64:        # keep the map tiny (windows roll fast)
                cutoff = now - 3600
                self._strikes = {
                    k: v for k, v in self._strikes.items() if v["ts"] >= cutoff
                }
        return float(strike)


def drift_signal(strike_price: Optional[float], btc_now: float,
                 time_remaining: Optional[float]) -> float:
    """Bounded, time-scaled BTC drift-from-strike signal in ``[-1, 1]``.

    ``>0`` = BTC above the strike (YES/Up favored), ``<0`` = below (NO/Down).
    Magnitude scales with the drift as a fraction of typical remaining-window
    volatility, so the same drift reads stronger as expiry nears.
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

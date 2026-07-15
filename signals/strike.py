"""BTC 'price to beat' (strike) registry + drift signal.

Polymarket BTC 5-min Up/Down markets resolve UP iff BTC's price at window CLOSE
exceeds its price at window OPEN — the "price to beat" (strike). The distance of
the current BTC price from that strike is the market's true moneyness and the
single strongest fundamental for these markets; the decision logic was flying
blind on it (it only saw the Polymarket price and BTC *momentum*, never BTC vs
the strike).

This module:
  * snapshots the strike the first time a window is observed LIVE (time_remaining
    <= the window length — a pre-window sighting must not set it early), and
  * turns ``(btc_now - strike)`` into a bounded, TIME-SCALED directional signal.

The signal is **regime-agnostic**: it favors YES when BTC is above the strike and
NO when below, self-correcting with the market rather than baking in a bias. It
grows more decisive as expiry nears (less time for BTC to revert), via a
z-score-style scale by the square root of the fraction of the window remaining.
"""

import math
import threading
import time
from typing import Optional

import config


class StrikeRegistry:
    """Per-market strike (BTC price at window open), captured at first live sight."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._strikes: dict[str, dict] = {}   # market_id -> {strike, ts, captured_tr}

    def observe(self, market_id: Optional[str], btc_price: float,
                time_remaining: Optional[float]) -> None:
        """Record the strike the first time a market is seen LIVE.

        Ignores pre-window sightings (``time_remaining`` greater than the window
        length) so the strike is anchored at/near the real window open, and never
        overwrites an existing strike.
        """
        if not market_id or not btc_price or btc_price <= 0:
            return
        window = config.MARKET_WINDOW_SEC
        # Only capture once the window is actually live (tr <= window + small
        # buffer). A market seen 20 min early (tr >> window) must not set it.
        if time_remaining is not None and time_remaining > window + 5:
            return
        now = time.time()
        with self._lock:
            if market_id not in self._strikes:
                self._strikes[market_id] = {
                    "strike": float(btc_price),
                    "ts": now,
                    "captured_tr": time_remaining,
                }
            # Opportunistic prune so the map stays tiny (windows roll every 5 min).
            if len(self._strikes) > 64:
                cutoff = now - 3600
                self._strikes = {
                    k: v for k, v in self._strikes.items() if v["ts"] >= cutoff
                }

    def strike(self, market_id: Optional[str]) -> Optional[float]:
        if not market_id:
            return None
        with self._lock:
            rec = self._strikes.get(market_id)
            return rec["strike"] if rec else None


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
    # Typical BTC stdev over the REMAINING window ~ full-window sigma * sqrt(frac).
    sigma_remaining = config.DRIFT_VOL_SCALE * math.sqrt(min(tr, window) / window)
    if sigma_remaining <= 0:
        return 0.0
    z = drift_pct / sigma_remaining          # how decisive the drift is right now
    return math.tanh(z)


_registry: Optional[StrikeRegistry] = None


def get_strike_registry() -> StrikeRegistry:
    global _registry
    if _registry is None:
        _registry = StrikeRegistry()
    return _registry

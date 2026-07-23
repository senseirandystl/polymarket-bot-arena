"""BTC derivatives context from Binance USD-M futures (public, no auth).

Three reads, refreshed by one background thread (never on the trading hot
path — ``get_signals()`` only returns the latest cached snapshot):

- **funding**: last funding rate from ``/fapi/v1/premiumIndex``. Persistent
  positive funding = crowded longs paying shorts (froth); negative = crowded
  shorts. Normalized so a +/-0.05%/8h rate reads near saturation.
- **oi_delta**: percent change in open interest across the refresh window
  (``/fapi/v1/openInterest``). Rising OI with rising price = trend fuel.
- **taker_delta**: taker buy/sell volume ratio from
  ``/futures/data/takerlongshortRatio`` (5m period) — executed aggression on
  the perp, the derivatives cousin of the spot CVD lane.

All outputs are smooth scores in (-1, 1) (signals/curves.py). LANE IS
KILL-SWITCHED (config.SIGNAL_WEIGHT_FUT = 0) until the offline harness
(tools/validate_signals.py) measures positive NET edge — house rule:
validate-before-weighting. The feed runs now so live readings accumulate in
trade reasoning logs for that validation.
"""

import logging
import threading
import time

from signals.curves import soft_saturate

logger = logging.getLogger(__name__)

FAPI_BASE = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
REFRESH_SEC = 60.0          # funding changes hourly; OI/taker move minute-scale
FUNDING_SCALE = 0.0005      # 0.05%/8h funding reads ~tanh(1) = 0.76
OI_DELTA_SCALE = 0.005      # 0.5% OI change per refresh window saturates
TAKER_RATIO_SCALE = 0.15    # |ratio-1| of 0.15 (60/40 split) reads ~0.76
STALE_SEC = 300.0           # snapshot older than this reports zeros


class FuturesMetaFeed:
    """Background-refreshing derivatives-context feed (singleton via get_feed)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot: dict = {}
        self._snapshot_ts = 0.0
        self._prev_oi: float | None = None
        self._running = False
        self._thread = None
        self._error_streak = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="futures-meta")
        self._thread.start()
        logger.info("Futures meta feed started")

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        while self._running:
            try:
                self._refresh()
                self._error_streak = 0
            except Exception as e:
                self._error_streak += 1
                # First failures at info, persistent outage at warning — a
                # dead lane must be visible without spamming transient blips.
                log = logger.warning if self._error_streak >= 5 else logger.info
                log(f"futures meta refresh failed (streak "
                    f"{self._error_streak}): {e}")
            # Back off up to 4x on persistent errors (rate-limit friendly).
            backoff = min(4.0, 1.0 + self._error_streak * 0.5)
            time.sleep(REFRESH_SEC * backoff)

    def _refresh(self) -> None:
        import requests

        funding = 0.0
        resp = requests.get(f"{FAPI_BASE}/fapi/v1/premiumIndex",
                            params={"symbol": SYMBOL}, timeout=10)
        resp.raise_for_status()
        funding = float(resp.json().get("lastFundingRate", 0.0) or 0.0)

        oi_delta_pct = 0.0
        resp = requests.get(f"{FAPI_BASE}/fapi/v1/openInterest",
                            params={"symbol": SYMBOL}, timeout=10)
        resp.raise_for_status()
        oi = float(resp.json().get("openInterest", 0.0) or 0.0)
        if self._prev_oi and self._prev_oi > 0 and oi > 0:
            oi_delta_pct = (oi - self._prev_oi) / self._prev_oi
        if oi > 0:
            self._prev_oi = oi

        taker_ratio = 1.0
        taker_params: dict[str, str | int] = {"symbol": SYMBOL, "period": "5m",
                                              "limit": 1}
        resp = requests.get(f"{FAPI_BASE}/futures/data/takerlongshortRatio",
                            params=taker_params, timeout=10)
        resp.raise_for_status()
        rows = resp.json() or []
        if rows:
            taker_ratio = float(rows[-1].get("buySellRatio", 1.0) or 1.0)

        snap = {
            # Positive funding = crowded longs. As a *contrarian froth* read
            # its sign is unresolved until validated — publish the raw lean
            # (long-crowding positive) and let the harness decide the sign.
            "funding": soft_saturate(funding, FUNDING_SCALE),
            "oi_delta": soft_saturate(oi_delta_pct, OI_DELTA_SCALE),
            "taker_delta": soft_saturate(taker_ratio - 1.0, TAKER_RATIO_SCALE),
            "funding_raw": funding,
            "oi_raw": oi,
            "taker_ratio_raw": taker_ratio,
        }
        with self._lock:
            self._snapshot = snap
            self._snapshot_ts = time.time()

    def get_signals(self) -> dict:
        """Latest cached snapshot; zeros when never fetched or stale."""
        with self._lock:
            snap = dict(self._snapshot)
            ts = self._snapshot_ts
        if not snap or (time.time() - ts) > STALE_SEC:
            return {"funding": 0.0, "oi_delta": 0.0, "taker_delta": 0.0,
                    "stale": True}
        snap["stale"] = False
        return snap


_feed: FuturesMetaFeed | None = None


def get_feed() -> FuturesMetaFeed:
    global _feed
    if _feed is None:
        _feed = FuturesMetaFeed()
    return _feed

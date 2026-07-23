"""Real-time BTC/ETH/SOL price data from Binance WebSocket.

One socket, three 1m-kline streams. ``get_signals`` keeps its original keys
(``prices``/``volumes``/``latest``/``stale``) for backward compatibility and
adds smooth derived metrics:

- ``momentum``: last-candle return, tanh-scaled to (-1, 1),
- ``acceleration``: change in candle-over-candle return (is the move
  speeding up or fading), tanh-scaled,
- ``mtf``: dict of raw 1m/3m/5m returns for multi-timeframe consumers.

ETH exists here for the cross-asset lane (signals/cross_asset.py) — it rides
the same socket, so the extra symbol costs no additional connection.
"""

import json
import time
import threading
import logging
from collections import deque

from signals.curves import soft_saturate

logger = logging.getLogger(__name__)

BINANCE_WS = "wss://stream.binance.com:9443/ws"
SYMBOLS = {"btc": "btcusdt", "eth": "ethusdt", "sol": "solusdt"}
MOMENTUM_SCALE = 0.002   # 0.2% one-candle move reads ~0.76 (~p97, see BUG #25)
ACCEL_SCALE = 0.001


class PriceFeed:
    def __init__(self, max_candles=100):
        self.prices = {sym: deque(maxlen=max_candles) for sym in SYMBOLS}
        self.volumes = {sym: deque(maxlen=max_candles) for sym in SYMBOLS}
        self.latest = {sym: 0.0 for sym in SYMBOLS}
        self._last_update = {sym: 0.0 for sym in SYMBOLS}
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Price feed started")

    def stop(self):
        self._running = False

    def _run(self):
        import websocket

        streams = "/".join(f"{s}@kline_1m" for s in SYMBOLS.values())
        url = f"{BINANCE_WS}/{streams}"
        backoff = 2.0

        while self._running:
            try:
                ws = websocket.WebSocket()
                ws.settimeout(10)
                ws.connect(url)
                logger.info(f"Connected to Binance WS: {url}")
                backoff = 2.0  # healthy connection resets the backoff

                while self._running:
                    try:
                        raw = ws.recv()
                    except Exception:
                        break

                    try:
                        msg = json.loads(raw)
                        kline = msg.get("k", {})
                        symbol = kline.get("s", "").lower()
                        close = float(kline.get("c", 0))
                        volume = float(kline.get("v", 0))
                        is_closed = kline.get("x", False)

                        # Map back to our symbol names
                        for name, binance_sym in SYMBOLS.items():
                            if symbol == binance_sym:
                                self.latest[name] = close
                                self._last_update[name] = time.time()
                                if is_closed:
                                    self.prices[name].append(close)
                                    self.volumes[name].append(volume)
                                break
                    except (KeyError, ValueError):
                        continue

                ws.close()
            except Exception as e:
                logger.error(f"Price feed error: {e} (retry in {backoff:.0f}s)")
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)  # exponential, capped

    def get_signals(self, symbol: str) -> dict:
        """Current price signals for a symbol (back-compat keys + derived)."""
        sym = symbol.lower()
        if sym not in self.prices:
            return {"prices": [], "volumes": [], "latest": 0}

        prices = list(self.prices[sym])
        stale = (time.time() - self._last_update.get(sym, 0)) > 60

        momentum = 0.0
        acceleration = 0.0
        if len(prices) >= 2 and prices[-2] > 0:
            r1 = (prices[-1] - prices[-2]) / prices[-2]
            momentum = soft_saturate(r1, MOMENTUM_SCALE)
            if len(prices) >= 3 and prices[-3] > 0:
                r0 = (prices[-2] - prices[-3]) / prices[-3]
                acceleration = soft_saturate(r1 - r0, ACCEL_SCALE)

        mtf = {}
        for horizon in (1, 3, 5):
            if len(prices) > horizon and prices[-1 - horizon] > 0:
                mtf[f"{horizon}m"] = (
                    (prices[-1] - prices[-1 - horizon]) / prices[-1 - horizon])

        return {
            "prices": prices,
            "volumes": list(self.volumes[sym]),
            "latest": self.latest.get(sym, 0),
            "stale": stale,
            "momentum": momentum,
            "acceleration": acceleration,
            "mtf": mtf,
        }


# Singleton
_feed = None


def get_feed() -> PriceFeed:
    global _feed
    if _feed is None:
        _feed = PriceFeed()
    return _feed

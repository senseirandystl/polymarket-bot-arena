"""Real-time crypto price data — Chainlink for BTC price, Binance for volume + xasset.

**BTC price** resolves on Polymarket against **Chainlink BTC/USD**, so the live
BTC level (latest, 1m candles for momentum/acceleration/mtf, drift ``btc_now``)
comes from Polymarket RTDS ``crypto_prices_chainlink`` / symbol ``btc/usd``.
That is the same oracle family as Price to Beat (``signals/strike.py``).

**BTC volume** is not on Chainlink. We subscribe to Binance ``btcusdt@kline_1m``
**volume-only** (never overwrite Chainlink price) for regime activity /
relative-volume context. Basis risk vs Chainlink is fine for volume.

**ETH / SOL** stay on Binance 1m klines for the cross-asset lane. One process
runs both sockets.

``get_signals`` keys are unchanged: ``prices``, ``volumes``, ``latest``,
``stale``, ``momentum``, ``acceleration``, ``mtf``. For BTC, ``prices`` are
Chainlink and ``volumes`` are Binance.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Deque, Optional

from signals.curves import soft_saturate

logger = logging.getLogger(__name__)

BINANCE_WS = "wss://stream.binance.com:9443/ws"
RTDS_WS = "wss://ws-live-data.polymarket.com"

# Cross-asset (non-resolution) price+volume on Binance.
BINANCE_SYMBOLS = {"eth": "ethusdt", "sol": "solusdt"}
# BTC volume-only on Binance (price stays Chainlink).
BINANCE_BTC_SYMBOL = "btcusdt"

MOMENTUM_SCALE = 0.002   # 0.2% one-candle move reads ~0.76 (~p97, see BUG #25)
ACCEL_SCALE = 0.001
STALE_SEC = 60.0
# Keep enough Chainlink ticks to rebuild ~2h of 1m candles + strike latch.
TICK_BUFFER_SEC = 7200
CANDLE_MAX = 100


class PriceFeed:
    def __init__(self, max_candles: int = CANDLE_MAX):
        self._max_candles = max_candles
        # Unified latest / candle streams for btc (Chainlink) + eth/sol (Binance)
        self.prices: dict[str, Deque[float]] = {
            "btc": deque(maxlen=max_candles),
            "eth": deque(maxlen=max_candles),
            "sol": deque(maxlen=max_candles),
        }
        self.volumes: dict[str, Deque[float]] = {
            "btc": deque(maxlen=max_candles),
            "eth": deque(maxlen=max_candles),
            "sol": deque(maxlen=max_candles),
        }
        self.latest: dict[str, float] = {"btc": 0.0, "eth": 0.0, "sol": 0.0}
        self._last_update: dict[str, float] = {"btc": 0.0, "eth": 0.0, "sol": 0.0}
        self._source: dict[str, str] = {
            "btc": "chainlink", "eth": "binance", "sol": "binance",
        }
        # Chainlink tick ring: (epoch_sec, price) for strike latch + diagnostics
        self._btc_ticks: Deque[tuple[float, float]] = deque(maxlen=TICK_BUFFER_SEC + 60)
        self._btc_candle_open_min: Optional[int] = None  # floor(epoch/60)
        self._btc_candle_last: Optional[float] = None

        self._running = False
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        t_cl = threading.Thread(
            target=self._run_chainlink_btc, name="price-feed-chainlink-btc",
            daemon=True,
        )
        t_bn = threading.Thread(
            target=self._run_binance_xasset, name="price-feed-binance-xasset",
            daemon=True,
        )
        self._threads = [t_cl, t_bn]
        for t in self._threads:
            t.start()
        logger.info(
            "Price feed started (BTC price=Chainlink RTDS, "
            "BTC volume+ETH/SOL=Binance klines)"
        )

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ BTC
    def _run_chainlink_btc(self) -> None:
        """RTDS Chainlink BTC/USD.

        Polymarket's RTDS often delivers a ~60s 1Hz snapshot on subscribe and
        few (or no) follow-up update frames. We therefore:
          * apply the snapshot (last point = live price)
          * if quiet for ``refresh_sec``, reconnect for a fresh snapshot
        so the live level stays within a few seconds of the oracle.

        Socket timeout is short (1s) so the quiet-refresh check is not delayed
        by a long blocking ``recv`` (a 5s timeout + 8s quiet ≈ 13s stale).
        """
        import websocket

        backoff = 2.0
        refresh_sec = 5.0
        # Quiet-refresh reconnects are normal (RTDS snapshot cadence) — log
        # INFO only on first connect and after a real error, DEBUG otherwise.
        # Otherwise arena.log fills with ~12 "Connected" lines/minute.
        first_connect = True
        after_error = False
        while self._running:
            ws = None
            try:
                ws = websocket.WebSocket()
                ws.settimeout(1.0)
                ws.connect(RTDS_WS)
                sub = {
                    "action": "subscribe",
                    "subscriptions": [{
                        "topic": "crypto_prices_chainlink",
                        "type": "*",
                        "filters": json.dumps({"symbol": "btc/usd"}),
                    }],
                }
                ws.send(json.dumps(sub))
                if first_connect or after_error:
                    logger.info(
                        "Connected to Polymarket RTDS Chainlink btc/usd%s",
                        " (recovered)" if after_error and not first_connect else "",
                    )
                    first_connect = False
                    after_error = False
                else:
                    logger.debug(
                        "RTDS Chainlink snapshot refresh reconnect btc/usd"
                    )
                backoff = 2.0
                last_ping = time.time()
                last_msg = time.time()

                while self._running:
                    now = time.time()
                    # RTDS application heartbeat
                    if now - last_ping >= 5.0:
                        try:
                            ws.send("PING")
                        except Exception:
                            break
                        last_ping = now
                    # Force reconnect for a fresh snapshot if the stream is quiet
                    if now - last_msg >= refresh_sec:
                        break
                    try:
                        raw = ws.recv()
                    except Exception:
                        # timeout → loop and check refresh/ping
                        continue
                    if not raw or raw in ("PONG", "pong"):
                        continue
                    last_msg = time.time()
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    self._ingest_chainlink_message(msg)

                try:
                    ws.close()
                except Exception:
                    pass
                # Brief pause before snapshot refresh reconnect (not error backoff)
                if self._running:
                    time.sleep(0.2)
                    backoff = 2.0
            except Exception as e:
                after_error = True
                logger.error(
                    "Chainlink BTC feed error: %s (retry in %.0fs)", e, backoff,
                )
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    def _ingest_chainlink_message(self, msg: dict) -> None:
        """Handle RTDS snapshot (type=subscribe, data[]) and live updates."""
        if not isinstance(msg, dict):
            return
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            return
        sym = str(payload.get("symbol") or "").lower().replace("-", "/")
        if sym and sym not in ("btc/usd", "btc"):
            return

        # Snapshot: historical 1Hz points
        data = payload.get("data")
        if isinstance(data, list) and data:
            for pt in data:
                self._on_btc_tick(pt)
            return

        # Live update: single point fields on payload
        if "value" in payload and "timestamp" in payload:
            self._on_btc_tick(payload)
            return
        # Some envelopes nest under payload.price / payload.update
        inner = payload.get("update") or payload.get("price")
        if isinstance(inner, dict):
            self._on_btc_tick(inner)

    def _on_btc_tick(self, pt: dict) -> None:
        try:
            ts_ms = pt.get("timestamp")
            val = pt.get("value")
            if val is None:
                return
            price = float(val)
            if price <= 0:
                return
            if ts_ms is None:
                ts = time.time()
            else:
                ts = float(ts_ms) / 1000.0 if float(ts_ms) > 1e12 else float(ts_ms)
        except (TypeError, ValueError):
            return

        with self._lock:
            self.latest["btc"] = price
            self._last_update["btc"] = time.time()
            self._btc_ticks.append((ts, price))
            # Drop ticks older than buffer
            cutoff = ts - TICK_BUFFER_SEC
            while self._btc_ticks and self._btc_ticks[0][0] < cutoff:
                self._btc_ticks.popleft()

            # Build 1m candles: on minute roll, append previous minute's last
            minute = int(ts // 60)
            if self._btc_candle_open_min is None:
                self._btc_candle_open_min = minute
                self._btc_candle_last = price
            elif minute > self._btc_candle_open_min:
                # Close prior minute with last seen Chainlink price.
                # Volume is filled separately from Binance BTC klines — do not
                # append 0.0 here (that polluted the activity series).
                if self._btc_candle_last and self._btc_candle_last > 0:
                    self.prices["btc"].append(self._btc_candle_last)
                self._btc_candle_open_min = minute
                self._btc_candle_last = price
            else:
                self._btc_candle_last = price

    def price_at(self, epoch_sec: float, *, tol_sec: float = 2.0) -> Optional[float]:
        """Nearest Chainlink BTC tick at/after ``epoch_sec`` (within ``tol_sec``).

        Used as a strike latch when the Polymarket openPrice REST call is down.
        Prefer the first tick with ``ts >= epoch_sec``; if none, nearest within
        tolerance.
        """
        with self._lock:
            ticks = list(self._btc_ticks)
        if not ticks:
            return None
        # Prefer first tick at or after window open (Chainlink sample after open).
        after = [t for t in ticks if t[0] >= epoch_sec - 0.5]
        if after:
            return float(after[0][1])
        # Nearest overall within tol
        best = min(ticks, key=lambda t: abs(t[0] - epoch_sec))
        if abs(best[0] - epoch_sec) <= tol_sec:
            return float(best[1])
        return None

    # -------------------------------------------------------------- Binance
    def _run_binance_xasset(self) -> None:
        """Binance 1m klines: ETH/SOL price+volume, BTC **volume only**.

        BTC price/resolution stays on Chainlink. We only harvest
        ``btcusdt`` base-asset volume for regime activity scoring.
        """
        import websocket

        streams = "/".join(
            f"{s}@kline_1m"
            for s in (*BINANCE_SYMBOLS.values(), BINANCE_BTC_SYMBOL)
        )
        # Combined stream endpoint (multiple streams on one connection)
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        backoff = 2.0

        while self._running:
            try:
                ws = websocket.WebSocket()
                ws.settimeout(10)
                ws.connect(url)
                logger.info(
                    "Connected to Binance WS (BTC vol + ETH/SOL): %s", url
                )
                backoff = 2.0

                while self._running:
                    try:
                        raw = ws.recv()
                    except Exception:
                        break
                    try:
                        msg = json.loads(raw)
                        # Combined stream wraps payload: {"stream":..., "data":{...}}
                        data = msg.get("data") if isinstance(msg, dict) else None
                        if isinstance(data, dict) and "k" in data:
                            msg = data
                        kline = msg.get("k", {})
                        symbol = (kline.get("s") or "").lower()
                        close = float(kline.get("c", 0) or 0)
                        volume = float(kline.get("v", 0) or 0)
                        is_closed = bool(kline.get("x", False))

                        # BTC: volume-only — never touch Chainlink price series
                        if symbol == BINANCE_BTC_SYMBOL:
                            if is_closed and volume >= 0:
                                with self._lock:
                                    self.volumes["btc"].append(volume)
                            continue

                        for name, binance_sym in BINANCE_SYMBOLS.items():
                            if symbol != binance_sym:
                                continue
                            with self._lock:
                                self.latest[name] = close
                                self._last_update[name] = time.time()
                                if is_closed and close > 0:
                                    self.prices[name].append(close)
                                    self.volumes[name].append(volume)
                            break
                    except (KeyError, ValueError, TypeError):
                        continue
                ws.close()
            except Exception as e:
                logger.error(
                    "Binance xasset feed error: %s (retry in %.0fs)", e, backoff,
                )
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    # --------------------------------------------------------------- API
    def get_signals(self, symbol: str) -> dict:
        """Current price signals for a symbol (back-compat keys + derived)."""
        sym = symbol.lower()
        if sym not in self.prices:
            return {"prices": [], "volumes": [], "latest": 0}

        with self._lock:
            prices = list(self.prices[sym])
            volumes = list(self.volumes[sym])
            latest = self.latest.get(sym, 0)
            last_up = self._last_update.get(sym, 0)
            source = self._source.get(sym, "unknown")
            # In-progress BTC minute: expose last tick as the working close so
            # consumers see a live level even before the minute rolls.
            if sym == "btc" and self._btc_candle_last and self._btc_candle_last > 0:
                latest = self._btc_candle_last

        stale = (time.time() - last_up) > STALE_SEC if last_up else True

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
                    (prices[-1] - prices[-1 - horizon]) / prices[-1 - horizon]
                )

        return {
            "prices": prices,
            "volumes": volumes,
            "latest": latest,
            "stale": stale,
            "momentum": momentum,
            "acceleration": acceleration,
            "mtf": mtf,
            "source": source,
        }


# Singleton
_feed: Optional[PriceFeed] = None


def get_feed() -> PriceFeed:
    global _feed
    if _feed is None:
        _feed = PriceFeed()
    return _feed

"""Real-time crypto price data — Chainlink spot + TWAP for BTC, Binance xasset.

**BTC resolution (2026-08-07+)** uses Chainlink **TWAP** (``TWAP_WINDOW_SEC``,
60s for 5-min markets). Both Price to Beat (open) and settlement (close)
come from the TWAP feed. Live path:

  * ``crypto_prices_twap_sixty`` / ``btc/usd`` → resolution ``btc_now`` /
    strike latch (``signals/twap.py``, ``signals/strike.py``)
  * ``crypto_prices_chainlink`` / ``btc/usd`` → 1m candles, momentum,
    regime tape, settlement-nowcast tick buffer (spot path still useful)

**BTC volume** is not on Chainlink. We subscribe to Binance ``btcusdt@kline_1m``
**volume-only** (never overwrite Chainlink price) for regime activity /
relative-volume context.

**ETH / SOL** stay on Binance 1m klines for the cross-asset lane.

``get_signals`` keys stay stable; BTC adds ``twap`` / ``twap_stale`` /
``resolution_source`` without breaking consumers.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Deque, Optional

import config
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
        # Chainlink spot tick ring: (epoch_sec, price) for candles + nowcast
        self._btc_ticks: Deque[tuple[float, float]] = deque(maxlen=TICK_BUFFER_SEC + 60)
        self._btc_candle_open_min: Optional[int] = None  # floor(epoch/60)
        self._btc_candle_last: Optional[float] = None

        # Official Chainlink TWAP (60s for 5m markets) — resolution path
        self._btc_twap: float = 0.0
        self._btc_twap_ts: float = 0.0          # observation epoch (sec)
        self._btc_twap_wall: float = 0.0        # local receive time
        self._btc_twap_window_s: int = int(
            getattr(config, "TWAP_WINDOW_SEC", 60) or 60
        )
        # TWAP observation ring for open latch (epoch_sec, twap_value)
        self._btc_twap_ticks: Deque[tuple[float, float]] = deque(
            maxlen=TICK_BUFFER_SEC + 60
        )

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
        t_twap = threading.Thread(
            target=self._run_chainlink_twap, name="price-feed-chainlink-twap",
            daemon=True,
        )
        t_bn = threading.Thread(
            target=self._run_binance_xasset, name="price-feed-binance-xasset",
            daemon=True,
        )
        self._threads = [t_cl, t_twap, t_bn]
        for t in self._threads:
            t.start()
        logger.info(
            "Price feed started (BTC spot+TWAP=Chainlink RTDS, "
            "BTC volume+ETH/SOL=Binance klines; TWAP window=%ss)",
            self._btc_twap_window_s,
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
        """Nearest Chainlink BTC **spot** tick at/after ``epoch_sec``.

        Prefer TWAP latch via :meth:`twap_at` for resolution strike under
        2026-08-07+ rules. Spot latch remains for diagnostics / fallback.
        """
        with self._lock:
            ticks = list(self._btc_ticks)
        return _nearest_tick(ticks, epoch_sec, tol_sec=tol_sec)

    def twap_at(self, epoch_sec: float, *, tol_sec: float = 2.0) -> Optional[float]:
        """Nearest official Chainlink **TWAP** observation at/after ``epoch_sec``.

        Preferred strike latch under TWAP resolution (open PTB is a TWAP print).
        """
        with self._lock:
            ticks = list(self._btc_twap_ticks)
        return _nearest_tick(ticks, epoch_sec, tol_sec=tol_sec)

    def latest_twap(self) -> tuple[float, float, int]:
        """Return ``(twap_value, observation_epoch, window_seconds)`` or zeros."""
        with self._lock:
            return (
                float(self._btc_twap or 0.0),
                float(self._btc_twap_ts or 0.0),
                int(self._btc_twap_window_s or 60),
            )

    def btc_spot_ticks(self) -> list[tuple[float, float]]:
        """Copy of the Chainlink spot tick ring (for settlement nowcast)."""
        with self._lock:
            return list(self._btc_ticks)

    def btc_twap_ticks(self) -> list[tuple[float, float]]:
        """Copy of the official TWAP observation ring."""
        with self._lock:
            return list(self._btc_twap_ticks)

    # -------------------------------------------------------------- TWAP RTDS
    def _run_chainlink_twap(self) -> None:
        """RTDS Chainlink BTC/USD TWAP (resolution path for 5m markets).

        Topic follows ``TWAP_WINDOW_SEC`` (60 → ``crypto_prices_twap_sixty``).
        See Polymarket chainlink-twap docs. No snapshot/history/replay —
        reconnect + resubscribe on quiet/error.
        """
        if not bool(getattr(config, "TWAP_RESOLUTION_ENABLED", True)):
            logger.info("TWAP RTDS feed disabled (TWAP_RESOLUTION_ENABLED=False)")
            return

        import websocket

        try:
            from signals.twap import rtds_topic as _rtds_topic
            topic = _rtds_topic(self._btc_twap_window_s)
        except Exception:
            topic = str(
                getattr(config, "TWAP_RTDS_TOPIC_60", "crypto_prices_twap_sixty")
                or "crypto_prices_twap_sixty"
            )
        symbol = str(getattr(config, "TWAP_SYMBOL", "btc/usd") or "btc/usd")
        backoff = 2.0
        refresh_sec = 8.0
        first_connect = True
        after_error = False

        while self._running:
            ws = None
            try:
                ws = websocket.WebSocket()
                ws.settimeout(1.0)
                ws.connect(RTDS_WS)
                # filters must be compact JSON, no spaces (Polymarket RTDS API)
                sub = {
                    "action": "subscribe",
                    "subscriptions": [{
                        "topic": topic,
                        "type": "update",
                        "filters": json.dumps(
                            {"symbol": symbol}, separators=(",", ":")
                        ),
                    }],
                }
                ws.send(json.dumps(sub))
                if first_connect or after_error:
                    logger.info(
                        "Connected to Polymarket RTDS TWAP %s %s%s",
                        topic, symbol,
                        " (recovered)" if after_error and not first_connect else "",
                    )
                    first_connect = False
                    after_error = False
                else:
                    logger.debug("RTDS TWAP refresh reconnect %s", topic)
                backoff = 2.0
                last_ping = time.time()
                last_msg = time.time()

                while self._running:
                    now = time.time()
                    if now - last_ping >= 5.0:
                        try:
                            ws.send("PING")
                        except Exception:
                            break
                        last_ping = now
                    if now - last_msg >= refresh_sec:
                        break
                    try:
                        raw = ws.recv()
                    except Exception:
                        continue
                    if not raw or raw in ("PONG", "pong"):
                        continue
                    last_msg = time.time()
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    # Topic-not-found / error frames before feed is live
                    if isinstance(msg, dict) and msg.get("error"):
                        err = str(msg.get("error") or msg)
                        if "not found" in err.lower() or "topic" in err.lower():
                            logger.warning(
                                "RTDS TWAP topic not ready (%s) — retry in %.0fs",
                                err[:160], backoff,
                            )
                            break
                        logger.debug("RTDS TWAP message error: %s", err[:160])
                        continue
                    self._ingest_twap_message(msg)

                try:
                    ws.close()
                except Exception:
                    pass
                if self._running:
                    time.sleep(0.3)
                    backoff = 2.0
            except Exception as e:
                after_error = True
                logger.error(
                    "Chainlink TWAP feed error: %s (retry in %.0fs)", e, backoff,
                )
                time.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    def _ingest_twap_message(self, msg: dict) -> None:
        """Handle RTDS TWAP update frames (and any snapshot-shaped payloads)."""
        if not isinstance(msg, dict):
            return
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            # Some envelopes put fields at top level
            if "value" in msg and ("timestamp" in msg or "window_s" in msg):
                payload = msg
            else:
                return

        sym = str(
            payload.get("symbol") or ""
        ).lower().replace("-", "/")
        if sym and sym not in ("btc/usd", "btc"):
            return

        # Snapshot-style list (if RTDS ever sends one)
        data = payload.get("data")
        if isinstance(data, list) and data:
            for pt in data:
                if isinstance(pt, dict):
                    self._on_btc_twap_tick(pt)
            return

        if "value" in payload:
            self._on_btc_twap_tick(payload)
            return

        inner = payload.get("update") or payload.get("price")
        if isinstance(inner, dict):
            self._on_btc_twap_tick(inner)

    def _on_btc_twap_tick(self, pt: dict) -> None:
        try:
            # Prefer full_accuracy_value (E18 string) when present
            raw_full = pt.get("full_accuracy_value")
            val = pt.get("value")
            if raw_full is not None and str(raw_full).strip() != "":
                try:
                    # E18 fixed-point integer string → float dollars
                    as_int = int(str(raw_full).strip())
                    price = as_int / 1e18
                except (TypeError, ValueError):
                    price = float(val) if val is not None else 0.0
            elif val is not None:
                price = float(val)
            else:
                return
            if price <= 0:
                return

            ts_ms = pt.get("timestamp")
            if ts_ms is None:
                ts = time.time()
            else:
                ts = float(ts_ms) / 1000.0 if float(ts_ms) > 1e12 else float(ts_ms)

            win = pt.get("window_s") or pt.get("windowSeconds") or pt.get(
                "window_seconds"
            )
            window_s = int(win) if win else int(
                getattr(config, "TWAP_WINDOW_SEC", 60) or 60
            )
        except (TypeError, ValueError):
            return

        with self._lock:
            self._btc_twap = price
            self._btc_twap_ts = ts
            self._btc_twap_wall = time.time()
            self._btc_twap_window_s = window_s
            self._btc_twap_ticks.append((ts, price))
            cutoff = ts - TICK_BUFFER_SEC
            while self._btc_twap_ticks and self._btc_twap_ticks[0][0] < cutoff:
                self._btc_twap_ticks.popleft()

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
            twap = float(self._btc_twap or 0.0) if sym == "btc" else 0.0
            twap_ts = float(self._btc_twap_ts or 0.0) if sym == "btc" else 0.0
            twap_wall = float(self._btc_twap_wall or 0.0) if sym == "btc" else 0.0
            twap_window = int(self._btc_twap_window_s or 60) if sym == "btc" else 0

        stale = (time.time() - last_up) > STALE_SEC if last_up else True
        twap_stale_sec = float(getattr(config, "TWAP_STALE_SEC", 15.0) or 15.0)
        twap_stale = True
        if sym == "btc" and twap > 0 and twap_wall > 0:
            twap_stale = (time.time() - twap_wall) > twap_stale_sec

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

        out = {
            "prices": prices,
            "volumes": volumes,
            "latest": latest,
            "stale": stale,
            "momentum": momentum,
            "acceleration": acceleration,
            "mtf": mtf,
            "source": source,
        }
        if sym == "btc":
            out["twap"] = twap
            out["twap_ts"] = twap_ts
            out["twap_stale"] = twap_stale
            out["twap_window_sec"] = twap_window
            # Prefer TWAP for resolution-aware consumers; spot remains in latest.
            if twap > 0 and not twap_stale:
                out["resolution_price"] = twap
                out["resolution_source"] = "rtds_twap"
            elif latest > 0:
                out["resolution_price"] = latest
                out["resolution_source"] = "spot"
            else:
                out["resolution_price"] = 0.0
                out["resolution_source"] = "none"
        return out


def _nearest_tick(
    ticks: list[tuple[float, float]],
    epoch_sec: float,
    *,
    tol_sec: float = 2.0,
) -> Optional[float]:
    """Tick nearest to ``epoch_sec``, only if within ``tol_sec``.

    **Critical (BUG TWAP-PTB):** never return a tick just because it is the
    first sample *after* ``epoch_sec``. After a mid-window restart the buffer
    only has current ticks; treating those as the open strike invents a
    false Price to Beat (dashboard ≠ Polymarket, drift inverted).
    """
    if not ticks:
        return None
    # Prefer the first tick at/after open, but only if it is still near open.
    after = [t for t in ticks if t[0] >= epoch_sec - 0.5]
    if after and (after[0][0] - epoch_sec) <= tol_sec:
        return float(after[0][1])
    # Otherwise nearest overall, still within tolerance.
    best = min(ticks, key=lambda t: abs(t[0] - epoch_sec))
    if abs(best[0] - epoch_sec) <= tol_sec:
        return float(best[1])
    return None


# Singleton
_feed: Optional[PriceFeed] = None


def get_feed() -> PriceFeed:
    global _feed
    if _feed is None:
        _feed = PriceFeed()
    return _feed

"""CF Benchmarks BRTI ticks + Kalshi last-60s settlement nowcast.

Kalshi KXBTC15M resolves on the simple average of ~60 one-second BRTI
prints in the final minute — not a rolling TWAP of the whole window, and
not Chainlink. Strike is the market's Price to Beat (floor), latched at
open from metadata, never a mid-window first sighting (BUG #23 analog).

Ticks live in the arena process. The dashboard is a separate process, so
the feed publishes a snapshot to arena_state (``brti_feed_status``) the
same way Chainlink TWAP uses ``price_feed_status``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, Sequence


SETTLEMENT_AVG_SEC = 60.0
STATE_KEY = "brti_feed_status"
CFB_BRTI_PAGE = "https://www.cfbenchmarks.com/data/indices/BRTI"

logger = logging.getLogger("signals.brti")

# In-process BRTI 1s prints: (unix_ts, price). Filled by WS/poll when Kalshi on.
_TICKS: list[tuple[float, float]] = []
_TICKS_MAX = 400
_TICKS_LOCK = None
_LAST_SOURCE = "none"
_LAST_AVG60: Optional[float] = None
_LAST_SETTLE60: Optional[float] = None


def _lock():
    global _TICKS_LOCK
    if _TICKS_LOCK is None:
        import threading
        _TICKS_LOCK = threading.Lock()
    return _TICKS_LOCK


def record_tick(ts: float, price: float, *, source: str | None = None,
                avg60: Optional[float] = None,
                settle60: Optional[float] = None) -> None:
    global _LAST_SOURCE, _LAST_AVG60, _LAST_SETTLE60
    if not price or float(price) <= 0:
        return
    with _lock():
        _TICKS.append((float(ts), float(price)))
        if len(_TICKS) > _TICKS_MAX:
            del _TICKS[: len(_TICKS) - _TICKS_MAX]
        if source:
            _LAST_SOURCE = str(source)
        if avg60 is not None and float(avg60) > 0:
            _LAST_AVG60 = float(avg60)
        if settle60 is not None and float(settle60) > 0:
            _LAST_SETTLE60 = float(settle60)


def stored_ticks() -> list[tuple[float, float]]:
    with _lock():
        return list(_TICKS)


def last_price() -> Optional[float]:
    with _lock():
        if not _TICKS:
            return None
        return _TICKS[-1][1]


def local_avg60(now: float | None = None) -> Optional[float]:
    """Mean of in-process BRTI prints in the last 60s. None if no ticks."""
    import time as _t
    tnow = float(now if now is not None else _t.time())
    with _lock():
        vals = [px for ts, px in _TICKS
                if tnow - SETTLEMENT_AVG_SEC <= float(ts) <= tnow
                and float(px) > 0]
    if not vals:
        return None
    return sum(vals) / len(vals)


def snapshot() -> dict:
    """In-process feed snapshot (arena). Dashboard must use ``load_published``."""
    import time as _t
    with _lock():
        last = _TICKS[-1][1] if _TICKS else None
        last_ts = _TICKS[-1][0] if _TICKS else None
        n = len(_TICKS)
        src = _LAST_SOURCE
        avg60 = _LAST_AVG60
        settle60 = _LAST_SETTLE60
    now = _t.time()
    if avg60 is None:
        avg60 = local_avg60(now)
        if avg60 is not None and src and "local60" not in str(src):
            src = f"{src}+local60" if src not in ("none", "") else "local60"
    age = (now - last_ts) if last_ts else None
    return {
        "ts": now,
        "last": last,
        "last_ts": last_ts,
        "age_sec": round(age, 1) if age is not None else None,
        "n_ticks": n,
        "source": src,
        "avg60": avg60,
        "settle60": settle60,
        "stale": bool(age is None or age > 30.0 or not last),
        "btc_now": settle60 or avg60 or last,
    }


def publish_status() -> dict:
    snap = snapshot()
    try:
        import db
        db.set_arena_state(STATE_KEY, json.dumps(snap))
    except Exception:
        pass
    return snap


def load_published() -> dict:
    """Dashboard-safe read of the arena-published BRTI snapshot."""
    try:
        import db
        raw = db.get_arena_state(STATE_KEY)
        if not raw:
            return {}
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_FEED_STARTED = False


def start_brti_feed() -> None:
    """Daemon poll/WS for CF BRTI when Kalshi is enabled. Idempotent."""
    global _FEED_STARTED
    if _FEED_STARTED:
        return
    _FEED_STARTED = True
    import threading
    threading.Thread(target=_brti_loop, name="brti-feed", daemon=True).start()


def _brti_loop() -> None:
    import time
    while True:
        try:
            from exchanges import KALSHI, exchange_enabled
            if not exchange_enabled(KALSHI):
                publish_status()
                time.sleep(5.0)
                continue
            if _try_ws_once():
                publish_status()
                continue
            if not _poll_cfb_once():
                _poll_once()
            publish_status()
            time.sleep(1.0)
            continue
        except Exception as e:
            logger.warning("BRTI feed loop: %s", e)
            try:
                publish_status()
            except Exception:
                pass
        time.sleep(1.0)


_BRTI_PX_KEYS = (
    "last_price", "index_value", "underlying_value", "underlying_price",
    "spot_price", "index_price", "value", "close",
)
# Strike / PTB must never be recorded as "now" (would pin drift at 0).
_BRTI_PX_SKIP = frozenset({
    "floor_strike", "price_to_beat", "strike", "yes_bid_dollars",
    "yes_ask_dollars", "no_bid_dollars", "no_ask_dollars",
})


def _btc_scale_px(obj) -> Optional[float]:
    if not isinstance(obj, dict):
        return None
    for key in _BRTI_PX_KEYS:
        if key in _BRTI_PX_SKIP:
            continue
        try:
            v = float(obj.get(key) or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v > 1000:  # BTC-scale, not a contract probability
            return v
    return None


def _f_px(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x > 1000 else None


def ingest_kalshi_ws_msg(msg: dict) -> Optional[dict]:
    """Parse a Kalshi ``cfbenchmarks_value`` frame.

    Official payload nests the print in ``msg.data`` (JSON string) and the
    trailing 60s mean in ``msg.avg_60s_data.value``. Settlement's last-60s
    of the 15m window is ``last_60s_windowed_average_15min`` (final minute).
    """
    if not isinstance(msg, dict):
        return None
    body = msg.get("msg") if isinstance(msg.get("msg"), dict) else msg
    data = body.get("data")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = None
    last = _btc_scale_px(data if isinstance(data, dict) else {})
    if last is None:
        last = _f_px(body.get("value") or body.get("price"))
    avg60 = None
    avg_block = body.get("avg_60s_data")
    if isinstance(avg_block, dict):
        avg60 = _f_px(avg_block.get("value"))
    settle60 = None
    settle_block = body.get("last_60s_windowed_average_15min")
    if isinstance(settle_block, dict):
        settle60 = _f_px(settle_block.get("value"))
    ts = body.get("received_at") or (data.get("time") if isinstance(data, dict) else None)
    try:
        ts = float(ts)
        if ts > 1e12:
            ts /= 1000.0
    except (TypeError, ValueError):
        import time as _t
        ts = _t.time()
    if last is None and avg60 is None:
        return None
    px = last or avg60
    record_tick(ts, px, source="kalshi_ws", avg60=avg60, settle60=settle60)
    return {"last": px, "avg60": avg60, "settle60": settle60, "ts": ts}


def parse_cfb_index_summary(html: str) -> Optional[float]:
    """Extract BRTI last from CF Benchmarks' public Next.js page (same index)."""
    if not html:
        return None
    m = re.search(
        r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
        html, re.S | re.I,
    )
    blob = m.group(1) if m else html
    try:
        data = json.loads(blob)
        val = (
            ((data.get("props") or {}).get("pageProps") or {})
            .get("indexSummary") or {}
        ).get("value")
        px = _f_px(val)
        if px:
            return px
    except Exception:
        pass
    m2 = re.search(r'"indexSummary"\s*:\s*\{[^}]*"value"\s*:\s*"([0-9]+(?:\.[0-9]+)?)"', blob)
    if m2:
        return _f_px(m2.group(1))
    return None


def _poll_cfb_once() -> bool:
    """Unauthenticated BRTI print from the public CF Benchmarks page.

    Same BRTI object Kalshi settles on — not Chainlink. Used when Kalshi
    WS keys are absent so paper still has a nowcast.
    """
    try:
        import http_client
        resp = http_client.get(
            CFB_BRTI_PAGE,
            timeout=12,
            headers={"User-Agent": "polymarket-bot-arena/brti"},
            retries=0,
        )
        if resp is None or resp.status_code >= 400:
            return False
        px = parse_cfb_index_summary(resp.text or "")
        if not px:
            return False
        import time as _t
        record_tick(_t.time(), px, source="cfb_page")
        return True
    except Exception as e:
        logger.debug("CFB BRTI page poll failed: %s", e)
        return False


def _poll_once() -> None:
    try:
        from kalshi_client import get_json
        import kalshi_markets
        payload = get_json(
            "/markets",
            params={"series_ticker": kalshi_markets.SERIES, "status": "open",
                    "limit": 3},
            timeout=8,
        )
        rows = (payload or {}).get("markets") if isinstance(payload, dict) else []
        import time as _t
        now = _t.time()
        for raw in rows or []:
            ev = raw.get("event") if isinstance(raw.get("event"), dict) else {}
            px = _btc_scale_px(raw) or _btc_scale_px(ev)
            if px:
                record_tick(now, px, source="kalshi_markets")
                return
    except Exception:
        return


def _try_ws_once() -> bool:
    """Subscribe Kalshi cfbenchmarks_value if keys exist. Returns True if connected."""
    try:
        from kalshi_client import has_auth
        if not has_auth():
            return False
        import time as _t
        import config
        import websocket
        from kalshi_client import _sign_headers
        url = str(getattr(config, "KALSHI_WS_BASE", "")).rstrip("/")
        if not url:
            return False
        headers = _sign_headers("GET", "/trade-api/ws/v2")
        hdr_list = [f"{k}: {v}" for k, v in headers.items()]
        ws = websocket.create_connection(url, header=hdr_list, timeout=10)
        ws.send(json.dumps({
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["cfbenchmarks_value"],
                "index_ids": ["BRTI"],
            },
        }))
        ws.settimeout(30)
        n = 0
        while True:
            raw = ws.recv()
            if not raw:
                break
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            got = ingest_kalshi_ws_msg(msg)
            if got:
                n += 1
                if n % 5 == 0:
                    publish_status()
        ws.close()
        return n > 0
    except Exception:
        return False


def last60_average(
    ticks: Sequence[tuple[float, float]],
    *,
    now: float,
    expiry: float,
    window_sec: float = SETTLEMENT_AVG_SEC,
) -> dict:
    """Mean of (ts, price) prints in [expiry − window_sec, now].

    ``ticks`` must be BRTI (or a test fixture of the same object).
    Incomplete coverage fills remaining seconds with the last print for the
    nowcast, and reports ``coverage`` in [0, 1].
    """
    lo = float(expiry) - float(window_sec)
    hi = min(float(now), float(expiry))
    in_win = [(float(ts), float(px)) for ts, px in (ticks or [])
              if lo <= float(ts) <= hi and float(px) > 0]
    in_win.sort(key=lambda x: x[0])
    n = len(in_win)
    in_settlement = (float(expiry) - float(now)) <= window_sec + 1e-9
    if n == 0:
        return {
            "brti_now": None,
            "coverage": 0.0,
            "n": 0,
            "in_settlement": in_settlement,
        }
    last_px = in_win[-1][1]
    secs = {int(ts) for ts, _ in in_win}
    # Fill-forward remaining seconds in the settlement window with last print.
    filled_n = n
    if in_settlement and last_px > 0:
        want = max(1, int(window_sec))
        filled_n = max(n, min(want, n + max(0, want - len(secs))))
        mean = (sum(p for _, p in in_win) + last_px * max(0, filled_n - n)) / filled_n
    else:
        mean = sum(p for _, p in in_win) / n
    coverage = min(1.0, len(secs) / max(1.0, window_sec))
    return {
        "brti_now": mean,
        "coverage": coverage,
        "n": n,
        "in_settlement": in_settlement,
        "last": last_px,
        "span": max(1e-9, hi - lo),
    }


def brti_certainty(*, coverage: float, elapsed_frac: float,
                   abs_drift: float) -> float:
    """0–1 lock quality in the settlement minute (elapsed × coverage × |d|)."""
    c = max(0.0, min(1.0, float(coverage)))
    e = max(0.0, min(1.0, float(elapsed_frac)))
    d = max(0.0, min(1.0, abs(float(abs_drift))))
    return max(0.0, min(1.0, e * c * (0.5 + 0.5 * min(1.0, d / 0.20))))


def latch_strike(floor_strike: Optional[float],
                 brti_at_open: Optional[float] = None) -> tuple[Optional[float], str]:
    """Official floor/price-to-beat wins; BRTI-at-open is fallback only."""
    try:
        fs = float(floor_strike) if floor_strike is not None else None
    except (TypeError, ValueError):
        fs = None
    if fs is not None and fs > 0:
        return fs, "kalshi_floor"
    try:
        b = float(brti_at_open) if brti_at_open is not None else None
    except (TypeError, ValueError):
        b = None
    if b is not None and b > 0:
        return b, "brti_open"
    return None, "none"

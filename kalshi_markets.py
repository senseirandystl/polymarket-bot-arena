"""Kalshi market discovery + book normalize for BTC 15m Up/Down.

Public REST (orderbook may require auth on some hosts). Market dicts are
stamped to the arena shape so ``make_decision`` can consume them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import config
from exchanges import KALSHI, namespace_market_id, stamp_exchange

logger = logging.getLogger("kalshi.markets")

SERIES = str(getattr(config, "KALSHI_SERIES_TICKER", "KXBTC15M"))
WINDOW_SEC = int(getattr(config, "KALSHI_WINDOW_SEC", 900) or 900)
SETTLEMENT = "brti_last60"


def _parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _f(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _levels_from_dollar_bids(rows) -> list[tuple[float, float]]:
    """Kalshi ``yes_dollars`` / ``no_dollars`` are [price, size] **bids**."""
    out: list[tuple[float, float]] = []
    for r in rows or []:
        try:
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                px, sz = float(r[0]), float(r[1])
            elif isinstance(r, dict):
                px, sz = float(r.get("price") or r.get("px")), float(
                    r.get("size") or r.get("count") or r.get("quantity") or 0
                )
            else:
                continue
        except (TypeError, ValueError, IndexError):
            continue
        if px > 1.0 + 1e-9:
            px = px / 100.0  # cents ladder, not yes_dollars
        if px > 0 and sz > 0:
            out.append((px, sz))
    out.sort(key=lambda x: x[0], reverse=True)  # best bid first
    return out


def normalize_kalshi_book(orderbook_fp: dict | None) -> dict:
    """Return YES/NO books in the arena CLOB shape (asks asc, bids desc).

    Kalshi ladders are bids on each outcome. Taker YES ask = 1 − best NO bid.
    """
    ob = orderbook_fp or {}
    yes_bids = _levels_from_dollar_bids(
        ob.get("yes_dollars") or ob.get("yes") or []
    )
    no_bids = _levels_from_dollar_bids(
        ob.get("no_dollars") or ob.get("no") or []
    )
    if not yes_bids and not no_bids:
        return {"valid": False}

    def _asks_from_other_bids(other_bids: list[tuple[float, float]]):
        # Buying this side lifts the other side's bids, priced as 1−bid.
        asks = []
        for px, sz in other_bids:
            ask = round(1.0 - px, 4)
            if ask > 0:
                asks.append((ask, sz))
        asks.sort(key=lambda x: x[0])  # lowest ask first
        return asks

    yes_asks = _asks_from_other_bids(no_bids)
    no_asks = _asks_from_other_bids(yes_bids)
    yes_best_bid = yes_bids[0][0] if yes_bids else None
    no_best_bid = no_bids[0][0] if no_bids else None
    yes_best_ask = yes_asks[0][0] if yes_asks else None
    no_best_ask = no_asks[0][0] if no_asks else None
    return {
        "valid": True,
        "yes": {
            "valid": True,
            "bids": yes_bids,
            "asks": yes_asks,
            "best_bid": yes_best_bid,
            "best_ask": yes_best_ask,
            "min_order_size": 1.0,
            "exchange": KALSHI,
        },
        "no": {
            "valid": True,
            "bids": no_bids,
            "asks": no_asks,
            "best_bid": no_best_bid,
            "best_ask": no_best_ask,
            "min_order_size": 1.0,
            "exchange": KALSHI,
        },
        "yes_bid": yes_best_bid,
        "yes_ask": yes_best_ask,
        "no_bid": no_best_bid,
        "no_ask": no_best_ask,
    }


def kalshi_taker_fee(contracts: float, price: float,
                     rate: float | None = None) -> float:
    """Quadratic taker fee, ceiled to the next cent per order."""
    import math
    if rate is None:
        rate = float(getattr(config, "KALSHI_TAKER_FEE_RATE", 0.07))
    c = max(0.0, float(contracts))
    p = max(0.0, min(1.0, float(price)))
    if c <= 0 or p <= 0 or p >= 1:
        return 0.0
    raw = rate * c * p * (1.0 - p)
    return math.ceil(raw * 100.0 - 1e-12) / 100.0


def _floor_strike(raw: dict) -> Optional[float]:
    for key in ("floor_strike", "floor_strike_dollars", "strike",
                "yes_strike", "price_to_beat"):
        v = _f(raw.get(key))
        if v and v > 0:
            return v
    # Nested event / product metadata
    for nest in ("event", "product_metadata", "rules_primary"):
        block = raw.get(nest)
        if isinstance(block, dict):
            v = _floor_strike(block)
            if v:
                return v
    return None


def normalize_market(raw: dict) -> Optional[dict]:
    """Kalshi market JSON → arena market dict (unstamped prices)."""
    if not raw:
        return None
    ticker = raw.get("ticker") or raw.get("market_ticker")
    if not ticker:
        return None
    close = (raw.get("close_time") or raw.get("expiration_time")
             or raw.get("latest_expiration_time"))
    open_t = raw.get("open_time") or raw.get("expected_expiration_time")
    end = _parse_iso(close)
    now = datetime.now(timezone.utc)
    time_rem = (end - now).total_seconds() if end else None
    strike = _floor_strike(raw)
    yes_bid = _f(raw.get("yes_bid_dollars") or raw.get("yes_bid"))
    yes_ask = _f(raw.get("yes_ask_dollars") or raw.get("yes_ask"))
    no_bid = _f(raw.get("no_bid_dollars") or raw.get("no_bid"))
    no_ask = _f(raw.get("no_ask_dollars") or raw.get("no_ask"))
    if yes_ask is None and no_bid is not None:
        yes_ask = round(1.0 - no_bid, 4)
    if no_ask is None and yes_bid is not None:
        no_ask = round(1.0 - yes_bid, 4)
    yes_mid = None
    if yes_bid is not None and yes_ask is not None:
        yes_mid = round((yes_bid + yes_ask) / 2.0, 4)
    elif yes_ask is not None:
        yes_mid = yes_ask
    m = {
        "native_id": ticker,
        "ticker": ticker,
        "event_ticker": raw.get("event_ticker"),
        "series_ticker": raw.get("series_ticker") or SERIES,
        "question": raw.get("title") or raw.get("subtitle") or ticker,
        "resolves_at": close,
        "event_start_time": open_t,
        "time_remaining_seconds": time_rem,
        "floor_strike": strike,
        "btc_strike": strike,
        "status": raw.get("status"),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "current_price": yes_mid,
        "no_price": (round(1.0 - yes_mid, 4) if yes_mid is not None else None),
        "closed": str(raw.get("status") or "").lower() in ("closed", "settled"),
        "result": raw.get("result"),
    }
    return stamp_exchange(m, KALSHI, window_sec=WINDOW_SEC, settlement=SETTLEMENT)


def discover_live(*, client=None, limit: int = 12) -> list[dict]:
    """Fetch open KXBTC15M markets. ``client`` is a callable GET for tests."""
    from exchanges import exchange_enabled
    if not exchange_enabled(KALSHI):
        return []
    getter = client
    if getter is None:
        import http_client
        base = str(getattr(config, "KALSHI_API_BASE", "")).rstrip("/")

        def getter(url, params=None, timeout=20):
            resp = http_client.get(url, params=params or {}, timeout=timeout)
            resp.raise_for_status()
            return resp.json()

        url = f"{base}/markets"
        params = {
            "series_ticker": SERIES,
            "status": "open",
            "limit": int(limit),
        }
        try:
            payload = getter(url, params=params)
        except Exception as e:
            logger.error("Kalshi discovery failed: %s", e)
            return []
    else:
        try:
            payload = getter()
        except Exception as e:
            logger.error("Kalshi discovery failed: %s", e)
            return []
    rows = payload.get("markets") if isinstance(payload, dict) else payload
    out = []
    for raw in rows or []:
        nm = normalize_market(raw)
        if nm:
            out.append(nm)
    out.sort(key=lambda m: float(m.get("time_remaining_seconds") or 1e12))
    return out


def select_current(markets: list[dict]) -> Optional[dict]:
    """Live window: shortest positive time remaining (or just expired)."""
    live = []
    for m in markets or []:
        tr = m.get("time_remaining_seconds")
        try:
            t = float(tr) if tr is not None else None
        except (TypeError, ValueError):
            t = None
        if t is None:
            continue
        if -30.0 <= t <= WINDOW_SEC + 5:
            live.append(m)
    if not live:
        return None
    live.sort(key=lambda m: abs(float(m.get("time_remaining_seconds") or 0)))
    return live[0]


def select_next(markets: list[dict], current: Optional[dict] = None) -> Optional[dict]:
    """Queued window after ``current`` (soonest later close)."""
    cur_id = (current or {}).get("id")
    try:
        cur_tr = float((current or {}).get("time_remaining_seconds") or 0.0)
    except (TypeError, ValueError):
        cur_tr = 0.0
    later = []
    for m in markets or []:
        if not m or m.get("id") == cur_id:
            continue
        try:
            t = float(m.get("time_remaining_seconds"))
        except (TypeError, ValueError):
            continue
        if t > cur_tr + 5.0:
            later.append(m)
    if not later:
        return None
    later.sort(key=lambda m: float(m.get("time_remaining_seconds") or 1e12))
    return later[0]


def namespaced_id(ticker: str) -> str:
    return namespace_market_id(KALSHI, ticker)


def current_prices(ticker: str | None) -> Optional[dict]:
    """YES/NO mids from the Kalshi book, or None if the book is empty."""
    books = get_order_book(ticker)
    if not books or not books.get("valid"):
        return None
    yes = books.get("yes") or {}
    no = books.get("no") or {}
    yb, ya = yes.get("best_bid"), yes.get("best_ask")
    nb, na = no.get("best_bid"), no.get("best_ask")
    yes_mid = None
    if yb is not None and ya is not None:
        yes_mid = round((float(yb) + float(ya)) / 2.0, 4)
    elif ya is not None:
        yes_mid = float(ya)
    no_mid = None
    if nb is not None and na is not None:
        no_mid = round((float(nb) + float(na)) / 2.0, 4)
    elif na is not None:
        no_mid = float(na)
    if yes_mid is None and no_mid is not None:
        yes_mid = round(1.0 - no_mid, 4)
    if no_mid is None and yes_mid is not None:
        no_mid = round(1.0 - yes_mid, 4)
    if yes_mid is None and no_mid is None:
        return None
    return {"yes": yes_mid, "no": no_mid}


def current_up_price(ticker: str | None):
    """YES midpoint for a Kalshi ticker, or None."""
    prices = current_prices(ticker)
    if not prices:
        return None
    return prices.get("yes")


def get_order_book(ticker: str | None, timeout: float | None = None) -> dict:
    """Normalized YES/NO books for a Kalshi ticker (arena CLOB shape per side)."""
    if not ticker:
        return {"valid": False}
    from exchanges import native_market_id
    native = native_market_id(str(ticker))
    to = float(timeout if timeout is not None
               else getattr(config, "BOOK_FETCH_TIMEOUT_SEC", 2.0))
    try:
        from kalshi_client import get_json
        payload = get_json(f"/markets/{native}/orderbook", timeout=to)
    except Exception as e:
        logger.debug("Kalshi book failed for %s: %s", native, e)
        return {"valid": False}
    if not payload:
        return {"valid": False}
    ob = payload.get("orderbook_fp") or payload.get("orderbook") or payload
    books = normalize_kalshi_book(ob if isinstance(ob, dict) else {})
    if not books.get("valid"):
        return {"valid": False}
    yes = dict(books["yes"])
    yes["exchange"] = KALSHI
    no = dict(books["no"])
    no["exchange"] = KALSHI
    return {
        "valid": True,
        "exchange": KALSHI,
        "yes": yes,
        "no": no,
        "yes_ask": books.get("yes_ask"),
        "no_ask": books.get("no_ask"),
        "yes_bid": books.get("yes_bid"),
        "no_bid": books.get("no_bid"),
    }


def apply_book_to_market(market: dict, books: dict) -> dict:
    """Lay Kalshi YES/NO books onto the market dict (asks for fills)."""
    if not books or not books.get("valid"):
        return market
    yes = books.get("yes") or {}
    no = books.get("no") or {}
    market["yes_book"] = yes
    market["no_book"] = no
    if yes.get("best_ask") is not None:
        market["yes_ask"] = yes["best_ask"]
    if no.get("best_ask") is not None:
        market["no_ask"] = no["best_ask"]
    yb, ya = yes.get("best_bid"), yes.get("best_ask")
    if yb is not None and ya is not None:
        market["current_price"] = round((yb + ya) / 2.0, 4)
    elif ya is not None:
        market["current_price"] = ya
    nb, na = no.get("best_bid"), no.get("best_ask")
    if nb is not None and na is not None:
        market["no_price"] = round((nb + na) / 2.0, 4)
    elif na is not None:
        market["no_price"] = na
    return market


def recent_resolutions() -> dict:
    """Namespaced ticker → True if Up/Yes resolved."""
    from kalshi_client import get_json
    payload = get_json(
        "/markets",
        params={"series_ticker": SERIES, "status": "settled", "limit": 40},
        timeout=20,
    )
    if not payload:
        return {}
    rows = payload.get("markets") if isinstance(payload, dict) else payload
    out = {}
    for raw in rows or []:
        ticker = raw.get("ticker") or raw.get("market_ticker")
        if not ticker:
            continue
        result = str(raw.get("result") or raw.get("settlement_result") or "").lower()
        if result in ("yes", "up"):
            up = True
        elif result in ("no", "down"):
            up = False
        else:
            continue
        out[namespaced_id(ticker)] = up
        out[ticker] = up
    return out


def fetch_market_outcome(market_id: str):
    """True=Up, False=Down, None=unresolved."""
    from exchanges import native_market_id
    native = native_market_id(str(market_id))
    from kalshi_client import get_json
    payload = get_json(f"/markets/{native}", timeout=15)
    if not payload:
        return None
    raw = payload.get("market") if isinstance(payload, dict) else payload
    if not isinstance(raw, dict):
        return None
    result = str(raw.get("result") or "").lower()
    if result in ("yes", "up"):
        return True
    if result in ("no", "down"):
        return False
    return None

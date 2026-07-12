"""Polymarket-native market data — discovery, fresh prices, order-book depth,
and resolution for the BTC 5-min up/down arena.

Replaces Simmer entirely. All reads here are public (no auth):
  * Gamma (``config.POLYMARKET_GAMMA_URL``) — discovery + resolution.
  * CLOB  (``config.POLYMARKET_HOST``)      — live order-book depth.

Market dicts are normalized to the shape the rest of the arena already expects
(``id``, ``question``, ``current_price``, ``polymarket_token_id`` [Up/YES],
``polymarket_no_token_id`` [Down/NO], ``resolves_at``, ``time_remaining_seconds``),
so ``base_bot.make_decision`` and ``market_utils.select_current_market`` work
unchanged.
"""

import json
import logging
from datetime import datetime, timezone

import requests

import config

logger = logging.getLogger("polymarket.markets")

GAMMA = config.POLYMARKET_GAMMA_URL
CLOB = config.POLYMARKET_HOST
SERIES_ID = config.POLYMARKET_BTC_5M_SERIES_ID


def _parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _as_list(v):
    """Gamma returns some array fields as JSON strings — normalize to list."""
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_markets(limit: int = 25) -> list:
    """Return normalized BTC 5-min markets for the current + upcoming windows.

    Orders by ``endDate`` ascending and filters ``end_date_min=now`` so the
    first results are the windows ending soonest (the live one, then the next),
    and stale never-resolved past markets are excluded. Prices are NOT fetched
    here (that would mean one book call per market); call :func:`refresh_price`
    on the selected current/next market instead.
    """
    now = datetime.now(timezone.utc).isoformat()
    try:
        resp = requests.get(
            f"{GAMMA}/events",
            params={
                "series_id": SERIES_ID,
                "closed": "false",
                "limit": limit,
                "order": "endDate",
                "ascending": "true",
                "end_date_min": now,
            },
            timeout=20,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        logger.error(f"Gamma discovery failed: {e}")
        return []

    out = []
    for ev in events or []:
        for m in ev.get("markets", []) or []:
            nm = _normalize(m)
            if nm:
                out.append(nm)
    return out


def _normalize(m: dict):
    cond = m.get("conditionId")
    toks = _as_list(m.get("clobTokenIds"))
    if not cond or not toks or len(toks) < 2:
        return None
    end = _parse_iso(m.get("endDate"))
    now = datetime.now(timezone.utc)
    time_rem = (end - now).total_seconds() if end else None
    return {
        "id": cond,
        "condition_id": cond,
        "question": m.get("question"),
        "polymarket_token_id": toks[0],       # "Up" / YES
        "polymarket_no_token_id": toks[1],    # "Down" / NO
        "polymarket_neg_risk": bool(m.get("negRisk")),
        "resolves_at": m.get("endDate"),
        "time_remaining_seconds": time_rem,
        "current_price": None,                # set by refresh_price()
        "outcome_prices": _as_list(m.get("outcomePrices")),
        "closed": bool(m.get("closed")),
    }


# ---------------------------------------------------------------------------
# Order book + fresh prices
# ---------------------------------------------------------------------------
def get_order_book(token_id: str) -> dict:
    """Fetch and NORMALIZE a token's CLOB order book.

    Polymarket returns bids ascending and asks descending (both worst→best), so
    ``bids[0]``/``asks[0]`` are the WORST prices — a trap the old client fell
    into. Here we sort explicitly: ``asks`` ascending (best/lowest first),
    ``bids`` descending (best/highest first), and expose ``best_bid``/``best_ask``.
    """
    try:
        resp = requests.get(f"{CLOB}/book", params={"token_id": token_id}, timeout=15)
        if resp.status_code != 200:
            return {"valid": False}
        b = resp.json()
    except Exception as e:
        logger.debug(f"order book fetch failed for {str(token_id)[:12]}…: {e}")
        return {"valid": False}

    def _levels(rows, reverse):
        out = []
        for r in rows or []:
            try:
                out.append((float(r["price"]), float(r["size"])))
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(out, key=lambda x: x[0], reverse=reverse)

    asks = _levels(b.get("asks"), reverse=False)   # lowest ask first
    bids = _levels(b.get("bids"), reverse=True)    # highest bid first
    if not asks and not bids:
        return {"valid": False}
    return {
        "valid": True,
        "asks": asks,
        "bids": bids,
        "best_ask": asks[0][0] if asks else None,
        "best_bid": bids[0][0] if bids else None,
        "tick_size": float(b.get("tick_size", 0.01) or 0.01),
        "min_order_size": float(b.get("min_order_size", 0) or 0),
        "neg_risk": bool(b.get("neg_risk", False)),
    }


def midpoint(book: dict):
    """Mid of a normalized book, or the one available side, or None."""
    bid, ask = book.get("best_bid"), book.get("best_ask")
    if bid is not None and ask is not None:
        return round((bid + ask) / 2, 4)
    return ask if ask is not None else bid


def current_up_price(condition_id: str):
    """Current Up-token (YES) mid price for a market, or ``None``.

    Reads the CLOB ``/markets/{condition_id}`` token list (``tokens[].price``
    is the live midpoint). Cheap single call keyed by condition id — used by the
    position monitor to price open positions for stop-loss / take-profit exits.
    """
    try:
        resp = requests.get(f"{CLOB}/markets/{condition_id}", timeout=10)
        if resp.status_code != 200:
            return None
        for t in resp.json().get("tokens", []) or []:
            if str(t.get("outcome", "")).lower() == "up":
                p = t.get("price")
                return float(p) if p is not None else None
    except Exception as e:
        logger.debug(f"current_up_price failed for {str(condition_id)[:12]}…: {e}")
    return None


def refresh_price(market: dict) -> dict:
    """Set ``current_price`` on ``market`` to the fresh Up-token mid. Returns it.

    ``current_price`` is the YES/Up probability the signal stack keys off, so a
    stale value means stale decisions — this is called right before each trade
    validation. Returns the market unchanged (current_price stays None) if the
    book is unavailable.
    """
    book = get_order_book(market.get("polymarket_token_id"))
    if book.get("valid"):
        mid = midpoint(book)
        if mid is not None:
            market["current_price"] = mid
    return market


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def recent_resolutions(limit: int = 100) -> dict:
    """Map ``condition_id -> True|False`` for recently resolved markets.

    ``True`` = Up won, ``False`` = Down won. Built from the series' closed
    events (Gamma ``outcomePrices`` — ``["1","0"]`` → Up, ``["0","1"]`` → Down),
    which is authoritative. Gamma's ``/markets`` endpoint cannot filter by
    condition id and the CLOB ``tokens[].winner`` flag is unreliable, so the
    resolver builds this map once per cycle and matches pending trades against
    it rather than doing per-market lookups.
    """
    try:
        resp = requests.get(
            f"{GAMMA}/events",
            params={
                "series_id": SERIES_ID,
                "closed": "true",
                "order": "endDate",
                "ascending": "false",
                "limit": limit,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            return {}
        events = resp.json()
    except Exception as e:
        logger.debug(f"recent_resolutions fetch failed: {e}")
        return {}

    out = {}
    for ev in events or []:
        for m in ev.get("markets", []) or []:
            cond = m.get("conditionId")
            prices = _as_list(m.get("outcomePrices"))
            if not cond or not prices or len(prices) < 2:
                continue
            try:
                up, down = float(prices[0]), float(prices[1])
            except (TypeError, ValueError):
                continue
            if up >= 0.99 and down <= 0.01:
                out[cond] = True
            elif down >= 0.99 and up <= 0.01:
                out[cond] = False
    return out

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
def discover_markets(limit: int = None) -> list:
    """Return normalized BTC 5-min markets for the current + upcoming windows.

    Orders by ``endDate`` ascending and filters ``end_date_min=now`` so the
    first results are the windows ending soonest (the live one, then the next),
    and stale never-resolved past markets are excluded. Prices are NOT fetched
    here (that would mean one book call per market); call :func:`refresh_price`
    on the selected current/next market instead.
    """
    if limit is None:
        limit = getattr(config, "POLYMARKET_DISCOVERY_LIMIT", 6)
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


# condition_id -> (up_token_id, down_token_id). Token ids are stable per market,
# so we resolve them once and cache to avoid a /markets call on every price tick.
_TOKEN_CACHE: dict = {}


def _token_ids(condition_id: str):
    """Return ``(up_token, down_token)`` for a market, cached. ``(None, None)``
    on failure."""
    if condition_id in _TOKEN_CACHE:
        return _TOKEN_CACHE[condition_id]
    up = down = None
    try:
        resp = requests.get(f"{CLOB}/markets/{condition_id}", timeout=10)
        if resp.status_code == 200:
            for t in resp.json().get("tokens", []) or []:
                oc = str(t.get("outcome", "")).lower()
                if oc == "up":
                    up = t.get("token_id")
                elif oc == "down":
                    down = t.get("token_id")
    except Exception as e:
        logger.debug(f"_token_ids failed for {str(condition_id)[:12]}…: {e}")
    if up:
        _TOKEN_CACHE[condition_id] = (up, down)
    return (up, down)


def midpoint_price(token_id: str):
    """Live CLOB midpoint for a token, or ``None``.

    IMPORTANT: uses the ``/midpoint`` endpoint, which tracks the live order book.
    The ``/markets/{cond}`` ``tokens[].price`` field is a STALE reference (it
    sticks near 0.50 and never updates) — do NOT use it for pricing.
    """
    if not token_id:
        return None
    try:
        resp = requests.get(f"{CLOB}/midpoint", params={"token_id": token_id}, timeout=10)
        if resp.status_code != 200:
            return None
        mid = resp.json().get("mid")
        return float(mid) if mid is not None else None
    except Exception as e:
        logger.debug(f"midpoint_price failed for {str(token_id)[:12]}…: {e}")
        return None


def current_prices(condition_id: str):
    """Fresh ``{"yes": up_mid, "no": down_mid}`` for a market, or ``None``.

    Uses the live ``/midpoint`` for each token; ``no`` falls back to ``1 - yes``
    if the Down book is momentarily empty.
    """
    up, down = _token_ids(condition_id)
    yes = midpoint_price(up)
    if yes is None:
        return None
    no = midpoint_price(down) if down else None
    if no is None:
        no = round(1.0 - yes, 4)
    return {"yes": yes, "no": no}


def current_up_price(condition_id: str):
    """Live Up-token (YES) midpoint for a market, or ``None``.

    Used by the position monitor to price open positions for SL/TP exits.
    """
    up, _ = _token_ids(condition_id)
    return midpoint_price(up)


def refresh_price(market: dict) -> dict:
    """Set ``current_price`` on ``market`` to the fresh YES/Up midpoint.

    Uses the live ``/midpoint`` endpoint (the ``tokens[].price`` field is stale);
    falls back to the order-book mid only if ``/midpoint`` is unavailable. Leaves
    ``current_price`` untouched if neither source responds.
    """
    tok = market.get("polymarket_token_id")
    yes = midpoint_price(tok)
    if yes is None:
        book = get_order_book(tok)
        yes = midpoint(book) if book.get("valid") else None
    if yes is not None:
        market["current_price"] = yes
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

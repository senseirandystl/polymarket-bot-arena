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
import http_client
from signals import clean_tick

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
def discover_markets(limit: int | None = None) -> list:
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
        resp = http_client.get(
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
        "event_start_time": m.get("eventStartTime"),  # window OPEN (ISO) — the strike anchor
        "time_remaining_seconds": time_rem,
        "current_price": None,                # set by refresh_price()
        "outcome_prices": _as_list(m.get("outcomePrices")),
        "closed": bool(m.get("closed")),
    }


# ---------------------------------------------------------------------------
# Order book + fresh prices
# ---------------------------------------------------------------------------
def get_order_book(token_id: str | None, timeout: float | None = None) -> dict:
    """Fetch and NORMALIZE a token's CLOB order book.

    Polymarket returns bids ascending and asks descending (both worst→best), so
    ``bids[0]``/``asks[0]`` are the WORST prices — a trap the old client fell
    into. Here we sort explicitly: ``asks`` ascending (best/lowest first),
    ``bids`` descending (best/highest first), and expose ``best_bid``/``best_ask``.

    ``timeout`` defaults to ``config.BOOK_FETCH_TIMEOUT_SEC`` (short) so the
    warmer cannot stall for 15s on a hung CLOB; cold paths may pass longer.
    """
    if not token_id:
        return {"valid": False}
    to = float(timeout if timeout is not None
               else getattr(config, "BOOK_FETCH_TIMEOUT_SEC", 2.0))
    try:
        resp = requests.get(f"{CLOB}/book", params={"token_id": token_id}, timeout=to)
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


def midpoint_price(token_id: str | None):
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


def midpoints_batch(token_ids: list) -> dict:
    """Live CLOB midpoints for many tokens in ONE call: ``{token_id: float}``.

    Uses the batch ``POST /midpoints`` endpoint so an entire discovery snapshot
    (current + upcoming markets, both tokens each) is priced with a single round
    trip instead of one ``/midpoint`` GET per token — far fewer failure points
    under the dashboard's 1-3s poll cadence, and an atomic snapshot so YES/NO
    for a market are consistent with each other. Tokens the book has no mid for
    are simply absent. Returns ``{}`` on failure so callers fall back cleanly.
    """
    ids = [str(t) for t in token_ids if t]
    if not ids:
        return {}
    try:
        resp = requests.post(
            f"{CLOB}/midpoints",
            json=[{"token_id": t} for t in ids],
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        raw = resp.json() or {}
    except Exception as e:
        logger.debug(f"midpoints_batch failed ({len(ids)} tokens): {e}")
        return {}
    out = {}
    for tok, mid in raw.items():
        try:
            out[tok] = float(mid)
        except (TypeError, ValueError):
            continue
    return out


def _yes_no_from_mids(up, down, mids: dict):
    """Resolve a ``{"yes","no"}`` dict from a batch-mids map, with fallbacks.

    Layered for reliability: batch mid → single ``/midpoint`` → order-book mid.
    ``no`` derives from ``1 - yes`` whenever the Down book has no independent mid.
    Returns ``None`` only when every source is exhausted.
    """
    yes = mids.get(str(up)) if up else None
    if yes is None:
        yes = midpoint_price(up)
    if yes is None:
        book = get_order_book(up)
        yes = midpoint(book) if book.get("valid") else None
    if yes is None:
        return None
    no = mids.get(str(down)) if down else None
    if no is None:
        no = round(1.0 - yes, 4)
    return {"yes": yes, "no": no}


def price_markets(markets: list) -> list:
    """Set ``current_price`` (YES/Up mid) and ``no_price`` (Down mid) on every
    market in ONE batch call. Markets whose tokens the batch didn't price keep
    their previous values. Returns the same list for chaining.

    This is the efficient path for the dashboard: instead of calling
    ``refresh_price`` (one /midpoint each) per visible market, price the whole
    snapshot at once.
    """
    real = [m for m in markets if m]
    tokens = []
    for m in real:
        tokens.append(m.get("polymarket_token_id"))
        tokens.append(m.get("polymarket_no_token_id"))
    mids = midpoints_batch(tokens)
    for m in real:
        up = m.get("polymarket_token_id")
        down = m.get("polymarket_no_token_id")
        yes = mids.get(str(up)) if up else None
        if yes is not None:
            m["current_price"] = yes
            no = mids.get(str(down)) if down else None
            m["no_price"] = no if no is not None else round(1.0 - yes, 4)
    return markets


def current_prices(condition_id: str):
    """Fresh ``{"yes": up_mid, "no": down_mid}`` for a market, or ``None``.

    Prices both tokens in a single batch ``/midpoints`` call, then falls back to
    per-token ``/midpoint`` and finally the order-book mid so a transient empty
    book on one side never blanks the card.
    """
    up, down = _token_ids(condition_id)
    if not up:
        return None
    mids = midpoints_batch([up, down] if down else [up])
    return _yes_no_from_mids(up, down, mids)


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
    # Clean-tick guard: reject an implausible single-tick jump (bad/stale data)
    # and drop the first tick from a freshly-seen token before it can move a bot.
    if tok:
        yes = clean_tick.clean_price(tok, yes)
    if yes is not None:
        market["current_price"] = yes
    return market


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
# Extreme outcomePrices mean the book has effectively decided. Gamma sometimes
# leaves ``closed=false`` for many minutes after endDate (CLOB tokens[].winner
# stays false too) — those markets never appear in ``closed=true`` series
# pages, which is how trades got stuck as "resolver stuck" for 15m+ while the
# prices sat at 0.9995 / 0.0005.
_RESOLVED_PRICE = 0.99
# Only accept de-facto (not-yet-closed) outcomes this many seconds after end.
_DEFACTO_GRACE_SEC = 120.0


def outcome_from_prices(prices) -> bool | None:
    """Map outcomePrices → True (Up) / False (Down) / None (not decided).

    Accepts list or JSON string. Requires an extreme book (one side ≥ 0.99,
    the other ≤ 0.01) so we never treat a live mid-window favorite as settled.
    """
    prices = _as_list(prices)
    if not prices or len(prices) < 2:
        return None
    try:
        up, down = float(prices[0]), float(prices[1])
    except (TypeError, ValueError):
        return None
    if up >= _RESOLVED_PRICE and down <= (1.0 - _RESOLVED_PRICE):
        return True
    if down >= _RESOLVED_PRICE and up <= (1.0 - _RESOLVED_PRICE):
        return False
    return None


def _end_is_past(end_iso, now: datetime | None = None,
                 grace_sec: float = _DEFACTO_GRACE_SEC) -> bool:
    end = _parse_iso(end_iso)
    if end is None:
        return False
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return (now - end).total_seconds() >= float(grace_sec)


def fetch_market_outcome(condition_id: str,
                         *, allow_defacto: bool = True) -> bool | None:
    """Direct Gamma lookup for one condition_id → Up/Down/None.

    Used as a fallback when a pending trade's market is missing from the
    closed-events bulk map (Gamma lag: extreme prices but still
    ``closed=false``). Safe for live markets: without extreme prices, or
    before endDate + grace, returns None.
    """
    if not condition_id:
        return None
    try:
        resp = http_client.get(
            f"{GAMMA}/markets",
            params={"condition_ids": condition_id},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        rows = resp.json()
    except Exception as e:
        logger.debug(f"fetch_market_outcome failed for {condition_id[:14]}…: {e}")
        return None
    if not isinstance(rows, list) or not rows:
        return None
    m = rows[0]
    outcome = outcome_from_prices(m.get("outcomePrices"))
    if outcome is None:
        return None
    # Officially closed → trust extreme prices immediately.
    if m.get("closed") is True:
        return outcome
    # De-facto: past end + extreme prices (the stuck-resolver class).
    if allow_defacto and _end_is_past(m.get("endDate") or m.get("end_date_iso")):
        return outcome
    return None


def recent_resolutions(limit: int = 100) -> dict:
    """Map ``condition_id -> True|False`` for recently resolved markets.

    ``True`` = Up won, ``False`` = Down won. Built from the series' closed
    events (Gamma ``outcomePrices`` — ``["1","0"]`` → Up, ``["0","1"]`` → Down),
    which is authoritative for markets that have flipped ``closed=true``.

    Gamma can leave a market at extreme prices for a long time without
    setting ``closed``; those never appear here. The resolver falls back to
    :func:`fetch_market_outcome` for any still-pending market_id.
    """
    try:
        resp = http_client.get(
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
            if not cond:
                continue
            outcome = outcome_from_prices(m.get("outcomePrices"))
            if outcome is not None:
                out[cond] = outcome
    return out

"""Order-book fill simulation + Polymarket taker/maker fees.

Used by BOTH venues so paper and live share identical pricing/fee math — the
only difference is that live actually submits the order. Paper "fills" are
computed by walking the real CLOB book so slippage and depth are respected.

Fee rules (Polymarket crypto tier):
  * makers (resting limit that does not cross) → fee 0
  * takers (marketable buy walking asks) → rate * shares * p * (1-p)
"""

from __future__ import annotations

import config


def taker_fee(shares: float, price: float,
              rate: float | None = None, *, exchange: str | None = None) -> float:
    """Taker fee in USDC (makers are never charged on PM; Kalshi ceils cents).

    Documented formula, symmetric around 50c so a 30c and 70c trade cost the
    same dollar fee::

        fee = rate * shares * price * (1 - price)

    ``rate`` defaults to ``config.POLYMARKET_TAKER_FEE_RATE`` (crypto tier).
    Isolated here so the exact coefficient can be tuned in one place.
    Kalshi uses the same quadratic then **ceils to the next cent per order**.
    """
    if (exchange or "").lower() == "kalshi":
        from kalshi_markets import kalshi_taker_fee
        return kalshi_taker_fee(shares, price, rate=rate)
    if rate is None:
        rate = config.POLYMARKET_TAKER_FEE_RATE
    price = max(0.0, min(1.0, price))
    return rate * shares * price * (1.0 - price)


def maker_fee(shares: float = 0.0, price: float = 0.0) -> float:
    """Polymarket maker fee — always 0 (documented; kept for call-site clarity)."""
    return 0.0


def trading_fee(shares: float, price: float, *, is_maker: bool) -> float:
    """Fee for a fill role — maker 0, taker the crypto-tier formula."""
    if is_maker:
        return maker_fee(shares, price)
    return taker_fee(shares, price)


def affordable_spend(available: float, price: float, *,
                     is_maker: bool) -> float:
    """Max USDC spend such that cost + fee ≤ ``available``."""
    if available <= 0 or price <= 0:
        return 0.0
    if is_maker:
        return float(available)
    rate = float(config.POLYMARKET_TAKER_FEE_RATE)
    # spend * (1 + rate*(1-p)) = available
    return float(available) / (1.0 + rate * (1.0 - float(price)))


def fee_per_share(price: float, *, is_maker: bool,
                  exchange: str | None = None) -> float:
    """Per-share fee used in edge math (1 share notional)."""
    if (exchange or "").lower() == "kalshi":
        return taker_fee(1.0, price, exchange="kalshi") if not is_maker else 0.0
    return trading_fee(1.0, price, is_maker=is_maker)


def _best_ask(book: dict) -> float | None:
    asks = book.get("asks") or []
    if not asks:
        return None
    return float(asks[0][0])


def _best_bid(book: dict) -> float | None:
    bids = book.get("bids") or []
    if not bids:
        return None
    return float(bids[0][0])


def limit_buy_price(book: dict, mid: float | None = None,
                    mode: str | None = None) -> float | None:
    """Choose a BUY limit price from the book + mode.

    Returns None when the book has no usable levels.
    """
    mode = mode or getattr(config, "LIMIT_PRICE_MODE", "passive_mid")
    tick = float(getattr(config, "LIMIT_TICK", 0.01) or 0.01)
    ask = _best_ask(book)
    bid = _best_bid(book)
    if ask is None and bid is None and mid is None:
        return None
    if mode == "join_bid":
        if bid is not None:
            return round(max(tick, bid), 4)
        if mid is not None:
            return round(max(tick, min(mid, (ask or mid) - tick)), 4)
        return round(max(tick, (ask or 0.5) - tick), 4)
    if mode in ("aggressive", "cap_ask"):
        # Immediate fill at the touch — limit-capped taker (no book walk
        # past the displayed ask). Honest fee = taker.
        if ask is not None:
            return round(ask, 4)
        if mid is not None:
            return round(mid, 4)
        return None
    # passive_mid (default): sit at mid, never above ask − tick
    m = mid if mid is not None else (
        (bid + ask) / 2.0 if (bid is not None and ask is not None)
        else (bid if bid is not None else ask)
    )
    if m is None:
        return None
    if ask is not None:
        m = min(m, ask - tick)
    if bid is not None:
        m = max(m, bid)  # at least join the bid queue
    return round(max(tick, min(1.0 - tick, m)), 4)


def simulate_fill(book: dict, amount_usdc: float) -> dict:
    """Walk a normalized order book's asks to fill ``amount_usdc`` of BUYs.

    Consumes ask levels cheapest-first, accumulating shares until the USDC
    budget is spent or the book is exhausted (realistic slippage + partial
    fills). Returns::

        {
          "filled":     bool,   # any shares at all filled
          "full":       bool,   # entire USDC budget was spendable on the book
          "shares":     float,  # shares acquired
          "cost":       float,  # USDC spent on shares (<= amount_usdc)
          "avg_price":  float,  # cost / shares
          "fee":        float,  # taker fee on the fill
          "is_maker":   bool,   # always False for marketable walk
        }

    A caller should skip the trade when ``filled`` is False (dead/empty book)
    or when ``shares`` is below the venue's ``min_order_size``.
    """
    empty = {"filled": False, "full": False, "shares": 0.0,
             "cost": 0.0, "avg_price": 0.0, "fee": 0.0, "is_maker": False}
    if not book.get("valid") or amount_usdc <= 0:
        return empty

    remaining = amount_usdc
    shares = 0.0
    cost = 0.0
    for price, size in book.get("asks", []):
        if remaining <= 1e-9 or price <= 0:
            break
        level_cost = price * size
        if level_cost <= remaining:
            # Take the whole level.
            shares += size
            cost += level_cost
            remaining -= level_cost
        else:
            # Partial take of this level with the leftover budget.
            take_shares = remaining / price
            shares += take_shares
            cost += remaining
            remaining = 0.0
            break

    if shares <= 0:
        return empty

    avg_price = cost / shares
    return {
        "filled": True,
        "full": remaining <= 1e-6,
        "shares": shares,
        "cost": cost,
        "avg_price": avg_price,
        "fee": taker_fee(shares, avg_price, exchange=book.get("exchange")),
        "is_maker": False,
    }


def simulate_fill_shares(book: dict, target_shares: float) -> dict:
    """Walk a normalized book's asks to buy exactly ``target_shares`` shares.

    The share-based sibling of :func:`simulate_fill`. Whereas ``simulate_fill``
    spends a fixed USDC budget (variable share count), this fills a fixed SHARE
    count (variable USDC cost) — essential for a market-neutral arbitrage where
    the two legs MUST end up share-matched to lock in ``$1`` per pair regardless
    of which leg walks deeper into its book. Returns the same shape as
    ``simulate_fill``; ``full`` means the whole share request was fillable::

        {"filled", "full", "shares", "cost", "avg_price", "fee", "is_maker"}

    If the book has less depth than ``target_shares`` the fill is partial
    (``full`` is False and ``shares`` < ``target_shares``); callers that need a
    matched pair should re-match on the smaller of the two legs' ``shares``.
    """
    empty = {"filled": False, "full": False, "shares": 0.0,
             "cost": 0.0, "avg_price": 0.0, "fee": 0.0, "is_maker": False}
    if not book.get("valid") or target_shares <= 0:
        return empty

    remaining = target_shares
    shares = 0.0
    cost = 0.0
    for price, size in book.get("asks", []):
        if remaining <= 1e-9 or price <= 0:
            break
        if size <= remaining:
            # Take the whole level.
            shares += size
            cost += price * size
            remaining -= size
        else:
            # Partial take of this level for the leftover share request.
            shares += remaining
            cost += price * remaining
            remaining = 0.0
            break

    if shares <= 0:
        return empty

    avg_price = cost / shares
    return {
        "filled": True,
        "full": remaining <= 1e-6,
        "shares": shares,
        "cost": cost,
        "avg_price": avg_price,
        "fee": taker_fee(shares, avg_price, exchange=book.get("exchange")),
        "is_maker": False,
    }


def simulate_limit_buy(book: dict, amount_usdc: float,
                       limit_price: float,
                       *, target_shares: float | None = None) -> dict:
    """Simulate a BUY limit at ``limit_price``.

    * If best ask ≤ limit → marketable: walk asks (only levels ≤ limit),
      **taker fee**.
    * Else resting: when ``LIMIT_PAPER_ASSUME_MAKER_FILL`` is True, fill at
      the limit as **maker (fee 0)** for the requested size/budget; otherwise
      return unfilled (live path posts a resting GTC instead).
    """
    empty = {"filled": False, "full": False, "shares": 0.0,
             "cost": 0.0, "avg_price": 0.0, "fee": 0.0, "is_maker": False}
    if not book.get("valid") or limit_price <= 0:
        return empty
    if amount_usdc <= 0 and (target_shares is None or target_shares <= 0):
        return empty

    ask = _best_ask(book)
    # Marketable limit: walk asks at or below the limit.
    if ask is not None and ask <= limit_price + 1e-9:
        if target_shares is not None:
            # Walk only levels ≤ limit_price
            remaining = target_shares
            shares = 0.0
            cost = 0.0
            for price, size in book.get("asks", []):
                if price > limit_price + 1e-9 or remaining <= 1e-9:
                    break
                take = min(size, remaining)
                shares += take
                cost += price * take
                remaining -= take
            if shares <= 0:
                return empty
            avg = cost / shares
            return {
                "filled": True,
                "full": remaining <= 1e-6,
                "shares": shares,
                "cost": cost,
                "avg_price": avg,
                "fee": taker_fee(shares, avg, exchange=book.get("exchange")),
                "is_maker": False,
            }
        # USD budget marketable walk, capped at limit
        remaining = amount_usdc
        shares = 0.0
        cost = 0.0
        for price, size in book.get("asks", []):
            if price > limit_price + 1e-9 or remaining <= 1e-9:
                break
            level_cost = price * size
            if level_cost <= remaining:
                shares += size
                cost += level_cost
                remaining -= level_cost
            else:
                take = remaining / price
                shares += take
                cost += remaining
                remaining = 0.0
                break
        if shares <= 0:
            return empty
        avg = cost / shares
        return {
            "filled": True,
            "full": remaining <= 1e-6,
            "shares": shares,
            "cost": cost,
            "avg_price": avg,
            "fee": taker_fee(shares, avg, exchange=book.get("exchange")),
            "is_maker": False,
        }

    # Resting maker path (paper assumption or explicit).
    if not getattr(config, "LIMIT_PAPER_ASSUME_MAKER_FILL", True):
        return empty

    if target_shares is not None and target_shares > 0:
        shares = float(target_shares)
        cost = shares * limit_price
    else:
        shares = amount_usdc / limit_price
        cost = amount_usdc
    if shares <= 0:
        return empty
    return {
        "filled": True,
        "full": True,
        "shares": shares,
        "cost": cost,
        "avg_price": float(limit_price),
        "fee": maker_fee(shares, limit_price),
        "is_maker": True,
    }

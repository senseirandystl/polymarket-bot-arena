"""Order-book fill simulation + Polymarket taker fees.

Used by BOTH venues so paper and live share identical pricing/fee math — the
only difference is that live actually submits the order. Paper "fills" are
computed by walking the real CLOB asks so slippage and depth are respected.
"""

import config


def taker_fee(shares: float, price: float,
              rate: float = None) -> float:
    """Polymarket taker fee in USDC (makers are never charged).

    Documented formula, symmetric around 50c so a 30c and 70c trade cost the
    same dollar fee::

        fee = rate * shares * price * (1 - price)

    ``rate`` defaults to ``config.POLYMARKET_TAKER_FEE_RATE`` (crypto tier).
    Isolated here so the exact coefficient can be tuned in one place.
    """
    if rate is None:
        rate = config.POLYMARKET_TAKER_FEE_RATE
    price = max(0.0, min(1.0, price))
    return rate * shares * price * (1.0 - price)


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
        }

    A caller should skip the trade when ``filled`` is False (dead/empty book)
    or when ``shares`` is below the venue's ``min_order_size``.
    """
    empty = {"filled": False, "full": False, "shares": 0.0,
             "cost": 0.0, "avg_price": 0.0, "fee": 0.0}
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
        "fee": taker_fee(shares, avg_price),
    }

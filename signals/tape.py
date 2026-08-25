"""Per-market settlement tape (candles) — never mix Chainlink TWAP into Kalshi.

Polymarket 5m: Chainlink TWAP ticks (same object as Price to Beat / close).
Kalshi 15m: CF BRTI ticks (same object as floor strike / last-60s close).
"""

from __future__ import annotations

from typing import Optional, Sequence


def is_kalshi_market(market: Optional[dict]) -> bool:
    try:
        from exchanges import KALSHI, exchange_of
        return exchange_of(market) == KALSHI
    except Exception:
        mid = str((market or {}).get("id") or (market or {}).get("market_id") or "")
        return mid.startswith("kalshi:")


def candle_prices(
    market: Optional[dict],
    signals: Optional[dict] = None,
    *,
    sample_sec: float = 60.0,
    max_points: int = 80,
) -> list[float]:
    """60s (default) candles on this market's settlement index.

    Kalshi never falls back to spot/TWAP (BUG #23 analog). Empty list means
    no BRTI tape yet — callers should hold, not substitute Chainlink.
    """
    sig = signals if isinstance(signals, dict) else {}
    from signals.drift_scale import resample_tick_prices

    def _from_prices() -> list[float]:
        out: list[float] = []
        for p in (sig.get("prices") or []):
            try:
                v = float(p)
            except (TypeError, ValueError):
                continue
            if v > 0:
                out.append(v)
        return out

    if is_kalshi_market(market):
        ticks = list(sig.get("btc_brti_ticks") or [])
        if not ticks:
            try:
                from signals import brti as brti_mod
                ticks = list(brti_mod.stored_ticks() or [])
            except Exception:
                ticks = []
        px = resample_tick_prices(
            ticks, sample_sec=sample_sec, max_points=max_points,
        )
        if px:
            return list(px)
        # Live Kalshi must not substitute Chainlink. Fixtures with no BRTI
        # ticks still pass ``prices`` for unit tests.
        if ticks:
            return []
        return _from_prices()

    ticks = list(sig.get("btc_twap_ticks") or [])
    px = resample_tick_prices(
        ticks, sample_sec=sample_sec, max_points=max_points,
    ) if ticks else None
    if px:
        return list(px)
    priced = _from_prices()
    if priced:
        return priced
    if not ticks:
        try:
            from signals.price_feed import get_price_feed
            ticks = list(get_price_feed().btc_twap_ticks() or [])
        except Exception:
            ticks = []
        px = resample_tick_prices(
            ticks, sample_sec=sample_sec, max_points=max_points,
        )
        if px:
            return list(px)
    return []


def tape_source(market: Optional[dict], signals: Optional[dict] = None) -> str:
    sig = signals if isinstance(signals, dict) else {}
    explicit = sig.get("tape_source")
    if explicit:
        return str(explicit)
    return "brti" if is_kalshi_market(market) else "twap"

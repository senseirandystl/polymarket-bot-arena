"""Exchange adapters: Polymarket vs Kalshi.

``exchange`` is where the contract lives (``polymarket`` | ``kalshi``).
``mode`` remains paper vs live fill. Market ids are namespaced so the two
books never collide in trades / exposure / MAX_BOTS.
"""

from __future__ import annotations

import json
import time
from typing import Iterable, Optional

import config

POLYMARKET = "polymarket"
KALSHI = "kalshi"
EXCHANGES = (POLYMARKET, KALSHI)

_TOGGLE_KEY = "exchange_toggles"
_TOGGLE_CACHE: tuple[float, dict] = (0.0, {})
_TOGGLE_TTL = 3.0


def namespace_market_id(exchange: str, native_id: str) -> str:
    native = str(native_id or "").strip()
    prefix = f"{exchange}:"
    if native.startswith(prefix):
        return native
    if ":" in native:
        return native
    return f"{prefix}{native}"


def native_market_id(market_id: str) -> str:
    s = str(market_id or "")
    if s.startswith("polymarket:") or s.startswith("kalshi:"):
        return s.split(":", 1)[1]
    return s


def exchange_of(market: dict | None, *, default: str = POLYMARKET) -> str:
    if not market:
        return default
    ex = (market.get("exchange") or market.get("venue") or "").strip().lower()
    if ex in EXCHANGES:
        return ex
    mid = str(market.get("id") or market.get("market_id") or "")
    if mid.startswith("kalshi:"):
        return KALSHI
    if mid.startswith("polymarket:"):
        return POLYMARKET
    return default


def stamp_exchange(market: dict, exchange: str, *, window_sec: int,
                   settlement: str) -> dict:
    """Return a new market dict with exchange identity fields set."""
    out = dict(market)
    native = out.get("native_id") or out.get("condition_id") or out.get("id")
    out["exchange"] = exchange
    out["venue"] = exchange
    out["native_id"] = str(native) if native else out.get("id")
    # Polymarket keeps raw condition_id as `id` (existing trades/DB).
    # Kalshi is always namespaced so the two never collide.
    if exchange == KALSHI:
        out["id"] = namespace_market_id(
            KALSHI, str(out["native_id"] or out.get("id") or ""),
        )
    else:
        out["id"] = out.get("id") or out["native_id"]
    out["window_sec"] = int(window_sec)
    out["settlement"] = settlement
    return out


def _defaults() -> dict:
    return {
        POLYMARKET: bool(getattr(config, "EXCHANGE_POLYMARKET_ENABLED", True)),
        KALSHI: bool(getattr(config, "EXCHANGE_KALSHI_ENABLED", True)),
    }


def load_toggles() -> dict:
    import time as _t
    global _TOGGLE_CACHE
    now = _t.time()
    ts, cached = _TOGGLE_CACHE
    if (now - ts) < _TOGGLE_TTL and cached:
        return dict(cached)
    data = dict(_defaults())
    try:
        import db
        raw = db.get_arena_state(_TOGGLE_KEY)
        if raw:
            parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if isinstance(parsed, dict):
                for k in EXCHANGES:
                    if k in parsed:
                        data[k] = bool(parsed[k])
    except Exception:
        pass
    _TOGGLE_CACHE = (now, data)
    return dict(data)


def save_toggles(updates: dict) -> dict:
    cur = load_toggles()
    for k, v in (updates or {}).items():
        if k in EXCHANGES:
            cur[k] = bool(v)
    import db
    db.set_arena_state(_TOGGLE_KEY, json.dumps(cur))
    global _TOGGLE_CACHE
    _TOGGLE_CACHE = (time.time(), dict(cur))
    return cur


def exchange_enabled(name: str) -> bool:
    return bool(load_toggles().get(str(name).lower(), False))


def enabled_exchanges() -> tuple[str, ...]:
    return tuple(ex for ex in EXCHANGES if exchange_enabled(ex))

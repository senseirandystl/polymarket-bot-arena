"""Clean-tick guard — reject implausible single-tick price jumps.

Raw market data (even REST midpoints polled every second) occasionally delivers
a stale snapshot, a duplicate, or an outright bad print. The profitable-bot
research (0xSurferX, Jun 2026) fixes this with a small hygiene layer: **drop the
first tick from a fresh source, and reject any tick that jumps more than ~15¢
from the last known-good price** (a real Polymarket YES mid does not teleport
15¢ between two reads a second apart — that is bad data).

This is the lightweight REST-era equivalent: a per-token guard that remembers
the last accepted price and rejects an outlier *once* (so a genuine fast reprice
still gets in on the very next read, which will be within threshold of the
rejected one only if it persists). Kept pure/stateful-by-token and free of any
network calls so both the trader hot path and the dashboard can share it.
"""

import threading
import time
from typing import Optional

import config

# token_id -> {"price": float, "ts": float, "seen": int}
_last_good: dict = {}
_lock = threading.Lock()


def clean_price(token_id: str, raw: Optional[float]) -> Optional[float]:
    """Return ``raw`` if it is a plausible next tick, else the last good price.

    Rules (thresholds from ``config``):
      1. ``raw is None`` → return the last good price (or ``None``).
      2. First accepted tick for a token is *dropped* (``DROP_FIRST_TICK``) —
         the initial REST read can be a stale cached snapshot; it seeds state
         and the next read is trusted.
      3. A jump ``> CLEAN_TICK_MAX_JUMP`` from the last good price is rejected
         **once** (returns last good). If the outlier persists past
         ``CLEAN_TICK_STALE_SEC`` it is accepted as a genuine reprice so the
         guard can never latch a token to a permanently wrong value.
    """
    max_jump = float(getattr(config, "CLEAN_TICK_MAX_JUMP", 0.15))
    stale_sec = float(getattr(config, "CLEAN_TICK_STALE_SEC", 10.0))
    drop_first = bool(getattr(config, "CLEAN_TICK_DROP_FIRST", True))
    now = time.time()

    with _lock:
        prev = _last_good.get(token_id)

        if raw is None:
            return prev["price"] if prev else None

        try:
            raw = float(raw)
        except (TypeError, ValueError):
            return prev["price"] if prev else None

        # First time we've seen this token.
        if prev is None:
            # Seed state. Optionally drop this first (possibly stale) tick.
            _last_good[token_id] = {"price": raw, "ts": now, "seen": 1}
            return None if drop_first else raw

        jump = abs(raw - prev["price"])
        if jump > max_jump and (now - prev["ts"]) < stale_sec:
            # Outlier and last good is still fresh — reject once, keep last good.
            return prev["price"]

        # Accept: within threshold, or the last good has gone stale.
        _last_good[token_id] = {"price": raw, "ts": now, "seen": prev["seen"] + 1}
        return raw


def reset(token_id: str | None = None) -> None:
    """Clear guard state (all tokens, or one). Used by tests and on rollover."""
    with _lock:
        if token_id:
            _last_good.pop(token_id, None)
        else:
            _last_good.clear()

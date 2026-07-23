"""Shared HTTP client: bounded retries + exponential backoff with jitter.

Every arena network read already wraps its request in ``try/except`` with a
safe fallback, so a failure never kills a thread. This adds RESILIENCE on top:
a transient blip (connection reset, timeout, 429/5xx) is retried a few times
with short exponential backoff instead of immediately falling back to
stale/empty data and forfeiting a whole cycle. On final failure the underlying
``requests`` exception (or the last response) is returned/raised so each
caller's existing handling behaves exactly as before.

Deliberately NOT used on the 1-second hot-path book/midpoint reads
(``polymarket_markets.get_order_book`` / ``midpoint*``): a retry-sleep there
would stall the trader tick, and those already fall back to the warm cache.
Use it for slow-cadence reads (discovery, resolution, CVD, PM history, strike).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests

import config

logger = logging.getLogger("arena.http")


def _backoff_seconds(attempt: int, base: float, cap: float) -> float:
    """Exponential backoff with full jitter: random in [0, min(cap, base·2^n)]."""
    ceiling = min(cap, base * (2 ** attempt))
    return random.uniform(0.0, ceiling)


def request_with_retry(
    method: str,
    url: str,
    *,
    retries: int | None = None,
    backoff_base: float | None = None,
    backoff_cap: float | None = None,
    retry_statuses: tuple[int, ...] | None = None,
    timeout: float = 10.0,
    **kwargs: Any,
) -> requests.Response:
    """Issue an HTTP request, retrying transient failures.

    Retries on any ``requests.RequestException`` and on the configured transient
    status codes (429/5xx). Returns the final :class:`requests.Response` (even a
    non-retryable error status — the caller inspects ``status_code`` as before).
    Raises the last exception only when every attempt raised.
    """
    retries = config.HTTP_MAX_RETRIES if retries is None else retries
    backoff_base = config.HTTP_BACKOFF_BASE if backoff_base is None else backoff_base
    backoff_cap = config.HTTP_BACKOFF_CAP if backoff_cap is None else backoff_cap
    retry_statuses = retry_statuses or config.HTTP_RETRY_STATUSES

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            delay = _backoff_seconds(attempt, backoff_base, backoff_cap)
            logger.debug(
                "http %s %s raised %s (attempt %d/%d) — retry in %.2fs",
                method, url, exc, attempt + 1, retries + 1, delay,
            )
            time.sleep(delay)
            continue

        if resp.status_code in retry_statuses and attempt < retries:
            delay = _backoff_seconds(attempt, backoff_base, backoff_cap)
            logger.debug(
                "http %s %s -> %d (attempt %d/%d) — retry in %.2fs",
                method, url, resp.status_code, attempt + 1, retries + 1, delay,
            )
            time.sleep(delay)
            continue

        return resp

    # Unreachable: the loop either returns a response or raises. Guard anyway.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("request_with_retry exhausted without a response")


def get(url: str, **kwargs: Any) -> requests.Response:
    """Convenience wrapper for a retrying GET."""
    return request_with_retry("GET", url, **kwargs)


def post(url: str, **kwargs: Any) -> requests.Response:
    """Convenience wrapper for a retrying POST."""
    return request_with_retry("POST", url, **kwargs)

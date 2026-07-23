"""Historical data loading for the backtester (network + gitignored cache).

Sources (all public, none touch bot_arena.db):
  * Gamma /events (series 10684) — resolved BTC 5-min markets + true outcome
  * Binance 1m klines — BTC opens (strike + drift) and closes (candle stream
    the strategies' analyze() consumes), batched over the whole span
  * CLOB /prices-history — the market's own Polymarket YES mid trajectory

Markets can be selected by count, by date range, or by an explicit
condition-id list. PM histories are cached per market in a size-capped JSON
under ``backtest/.cache/`` (same pattern as the Signal Lab harness cache).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

import config
from tools.lane_candidates import Series

logger = logging.getLogger("backtest.data")

GAMMA = "https://gamma-api.polymarket.com/events"
BINANCE = "https://api.binance.com/api/v3/klines"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
SERIES_ID = "10684"

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_FILE = CACHE_DIR / "history.json"
CACHE_MAX_MARKETS = 3000


@dataclass(frozen=True)
class MarketRecord:
    """One resolved BTC 5-min market."""
    id: str
    question: str
    open_ts: float               # epoch seconds of eventStartTime
    close_ts: float              # epoch seconds of endDate
    yes_won: bool
    up_token: str | None


@dataclass
class HistoricalData:
    """Everything the engine needs to replay a set of markets."""
    markets: list                                  # [MarketRecord] chronological
    btc_opens: Series = field(default_factory=lambda: Series([]))
    btc_closes: Series = field(default_factory=lambda: Series([]))
    pm_prices: dict = field(default_factory=dict)  # market_id -> [(ts, yes_mid)]


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def fetch_resolved_markets(limit: int | None = None,
                           start: datetime | None = None,
                           end: datetime | None = None,
                           market_ids: list | None = None) -> list:
    """Resolved markets, newest-first from Gamma, returned CHRONOLOGICAL.

    Selection is by ``limit`` (most recent N), by ``[start, end]`` date range
    (on the window close time), or by an explicit ``market_ids`` condition-id
    list — combinable; a market must satisfy every provided filter.
    """
    wanted = set(market_ids) if market_ids else None
    out: list = []
    offset = 0
    while True:
        r = requests.get(GAMMA, params={
            "series_id": SERIES_ID, "closed": "true", "limit": 100,
            "offset": offset, "order": "endDate", "ascending": "false",
        }, timeout=20)
        r.raise_for_status()
        events = r.json()
        if not events:
            break
        page_oldest = None
        for e in events:
            for m in (e.get("markets") or []):
                start_iso, end_iso = m.get("eventStartTime"), m.get("endDate")
                prices = m.get("outcomePrices")
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if not (start_iso and end_iso and prices and len(prices) == 2):
                    continue
                if prices[0] not in ("0", "1"):
                    continue
                close_ts = _ts(end_iso)
                page_oldest = (close_ts if page_oldest is None
                               else min(page_oldest, close_ts))
                if wanted is not None and m.get("conditionId") not in wanted:
                    continue
                if start is not None and close_ts < start.timestamp():
                    continue
                if end is not None and close_ts > end.timestamp():
                    continue
                tokens = m.get("clobTokenIds")
                if isinstance(tokens, str):
                    try:
                        tokens = json.loads(tokens)
                    except (TypeError, ValueError):
                        tokens = None
                out.append(MarketRecord(
                    id=m.get("conditionId"),
                    question=m.get("question") or "",
                    open_ts=_ts(start_iso),
                    close_ts=close_ts,
                    yes_won=prices[0] == "1",
                    up_token=tokens[0] if tokens else None,
                ))
        offset += 100
        if limit is not None and wanted is None and start is None and len(out) >= limit:
            break
        # Gamma pages newest-first: once the page is entirely older than the
        # requested range start, no further page can match.
        if start is not None and page_oldest is not None \
                and page_oldest < start.timestamp():
            break
        if offset >= 5000:                     # hard stop — never crawl forever
            logger.warning("fetch_resolved_markets: pagination cap hit")
            break
    if limit is not None:
        out = out[:limit]
    return sorted(out, key=lambda m: m.close_ts)


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except (OSError, ValueError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    if len(cache) > CACHE_MAX_MARKETS:
        for k in list(cache.keys())[: len(cache) - CACHE_MAX_MARKETS]:
            cache.pop(k, None)
    CACHE_FILE.write_text(json.dumps(cache))


def _fetch_btc_series(start_ms: int, end_ms: int) -> tuple:
    """Batched BTC 1m klines over the span: (opens_by_open_time, closes_by_close_time).

    Opens keyed by candle OPEN time give the exact strike (Binance open at
    eventStartTime — the BUG #23 lesson) and the harness-identical intra-window
    trajectory; closes keyed by CLOSE time feed analyze()'s candle stream
    without look-ahead (a candle is only visible once finished).
    """
    opens, closes = [], []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(BINANCE, params={
            "symbol": "BTCUSDT", "interval": "1m",
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        }, timeout=20)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        for c in rows:
            opens.append((c[0] / 1000.0, float(c[1])))
            closes.append(((c[6] + 1) / 1000.0, float(c[4])))
        cursor = rows[-1][6] + 1
        if len(rows) < 1000:
            break
    return Series(opens), Series(closes)


def _fetch_pm_history(mkt: MarketRecord) -> list:
    if not mkt.up_token:
        return []
    r = requests.get(CLOB_HISTORY, params={
        "market": mkt.up_token,
        "startTs": int(mkt.open_ts), "endTs": int(mkt.close_ts),
        "fidelity": 1,
    }, timeout=20)
    r.raise_for_status()
    hist = r.json().get("history") or []
    return [[float(h["t"]), float(h["p"])] for h in hist]


def load_historical_data(markets: list, use_cache: bool = True,
                         lookback_min: int = 65) -> HistoricalData:
    """Fetch BTC klines (with analyze() warmup lookback) + PM mid histories."""
    if not markets:
        return HistoricalData(markets=[])
    span_start = int(min(m.open_ts for m in markets) * 1000)
    span_end = int(max(m.close_ts for m in markets) * 1000)
    logger.info(f"Fetching BTC 1m klines for span "
                f"({(span_end - span_start) / 3600e3:.1f}h + {lookback_min}m lookback)")
    opens, closes = _fetch_btc_series(
        span_start - lookback_min * 60 * 1000, span_end)

    cache = _load_cache() if use_cache else {}
    pm: dict = {}
    fetched = 0
    for i, mkt in enumerate(markets):
        key = f"pm:{mkt.id}"
        if use_cache and key in cache:
            pm[mkt.id] = [(float(t), float(p)) for t, p in cache[key]]
            continue
        try:
            rows = _fetch_pm_history(mkt)
        except requests.RequestException as e:
            logger.warning(f"PM history fetch failed for {str(mkt.id)[:12]}…: {e}")
            rows = []
        pm[mkt.id] = [(t, p) for t, p in rows]
        if use_cache:
            cache[key] = rows
        fetched += 1
        if use_cache and fetched % 25 == 0:
            _save_cache(cache)
    if use_cache:
        _save_cache(cache)
    logger.info(f"Loaded {len(markets)} markets "
                f"({fetched} PM histories fetched, {len(markets) - fetched} cached)")
    return HistoricalData(markets=list(markets), btc_opens=opens,
                          btc_closes=closes, pm_prices=pm)

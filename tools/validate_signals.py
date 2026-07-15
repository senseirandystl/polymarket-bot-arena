#!/usr/bin/env python3
"""Offline signal-validation harness — does a candidate signal actually predict?

Uses REAL data, writes nothing to the runtime DB:
  * resolved BTC 5-min markets from Polymarket Gamma (window times + true outcome)
  * BTC price trajectory from Binance 1m klines (Chainlink-proxy; basis ~0.005%)

The accurate "price to beat" (strike) is the Binance open at the market's
``eventStartTime`` (the exact window open) — NOT a mid-window "first sighting",
which is the bug that made the live drift signal read inverted (BUG #23).

Run:
    .venv/bin/python3 tools/validate_signals.py --markets 200
    .venv/bin/python3 tools/validate_signals.py --markets 300 --no-cache

Storage: klines are cached in a gitignored, size-capped JSON
(``tools/.signal_cache/klines.json``, <=CACHE_MAX markets) so re-runs are fast
without growing the market DB. Nothing here touches bot_arena.db.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.signal_validation import build_samples, predictiveness, time_buckets

GAMMA = "https://gamma-api.polymarket.com/events"
BINANCE = "https://api.binance.com/api/v3/klines"
SERIES_ID = "10684"
WINDOW_SEC = 300
CACHE_DIR = Path(__file__).resolve().parent / ".signal_cache"
CACHE_FILE = CACHE_DIR / "klines.json"
CACHE_MAX = 1500                     # cap cached markets — keep storage tiny


def _ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


def fetch_resolved_markets(n: int) -> list:
    """Most-recent ``n`` resolved BTC 5-min markets (window + outcome)."""
    out, offset = [], 0
    while len(out) < n:
        r = requests.get(GAMMA, params={
            "series_id": SERIES_ID, "closed": "true", "limit": 100,
            "offset": offset, "order": "endDate", "ascending": "false",
        }, timeout=20)
        r.raise_for_status()
        evs = r.json()
        if not evs:
            break
        for e in evs:
            for m in (e.get("markets") or []):
                start, end = m.get("eventStartTime"), m.get("endDate")
                prices = m.get("outcomePrices")
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if not (start and end and prices and len(prices) == 2):
                    continue
                # ["1","0"] => Up won; ["0","1"] => Down won.
                if prices[0] not in ("0", "1"):
                    continue
                out.append({"id": m.get("conditionId"), "start": start,
                            "end": end, "yes_won": prices[0] == "1"})
        offset += 100
    return out[:n]


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    if len(cache) > CACHE_MAX:                      # prune oldest, keep bounded
        for k in list(cache.keys())[: len(cache) - CACHE_MAX]:
            cache.pop(k, None)
    CACHE_FILE.write_text(json.dumps(cache))


def fetch_trajectory(mkt: dict, cache: dict, use_cache: bool) -> list:
    """Binance 1m opens over the window as ``(seconds_from_open, btc)`` points."""
    key = mkt["id"]
    if use_cache and key in cache:
        rows = cache[key]
    else:
        st, en = _ms(mkt["start"]), _ms(mkt["end"])
        r = requests.get(BINANCE, params={
            "symbol": "BTCUSDT", "interval": "1m",
            "startTime": st, "endTime": en, "limit": 10,
        }, timeout=20)
        r.raise_for_status()
        k = r.json()
        rows = [[c[0], c[1]] for c in k]            # [open_time_ms, open_price]
        if use_cache:
            cache[key] = rows
    if not rows:
        return []
    open_ms = _ms(mkt["start"])
    return [((row[0] - open_ms) / 1000.0, float(row[1])) for row in rows]


def _fmt(res: dict) -> str:
    wr = res["follow_winrate"]
    wr_s = f"{wr*100:5.1f}%" if wr is not None else "  n/a"
    verdict = ""
    if wr is not None and res["n"] >= 20:
        verdict = "  <-- PREDICTIVE" if wr >= 0.55 else ("  <-- INVERTED" if wr <= 0.45 else "")
    return (f"    n={res['n']:4d}  follow-WR={wr_s}   "
            f"(up n={res['up_n']} wr={_p(res['up_winrate'])}, "
            f"down n={res['down_n']} wr={_p(res['down_winrate'])}){verdict}")


def _p(x):
    return f"{x*100:.0f}%" if x is not None else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markets", type=int, default=200, help="resolved markets to sample")
    ap.add_argument("--no-cache", action="store_true", help="ignore + skip the kline cache")
    args = ap.parse_args()

    use_cache = not args.no_cache
    print(f"Fetching {args.markets} resolved BTC 5-min markets from Gamma...")
    markets = fetch_resolved_markets(args.markets)
    print(f"  got {len(markets)} resolved markets")

    cache = _load_cache() if use_cache else {}
    all_samples = []
    up_count = 0
    for i, mkt in enumerate(markets):
        try:
            traj = fetch_trajectory(mkt, cache, use_cache)
        except Exception as e:
            print(f"  skip {str(mkt['id'])[:10]}: {e}")
            continue
        if len(traj) < 3:
            continue
        strike = traj[0][1]                         # open @ eventStartTime = true strike
        up_count += 1 if mkt["yes_won"] else 0
        all_samples.extend(build_samples(mkt["id"], strike, traj,
                                          mkt["yes_won"], WINDOW_SEC))
        if use_cache and i % 25 == 0:
            _save_cache(cache)
        if not use_cache:
            time.sleep(0.05)                        # be gentle without a cache
    if use_cache:
        _save_cache(cache)

    n_mkts = up_count + (len([1 for m in markets]) - up_count)  # informational
    print(f"\nBuilt {len(all_samples)} decision-samples from {len(markets)} markets "
          f"(UP base rate {100*up_count/max(len(markets),1):.0f}%).\n")

    signals = ["drift_raw", "drift_prod", "mom2"]
    print("=== Overall predictiveness (follow-the-signal win rate; >55% good, <45% inverted) ===")
    for sig in signals:
        print(f"  [{sig}]")
        print(_fmt(predictiveness(all_samples, sig)))

    print("\n=== drift_prod by time-remaining bucket (is it salvageable near expiry?) ===")
    for b in time_buckets(all_samples, "drift_prod"):
        print(f"  {b['bucket']:>10}: {_fmt(b).strip()}")

    print("\nNote: nothing was written to bot_arena.db. "
          f"Kline cache: {'on' if use_cache else 'off'} ({CACHE_FILE}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

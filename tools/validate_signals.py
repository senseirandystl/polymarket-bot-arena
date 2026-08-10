#!/usr/bin/env python3
"""Offline signal-validation harness — does a candidate signal actually predict?

Uses REAL data, writes nothing to the runtime DB:
  * resolved BTC 5-min markets from Polymarket Gamma (window times + true outcome)
  * BTC price trajectory from Binance 1m klines (proxy for ranking only)

Live resolution (2026-08-07+) is Chainlink **30s TWAP** at open vs close, not a
single Binance print. This harness still reconstructs strike as the Binance open
at ``eventStartTime`` and trajectories from 1m klines — fine for *ordering*
signals and relative net-edge, **not** absolute live P&L. Never use mid-window
"first sighting" as strike (BUG #23). For production moneyness see
``signals/strike.py`` + ``signals/twap.py``.

Run:
    .venv/bin/python3 tools/validate_signals.py --markets 200
    .venv/bin/python3 tools/validate_signals.py --markets 300 --no-cache
    .venv/bin/python3 tools/validate_signals.py --markets 300 --candidates
    .venv/bin/python3 tools/validate_signals.py --markets 300 --propose

``--candidates`` backfills the kill-switched candidate lanes (fut/tech/xasset
— tools/lane_candidates.py) and reports their follow-WR + net edge.
``--propose`` additionally records the run in bot_arena.db
(lane_validation_runs) and files a PENDING lane proposal for any candidate
clearing the promotion thresholds — reviewed and approved/denied by a human
in the dashboard Signal Lab (approval activates the lane live via the
DB override; see db.decide_lane_proposal). This is the ONLY mode that writes
to bot_arena.db, and it never touches trade tables.

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
from tools.signal_validation import (
    build_samples, decay_analysis, information_coefficient,
    magnitude_distribution, net_edge, net_edge_time_buckets,
    predictiveness, rank_signals, regime_split, time_buckets,
)
from polymarket_fills import taker_fee

GAMMA = "https://gamma-api.polymarket.com/events"
BINANCE = "https://api.binance.com/api/v3/klines"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
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
                tokens = m.get("clobTokenIds")
                if isinstance(tokens, str):
                    try:
                        tokens = json.loads(tokens)
                    except Exception:
                        tokens = None
                up_token = tokens[0] if tokens else None
                out.append({"id": m.get("conditionId"), "start": start,
                            "end": end, "yes_won": prices[0] == "1",
                            "up_token": up_token})
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


def fetch_pm_prices(mkt: dict, cache: dict, use_cache: bool) -> list:
    """Polymarket YES(Up) mid over the window as ``(seconds_from_open, p)``.

    Cached under ``pm:<condition_id>`` in the same size-capped kline cache so
    storage stays bounded.
    """
    if not mkt.get("up_token"):
        return []
    key = f"pm:{mkt['id']}"
    if use_cache and key in cache:
        rows = cache[key]
    else:
        st, en = _ms(mkt["start"]) // 1000, _ms(mkt["end"]) // 1000
        r = requests.get(CLOB_HISTORY, params={
            "market": mkt["up_token"], "startTs": st, "endTs": en, "fidelity": 1,
        }, timeout=20)
        r.raise_for_status()
        hist = r.json().get("history") or []
        rows = [[h["t"], h["p"]] for h in hist]
        if use_cache:
            cache[key] = rows
    open_s = _ms(mkt["start"]) / 1000.0
    return [(float(t) - open_s, float(p)) for t, p in rows]


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
    ap.add_argument("--candidates", action="store_true",
                    help="backfill + evaluate the kill-switched candidate lanes "
                         "(fut/tech/xasset) — extra Binance series fetches")
    ap.add_argument("--propose", action="store_true",
                    help="record the run in bot_arena.db and file lane "
                         "proposals for candidates clearing the promotion "
                         "thresholds (implies --candidates)")
    ap.add_argument("--rank", action="store_true",
                    help="re-validate EVERY signal (live lanes, candidate "
                         "lanes, and the expanded feature suite) and produce "
                         "a ranked scorecard: IC, follow-WR, net edge at mid "
                         "and after slippage, decay, and regime splits. "
                         "Implies --candidates. Writes a markdown report.")
    ap.add_argument("--slippage", type=float, default=0.005,
                    help="per-share execution penalty (fraction of $1) for "
                         "the slippage-adjusted net-edge column "
                         "(default 0.005 = 0.5c)")
    ap.add_argument("--report", type=str, default=None,
                    help="markdown report path for --rank "
                         "(default logs/signal_report.md)")
    args = ap.parse_args()
    if args.propose or args.rank:
        args.candidates = True

    use_cache = not args.no_cache
    print(f"Fetching {args.markets} resolved BTC 5-min markets from Gamma...")
    markets = fetch_resolved_markets(args.markets)
    print(f"  got {len(markets)} resolved markets")

    # Candidate-lane backfill: fetch each auxiliary series ONCE for the whole
    # span (paginated), then look values up per decision point — instead of
    # per-market requests.
    series = {}
    if args.candidates and markets:
        from tools import lane_candidates as lc
        span_start = min(_ms(m["start"]) for m in markets)
        span_end = max(_ms(m["end"]) for m in markets)
        hist_start = span_start - 65 * 60 * 1000    # 60 candles of BTC history
        print("Fetching candidate-lane series (BTC/ETH/SOL klines + "
              "funding/OI/taker)...")
        try:
            series["btc_close"] = lc.fetch_klines_series("BTCUSDT", hist_start, span_end)
            series["eth_close"] = lc.fetch_klines_series("ETHUSDT", span_start - 5*60*1000, span_end)
            series["sol_close"] = lc.fetch_klines_series("SOLUSDT", span_start - 5*60*1000, span_end)
            # Funding prints only every 8h — reach back 9h so every sample has
            # a last-at-or-before funding reading; OI/taker are 5m series.
            series.update(lc.fetch_futures_series(span_start - 9*3600*1000, span_end))
            for k, v in series.items():
                print(f"  {k}: {len(v)} points")
        except Exception as e:
            print(f"  candidate series fetch failed ({e}) — candidate lanes "
                  f"will be partial/empty")

    cache = _load_cache() if use_cache else {}
    all_samples = []
    up_count = 0
    for i, mkt in enumerate(markets):
        try:
            traj = fetch_trajectory(mkt, cache, use_cache)
            pm = fetch_pm_prices(mkt, cache, use_cache)
        except Exception as e:
            print(f"  skip {str(mkt['id'])[:10]}: {e}")
            continue
        if len(traj) < 3:
            continue
        strike = traj[0][1]                         # open @ eventStartTime = true strike
        up_count += 1 if mkt["yes_won"] else 0
        mkt_samples = build_samples(mkt["id"], strike, traj,
                                    mkt["yes_won"], WINDOW_SEC,
                                    pm_prices=pm, market_seq=i)
        if args.candidates and series:
            from tools import lane_candidates as lc
            lc.attach_candidates(mkt_samples, _ms(mkt["start"]) / 1000.0, series)
            # Always attach multiscale/session features in candidate mode so
            # ms_mom (and lag dual-path) can clear the promotion bar — not
            # only under --rank.
            lc.attach_features(mkt_samples, _ms(mkt["start"]) / 1000.0, series)
        all_samples.extend(mkt_samples)
        if use_cache and i % 25 == 0:
            _save_cache(cache)
        if not use_cache:
            time.sleep(0.05)                        # be gentle without a cache
    if use_cache:
        _save_cache(cache)

    n_mkts = up_count + (len([1 for m in markets]) - up_count)  # informational
    print(f"\nBuilt {len(all_samples)} decision-samples from {len(markets)} markets "
          f"(UP base rate {100*up_count/max(len(markets),1):.0f}%).\n")

    signals = ["drift_raw", "drift_prod", "mom2", "pm_mom"]
    print("=== Overall predictiveness (follow-the-signal win rate; >55% good, <45% inverted) ===")
    for sig in signals:
        print(f"  [{sig}]")
        print(_fmt(predictiveness(all_samples, sig)))

    print("\n=== pm_mom |magnitude| distribution (per-minute PM price move) ===")
    dist = magnitude_distribution(all_samples, "pm_mom")
    print(f"    n={dist['n']}  " + "  ".join(
        f"p{p}={dist[f'p{p}']:.4f}" if dist[f'p{p}'] is not None else f"p{p}=n/a"
        for p in (50, 75, 90, 97)))
    print("    (live lane saturates at 0.0019/step — compare to p50/p97 above)")

    print("\n=== drift_prod by time-remaining bucket (is it salvageable near expiry?) ===")
    for b in time_buckets(all_samples, "drift_prod"):
        print(f"  {b['bucket']:>10}: {_fmt(b).strip()}")

    # --- NET-EDGE metrics: what a bot actually earns AFTER paying the PM price
    # + taker fee. A signal can be predictive yet worthless once priced in.
    def _ne_fmt(res):
        if not res["n"]:
            return "    n=   0"
        return (f"    n={res['n']:4d}  wr={res['winrate']*100:5.1f}%  "
                f"avg_price={res['avg_price']:.3f}  "
                f"EV/share={res['ev_per_share']*100:+.2f}c")

    def fav_side(s):
        if abs(s.pm_yes - 0.5) < 0.02:
            return None
        return "yes" if s.pm_yes > 0.5 else "no"

    def drift_side(s):
        d = s.signals["drift_prod"]
        return None if abs(d) < 1e-6 else ("yes" if d > 0 else "no")

    def mom_side(s):
        m = s.signals["mom2"]
        return None if abs(m) < 1e-9 else ("yes" if m > 0 else "no")

    def drift_lag_side(s):
        """Drift side only when the PM price has NOT priced it in yet."""
        side = drift_side(s)
        if side is None:
            return None
        price = s.pm_yes if side == "yes" else 1.0 - s.pm_yes
        return side if price <= 0.58 else None

    def pm_side(s):
        v = s.signals.get("pm_mom")
        if v is None or abs(v) < 1e-9:
            return None
        return "yes" if v > 0 else "no"

    def ignorance_fade_side(s):
        """Reproduce the live leak: with drift weak (model near-ignorant),
        buy the market UNDERDOG — the trade the fair-blend manufactures when
        P_model~0.5 and the mid has moved away. Expect strongly negative EV."""
        if abs(s.signals["drift_prod"]) >= 0.15:
            return None
        if abs(s.pm_yes - 0.5) < 0.05:
            return None
        return "no" if s.pm_yes > 0.5 else "yes"

    rules = [("buy the favorite (tilt lane)", fav_side),
             ("follow drift", drift_side),
             ("follow drift ONLY when side <=58c (market lags)", drift_lag_side),
             ("follow 1-candle BTC momentum", mom_side),
             ("follow PM in-market momentum (pm lane)", pm_side),
             ("IGNORANCE-FADE: buy underdog when |drift|<0.15 (live leak)",
              ignorance_fade_side)]
    print("\n=== NET EDGE vs the actual PM price (per-share EV after taker fee) ===")
    for label, rule in rules:
        print(f"  [{label}]")
        print(_ne_fmt(net_edge(all_samples, rule, taker_fee)))
        for b in net_edge_time_buckets(all_samples, rule, taker_fee):
            print(f"      {b['bucket']:>10}: {_ne_fmt(b).strip()}")

    # --- Candidate lanes (kill-switched): follow-WR + NET edge per lane ---
    wrote_db = False
    if args.candidates:
        from tools import lane_candidates as lc
        results = lc.evaluate_candidates(all_samples, taker_fee)
        print("\n=== CANDIDATE lanes (kill-switched; promotion bar: "
              f"n>={lc.MIN_SAMPLES}, follow-WR>={lc.MIN_FOLLOW_WR:.0%}, "
              f"net>={lc.MIN_NET_EDGE*100:.1f}c/share on the LIVE key) ===")
        for key in lc.CANDIDATE_KEYS:
            m = results[key]
            wr = f"{m['follow_wr']*100:5.1f}%" if m["follow_wr"] is not None else "  n/a"
            ev = (f"{m['ev_per_share']*100:+.2f}c"
                  if m["ev_per_share"] is not None else "n/a")
            live = "  (LIVE key)" if key in lc.LIVE_LANE_KEYS.values() else ""
            print(f"  [{key:<11}] n={m['n']:5d}  follow-WR={wr}  "
                  f"net n={m['net_n']:5d}  EV/share={ev}{live}")

        proposals = lc.build_proposals(results)
        if args.propose:
            import db
            db.init_db()
            run_id = db.record_lane_validation_run(
                len(markets), len(all_samples), results)
            wrote_db = True
            if proposals:
                for p in proposals:
                    pid = db.create_lane_proposal(
                        p["lane"], p["metrics"], p["proposal"], run_id=run_id)
                    state = f"proposal #{pid}" if pid else "already approved — skipped"
                    print(f"  >> {p['lane']}: cleared promotion bar -> {state}")
            else:
                print("  >> no lane cleared the promotion bar this run")
            print(f"  (run #{run_id} recorded; review pending proposals in the "
                  f"dashboard Signal Lab)")
        elif proposals:
            print("  >> would propose: "
                  + ", ".join(p["lane"] for p in proposals)
                  + "  (re-run with --propose to file)")

    # --- Ranked full-suite scorecard (--rank): IC + edge after fees/slippage
    # + decay + regime-specific value, for every signal the arena knows.
    if args.rank:
        from tools import lane_candidates as lc
        directional = (["drift_prod", "drift_raw", "mom2", "pm_mom"]
                       + lc.CANDIDATE_KEYS + lc.FEATURE_DIRECTIONAL_KEYS)
        rows = rank_signals(all_samples, directional, taker_fee,
                            slippage=args.slippage)

        def _c(x, pct=False):
            if x is None:
                return "   n/a"
            return f"{x*100:+6.2f}" if not pct else f"{x*100:5.1f}%"

        def _ic(x):
            return "  n/a" if x is None else f"{x:+.2f}"

        print(f"\n=== RANKED signal scorecard (slippage {args.slippage*100:.1f}c/share; "
              "keep/weight ONLY positive ev_slip with a healthy recent slice) ===")
        header = (f"  {'signal':<12} {'n':>6} {'IC':>7} {'follow':>7} "
                  f"{'ev_mid c':>9} {'ev_slip c':>10} {'recentWR':>9}  verdict")
        print(header)
        for r in rows:
            ev_slip = r["ev_slip"]
            wr_recent = r["recent_wr"]
            if ev_slip is None or r["net_n"] < 50:
                verdict = "insufficient"
            elif ev_slip >= 0.005 and (wr_recent or 0) >= 0.53:
                verdict = "POSITIVE EDGE"
            elif ev_slip > 0:
                verdict = "marginal"
            else:
                verdict = "no edge"
            print(f"  {r['signal']:<12} {r['n']:>6} "
                  f"{_ic(r['ic']):>7} {_c(r['follow_wr'], pct=True):>7} "
                  f"{_c(r['ev_mid']):>9} {_c(ev_slip):>10} "
                  f"{_c(wr_recent, pct=True):>9}  {verdict}")

        print("\n=== Performance decay (chronological thirds; recent first) ===")
        decay_keys = ["drift_prod", "mom2"] + [
            r["signal"] for r in rows[:3]
            if r["signal"] not in ("drift_prod", "mom2")]
        for key in decay_keys:
            parts = []
            for b in decay_analysis(all_samples, key):
                wr = b["follow_winrate"]
                parts.append(f"{b['bucket']}: n={b['n']} "
                             f"wr={_c(wr, pct=True).strip()} "
                             f"ic={_ic(b['ic'])}")
            print(f"  [{key:<12}] " + "  |  ".join(parts))

        print("\n=== Regime-specific value (terciles of context features) ===")
        regime_pairs = [("drift_prod", "regime_trend"),
                        ("drift_prod", "ms_rvol_5m"),
                        ("mom2", "regime_trend"),
                        ("mom2", "ms_rvol_5m")]
        regime_results = {}
        for sig, ctx in regime_pairs:
            buckets = regime_split(all_samples, sig, ctx, taker_fee,
                                   slippage=args.slippage)
            regime_results[(sig, ctx)] = buckets
            for b in buckets:
                print(f"  [{sig:<10} | {b['regime']:<20}] n={b['n']:5d} "
                      f"wr={_c(b['follow_winrate'], pct=True)} "
                      f"ic={_ic(b['ic']):>6} "
                      f"ev_slip={_c(b['ev_per_share'])}c")

        report_path = Path(args.report) if args.report else (
            Path(__file__).resolve().parent.parent / "logs" / "signal_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Signal validation report",
            f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
            f"- markets: {len(markets)}  samples: {len(all_samples)}  "
            f"UP base rate: {100*up_count/max(len(markets),1):.0f}%",
            f"- slippage assumption: {args.slippage*100:.1f}c/share "
            "(added to stale PM mids; fee = canonical taker fee)",
            "",
            "## Ranked scorecard (by net edge after fees + slippage)",
            "",
            "| signal | n | IC | follow-WR | EV@mid c | EV@slip c | recent WR |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['signal']} | {r['n']} | {_ic(r['ic'])} | "
                f"{_c(r['follow_wr'], pct=True).strip()} | "
                f"{_c(r['ev_mid']).strip()} | {_c(r['ev_slip']).strip()} | "
                f"{_c(r['recent_wr'], pct=True).strip()} |")
        lines += ["", "## Regime splits", ""]
        for (sig, ctx), buckets in regime_results.items():
            for b in buckets:
                lines.append(
                    f"- `{sig}` in `{b['regime']}`: n={b['n']}, "
                    f"WR={_c(b['follow_winrate'], pct=True).strip()}, "
                    f"IC={_ic(b['ic'])}, "
                    f"EV@slip={_c(b['ev_per_share']).strip()}c")
        lines += [
            "",
            "## Policy",
            "Only signals with positive EV after fees + slippage AND a "
            "healthy recent slice may carry (or grow) live weight. "
            "Candidate lanes still graduate exclusively through the "
            "lane-proposal pipeline (harness nominates, live attribution "
            "judges).",
        ]
        report_path.write_text("\n".join(lines) + "\n")
        print(f"\nReport written to {report_path}")

    if wrote_db:
        print("\nNote: wrote lane_validation_runs/lane_proposals only — trade "
              "tables untouched. "
              f"Kline cache: {'on' if use_cache else 'off'} ({CACHE_FILE}).")
    else:
        print("\nNote: nothing was written to bot_arena.db. "
              f"Kline cache: {'on' if use_cache else 'off'} ({CACHE_FILE}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

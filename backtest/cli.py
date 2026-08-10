"""Command-line interface for the backtester.

Examples::

    .venv/bin/python3 -m backtest --days 3
    .venv/bin/python3 -m backtest --markets 200 --bots momentum,hybrid,meanrev
    .venv/bin/python3 -m backtest --from 2026-07-18 --to 2026-07-21 --to-db
    .venv/bin/python3 -m backtest --market-ids ids.txt --json out.json
    .venv/bin/python3 -m backtest --days 5 --walk-forward --folds 3 --top-k 3

Market selection: ``--days N`` (last N days), ``--from/--to`` (dates),
``--markets N`` (most recent N), or ``--market-ids FILE`` (one condition id
per line). Reports print to stdout, write JSON under backtest/reports/, and
``--to-db`` additionally records a summary row in bot_arena.db
(``backtest_runs`` — the Signal Lab run-record pattern; trade tables are
never touched).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Short bot-selection aliases -> (module, class, default instance name).
BOT_ALIASES = {
    "momentum":   ("bots.bot_momentum", "MomentumBot", "momentum-v1"),
    "meanrev":    ("bots.bot_mean_rev", "MeanRevBot", "meanrev-v1"),
    "meanrev-tp": ("bots.bot_meanrev_tp", "MeanRevTPBot", "meanrev-tp-v1"),
    "sniper":     ("bots.bot_sniper", "SniperBot", "sniper-v1"),
    "phantom":    ("bots.bot_phantom", "PhantomBot", "phantom-v1"),
    "hybrid":     ("bots.bot_hybrid", "HybridBot", "hybrid-v1"),
    "lag":        ("bots.bot_lag_residual", "LagResidualBot", "lag-residual-v1"),
    "regime":     ("bots.bot_regime_specialist", "RegimeSpecialistBot", "regime-specialist-v1"),
    "no-lag":     ("bots.bot_no_lag", "NoLagBot", "no-lag-v1"),
    "sweeper":    ("bots.bot_sweeper", "SweeperBot", "sweeper-v1"),
}
DEFAULT_BOTS = ["momentum", "phantom", "meanrev", "hybrid", "sniper"]


def make_bot_factory(names: list):
    """Zero-arg factory returning FRESH instances of the selected bots."""
    import importlib

    specs = []
    for alias in names:
        if alias not in BOT_ALIASES:
            raise SystemExit(f"Unknown bot '{alias}'. "
                             f"Choose from: {', '.join(sorted(BOT_ALIASES))}")
        mod_name, cls_name, bot_name = BOT_ALIASES[alias]
        cls = getattr(importlib.import_module(mod_name), cls_name)
        specs.append((cls, bot_name))

    def factory():
        return [cls(name=bot_name) for cls, bot_name in specs]
    return factory


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="backtest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_argument_group("market selection")
    sel.add_argument("--days", type=float, help="replay the last N days")
    sel.add_argument("--from", dest="date_from", type=_parse_date,
                     help="range start (YYYY-MM-DD)")
    sel.add_argument("--to", dest="date_to", type=_parse_date,
                     help="range end (YYYY-MM-DD)")
    sel.add_argument("--markets", type=int, help="most recent N markets")
    sel.add_argument("--market-ids", type=Path,
                     help="file of condition ids, one per line")
    ap.add_argument("--bots", default=",".join(DEFAULT_BOTS),
                    help=f"comma list of {', '.join(sorted(BOT_ALIASES))} "
                         f"(default: {','.join(DEFAULT_BOTS)})")
    ap.add_argument("--bankroll", type=float, help="starting bankroll (USDC)")
    ap.add_argument("--kelly", type=float, help="Kelly fraction override")
    ap.add_argument("--tick-sec", type=int, help="decision-tick spacing")
    ap.add_argument("--compound", action="store_true",
                    help="size Kelly bets off the compounding pool (default: "
                         "fixed notional, so P&L reads as edge not compounding)")
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass the PM-history cache")
    ap.add_argument("--walk-forward", action="store_true",
                    help="walk-forward mode (train/test folds)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--json", dest="json_path",
                    help="explicit JSON report path (default: backtest/reports/)")
    ap.add_argument("--to-db", action="store_true",
                    help="record the run summary in bot_arena.db (backtest_runs)")
    ap.add_argument("--label", default="", help="run label for report/DB")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    from backtest import data as bt_data
    from backtest import report as bt_report
    from backtest.engine import run_backtest
    from backtest.metrics import summarize
    from backtest.walkforward import format_walkforward, walk_forward

    if not any((args.days, args.date_from, args.markets, args.market_ids)):
        args.days = 1.0  # sensible default: replay the last day

    date_from, date_to = args.date_from, args.date_to
    if args.days is not None:
        date_to = date_to or datetime.now(timezone.utc)
        date_from = date_to - timedelta(days=args.days)
    market_ids = None
    if args.market_ids:
        market_ids = [line.strip() for line in
                      args.market_ids.read_text().splitlines() if line.strip()]

    print("Fetching resolved markets from Gamma…")
    markets = bt_data.fetch_resolved_markets(
        limit=args.markets, start=date_from, end=date_to,
        market_ids=market_ids)
    if not markets:
        print("No resolved markets matched the selection.")
        return 1
    span_h = (markets[-1].close_ts - markets[0].close_ts) / 3600
    print(f"  {len(markets)} markets "
          f"({datetime.fromtimestamp(markets[0].close_ts, tz=timezone.utc):%Y-%m-%d %H:%M} "
          f"→ {datetime.fromtimestamp(markets[-1].close_ts, tz=timezone.utc):%Y-%m-%d %H:%M} UTC, "
          f"{span_h:.1f}h)")

    data = bt_data.load_historical_data(markets, use_cache=not args.no_cache)
    bot_names = [b.strip() for b in args.bots.split(",") if b.strip()]
    factory = make_bot_factory(bot_names)
    run_kwargs = dict(bankroll=args.bankroll, kelly_fraction=args.kelly,
                      tick_sec=args.tick_sec, compound=args.compound)
    label = args.label or (
        ("wf-" if args.walk_forward else "") + "-".join(bot_names))

    if args.walk_forward:
        wf = walk_forward(factory, data, folds=args.folds,
                          train_frac=args.train_frac, top_k=args.top_k,
                          **run_kwargs)
        print("\n" + format_walkforward(wf))
        path = bt_report.write_json(wf, label, args.json_path)
        print(f"\nJSON report: {path}")
        if args.to_db:
            agg = wf["aggregate"]
            row = {"mode": "walk_forward", **wf["params"], "aggregate": agg,
                   "folds": [{k: v for k, v in f.items() if k != "test_summary"}
                             for f in wf["folds"]]}
            rid = bt_report.record_to_db(
                {"markets_replayed": len(markets),
                 "overall": {"n": agg["baseline_trades"]}, **row},
                label, path)
            print(f"Recorded backtest_runs row #{rid}")
        return 0

    result = run_backtest(factory(), data, **run_kwargs)
    summary = summarize(result)
    print("\n" + bt_report.format_report(summary, title=f"Backtest [{label}]"))
    path = bt_report.write_json(summary, label, args.json_path)
    print(f"\nJSON report: {path}")
    if args.to_db:
        rid = bt_report.record_to_db(summary, label, path)
        print(f"Recorded backtest_runs row #{rid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

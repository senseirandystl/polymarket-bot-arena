# Backtesting Framework (`backtest/`)

Offline replay of resolved BTC 5-min markets through the arena's **real
decision path** — the same `BaseBot.make_decision` signal stack, guards,
Kelly sizing, depth-walked fills and taker-fee math that paper/live use.
Nothing in a backtest writes to the live trade tables; the only DB write is
the opt-in run record (`backtest_runs`, mirroring the Signal Lab's
`lane_validation_runs`).

## Quick start

```bash
.venv/bin/python3 -m backtest --days 2                       # last 2 days, default slate
.venv/bin/python3 -m backtest --markets 200 --bots momentum,hybrid
.venv/bin/python3 -m backtest --from 2026-07-18 --to 2026-07-21 --to-db
.venv/bin/python3 -m backtest --market-ids ids.txt --json out.json
.venv/bin/python3 -m backtest --days 5 --walk-forward --folds 3 --top-k 3
```

Market selection: `--days N`, `--from/--to` (dates, on window close), the
most recent `--markets N`, or an explicit `--market-ids FILE` (one condition
id per line). Replayable bots: `momentum, phantom, meanrev, meanrev-tp,
sniper, sentiment, hybrid`. The arbitrage and maker bots are excluded — they
execute against live warm-book microstructure the historical record does not
carry.

## From code (arena / tools)

```python
from backtest import run_backtest, walk_forward
from backtest.data import fetch_resolved_markets, load_historical_data
from backtest.metrics import summarize
from backtest.cli import make_bot_factory

markets = fetch_resolved_markets(limit=100)
data = load_historical_data(markets)
result = run_backtest(make_bot_factory(["momentum", "hybrid"])(), data)
summary = summarize(result)          # JSON-serializable metrics bundle
```

`run_backtest` also accepts `lane_overrides=` (same shape as
`db.get_lane_overrides()`) so a candidate-lane configuration can be replayed
before/after approval — the offline complement to the Signal Lab pipeline.

## Architecture

| Module | Role |
|---|---|
| `backtest/data.py` | Gamma resolved markets + batched Binance 1m klines (65-min lookback for analyze() warmup; strike = the open at `eventStartTime`, the BUG #23 lesson) + CLOB `prices-history` PM mids. Size-capped cache in `backtest/.cache/`. |
| `backtest/books.py` | Synthetic ask ladder anchored on the recorded PM mid (`config.BACKTEST_HALF_SPREAD` / `BACKTEST_BOOK_DEPTH`) — historical book depth is not archived. |
| `backtest/broker.py` | Shared virtual pool, fills via production `polymarket_fills.simulate_fill`, taker fee, symmetric slippage band (BUG #28), shared-pool concentration cap (BUG #27), resolution. |
| `backtest/runtime.py` | Context manager that swaps the DB-backed bankroll / Kelly / learned-bias / lane-override hooks in `bots.base_bot` for backtest-local ones — a replay can never touch `bot_arena.db` runtime state. |
| `backtest/engine.py` | Tick loop (default one decision per 1-min candle, the harness's decision cadence): rebuilds the `market` + `signals` dicts exactly as `arena/signals.py` shapes them, calls each bot's real `make_decision`, one trade per (bot, market). |
| `backtest/metrics.py` | Expectancy, WR, profit factor, per-trade Sharpe, max drawdown, break-even gap, per-bot/side/regime splits (vol regime, drift band, entry bucket, time bucket), per-signal contribution. |
| `backtest/walkforward.py` | Train-on-A / test-on-B folds. "Training" = slate SELECTION (what evolution does live): rank bots on the train window, measure the top-k slate out of sample vs the all-bots baseline. |
| `backtest/report.py` | Console report, JSON to `backtest/reports/`, optional `backtest_runs` DB row (dashboard: `GET /api/backtests`). |

## Honesty caveats (read before trusting a number)

1. **Stale mids** — CLOB `prices-history` is fidelity-1 (minute buckets) and
   mids lag the BTC trajectory the replay also sees, so the replayed WR is an
   **optimistic upper bound** (same caveat as `tools/validate_signals.py`).
   Use results for ordering/sign, regime splits and relative bot ranking; the
   live DB is ground truth for absolute P&L.
2. **Synthetic depth** — fills walk an assumed ladder; stress the liquidity
   assumption via `config.BACKTEST_HALF_SPREAD` / `BACKTEST_BOOK_DEPTH`.
3. **Non-compounding by default** — Kelly sizes off the fixed initial
   bankroll so P&L reads as edge, not compounding (`--compound` restores
   arena-like pool sizing).
4. Killed lanes (pm/cvd/obi) are fed neutral zeros exactly as their live
   weight is zero; fut/xasset are stale-neutral. `vol_regime`/`technicals`
   run the production compute off the reconstructed candle stream.

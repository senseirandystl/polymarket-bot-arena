# Polymarket Bot Arena

An automated trading bot arena that runs competing strategies on Polymarket's BTC 5-minute up/down markets. Paper mode simulates against **real Polymarket order books** (depth-walked fills, real taker fees, real resolutions); live mode submits real CLOB orders. Directional bots evolve every few hours — the bottom performers are replaced by mutated copies of the top ones.

## How It Works

**The default slate is 7 bots** (selectable at startup — see below):

| Bot | Strategy | Description |
|-----|----------|-------------|
| `momentum-v1` | Trend following | Drift anchor + heavy BTC/PM momentum and flow lanes |
| `phantom-v1` | Trend following (fast) | EMA 9/26 trend filter + 10-candle breakout |
| `meanrev-sl25-v1` | Mean reversion | Fundamentals-only model (near-pure drift); fades price moves BTC doesn't back |
| `hybrid-v1` | Ensemble | Weighted vote over momentum / mean-rev / sentiment sub-analyzers |
| `arbitrage-v1` | Market-neutral | Buys YES+NO share-matched when the depth-walked pair cost < $1 − fees |
| `late-window-maker-v1` | Late-window | Final-150s entries, side picked by drift conviction |
| `fee-zone-maker-v1` | Fee-zone | Quotes the 56–86¢ zone, only when drift backs the side |

Additional selectable strategies: plain mean-reversion, a 2× take-profit variant, a price-zone **sniper**, and a sentiment (in-market flow) bot.

**Signals.** Each bot computes a model probability from normalized lanes — **BTC drift from the window's "price to beat"** (the validated fundamental: Binance open at `eventStartTime`), BTC/Polymarket momentum, CVD executed-flow, and its own strategy thesis — weighted per-strategy. Fair value is a market-vs-model blend: `fair = mid + trust · (P_model − mid)`, so **edge exists only where the model disagrees with the price**. Bots never trade against a non-trivial drift reading, and never fade the market on an ignorant model. Candidate signals must pass the offline validation harness (`tools/validate_signals.py` — real resolved markets, net-edge after price+fee) before earning a live weight.

**Sizing.** Pure fractional-**Kelly**: `bet = fraction × edge/(1−price) × live bankroll`. The Kelly fraction (default 0.25) is editable live in the dashboard **Settings** tab; there are no per-trade caps in paper mode (the shared pool is the only spend limit). Live mode keeps a hard per-trade cap.

**Evolution.** Directional bots are ranked periodically; the bottom performers are replaced with mutated copies of the winners. The arbitrage and maker bots are evolution-exempt.

## Architecture

```
arena.py               # Coordinator: interactive startup, threads, evolution
arena/                 # Trader tick, discovery, market-data warmer, session filter
bots/                  # base_bot (model-blend + Kelly sizing) + one file per strategy
venues/                # paper (order-book sim vs real CLOB) / live (real orders)
signals/               # BTC price feed, strike/drift, order-flow, clean-tick
polymarket_markets.py  # Discovery (Gamma), books/prices (CLOB), resolutions
polymarket_fills.py    # Depth-walked fill simulation + canonical taker fee
tools/validate_signals.py  # Offline signal-validation harness (run before weighting anything)
dashboard/             # FastAPI backend + web dashboard (port 8501)
db.py                  # SQLite: trades, bot configs, evolution, settings
```

## Setup

### Prerequisites

- Python 3.10+ — no accounts or API keys needed for paper trading (market data is public)

### Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Run

Use the `bin/arena` wrapper whenever you want to start the stack from a terminal:

1. Auto-selects the project-local venv's Python interpreter — system `python3` doesn't have `cryptography`, but the wrapper picks `<repo>/.venv/bin/python3` so you stop hitting `ModuleNotFoundError`.
2. Auto-spawns `dashboard/server.py` in the background if port 8501 isn't already serving, then waits up to ~3s for the endpoint to come up before launching `arena.py`.

```bash
# From the repo root (most common)
./bin/arena

# Or if you've symlinked it into ~/bin/ (absolute symlink only — see header
# in bin/arena for the relative-symlink caveat)
arena
```

On an interactive terminal the arena asks **Continue** (resume the previous slate) or **Start fresh** (wipe + choose bots: Enter for the 7-bot default, or a manual selection like `1,3,5` / `1-6`). Under launchd there is no prompt — it resumes the existing DB config.

Useful env-var overrides:

```bash
ARENA_NO_DASHBOARD=1 ./bin/arena     # already have the dashboard running on :8501 (e.g. via launchd) — leave it alone
DASHBOARD_PORT=8502     ./bin/arena   # probe + spawn on a non-default port (both sides honor the same env var)
DASHBOARD_LOG=$HOME/var/log/arena-dash.log ./bin/arena  # override the per-run dashboard log path
```

The wrapper also sets `PYTHONUNBUFFERED=1` so output line-flushes when redirected to a log file — without it, `bin/arena > /tmp/run.log` looks empty until ~4KB of output accumulates (same env var the launchd plists set).

If you can't or don't want to use the wrapper, the equivalent calls are the venv python explicitly:

```bash
# Start the arena (paper trading)
.venv/bin/python3 arena.py # MacOS
.venv\Scripts\python.exe arena.py # Windows

# Start the dashboard (separate terminal)
.venv/bin/python3 dashboard/server.py # MacOS
.venv\Scripts\python.exe dashboard\server.py # Windows

# Open http://localhost:8501  (HTTP-Basic auth — creds at the top of dashboard/server.py)
```

For persistent (auto-restarting) operation on macOS, symlink and load the two launchd plists in the repo root — see `CLAUDE.md` → *launchd Services*.

## Dashboard

Real-time web dashboard (port 8501) showing P&L stats, per-bot performance and win rates, active market countdowns, recent trades, evolution history, and entry-price bucket ROI. The **Settings** tab lets you:

- **Paper Balance** — top the shared virtual USDC pool up to any figure (history and open positions preserved)
- **Kelly Fraction** — the live sizing multiplier (0.25 = quarter-Kelly, 1.0 = full Kelly); picked up by the arena within seconds
- **Live credentials** — Polymarket keys, only needed for live mode

## Paper vs Live Trading

The system starts in **paper mode**: one shared virtual USDC pool, fills simulated by walking the real CLOB book (slippage + taker fees included), resolved against real market outcomes. To go live: add your Polymarket credentials in Settings and toggle the bot(s) to live. Live mode keeps a hard per-trade cap (`config.LIVE_MAX_POSITION`, default $10) and daily loss limits.

## License

MIT

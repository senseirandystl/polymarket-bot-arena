# Polymarket Bot Arena — Developer Guide

## What This Is

An automated trading bot arena that runs competing bots on **Polymarket's** BTC 5-minute up/down markets. The **default slate is 7 bots** (four directional defaults + the market-neutral **arbitrage** bot + two **maker** bots — late-window and fee-zone); a terminal launch can instead select any subset of strategies (see **Startup flow** below). The maker bots are first-class members of the slate but run on the discovery-cycle (maker) cadence, not the 1s trader tick. Directional bots evolve every 4 hours — the bottom performers are replaced by mutated copies of the top ones; the arbitrage bot is **evolution-exempt** (`arena.EVOLUTION_EXEMPT_TYPES`) and the maker bots are excluded from evolution too (they are partitioned out of the trader/evolution list in `main_loop`). **Paper mode simulates against real Polymarket order books** (discovery, prices, depth-based fills, fees, resolution — everything except order submission); **live mode** submits real CLOB orders. Simmer has been fully removed (its 5-min market feed was inconsistent and its free tier capped at 50 buys/day). See [BUG_HISTORY.md](./BUG_HISTORY.md) #10.

### Startup flow (terminal launches only — `arena.py` / `bin/arena`)
On an interactive tty, `arena/startup.py` runs before the threads boot. If the DB holds a previous run it asks **Continue** (resume the exact prior slate) or **Start fresh** (wipe DB rows via `db.wipe_all()` + truncate `logs/*.log`, then choose bots). Bot choice is **Default** (Enter → the 7-bot slate incl. both makers) or **Manual** (numbered strategy menu — now includes the two maker bots as selectable entries; accepts `1,3,5`, `1-6`, or a mix → launches exactly those). Under launchd / any non-tty parent there is no prompt — it silently resumes the existing DB config, so the service never blocks.

## Current State (v4 — Feb 15, 2026)

**GitHub:** https://github.com/senseirandystl/polymarket-bot-arena.git (branch: main)

### Performance (historical baseline)
Absolute numbers from the Feb 2026 v4 baseline (276 resolved trades, total P&L `-52.10`, per-bot WR/P&L split) cannot be reproduced against the current `<repo>/bot_arena.db` — the historical source file is not in the repo tree and the live DB starts fresh. Once ~50+ trades accumulate against the v4 fixes (consensus guard, bet sizing cap, aggression normalisation), re-run the **Priority 3** query below to recompute. Qualitative takeaways from that baseline (the ones that actually drive arena design) are preserved in **Key Data Insights** directly underneath — they remain valid guidance while the live dataset rebuilds.

### Key Data Insights (use these for future iterations)
- **Market price is the strongest signal** — when YES is priced >65c, YES wins ~100% of the time
- **Contrarian/mean-reversion strategies lose money** in 5-min markets
- **Confidence 0.30-0.50 is the sweet spot** — 67.9% WR, +$48 total
- **Confidence >0.50 LOSES money** — 48.6% WR but large bet sizes = big losses
- **NO bets had 44.9% WR vs YES at 49.2%** in the Simmer-era data — this was the
  basis for the old blanket NO ban, now **removed** (BUG_HISTORY #20). That stat
  came from Simmer's inconsistent 5-min feed and a YES-centric decision path, so
  it did not measure a fair NO decision. NO is now traded on a cost-adjusted
  per-side net edge; re-evaluate NO vs YES WR once Polymarket-native trades
  accumulate (Priority 3 query).
- **Buying cheap YES (<40c) against market consensus = 0-10% WR** (catastrophic)
  — now enforced symmetrically for both sides via the consensus guard (0.35).

### What's Running
- **Arena process:** launchd service `com.polymarket.botarena` (loadable via `launchctl load -w ~/Library/LaunchAgents/com.polymarket.botarena.plist`; auto-restarts on crash via `KeepAlive`). Check status with `launchctl list | grep polymarket`.
- **Dashboard:** launchd service `com.polymarket.dashboard` on FastAPI port 8501 (loadable via `launchctl load -w ~/Library/LaunchAgents/com.polymarket.dashboard.plist`). For **manual/terminal runs on any OS**, `arena.py`'s `start_dashboard()` now auto-manages it: it probes `http://127.0.0.1:8501/api/status`, and if nothing is answering it spawns `dashboard/server.py` with `sys.executable` (the same venv interpreter running the arena — `.venv/bin/python3` on macOS/Linux, `.venv\Scripts\python.exe` on Windows), waits up to ~30s for uvicorn to bind, then opens the browser only once the server responds. The spawned child is terminated via an `atexit` hook when the arena exits (no orphan on :8501), and its stdout/stderr are captured to `<LOG_DIR>/dashboard.log`. If the port is *already* served (launchd service, or a manual `dashboard/server.py`), `start_dashboard()` detects it and does **not** double-spawn. Set `ARENA_NO_DASHBOARD=1` to disable auto-spawn (e.g. when the launchd service owns the dashboard). The dashboard is gated by HTTP-Basic auth (`admin` / `Thor` — defined as `DASHBOARD_USER` / `DASHBOARD_PASS` at the top of `dashboard/server.py`); the browser will prompt for credentials on first visit.
- **Remote access:** localtunnel (not persistent, needs manual restart: `npx localtunnel --port 8050`)
- **Price feed:** Binance WebSocket for BTC/USDT 1-min candles

> **Deployment note:** The two plists in `~/Library/LaunchAgents/` are symlinks back to the project tree (see [launchd Services](#launchd-services) below), so the repo is the single source of truth — `git pull` automatically propagates plist edits. Logs live in `~/Library/Logs/`, not in the repo.

### Simmer API Keys (4 accounts, slot-based)
Stored at `~/.config/simmer/bot_keys.json` — keys mapped to slot_0 through slot_3. When evolution kills a bot, the replacement inherits the dead bot's slot (and API key). Default key at `~/.config/simmer/simmer_api_key.json`.

## Architecture

### Signal Hierarchy (make_decision in base_bot.py)
```
combined = (
    market_price_edge * 0.50    # Strongest: follow the market price (edge = fair − mkt)
    + btc_momentum * 0.15       # BTC spot momentum (Binance candles)
    + pm_momentum * 0.10        # Polymarket in-market YES price momentum
    + strategy_signal * 0.15    # Per-bot strategy differentiation
    + obi_signal * 0.10         # Order-book imbalance (resting bid vs ask depth)
    + cvd_signal * 0.10         # Cumulative volume delta (market buys − sells)
    + learning_bias * variable  # Grows from 5% to 30% weight with data
)
```
OBI + CVD (`signals/orderflow_signals.py`) are the two order-flow reads the
profitable-bot research favors over price-history indicators — they describe
pressure that hasn't hit the price yet. OBI is computed once per discovery cycle
from the Up-token CLOB book; CVD is fetched per market from the data-api trade
tape (`data-api.polymarket.com/trades`), cached ~20s. Both are in `[-1, 1]`
(positive = upward/YES) and clamped into the secondary-lane band before weighting.
Weights are tunable via `config.SIGNAL_WEIGHT_OBI` / `SIGNAL_WEIGHT_CVD`.

**Two-sided (YES/NO) net-edge selection.** The `combined` sum above is
reinterpreted as a **fair YES probability**: `fair_yes = yes_mid + price_tilt +
alpha`, where `price_tilt = (yes_mid−0.5)·aggression·config.K_TILT` is the
favorite-following lane (the old `market_price_edge*0.50`, with `K_TILT=0.5`
keeping it numerically identical at aggression 1.0) and `alpha` is the summed
secondary lanes. The bot then computes a **cost-adjusted net edge on BOTH sides**
— `edge = prob − side_price − taker_fee(1, side_price)` for YES and NO, each
using that side's own book mid (`market["no_price"]` for NO) and the canonical
`polymarket_fills.taker_fee` — and **buys whichever side has the larger positive
edge** above a per-strategy `BaseBot.MIN_EDGE` floor (skips if neither clears it).
YES and NO are evaluated on their own prices/fees, so NO is a first-class
decision, not a mirror of YES; when the two mids are exactly complementary the
comparison reduces to the old sign decision. A directional bot takes at most one
side per market (argmax) — arbitrage is the only two-legged bot. Sizing,
`entry_price`, and the slippage limit all key off the chosen side's price. The
old blanket **NO ban is gone** (see BUG_HISTORY #20).

### Safeguards
- **Symmetric side guards:** A bot never buys a *chosen* side priced above
  `config.HIGH_PRICE_GUARD` (0.72 — bad risk/reward) or below
  `config.CONSENSUS_GUARD` (0.35 — fighting strong consensus). Both key off the
  side actually being bought, so YES and NO are protected identically (replaces
  the old YES-only NO-ban + one-sided consensus guard).
- **Session-timing skip:** Sit flat during high-flip session handovers (NYSE
  open/close, ET) — the trader gates all taker bots once per tick
  (`arena/session_filter.py`, `config.SESSION_SKIP_*`). "Build the skip, default
  flat." Skip reasons are tallied in shared state and flushed to `arena_state`
  every 30s (dashboard `/api/skips`).
- **Clean-tick guard:** Reject implausible single-tick price jumps (>15¢)
  (`signals/clean_tick.py`, `config.CLEAN_TICK_*`); applied to both YES and NO
  prices in the market-data warmer (and the fallback `refresh_price`).
  Drop-first-tick is OFF (`CLEAN_TICK_DROP_FIRST=False`) — REST/warmer reads are
  already current, so dropping the first would blank a new market for a cycle.
- **Shares-first sizing:** `make_decision` sizes in exact shares first, then
  derives USD (`amount = target_shares × price`) — never USD → shares, which
  rounds away PnL at low prices. `target_shares` is returned on the signal.
- **Entry-price-bucket ROI:** `db.get_entry_price_buckets()` + dashboard
  `/api/entry-buckets` report count/WR/ROI and the **break-even gap** (WR − avg
  entry) per bucket — a high WR bought at high prices still loses; the gap must
  be ≥5¢ to break even, ≥10¢ to profit.
- **Bet sizing cap:** Confidence capped at 0.45 for sizing (prevents overconfident large bets)
- **No stale expiry:** Pending trades stay pending until the market actually resolves (Simmer can take up to a day). The old 1h auto-expire was removed — it threw away real outcomes. See BUG_HISTORY #10.
- **Daily loss limits:** Uncapped for paper trading (was $10/bot, $25 total)
- **Dedup:** Loads recent (bot, market) pairs from DB to prevent duplicates across restarts

### Market-data warmer (`arena/market_data.py`) — the hot-path fast lane
One background thread (`MarketDataWarmer`, `config.MARKET_DATA_INTERVAL_SEC`, default **1s**) is the **single owner of all per-market network reads**: YES+NO books, YES+NO prices, OBI, CVD, PM in-market momentum → written into a shared `MarketDataStore` (`market_data.store()`). The Trader's 1s tick and the arbitrage bot read this **warm cache** (zero network on the hot path); `build_combined_signals(..., warm=...)` takes the warm values directly. So every trading-decision input stays **≤1s fresh**. The per-signal feed caches (CVD tape, PM history) are now just coalescing guards — TTL `config.SIGNAL_CACHE_TTL_SEC` (≈0.8s) so the warmer refreshes them every cycle. Cost: ~4 CLOB/data-api calls/sec for the live market; dial `MARKET_DATA_INTERVAL_SEC` up to back off. The maker section (20s hook) still uses the cold path (`warm=None`).

**Hot-path DB caches:** `make_decision` used to run two SQLite queries per bot per second (resolved-trade count for the learning weight; the `bot_learning` table for the learned bias). Both now cache for `config.HOTPATH_CACHE_TTL_SEC` (30s — they only change on resolution); `learning.record_outcome` busts the bias cache. `db.get_bot_mode` caches for `config.BOT_MODE_CACHE_TTL_SEC` (3s) and is invalidated on `set_bot_mode`/`retire_bot`.

### Market data (Polymarket-native)
`polymarket_markets.py` owns all market data (public, no auth):
- **Discovery:** Gamma `/events?series_id=10684` ("BTC Up or Down 5m") → normalized market dicts; the live window is picked by real `resolves_at` (`market_utils.select_current_market`).
- **Fresh prices / depth:** CLOB `/book` — normalized so `best_bid`/`best_ask` are correct (the raw feed is worst→best ordered, a trap). Consumed on the hot path via the warmer above, not fetched per bot.
- **Resolution:** `recent_resolutions()` builds a `condition_id → outcome` map from the series' closed events' `outcomePrices` (`["1","0"]`=Up). The CLOB `tokens[].winner` flag is unreliable — do not use it.

### Execution venues (paper vs live)
Order placement is split by venue so the two never intermix — `base_bot.execute()` picks an engine via `venues.get_engine(mode)`. Both use identical pricing/fill/fee math (`polymarket_fills.py`):
- **Paper** (`venues/paper.py`, `fill_source='paper_sim'`): **simulates against the real CLOB order book** — walks the asks for depth/slippage, applies the Polymarket taker fee, and never submits. All paper bots share ONE virtual USDC pool. `available = bankroll + realized_paper_pnl − reserved_open_cost` (`db.get_paper_available`); a bot can't spend cash the pool lacks. The dashboard Settings "Balance" field **tops the pool up to the entered figure** via `db.topup_paper_bankroll` — it back-solves the underlying `bankroll` so `available` equals what you type, *preserving* trade history and open positions (entering $200 when the pool is at $45 sets available to exactly $200). Resolves against the real market outcome.
- **Live** (`venues/live.py` → `polymarket_client.py`): real CLOB `create_market_order`/`MarketOrderArgs` (auto tick-size / neg-risk / fee). Uses the real wallet USDC balance. Fully wired but only used when a bot's `trading_mode` is `live` (arena starts in paper).
- **Fees:** `polymarket_fills.taker_fee()` is the **single source of truth** for fee math — makers free, takers pay `feeRate × shares × p × (1−p)` per the [official Polymarket docs](https://docs.polymarket.com/trading/fees) (symmetric around 50¢; crypto tier `config.POLYMARKET_TAKER_FEE_RATE = 0.07`, peaking at $1.75/100 shares at 50¢). Any bot needing a fee estimate must call this, never re-derive it (see BUG_HISTORY #17). Factored into resolved P&L (`payout − amount − fee`). Trade columns `fill_source`/`entry_price`/`fee` record each fill.

### Per-Strategy Differentiation
| Strategy | Aggression | Prior | Min Confidence |
|----------|-----------|-------|----------------|
| momentum | 1.2 (follows price strongly) | 0.52 (slight YES) | 0.01 (trades almost everything) |
| mean_reversion | 0.95 (nearly follows, was 0.6) | 0.48 (slight NO) | 0.06 |
| sentiment | 1.0 (neutral) | 0.50 | 0.03 |
| hybrid | 1.0 (neutral, was 0.9) | 0.50 | 0.05 |
| arbitrage | n/a — **overrides** `make_decision`/`execute` (market-neutral, two-legged) | n/a | n/a |

**Every bot trades both sides now (BUG_HISTORY #20).** The four directional bots (+SL/TP variants) pick YES or NO via the two-sided net-edge comparison above. The **sniper** overrides `make_decision` but applies its cheap/strong price zones symmetrically to the NO token (down-momentum confirmed). Both **makers** quote whichever side's price is in their band — late-window on down-momentum, fee-zone on the symmetric fee zone. A directional/sniper/maker bot takes **at most one side per market**; only arbitrage is two-legged. The sniper/maker NO branches are symmetric mirrors not yet validated by live NO data (recheck once NO trades accumulate).

**Arbitrage bot** (`bots/bot_arbitrage.py`): buys **both** YES and NO on one market when the market-neutral edge clears `config.ARBITRAGE_MIN_MARGIN` per matched share pair, locking in `1 − cost` regardless of outcome. **Two things make the edge real (see BUG_HISTORY #11):** (1) the edge is measured from the **depth-walked VWAP** of filling the intended size on each book — *not* the thin top-of-book `best_ask`, which lies about cost once you size past one share; (2) both legs are **share-matched** — sized to the *same* share count (the smaller of the two books' fillable depth, capped by `max_pos` and the shared bankroll), and filled by an exact-share path (`polymarket_fills.simulate_fill_shares` → `engine.place(..., target_shares=...)`) so the position is genuinely neutral. It bypasses the directional signal stack/guards, places both legs in one `execute()` (success only if **both** fill — a one-legged fill is logged as naked risk), reads warm books from the market-data store, and is evolution-exempt.

### Learning System
- Features extracted at TRADE TIME (not resolution time — this was a critical bug fix)
- Stored in `trade_features` column in trades table
- Learning records win/loss by feature bucket (price level + momentum)
- Weight ramps from 10% to 60% as bot accumulates resolved trades
- Learning data in `bot_learning` table

## Key Files

```
arena.py              # Coordinator: interactive startup, boot threads, evolution cycle
arena/market_data.py  # MarketDataWarmer (1s) — sole owner of per-market network reads → warm store
arena/startup.py      # Interactive continue/fresh + default/manual bot selection (tty only)
bots/base_bot.py      # BaseBot with make_decision() signal hierarchy + execute() → venue engine
bots/bot_momentum.py  # MomentumBot (follows trends)
bots/bot_mean_rev.py  # MeanRevBot (was contrarian, now nearly neutral)
bots/bot_sentiment.py # SentimentBot
bots/bot_hybrid.py    # HybridBot
bots/bot_arbitrage.py # ArbitrageBot: market-neutral YES+NO cross-book arb (evolution-exempt)
venues/__init__.py    # get_engine(mode) + TradeResult — paper vs live split
venues/paper.py       # PaperEngine: simulate fills vs real CLOB book + shared bankroll
venues/live.py        # LiveEngine: Polymarket CLOB order placement
polymarket_markets.py # Market data: discovery (Gamma), book/prices, resolution
polymarket_fills.py   # Order-book fill simulation + taker fee formula
polymarket_client.py  # CLOB client: market/limit orders, balances, order book
config.py             # All config: paths, limits, evolution interval, API URLs, SIMMER_MIRROR_ENABLED
db.py                 # SQLite: trades (+ fill_source/entry_price), bot_configs, evolution, bot_learning
learning.py           # Feature extraction, bias calculation, outcome recording
signals/price_feed.py # Binance WS for BTC candles (staleness detection)
signals/sentiment.py  # Sentiment signals
signals/orderflow.py  # Order flow signals
dashboard/server.py   # FastAPI dashboard backend
dashboard/index.html  # Dashboard frontend
copytrading/          # Wallet tracking + copy trading (not actively used)
```

### launchd Services
The plists in `~/Library/LaunchAgents/` are symlinks back to the project tree, so the repo is the single source of truth — `git pull` automatically propagates plist edits without manual copy or relink.

```
~/Library/LaunchAgents/com.polymarket.botarena.plist   →  com.polymarket.botarena.plist  (in repo)
~/Library/LaunchAgents/com.polymarket.dashboard.plist  →  com.polymarket.dashboard.plist (in repo)
~/Library/Logs/com.polymarket.botarena.out.log, com.polymarket.botarena.err.log
~/Library/Logs/com.polymarket.dashboard.out.log, com.polymarket.dashboard.err.log
```

To reload a service after editing its plist in the repo:

```bash
launchctl unload ~/Library/LaunchAgents/com.polymarket.botarena.plist
launchctl load   -w ~/Library/LaunchAgents/com.polymarket.botarena.plist
# (same for com.polymarket.dashboard)
```

Why symlinks instead of copies: deploying via **copy** would let the two plists drift out of sync — someone updates the repo plist and forgets to re-copy it to `~/Library/LaunchAgents/`, or vice versa. The symlink removes that failure mode: plist edits land in the repo via `git pull` and launchd always reads the same file. Independently of the deployment mechanism: this does *not* protect against in-repo authoring bugs — the original `/Users/ben/...` mistake would have broken a symlink just as silently as a copy; that was a content bug, not a deployment one.

### Python & Dependencies

The project runs under a project-local virtualenv at `<repo>/.venv` so the launchd services always see the same Python + the same packages — regardless of which system Python is in `$PATH` on the host Mac.

Install (one-time):

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install -r requirements.txt
```

For *manual* invocations from the shell (e.g. `python3 arena.py`), the same rule applies: use `<repo>/.venv/bin/python3` and NOT system `python3`. The encrypted credential store in `credentials_store.py` requires the `cryptography` package (added to `requirements.txt` in v5), which is installed into `.venv/` via `.venv/bin/pip install -r requirements.txt` above but is not available on the host's default `python3` — that's why a bare-system invocation fails with `ModuleNotFoundError: No module named 'cryptography'` on a fresh clone. Under launchd this never bites because the plists already point at `.venv/bin/python3`; for manual runs, just call `.venv/bin/python3 arena.py` (or `source .venv/bin/activate && python3 arena.py`). If a manual run ever does hit that traceback, the error message from `credentials_store.py` will tell you the same fix.

The launchd plists pin `ProgramArguments[0]` to `<repo>/.venv/bin/python3` (not `/usr/bin/env python3` — `~` and PATH lookups under launchd are brittle), and set `EnvironmentVariables.PYTHONUNBUFFERED=1` so log output is line-flushed (otherwise stdout/stderr redirect to a log file is fully buffered and the logs stay empty until the buffer fills). `requirements.txt` pins none of fastapi / uvicorn / requests / websocket-client / websockets / py-clob-client / py-order-utils / cryptography — versions stay unpinned to match the airy `pip install ...` line in README.md; add pins here if you need reproducibility.

> **Heads-up for fresh clones:** the plist Python paths are hardcoded to `/Users/randalljames/Documents/GitHub/pba/.venv/bin/python3` (this specific user's home directory), so on any other host `launchctl load` will fail silently with exit code 78 ("no such file") — same class of failure as the original `/Users/ben/...` path bug documented in the `launchd Services` block above. One-shot sed-replace the python path before `launchctl load -w`. The pattern targets **only** the `.venv/bin/python3` line (the log paths under `~/Library/Logs/` are intentionally left absolute — `launchd` does *not* expand `$HOME` in plist values, so a broader `s|/Users/randalljames|$HOME|` would silently break log writes):
>
> ```bash
> sed -i '' "s|/Users/randalljames/Documents/GitHub/pba/.venv/bin/python3|$HOME/Documents/GitHub/pba/.venv/bin/python3|" com.polymarket.*.plist
> grep -n '\.venv/bin/python3' com.polymarket.*.plist   # verify both plists now point at your venv
> ```

### Manual invocation: the `bin/arena` wrapper

For TERMINAL invocations (when you don't want to use the launchd services), use the `bin/arena` shell wrapper at the repo root instead of calling `arena.py` directly. It does more than pin the venv python — it brings up the full stack from a single command:

```bash
./bin/arena                # from the repo root
bin/arena                  # relative-path
arena                      # if symlinked into ~/bin/ (ABSOLUTE symlink only)
```

What it does under the hood:

1. **Symlink-aware path resolution.** Pure-bash `BASH_SOURCE` walk that works whether invoked as `./bin/arena`, `bin/arena`, or via a `$PATH` symlink pointing back at this file (e.g. `~/bin/arena -> /abs/path/to/repo/bin/arena`). **Relative** symlinks (e.g. `~/bin/arena -> repo/bin/arena`) don't work — they resolve through the symlink's own directory and end up at `~/bin/repo/...`, not the real repo. Use absolute paths when linking: `ln -sfn /abs/path/to/repo/bin/arena ~/bin/arena`.
2. **Venv sanity-check.** Prints a friendly install hint and exits 1 if `<repo>/.venv/bin/python3` is missing (instead of failing later with bash's cryptic `python3: No such file or directory` from inside an exec call).
3. **PYTHONUNBUFFERED=1** exported in the wrapper's environment, so stdout line-flushes when redirected to a log file (matches the launchd plists — without this, `bin/arena > /tmp/run.log` looks empty until ~4KB of output accumulates).
4. **Orphan reaping.** Before doing anything else, reads `<repo>/.dashboard.pid`. If a previous `bin/arena` was SIGKILL'd (`kill -9`, Activity Monitor "Force Quit"), the cleanup trap below never fires and `dashboard/server.py` can be left bound on :$DASHBOARD_PORT indefinitely. The reap logic verifies the PID's `ps -o command=` still matches `dashboard/server.py` (guards against PID recycling onto an unrelated process) and kills it if so — plus gives uvicorn up to ~1s to release the port.
5. **Dashboard auto-spawn.** Probes `http://localhost:${DASHBOARD_PORT}/api/status`. If alive, leaves the existing dashboard alone (avoids double-bind collisions). If not, spawns `dashboard/server.py` in the background, waits up to ~3s for the endpoint to come up, then runs `arena.py` in the foreground. Per-run logs go to `${DASHBOARD_LOG:-<repo>/dashboard.log}` — note the `>` not `>>`, so each fresh run gets a clean log correlating to that run only.
6. **Cleanup trap (`trap cleanup EXIT INT TERM`).** When the wrapper exits (clean stop or Ctrl-C), kills the backgrounded dashboard child (graceful SIGTERM, since the wrapper shares a process group with it) and removes the pidfile so the next invocation starts clean.

Env-var interfaces (all set in the calling shell — the wrapper reads them at runtime, doesn't accept CLI flags):

| Env var | Default | Effect |
|---|---|---|
| `ARENA_NO_DASHBOARD` | unset (= `0`) | Skip step 5 entirely. Use when you manage the dashboard yourself, e.g. `com.polymarket.dashboard.plist` is loaded under launchd and you don't want the wrapper to fight it. |
| `DASHBOARD_PORT` | `8501` | Probe + spawn port. The dashboard server itself honors the same env var (read at the top of `dashboard/server.py`), so on both ends the override sticks. |
| `DASHBOARD_LOG` | `<repo>/dashboard.log` | Path for the spawned dashboard's stdout/stderr capture. Truncated per run. |

The HTTP-Basic credentials used by the probe (`admin` / `Thor`) are hardcoded to match `DASHBOARD_USER` / `DASHBOARD_PASS` at the top of `dashboard/server.py`. If those constants change there, update the probe line in `bin/arena` too — the probe will otherwise either false-negative (different creds → curl -u admin:Thor gets 401, still returns 0 — still considered alive, leaves your updated dashboard alone) or false-positive (no creds match → curl gets healthy response but you can't log in from the browser). One site in `bin/arena`, one site in `dashboard/server.py`.

**This wrapper does NOT replace the launchd services** — under steady-state you want KeepAlive handling and persistent logs. Use the wrapper for ad-hoc terminal runs (debugging after a config tweak, testing a strategy change, inspecting a stuck bot). Once you're happy, reload the plist instead.

### Fresh-clone Setup

> ⚠️ This starts the bot trading on completion. Make sure the Simmer keys exist at `~/.config/simmer/simmer_api_key.json` and at `~/.config/simmer/bot_keys.json` (all four `slot_0…slot_3` entries populated) first. Otherwise the loaded program will fail to start and `KeepAlive` will keep relaunching it — the throttle is 30s for the arena and 10s for the dashboard, so the stderr logs (`~/Library/Logs/com.polymarket.botarena.err.log`, `~/Library/Logs/com.polymarket.dashboard.err.log`) fill quickly with `FileNotFoundError` lines.

`WorkingDirectory` is intentionally unset in both plists — `config.py` (`DB_PATH`, `LOG_DIR`) and `dashboard/server.py` (`Path(__file__).parent / "index.html"`) anchor every path on `__file__`, so launchd's default cwd of `/` is harmless. If you specifically want the courtesy-chdir before exec, add `<key>WorkingDirectory</key><string>/your/absolute/repo/path</string>` to both plists manually before `launchctl load -w`.

First-time bootstrap for a new contributor (ensures the logs dir exists, creates the symlinks back to the repo, and loads both services persistently):

```bash
cd /Users/randalljames/Documents/GitHub/pba   # or wherever the repo lives
mkdir -p ~/Library/Logs
ln -sfn "$PWD/com.polymarket.botarena.plist"  ~/Library/LaunchAgents/com.polymarket.botarena.plist
ln -sfn "$PWD/com.polymarket.dashboard.plist" ~/Library/LaunchAgents/com.polymarket.dashboard.plist
launchctl load -w ~/Library/LaunchAgents/com.polymarket.botarena.plist
launchctl load -w ~/Library/LaunchAgents/com.polymarket.dashboard.plist
launchctl list | grep polymarket   # both should show a PID; `-` means it failed
```

Reload one of them after a repo-side plist edit:

```bash
launchctl unload ~/Library/LaunchAgents/com.polymarket.botarena.plist
launchctl load   -w ~/Library/LaunchAgents/com.polymarket.botarena.plist
# (same for com.polymarket.dashboard)
```

### Database
SQLite at `<repo>/bot_arena.db` (= `config.DB_PATH`) — tables: trades, bot_configs, evolution_events, daily_stats, bot_learning, copytrading_wallets, copytrading_trades

## Bug History (avoid re-introducing)

Moved to **[BUG_HISTORY.md](./BUG_HISTORY.md)** to keep this guide lean. Read it before touching the resolver, discovery, learning, or P&L code — it records nine already-fixed bugs (circular learning, the various `$0` P&L causes, stale-trade clogging, and the next-day/15-min market selection bug) and the reasoning behind each fix.

## Next Steps for Iteration

### Priority 1: Let v4 accumulate data
The v4 fixes (consensus guard, bet sizing cap, aggression fix) need 50+ resolved trades with stored features to evaluate. Check after ~2-4 hours of running.

### Priority 2: Verify learning is working correctly
Once trades with stored features start resolving, verify:
```python
# In python3 from trading_bot dir:
import db
with db.get_conn() as conn:
    rows = conn.execute("SELECT * FROM bot_learning ORDER BY updated_at DESC LIMIT 20").fetchall()
    for r in rows: print(dict(r))
```

### Priority 3: Analyze v4 performance
After 50+ resolved trades with features, run the analysis:
```python
import db
with db.get_conn() as conn:
    # Compare pre-v4 vs post-v4 by checking trades after the restart
    rows = conn.execute('''
        SELECT bot_name, side, COUNT(*) as trades,
            SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins,
            ROUND(SUM(pnl), 2) as pnl
        FROM trades WHERE outcome IN ('win','loss') AND trade_features IS NOT NULL
        GROUP BY bot_name
    ''').fetchall()
    for r in rows: print(dict(r))
```

### Priority 4: Future improvements to explore
- **Time-of-day analysis:** Do certain hours have better WR?
- **BTC volatility filter:** Skip trading during low-volatility periods (no edge)
- **Adaptive confidence thresholds:** Adjust min_confidence based on recent WR
- **Ensemble voting (optional):** When 3+ bots agree, increase bet size
- **Live trading readiness:** Once consistently profitable in paper, consider switching to live

### User Incentive
"$10 in tokens for every $100 earned" — both user and bot benefit from profitability.

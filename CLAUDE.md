# Polymarket Bot Arena — Developer Guide

## What This Is

An automated trading bot arena that runs 4 competing bots on Polymarket's BTC 5-minute up/down markets via the Simmer paper trading platform. Bots evolve every 4 hours — the bottom 2 are replaced by mutated copies of the top 2. Each bot has its own Simmer account for independent trading and real performance comparison.

## Current State (v4 — Feb 15, 2026)

**GitHub:** https://github.com/senseirandystl/polymarket-bot-arena.git (branch: main)

### Performance (historical baseline)
Absolute numbers from the Feb 2026 v4 baseline (276 resolved trades, total P&L `-52.10`, per-bot WR/P&L split) cannot be reproduced against the current `<repo>/bot_arena.db` — the historical source file is not in the repo tree and the live DB starts fresh. Once ~50+ trades accumulate against the v4 fixes (consensus guard, bet sizing cap, aggression normalisation), re-run the **Priority 3** query below to recompute. Qualitative takeaways from that baseline (the ones that actually drive arena design) are preserved in **Key Data Insights** directly underneath — they remain valid guidance while the live dataset rebuilds.

### Key Data Insights (use these for future iterations)
- **Market price is the strongest signal** — when YES is priced >65c, YES wins ~100% of the time
- **Contrarian/mean-reversion strategies lose money** in 5-min markets
- **Confidence 0.30-0.50 is the sweet spot** — 67.9% WR, +$48 total
- **Confidence >0.50 LOSES money** — 48.6% WR but large bet sizes = big losses
- **NO bets have 44.9% WR vs YES at 49.2%** — slight YES bias is profitable
- **Buying cheap YES (<40c) against market consensus = 0-10% WR** (catastrophic)

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
    market_price_edge * 0.50    # Strongest: follow the market price
    + btc_momentum * 0.20       # BTC price movement direction
    + strategy_signal * 0.15    # Per-bot strategy differentiation
    + learning_bias * variable  # Grows from 10% to 60% weight with data
)
```

### Safeguards
- **Market consensus guard:** Never bet against prices >65c or <35c
- **Bet sizing cap:** Confidence capped at 0.45 for sizing (prevents overconfident large bets)
- **No stale expiry:** Pending trades stay pending until the market actually resolves (Simmer can take up to a day). The old 1h auto-expire was removed — it threw away real outcomes. See BUG_HISTORY #10.
- **Daily loss limits:** Uncapped for paper trading (was $10/bot, $25 total)
- **Dedup:** Loads recent (bot, market) pairs from DB to prevent duplicates across restarts

### Execution venues (paper vs live)
Order placement is split by venue so the two never intermix — `base_bot.execute()` picks an engine via `venues.get_engine(mode)`:
- **Paper** (`venues/paper.py`): fills are computed **locally** from the real market price (`shares = amount / entry_price`, `fill_source='local_sim'`) and resolved against the real market outcome. This makes paper trading **unlimited** and independent of Simmer's **50-buys/day free-tier cap** (which previously caused mislogged "phantom" fills — BUG_HISTORY #10). Simmer is an opt-in cross-check only via `config.SIMMER_MIRROR_ENABLED` (default off; checks the response's `success` flag before trusting it).
- **Live** (`venues/live.py` → `polymarket_client.py`): Polymarket CLOB via `create_market_order`/`MarketOrderArgs` (auto tick-size / neg-risk / fee). Fully wired but only used when a bot's `trading_mode` is `live` (arena starts in paper). The `fill_source`/`entry_price` trade columns record how each trade filled.

### Per-Strategy Differentiation
| Strategy | Aggression | Prior | Min Confidence |
|----------|-----------|-------|----------------|
| momentum | 1.2 (follows price strongly) | 0.52 (slight YES) | 0.01 (trades almost everything) |
| mean_reversion | 0.95 (nearly follows, was 0.6) | 0.48 (slight NO) | 0.06 |
| sentiment | 1.0 (neutral) | 0.50 | 0.03 |
| hybrid | 1.0 (neutral, was 0.9) | 0.50 | 0.05 |

### Learning System
- Features extracted at TRADE TIME (not resolution time — this was a critical bug fix)
- Stored in `trade_features` column in trades table
- Learning records win/loss by feature bucket (price level + momentum)
- Weight ramps from 10% to 60% as bot accumulates resolved trades
- Learning data in `bot_learning` table

## Key Files

```
arena.py              # Main loop: discover markets, run bots, resolve trades, evolve
bots/base_bot.py      # BaseBot with make_decision() signal hierarchy + execute() → venue engine
bots/bot_momentum.py  # MomentumBot (follows trends)
bots/bot_mean_rev.py  # MeanRevBot (was contrarian, now nearly neutral)
bots/bot_sentiment.py # SentimentBot
bots/bot_hybrid.py    # HybridBot
venues/__init__.py    # get_engine(mode) + TradeResult — paper vs live split
venues/paper.py       # PaperEngine: local-sim fills (unlimited) + optional Simmer mirror
venues/live.py        # LiveEngine: Polymarket CLOB order placement
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

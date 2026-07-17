"""
Polymarket Bot Arena Configuration
"""

import os
from pathlib import Path

# Re-export encrypted credentials helpers so callers can
# `from config import get_credential` (consistent with the rest of the
# codebase) rather than `import credentials_store`. The Simmer API key,
# per-bot keys, and the Polymarket L2 credential bundle all live in the
# encrypted store now; the constants below point at *legacy plaintext
# locations* which were auto-migrated to the store on first run.
from credentials_store import (
    get_credential,
    set_credentials,
    credentials_status,
    is_credential_configured,
    CREDENTIALS_FILE,
    CREDENTIALS_KEY_FILE,
)

# Trading Mode: "paper" (default, uses $SIM) or "live" (real USDC)
TRADING_MODE = "paper"  # MUST start in paper mode

# Polymarket Direct CLOB (live trading + all market data).
# Legacy plaintext location — the active source of truth is the encrypted
# credentials store (CREDENTIALS_FILE above); reads go through get_credential().
POLYMARKET_KEY_PATH = Path.home() / ".config/polymarket/credentials.json"
POLYMARKET_HOST = "https://clob.polymarket.com"
POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"  # discovery + resolution
POLYMARKET_CHAIN_ID = 137  # Polygon

# BTC 5-min up/down markets live under this recurring Gamma series ("BTC Up or
# Down 5m"). Discovery lists this series' open events; the live 5-min window is
# then selected by its real resolves_at timestamp (see arena/market_utils).
POLYMARKET_BTC_5M_SERIES_ID = "10684"

# Taker fee model (makers are never charged). Polymarket's documented taker fee
# is symmetric around 50c: fee_usdc = rate * shares * price * (1 - price). Crypto
# is the highest tier. The rate is isolated here so it can be tuned in one place;
# both paper (simulated) and live use it. See polymarket_fills.taker_fee().
POLYMARKET_TAKER_FEE_RATE = 0.07

# Paper mode is a full simulation against real Polymarket order books (no order
# is submitted). All paper bots share ONE virtual USDC bankroll, set by the user
# in the dashboard Settings tab (arena_state key 'paper_bankroll'); this default
# is used until they set one. Live mode uses the real wallet USDC balance.
PAPER_BANKROLL_DEFAULT = 100.0

# Database
DB_PATH = Path(__file__).parent / "bot_arena.db"

# Target Market: BTC 5-min up/down
TARGET_MARKET_QUERY = "btc"  # Search term for market discovery
TARGET_MARKET_KEYWORDS = ["5 min", "5-min", "5min", "up or down", "up/down"]
BTC_5MIN_MARKET_ID = None  # Will be populated by setup.py

# Risk Limits - Paper Mode (default) — no caps, let bots compete freely
# NOTE (2026-07-17): directional bets are PURE-KELLY sized and no longer capped
# by PAPER_MAX_POSITION — it now only scales the maker/arb position_size_pct
# knobs (via get_max_position()).
PAPER_MAX_POSITION = 50.0  # $SIM sizing base for maker/arb bots
PAPER_MAX_DAILY_LOSS_PER_BOT = 999999.0  # Uncapped for paper
PAPER_MAX_DAILY_LOSS_TOTAL = 999999.0  # Uncapped for paper
PAPER_STARTING_BALANCE = 10000.0  # $SIM

# Risk Limits - Live Mode (stricter)
LIVE_MAX_POSITION = 10.0  # USDC per trade
LIVE_MAX_DAILY_LOSS_PER_BOT = 50.0  # USDC
LIVE_MAX_DAILY_LOSS_TOTAL = 100.0  # USDC

# General Risk Rules (both modes)
# No longer caps directional bets (pure Kelly, 2026-07-17). Still used to
# derive the live-mode notional bankroll for sizing (LIVE_MAX_POSITION / pct).
MAX_POSITION_PCT_OF_BALANCE = 0.10
MAX_TRADES_PER_HOUR_PER_BOT = 60  # Bots trade every 5-min market they find

# Evolution Settings
EVOLUTION_INTERVAL_HOURS = 2
MUTATION_RATE = 0.15  # 15% random adjustment to params
NUM_BOTS = 4
SURVIVORS_PER_CYCLE = 1  # Top 1 survives, bottom 3 replaced
MIN_TRADES_FOR_JUDGMENT = 20   # Bots with fewer resolved trades are immune
MIN_WIN_RATE = 0.65            # 65% WR threshold to survive evolution

# Signal Feed Settings
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
PRICE_UPDATE_INTERVAL_SEC = 1  # Real-time price updates

# --- Arbitrage bot (bots/bot_arbitrage.py) ---
# Classic Polymarket cross-book arb: buy YES and NO on the SAME market whenever
# YES_ask + NO_ask (+ taker fees on both legs) < $1.00 with enough margin. At
# resolution exactly one side pays $1/share, so a matched pair locks in
# 1 - (yes_ask + no_ask + fees) per share regardless of outcome — market-neutral.
# ARBITRAGE_MIN_MARGIN is the required net profit per matched share pair AFTER
# fees; below it the (usually fleeting) edge doesn't clear execution risk.
ARBITRAGE_MIN_MARGIN = 0.02     # min net USDC profit per matched share pair
ARBITRAGE_TARGET_SHARES = 20    # shares per leg to take when an opportunity appears
ARBITRAGE_BOOK_CACHE_SEC = 1.0  # micro-cache on the per-leg book reads (hot path)

# --- Fill slippage guard (all venues) ---
# A bot decides on one order-book snapshot but the fill is simulated/placed a
# moment later against a possibly-moved book. MAX_FILL_SLIPPAGE is how far (in
# ¢) a BUY's realized avg fill price may exceed the price the decision expected
# before the fill is REJECTED (reason "slippage_exceeded"). This kills the class
# of loss where a razor-thin edge (esp. the arbitrage bot's ~1-2¢/pair) is wiped
# out by adverse drift between decision and fill. The arbitrage bot additionally
# re-validates the *combined* edge and fills both legs against the exact snapshot
# it validated (passed to the engine), so its two legs stay atomic.
MAX_FILL_SLIPPAGE = 0.03

# --- Order-flow signal weights (base_bot.make_decision) ---
# Re-weighted from the 2026-07-15 overnight run (460 directional trades):
# measured per-signal predictiveness (confirms-side WR vs contradicts-side WR):
#   CVD  66.9% vs 52.4%  -> the ONE real flow edge      -> weighted up
#   OBI  58.1% vs 66.7%  -> INVERTED (resting-depth fade) -> zeroed out
# CVD = executed aggression (predicts); OBI = resting depth (fades). See
# docs/superpowers/specs/2026-07-15-strategy-rootcause-improvements-design.md.
# OBI re-disabled (2026-07-15): restored at 0.10 with natural sign, but it
# measured anti-predictive AGAIN (confirms-side WR 22% vs contradicts 50%) — the
# same inversion as the pre-#21 clean run. So OBI as computed here (top-of-book
# resting depth) is a FADE signal in this venue, not upward pressure. Kept wired
# at weight 0 pending an OFFLINE validation of the fade sign before any re-enable.
# NOTE (2026-07-16): per-lane weights moved into the per-strategy model
# profiles (bots/base_bot.py STRATEGY_SIGNAL_PROFILE) so strategies genuinely
# differ. SIGNAL_WEIGHT_OBI remains as a GLOBAL kill-switch multiplied onto the
# OBI lane for every strategy — keep 0.0 until a fade-sign OBI is validated
# offline.
SIGNAL_WEIGHT_OBI = 0.0

# --- BTC drift-from-strike ("price to beat") signal (signals/strike.py) ---
# The dominant fundamental for these markets: where BTC sits vs the window's open
# price. Regime-agnostic (favors whichever side BTC is actually on) and time-
# scaled (more decisive near expiry). Fed into fair value at SIGNAL_WEIGHT_DRIFT.
MARKET_WINDOW_SEC = 300           # 5-min window length
DRIFT_VOL_SCALE = 0.0015          # typical BTC move (fraction) over a full window
# RE-ENABLED (2026-07-16) after the #23 blow-up was traced to a MISCALCULATED
# strike (mid-window "first sighting"), not a bad signal. With the accurate
# strike (Binance open @ eventStartTime) the offline harness
# (tools/validate_signals.py, 300 resolved markets, 50% UP base rate) measures
# drift ~76% predictive — symmetric and 86% near expiry. Drift is now weighted
# per-strategy inside STRATEGY_SIGNAL_PROFILE (bots/base_bot.py); it is the
# anchor lane of every strategy's model.

# --- Two-sided (YES/NO) net-edge side selection: MODEL-BLEND fair value ---
# fair_yes = yes_mid + trust * (P_model - yes_mid). Edge exists ONLY when the
# bot's model probability diverges from the market price (market lags BTC) —
# never by construction. This replaced the additive tilt/alpha stack after the
# 2026-07-16 live run (136 resolved trades): the flat +6c favorite tilt cleared
# the MIN_EDGE gate at window open on its own, so every bot bought the 58-65c
# favorite in the first minute (107 early trades, 49% WR, -$79.53; the 60-70c
# bucket alone was -$64.55 at 47% WR — no favorite premium exists at taker
# prices). The net-edge harness (tools/validate_signals.py, PM price history)
# confirms: "buy the favorite" EV is negative above ~0.67 and marginal
# elsewhere, while "follow drift only when the market lags" is the top rule.
# Weight of each strategy's analyze() lean inside P_model (all strategies).
STRATEGY_SIGNAL_WEIGHT = 0.15
# Sanity clamp on P_model.
MODEL_PROB_MIN = 0.02
MODEL_PROB_MAX = 0.98
# Drift veto: a directional bot never buys the side that CONTRADICTS a drift
# reading of at least this magnitude. Live evidence (2026-07-16 overnight run):
# drift-contradicting trades 26% WR / -$55 vs 52% agreeing. Below the floor
# (drift ~ 0) flow-only trades are allowed — they measured break-even.
DRIFT_VETO_MIN = 0.05
# When drift is below the veto floor (flow-only trade), the MIN_EDGE bar is
# multiplied by this — a claim resting purely on the noisy flow/momentum lanes
# must be proportionally stronger (flow-only cheap-side trades ran 29% WR).
FLOW_ONLY_EDGE_MULT = 2.0

# --- Fractional-Kelly bet sizing (base_bot.make_decision) ---
# For a binary market, buying a side at price c with true probability p, the
# growth-optimal bankroll fraction is f* = (p - c)/(1 - c); with our
# fee-adjusted edge (= p - c - fee) that is f* = edge/(1 - price). Full Kelly
# over-bets on estimation error (our p is a model output), so we bet a
# fraction of it. Size therefore scales with edge, odds, AND the live
# bankroll (compounding) — replacing the old flat 5-9.5%-of-max-position
# formula that ignored all three (win avg $3.83 vs loss avg $3.76 overnight).
# This constant is only the DEFAULT: the live value is stored in the DB
# (db.get_kelly_fraction) and editable in the dashboard Settings tab —
# changes take effect within SIZING_BANKROLL_CACHE_SEC, no restart. Bets are
# PURE Kelly (2026-07-17): no per-trade or %-of-balance caps in paper mode
# (the shared-pool gate is the only spend limit); live keeps LIVE_MAX_POSITION.
KELLY_FRACTION = 0.25
# How long make_decision may reuse the last bankroll read (it runs per-bot
# per-second; the pool changes only on fills/resolutions).
SIZING_BANKROLL_CACHE_SEC = 5.0
# Live learning bias: the raw-YES-WR learner was anti-predictive (-24pp) and
# double-counted price. Disabled in live decisions (outcomes still recorded)
# pending the edge-calibrated redesign. See spec R5.
LEARNING_ENABLED = False
# Fallback minimum cost-adjusted edge (probability units) to place a trade.
MIN_EDGE_DEFAULT = 0.02
# Maps the chosen side's edge -> sizing confidence (~0.10 edge -> 0.45 cap).
EDGE_TO_CONFIDENCE = 4.5
# A bot never buys a side priced above HIGH_PRICE_GUARD (bad risk/reward) or
# below CONSENSUS_GUARD (fighting strong market consensus). Symmetric per side.
HIGH_PRICE_GUARD = 0.72
CONSENSUS_GUARD = 0.35

# --- Session-timing skip filter (arena/session_filter.py) ---
# 'Build the skip': sit flat during high-flip session handovers. Defaults are
# the research's known-bad windows (NYSE open/close, in ET). Weekends off by
# default (crypto trades weekends; no v2 weekend data yet). Tighten to the
# arena's own flip-heavy slots once logs accumulate.
SESSION_SKIP_ENABLED = True
SESSION_SKIP_WEEKENDS = False
SESSION_SKIP_WINDOWS_ET = [
    "09:30-10:15",   # NYSE open — highest direction-flip count per window
    "15:45-16:15",   # NYSE close — second flip spike
]

# --- Clean-tick guard (signals/clean_tick.py) ---
# Reject implausible single-tick price jumps and drop the first (possibly stale)
# tick from a fresh token. A real Polymarket YES mid does not move >15¢ between
# two reads a second apart — that is bad data, not a reprice.
CLEAN_TICK_MAX_JUMP = 0.15   # reject a jump larger than this (in probability)
CLEAN_TICK_STALE_SEC = 10.0  # ...unless last good is older than this (real reprice)
# Drop-first-tick is a *WebSocket* hygiene rule (a freshly-opened socket replays
# a stale cached snapshot). We poll fresh REST /midpoint reads, where the first
# read is already current — dropping it would just blank a new market's price
# for a whole cycle (makers then hit `None - price`). Off by default here; the
# jump-rejection above is the part that matters for REST polling.
CLEAN_TICK_DROP_FIRST = False # drop the first tick from a newly-seen token

# Copy Trading Settings
COPYTRADING_ENABLED = True
COPYTRADING_MAX_WALLETS_TO_TRACK = 10
COPYTRADING_POSITION_SIZE_FRACTION = 0.5  # Copy 50% of whale's position size
COPYTRADING_DAILY_LOSS_LIMIT = 50.0     # Max USDC in realized losses per calendar day (wins are unlimited)
COPYTRADING_MAX_TRADES_PER_CYCLE = 5    # Max trades to execute per arena loop cycle
COPYTRADING_MIN_PRICE = 0.40            # Skip trades where whale's entry price < this
COPYTRADING_MAX_PRICE = 0.65            # Skip trades where whale's entry price > this (expensive bets need 65%+ WR to break even)
COPYTRADING_COPY_NO_BETS = False        # Copy NO bets — data shows NO side loses money, skip by default
COPYTRADING_BLOCKED_HOURS_UTC = [22]    # UTC hours to skip entirely (22:00 = -$76 in data)

# Dashboard Settings
DASHBOARD_PORT = 8501
DASHBOARD_HOST = "0.0.0.0"

# Arena Loop Cadences
# Each loop is its own daemon thread; root arena.py starts them all up.  Before
# this split, all four concerns ran in one 15s main_loop which (a) re-scanned
# the same markets every cycle and (b) meant bots only re-evaluated every 15s.
# After the split:
#   - discovery   : ~1-2 HTTPS calls every 20s (window selection only)
#   - market data : all per-market reads (YES+NO books, OBI, CVD, PM momentum)
#                   every 1s in one warmer thread -> shared warm cache
#   - trader      : zero network calls per tick (1s) except on bot.execute
#   - resolver    : 1 HTTPS call every 60s
#   - pos monitor : 0.5s SL/TP exit loop (hard-realtime; see arena/position_monitor.py)
DISCOVERY_INTERVAL_SEC = 20       # Gamma discovery + window selection. 5-min
                                  # windows roll every 300s; 20s keeps the
                                  # current/next selection fresh and turnover
                                  # snappy without hammering the API.
TRADE_LOOP_INTERVAL_SEC = 1.0     # bot eval / trade-execution loop
RESOLVE_INTERVAL_SEC = 60         # trade resolution (Polymarket closed events)
ORDERFLOW_CACHE_SECONDS = 30      # (unused since Simmer removal; kept for compat)

# --- Market-data warmer (arena/market_data.py) ---
# One background thread owns EVERY per-market network read so the trader hot
# path and the arbitrage bot both read warm, in-memory data (zero network on
# the 1s tick). Refreshed for the live market every MARKET_DATA_INTERVAL_SEC so
# all trading-decision inputs — YES+NO prices, both books, OBI, CVD, PM
# momentum — stay <=1s fresh. Lower = fresher but more HTTPS/sec to the CLOB.
MARKET_DATA_INTERVAL_SEC = 1.0

# --- Hot-path DB caches ---
# make_decision runs every 1s per bot and used to issue two SQLite queries each
# time (resolved-trade count for the learning weight, and the bot_learning
# feature table for the learned bias) — data that only changes when a trade
# RESOLVES (~60s cadence). Cache both per bot for this TTL to take the per-tick
# DB load from 2*N_bots queries/sec down to a trickle. get_bot_mode is cached
# separately (shorter TTL) so dashboard live/paper toggles still apply promptly.
HOTPATH_CACHE_TTL_SEC = 30
BOT_MODE_CACHE_TTL_SEC = 3
# The per-signal feed caches (CVD trade tape, PM price history) are coalescing
# guards only now — the warmer is effectively their sole caller and refreshes
# every cycle, so their TTL is kept just under the warm interval.
SIGNAL_CACHE_TTL_SEC = 0.8

# Polymarket enforces a per-order minimum of 5 shares. Bet sizing floors the
# spend so a trade always clears this (5 shares × price × buffer) — otherwise
# small-edge bets get rejected 'below_min_size' and never fill.
POLYMARKET_MIN_SHARES = 5
# How many BTC 5-min markets to pull per discovery cycle (current + next few).
POLYMARKET_DISCOVERY_LIMIT = 6
MAKER_UPCOMING_WINDOW_SEC = 1200  # ≤N seconds in the future the maker section is
                                  # allowed to fall back to (i.e. quote on a
                                  # market whose window hasn't opened yet).
                                  # 1200s = 20min, matches the pre-refactor
                                  # tradeoff: long enough to warm up bid/ask
                                  # ahead of the next window, short enough
                                  # to keep signal convergence meaningful.
STALENESS_DISPLAY_MAX_SEC = 300  # Upper clamp on the staleness value shown
                                  # in the dashboard's Maker Section card.
                                  # Without this, forward clock skew between
                                  # the arena and the dashboard process inflates
                                  # observed staleness ("last arena update
                                  # 5m ago" when it's really 30s ago).  Caps at
                                  # 5min -- enough headroom beyond the 120s
                                  # STALE-display threshold that the card still
                                  # flips to STALE for any snapshot older than
                                  # that, but values shown to operators stay
                                  # honest.  Operates as a sanity ceiling, not
                                  # an STALE policy.

# Logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def get_current_mode():
    """Get current trading mode"""
    return TRADING_MODE


def get_max_position():
    """Get max position size based on current mode"""
    return LIVE_MAX_POSITION if TRADING_MODE == "live" else PAPER_MAX_POSITION


def get_max_daily_loss_per_bot():
    """Get max daily loss per bot based on current mode"""
    return LIVE_MAX_DAILY_LOSS_PER_BOT if TRADING_MODE == "live" else PAPER_MAX_DAILY_LOSS_PER_BOT


def get_max_daily_loss_total():
    """Get max total daily loss based on current mode"""
    return LIVE_MAX_DAILY_LOSS_TOTAL if TRADING_MODE == "live" else PAPER_MAX_DAILY_LOSS_TOTAL


def get_venue():
    """Trading venue — always Polymarket now (paper simulates against its books)."""
    return "polymarket"


def set_trading_mode(mode: str):
    """
    Set trading mode (paper or live)
    NOTE: This only updates the runtime config, not the config.py file
    For persistence, use the dashboard or manually edit config.py
    """
    global TRADING_MODE
    if mode not in ["paper", "live"]:
        raise ValueError("Mode must be 'paper' or 'live'")
    TRADING_MODE = mode
    return TRADING_MODE

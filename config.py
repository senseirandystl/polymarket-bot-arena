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

# Simmer API Configuration
# Legacy plaintext location — kept as a documentation breadcrumb only.
# The active source of truth is the encrypted credentials store
# (CREDENTIALS_FILE above). Use `config.get_credential("simmer_api_key")`.
SIMMER_API_KEY_PATH = Path.home() / ".config/simmer/simmer_api_key.json"
SIMMER_BASE_URL = "https://api.simmer.markets"

# Multi-agent: each bot gets its own Simmer account for independent trading
# Keys are mapped bot_name -> api_key. Falls back to the default key.
# Legacy plaintext location — see SIMMER_API_KEY_PATH note above.
SIMMER_BOT_KEYS_PATH = Path.home() / ".config/simmer/bot_keys.json"

# Polymarket Direct CLOB (for live trading)
# Legacy plaintext location — see SIMMER_API_KEY_PATH note above. Reads in
# the codebase now go through the encrypted store.
POLYMARKET_KEY_PATH = Path.home() / ".config/polymarket/credentials.json"
POLYMARKET_HOST = "https://clob.polymarket.com"
POLYMARKET_CHAIN_ID = 137  # Polygon

# Database
DB_PATH = Path(__file__).parent / "bot_arena.db"

# Target Market: BTC 5-min up/down
TARGET_MARKET_QUERY = "btc"  # Search term for market discovery
TARGET_MARKET_KEYWORDS = ["5 min", "5-min", "5min", "up or down", "up/down"]
BTC_5MIN_MARKET_ID = None  # Will be populated by setup.py

# Risk Limits - Paper Mode (default) — no caps, let bots compete freely
PAPER_MAX_POSITION = 50.0  # $SIM per trade
PAPER_MAX_DAILY_LOSS_PER_BOT = 999999.0  # Uncapped for paper
PAPER_MAX_DAILY_LOSS_TOTAL = 999999.0  # Uncapped for paper
PAPER_STARTING_BALANCE = 10000.0  # $SIM

# Risk Limits - Live Mode (stricter)
LIVE_MAX_POSITION = 10.0  # USDC per trade
LIVE_MAX_DAILY_LOSS_PER_BOT = 50.0  # USDC
LIVE_MAX_DAILY_LOSS_TOTAL = 100.0  # USDC

# General Risk Rules (both modes)
MAX_POSITION_PCT_OF_BALANCE = 0.10  # Never bet more than 10% of balance
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
#   - discovery : up to 2 HTTPS calls every 60s
#   - trader    : zero network calls per tick (1s) except on bot.execute
#   - resolver  : 1 HTTPS call every 60s
#   - pos monitor: 0.5s SL/TP exit loop (hard-realtime; see arena/position_monitor.py)
DISCOVERY_INTERVAL_SEC = 60       # market discovery + orderflow refresh
TRADE_LOOP_INTERVAL_SEC = 1.0     # bot eval / trade-execution loop
RESOLVE_INTERVAL_SEC = 60         # trade resolution + stale-trade sweep
ORDERFLOW_CACHE_SECONDS = 30      # per-market /api/sdk/context refresh window
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
    """Get trading venue based on current mode"""
    return "polymarket" if TRADING_MODE == "live" else "simmer"


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

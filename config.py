"""
Polymarket Bot Arena Configuration
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

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

# --- Shared HTTP retry policy (http_client.request_with_retry) ---
# Bounded retries + exponential backoff for SLOW-cadence reads (discovery,
# resolution, CVD, PM history, strike). NOT applied to the 1s hot-path book/
# midpoint reads — a retry-sleep there would stall the trader tick, and those
# calls are already best-effort with a warm-cache fallback. Worst-case added
# latency per call ≈ backoff_base·(2^0 + 2^1) ≈ 1.2s at the defaults.
HTTP_MAX_RETRIES = 2                       # attempts after the first = 3 total tries
HTTP_BACKOFF_BASE = 0.4                    # seconds; grows 0.4, 0.8, ... (capped)
HTTP_BACKOFF_CAP = 2.0                     # per-sleep ceiling
HTTP_RETRY_STATUSES = (429, 500, 502, 503, 504)  # transient server/rate-limit codes

# BTC 5-min up/down markets live under this recurring Gamma series ("BTC Up or
# Down 5m"). Discovery lists this series' open events; the live 5-min window is
# then selected by its real resolves_at timestamp (see arena/market_utils).
POLYMARKET_BTC_5M_SERIES_ID = "10684"

# Taker fee model (makers are never charged). Polymarket's documented taker fee
# is symmetric around 50c: fee_usdc = rate * shares * price * (1 - price). Crypto
# is the highest tier. The rate is isolated here so it can be tuned in one place;
# both paper (simulated) and live use it. See polymarket_fills.taker_fee().
POLYMARKET_TAKER_FEE_RATE = 0.07

# --- Dual exchange (Polymarket 5m + Kalshi 15m) ---
# Settings toggles persist in arena_state `exchange_toggles`; these are defaults
# when the operator has never saved. Off ⇒ no discovery/eval/fills for that
# exchange (paper included).
EXCHANGE_POLYMARKET_ENABLED = True
EXCHANGE_KALSHI_ENABLED = True
KALSHI_API_BASE = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_WS_BASE = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
KALSHI_SERIES_TICKER = "KXBTC15M"
KALSHI_WINDOW_SEC = 900
KALSHI_SETTLEMENT_AVG_SEC = 60
KALSHI_TAKER_FEE_RATE = 0.07
# 15m σ seed (~√3 vs 5m prior if the process is similar). Soak-tune later.
KALSHI_DRIFT_VOL_SCALE = 0.0038
KALSHI_DRIFT_MIN_ABS_PCT = 0.00080
KALSHI_DRIFT_MIN_ABS_Z = 0.35
KALSHI_MOMENTUM_LATE_SKIP_SEC = 120
KALSHI_SNIPER_MIN_AGE_SEC = 90

# Paper mode is a full simulation against real Polymarket order books (no order
# is submitted). All paper bots share ONE virtual USDC bankroll, set by the user
# in the dashboard Settings tab (arena_state key 'paper_bankroll'); this default
# is used until they set one. Live mode uses the real wallet USDC balance.
PAPER_BANKROLL_DEFAULT = 200.0

# Database — override with ARENA_DB_PATH for Docker / non-default layouts.
DB_PATH = Path(os.environ.get("ARENA_DB_PATH") or (Path(__file__).parent / "bot_arena.db"))

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
MUTATION_RATE = 0.15  # legacy exploratory rate (used as GA fallback ceiling)
# Directed / GA default gene-flip probability (BUG #31 → full GA 2026-07-23):
# with elitism protecting top performers and crossover blending parents,
# per-gene mutation should stay modest so offspring stay near proven regions.
MUTATION_RATE_DIRECTED = 0.07
NUM_BOTS = 4
SURVIVORS_PER_CYCLE = 1  # legacy; GA uses GA_ELITE_COUNT instead
# Judgment WINDOW is decoupled from the 2h cycle CADENCE (2026-07-19): judging
# on the 2h window with a 20-trade floor made every bot permanently IMMUNE
# (bots average 5-12 trades per 2h), so zero evolutions fired in the whole
# 24h v5 run while momentum-v1 bled -$86. The window is what a bot is judged
# ON; the interval is only how often the judgment runs.
EVOLUTION_WINDOW_HOURS = 72        # 24→72 (audit: 24h overfits current regime; 72h covers multiple regime shifts)
# Raised 15 -> 30 (2026-07-21): a 5-min-market window of 15-20 resolved trades
# is dominated by noise — cycle 6 killed sniper-v1 on a 17-trade / -$8.49 dip
# one cycle after it survived at 61% WR, and mutated survivors were "judged" on
# 1-2 trades. Empirically the per-bucket WR/P&L numbers in the run only
# stabilized past ~30 samples, so a bot needs at least that many resolved this
# window before it can be replaced.
# Raised 30 → 40 (2026-08 soak): GA culled meanrev/mom/sniper/phantom on
# noisy 30-trade dips mid-run; mutants then bled the book. Need a fuller
# window sample before a bot is eligible for replacement.
MIN_TRADES_FOR_JUDGMENT = 40   # Fewer resolved trades in the window = immune
# Post-TWAP trade rate can sit at 1–13 fills / 72h. Adaptive floor lets the
# most-active directional be judged without waiting 40 samples; never below
# GA_MIN_TRADES_FLOOR (starved slates stay immune).
GA_MIN_TRADES_ADAPTIVE = True
GA_MIN_TRADES_FLOOR = 20
# Deep-red early cull: bots with n in [GA_EARLY_CULL_MIN_TRADES, MIN_TRADES)
# can still be replaceable when P&L and BE gap are catastrophically bad —
# prevents "IMMUNE forever" while a bot bleeds mid-session (2026-08-11).
GA_EARLY_CULL_ENABLED = True
GA_EARLY_CULL_MIN_TRADES = 15
GA_EARLY_CULL_PNL = -15.0
GA_EARLY_CULL_BE_GAP = -0.10
# Survival bar is the BREAK-EVEN GAP (win_rate - avg_entry_price), not a flat
# WR threshold: 65% WR bought at 70c loses money while 55% bought at 45c
# prints. A bot survives if its gap clears this floor OR its window P&L is
# positive (good sizing can rescue a thin gap). The old MIN_WIN_RATE=0.65
# would have culled every bot in the v5 run including the profitable ones
# (best WR was 63.3%). Still used by the GA as the *replacement eligibility*
# bar. Elites must ALSO pass this bar (GA_ELITE_REQUIRE_SURVIVAL).
EVOLUTION_BE_GAP_MIN = 0.03    # survive if WR beats avg entry by >= 3c
# Soft P&L floor: small negative window P&L is noise, not a cull signal.
# 2026-08: meanrev-v1 at −$3 was replaced by a worse mutant.
EVOLUTION_PNL_CULL_MAX = -12.0  # only replaceable if window PnL ≤ this
# Recency-weighted survival: blend long-window P&L with a short window so a
# recent session bleed is not buried under earlier green (2026-08-11).
GA_SURVIVAL_RECENCY_HOURS = 24.0
GA_SURVIVAL_RECENCY_WEIGHT = 0.55   # weight on short-window pnl for cull bar
# Gen-0 / default-lineage bots: only replace when deeply underwater (founders
# carry the slate until a mutant clearly earns the seat). Decays after
# GA_FOUNDER_PROTECT_MAX_CYCLES evolution cycles (0 = never decay).
GA_PROTECT_FOUNDERS = True
GA_FOUNDER_CULL_PNL = -20.0     # gen0 replaceable only if pnl ≤ this
GA_FOUNDER_CULL_BE_GAP = -0.02  # or BE gap worse than −2¢ with enough n
GA_FOUNDER_PROTECT_MAX_CYCLES = 20  # after this, founders use normal cull bar

# --- Genetic Algorithm (replaces simple mutate-from-winner, 2026-07-23) ---
GA_ELITE_COUNT = 2             # top-N by multi-obj fitness preferred as parents
# Elites are parents / gene-bank seeds, NOT immortal seats. When True, a bot
# that fails the economic survival bar is replaceable even if top-N fitness.
GA_ELITE_REQUIRE_SURVIVAL = True
GA_TOURNAMENT_K = 3            # tournament selection size
GA_MUTATION_RATE = 0.20        # per-gene probability of Gaussian noise
GA_MUTATION_SIGMA = 0.12       # noise scale as fraction of param range
GA_CROSSOVER_ALPHA_LO = 0.30   # blend weight range for parent A
GA_CROSSOVER_ALPHA_HI = 0.70
GA_CONSISTENCY_BLOCK = 10      # trades per consistency window
# Multi-objective weights (re-normalized at score time). Higher = more influence.
GA_FITNESS_WEIGHTS = {
    "pnl": 0.35,
    "sharpe": 0.20,
    "drawdown": 0.18,
    "consistency": 0.12,
    "regime_robustness": 0.15,  # cross-regime stability (stamped trade features)
}
GA_REGIME_CONDITION = True
GA_REGIME_MIN_TRADES = 5       # min samples in a regime to score robustness

# --- Regime detector (signals/regime_detector.py) ---
REGIME_EMA_ALPHA = 0.25        # feature EMA (higher = faster)
# Hysteresis (1s warm-path ticks). 3 was too sticky-thin overnight: relative
# vol/direction bounced across normal↔low_vol_range↔low_vol_trend every few
# minutes and portfolio rebalanced on every flip. 20 ticks ≈ 20s hold.
REGIME_HOLD_TICKS = 20
# Downstream (adapt / router / portfolio / hybrid / GA) only act when the
# committed label is this confident and has been held this long.
REGIME_ACTION_MIN_CONF = 0.50
REGIME_ACTION_MIN_HOLD_SEC = 20.0
REGIME_SWITCH_MARGIN = 0.12    # required confidence edge to start switch
REGIME_USE_CENTROIDS = True    # lightweight online clustering soft vote
MOM_CHOP_REGIME_DAMP = 0.45    # mom lane damp under high_vol_chop
STRAT_CHOP_REGIME_DAMP = 0.70  # strat lane damp under high_vol_chop
# --- Relative multi-factor regime calibration (PLAN 2026-08-05) ---
# "High vol" = high *for recent BTC*, not a fixed absolute threshold.
REGIME_USE_RELATIVE = True
REGIME_REL_RESERVOIR_MAX = 20_000
REGIME_REL_MIN_SAMPLES = 500
REGIME_REL_WINDOW_DAYS = 14          # rolling window of unique 1m candles
REGIME_REL_WINDOW_DAYS_SLOW = 60     # reserved; not used this pass
REGIME_CLASSIFY_VOL_HI = 0.70        # percentile / relative score
REGIME_CLASSIFY_VOL_LO = 0.30
REGIME_CLASSIFY_DIR_HI = 0.55        # directionality composite
REGIME_CLASSIFY_DIR_LO = 0.40
# Per-regime lane profiles (core tuner writes by_regime)
REGIME_PROFILE_ADAPT_ENABLED = True
REGIME_PROFILE_SEEDS_ENABLED = True
CORE_TUNE_MIN_TRADES_REGIME = 40
# Never UP a lane when strategy×regime P&L is negative with enough fills.
# Regime-local cells are thin overnight (n=3–8 trades) while attribution n is
# large — use a lower bar so red $ blocks UP before accuracy alone pumps drift.
CORE_TUNE_PNL_GATE = True
CORE_TUNE_PNL_MIN_TRADES = 15
CORE_TUNE_PNL_MIN_TRADES_REGIME = 5
# Primary objective: lane net EV (signed-agreement $), not sign accuracy.
# Accuracy is a one-way UP veto (must clear UP_ACC_FLOOR). Missing EV → HOLD,
# never an accuracy-led UP (soak 2026-08-24: mom 0.20→0.40 on 56.9% WR, −1.45¢).
CORE_TUNE_EV_PRIMARY = True
CORE_TUNE_EV_MIN_TRADES = 20
CORE_TUNE_EV_UP_MIN = 0.0          # need mean attributed $ ≥ this to UP
CORE_TUNE_EV_DOWN_MAX = -0.05      # mean attributed $ ≤ this → DOWN
CORE_TUNE_UP_ACC_FLOOR = 0.50      # never UP a lane that is coin-flip or worse
CORE_TUNE_RESET_SEED_ON_RED = True # snap elevated weight to seed when EV red/missing
# Unique-market scorecard net ¢/share (preferred judge over sign accuracy).
CORE_TUNE_SCORECARD_MIN = 20
CORE_TUNE_SCORECARD_DOWN_MAX = 0.0   # net ≤ this → block UP
CORE_TUNE_SCORECARD_FORCE_DOWN = -0.005  # net ≤ this → step DOWN
CORE_TUNE_SCORECARD_MAX_ENTRY = 0.62  # ignore 84¢+ dual-gate rows in overlay
# Timeout UP disabled (was re-pumping red lanes after 6h of hold_pnl_gate).
CORE_TUNE_PNL_GATE_TIMEOUT_HOURS = 0.0  # 0 = never timeout-up
# Soften drift floor: allow collapse toward this when EV is red (was hard 0.15).
CORE_TUNE_DRIFT_FLOOR = 0.10
CORE_TUNE_DRIFT_FLOOR_WHEN_RED = 0.05
# Renormalize each strategy profile to sum=1 after a tune step.
CORE_TUNE_NORMALIZE_PROFILE = True
# Continuous residual w = w0 + B·F (off until sample mass)
REGIME_CONTINUOUS_BLEND = False
REGIME_CONTINUOUS_MAX_DELTA = 0.08
REGIME_CONTINUOUS_MIN_SAMPLES = 200
REGIME_CONTINUOUS_ETA = 0.002
# Strategy routing (capital / GA) — not hard-skip
REGIME_ROUTER_MIN_TRADES = 12
REGIME_ROUTER_GA_BLEND = 0.35
# Blend TWAP path into regime trend/mom (0=spot only, 0.45 default)
REGIME_TWAP_BLEND = 0.45
# Soft frequency target (optional edge ease; default OFF)
REGIME_FREQ_TARGET_ENABLED = False
REGIME_FREQ_TARGET_FILLS_PER_HOUR = 4.0
REGIME_FREQ_EDGE_EASE_MAX = 0.15
# Live strategy×regime style-skip (data-driven; default ON after 2026-08-06 soak)
# Dual-path (2026-08-07): fast window enters skip on ~10 bad fills so 5m
# markets can stand down same session; slow path keeps overnight stability.
REGIME_STYLE_SKIP_ENABLED = True
REGIME_STYLE_SKIP_MIN_TRADES = 18       # slow (long-window) enter n
REGIME_STYLE_SKIP_WR = 0.42             # slow enter WR bar
REGIME_STYLE_SKIP_CLEAR_WR = 0.48
# Fast path: overnight cells are thin — n=6 at WR≤38% + red $ is enough to
# stand a toxic strategy down without waiting for n=18 on the slow path.
REGIME_STYLE_SKIP_FAST_MIN_N = 6
REGIME_STYLE_SKIP_FAST_WR = 0.38        # fast-path enter WR (stricter)
# Never style-skip these types (empty = pure data for all directionals)
REGIME_STYLE_SKIP_EXEMPT_TYPES = ()
# Side-aware continuous tax / binary side-skip (strategy×regime×side)
REGIME_SIDE_SKIP_ENABLED = True
REGIME_SIDE_CONT_MIN_N = 8
# Live NO-side tax when NO is toxic in a regime
REGIME_NO_SIDE_MIN_TRADES = 15
REGIME_NO_SIDE_WR = 0.42
REGIME_NO_SIDE_EDGE_MULT = 1.55
REGIME_NO_SIDE_EXTRA_DRIFT = 0.06
# Trade-stats cache for style-skip / tuner P&L gate
# Dual window: long = stability; fast = last N hours for continuous adapt.
REGIME_STATS_CACHE_SEC = 15.0
REGIME_STATS_LOOKBACK_HOURS = 72.0
REGIME_STATS_FAST_HOURS = 2.5
REGIME_STATS_MAX_TRADES = 4000
# Continuous adapt from blended WR (before binary style-skip).
# Lower bar so overnight strategy×regime toxicity raises min_edge / cuts size
# before a full style-skip cell is earned (auto per-regime rules).
REGIME_ADAPT_CONT_MIN_N = 5
REGIME_ADAPT_FAST_BLEND = 0.65          # weight on fast_wr when both thick
# Performance-triggered early evolution (in addition to EVOLUTION_INTERVAL_HOURS).
GA_PERF_TRIGGER_ENABLED = False
GA_FREEZE_DEFAULT_ROSTER = True
GA_PERF_TRIGGER_PNL = -25.0    # fire early if pool window P&L ≤ this
GA_PERF_TRIGGER_MIN_TRADES = 40
GA_PERF_TRIGGER_DROP = 40.0    # optional: fire if pool P&L drops this much vs last check
# Minimum seconds between GA cycles even when performance-triggered (anti-thrash).
GA_MIN_INTERVAL_SEC = 30 * 60

# --- GA upgrades (gene bank, type alloc, backtest gate, adaptive mutation) ---
GA_GENE_BANK_SIZE = 20            # max shadow elites kept as future parents
GA_TYPE_ALLOC_ENABLED = True      # tier-1: sample strategy_type by fitness softmax
# High stickiness (2026-08): cross-type swaps (phantom→sentiment) destroyed
# slate coherence. Prefer same-type offspring; still allow rare type shift.
GA_TYPE_STICKINESS = 0.80         # mass kept on the culled slot's original type
GA_TYPE_ALLOC_TEMPERATURE = 0.35  # lower = greedier toward high-fitness types
# Hard same-type only when True (ignore fitness softmax for type pick).
GA_TYPE_SAME_TYPE_ONLY = False
# Types excluded from spawn until their lanes are live (sentiment needs pm/cvd).
GA_SPAWN_EXCLUDE_TYPES = ()       # filled dynamically if empty — see type_alloc
GA_RECENCY_WEIGHTING = True       # fitness favors recent + current-regime trades
GA_REGIME_RECENCY_HALFLIFE_H = 6.0
GA_REGIME_MATCH_BOOST = 1.5       # multiplier for trades stamped with live regime
GA_ADAPTIVE_MUTATION = True       # tier-2: sample near elite cloud (TPE-ish)
GA_ELITE_SAMPLE_RATE = 0.55       # P(sample from elite cloud vs local Gaussian)
GA_BACKTEST_GATE_ENABLED = True   # promote only after offline backtest clears
GA_BACKTEST_REQUIRED = True       # fail-closed: reject spawn when history unavailable
GA_BACKTEST_MARKETS = 40          # recent resolved markets for the gate
GA_BACKTEST_CACHE_SEC = 3600.0    # reuse fetched history within this TTL
GA_BACKTEST_BEAT_BASELINE = True  # child must not be worse than replaced bot
GA_BACKTEST_EPS = 0.50            # $ noise band when comparing to baseline
GA_BACKTEST_MIN_PNL = None        # optional absolute floor (None = off)
GA_SPAWN_ATTEMPTS = 3             # type/param samples before fallback defaults
# Cap identical strategy_types introduced in one cycle (kept survivors count
# toward the cap). Prevents a monoculture elite from filling every open slot
# with two identical hybrids (cycle-4 2026-07-29: hybrid-g4-158 + hybrid-g4-259).
GA_MAX_PER_TYPE_PER_CYCLE = 1
# Gene bank: max elites retained per strategy_type (then global GA_GENE_BANK_SIZE).
GA_GENE_BANK_MAX_PER_TYPE = 3
# Don't deposit elites with fewer resolved trades than this (rank-fitness on
# n=2 is noise — sniper-v1 sat in the bank at 50% WR / -$0.46 with n=2).
GA_GENE_BANK_MIN_TRADES = 5
# When True, prune bank entries that have ≥ MIN_TRADES and pnl < 0 so a once-
# elite genome that later looks like a loser cannot keep parent-tainting.
GA_GENE_BANK_DROP_NEG_PNL = True
# Spawn diversity: min normalized L1 distance vs live same-type peers (0 = off).
GA_DIVERSITY_MIN_DISTANCE = 0.08
# Backtest gate: also require not-worse vs baseline in the live regime subset.
GA_BACKTEST_REGIME_MIX = True
GA_BACKTEST_REGIME_EPS = 0.50
GA_BACKTEST_REGIME_MIN_TRADES = 3
# Extra frozen gene names (merged with evolution.frozen defaults)
GA_FROZEN_GENES = ()

# --- Sniper ask quality ---
# Max (ask − mid) on the chosen side; wider spreads mean the lag thesis on
# mid is already stale at the executable price (2026-07-29 mid0.54/ask0.75).
SNIPER_MAX_ASK_MID_SPREAD = 0.03
# Style-skip seeds (no wait for 12h toxic WR). Tuple (regime, strategy).
REGIME_STYLE_SKIP_SEEDS = {
    ("high_vol_chop", "momentum"): True,
    ("low_vol_range", "momentum"): True,
    ("high_vol_trend", "mean_reversion"): True,
}

# --- Portfolio explore floor for new gN bots ---
# Until a post-evolution bot has this many resolved trades, cap its capital
# weight so cold mutants cannot eat a full Kelly slice immediately.
PORTFOLIO_EXPLORE_MIN_TRADES = 8
PORTFOLIO_EXPLORE_MAX_WEIGHT = 0.06   # per cold bot
# Total capital budget shared by ALL not-ready / cold bots (prevents 3×24%).
PORTFOLIO_EXPLORE_TOTAL_BUDGET = 0.12
# Proven bots (long-window exp>0 or gen0 with n≥floor) keep at least this
# weight even if short-window expectancy dips slightly negative.
PORTFOLIO_PROVEN_FLOOR = 0.06
PORTFOLIO_PROVEN_MIN_TRADES = 10
# Anti-starvation floor (audit 2b): active bots with few recent fills keep
# at least this weight so flat-market hysteresis doesn't zero their capital.
PORTFOLIO_ACTIVE_MIN_TRADES = 3
PORTFOLIO_ACTIVE_MIN_WEIGHT = 0.05

# --- Learned trade rules (decision_events → auto skip/go) ---
# Mines regime×price×drift×side cells; promotes skip when buys lose and go
# (softer min_edge + size boost) when buys print or skips miss winners.
LEARNED_RULES_ENABLED = True
LEARNED_RULES_MIN_N = 25
LEARNED_RULES_SKIP_WR_MAX = 0.47
LEARNED_RULES_SKIP_HYP_MAX = -0.005
LEARNED_RULES_GO_WR_MIN = 0.58
LEARNED_RULES_GO_HYP_MIN = 0.01
LEARNED_RULES_MISSED_WR_MIN = 0.60
LEARNED_RULES_DEMOTE_SKIP_WR = 0.53
LEARNED_RULES_DEMOTE_GO_WR = 0.50
LEARNED_RULES_GO_SIZE_MULT = 1.15
LEARNED_RULES_GO_EDGE_MULT = 0.85
LEARNED_RULES_MISSED_EDGE_MULT = 0.80
# Never ease (GO) high-price cells from skip counterfactuals alone — that
# re-opened expensive favorites (2026-08: ≥0.72 mid −$38). Require real buy
# evidence with positive hyp (fee-aware).
LEARNED_RULES_BAN_GO_HIGH_FROM_SKIP = True
LEARNED_RULES_GO_HIGH_MIN_BUY_N = 15
LEARNED_RULES_GO_HIGH_MIN_HYP = 0.02
LEARNED_RULES_MAX = 40
LEARNED_RULES_MAX_CONTINUOUS = 30
LEARNED_RULES_CACHE_SEC = 30.0
# Per-strategy cells: True forces on; False + AUTO enables when sample mass
# clears LEARNED_RULES_PER_STRATEGY_MIN_* (no manual flip needed in normal use).
LEARNED_RULES_PER_STRATEGY = False
LEARNED_RULES_PER_STRATEGY_AUTO = True
LEARNED_RULES_PER_STRATEGY_MIN_RESOLVED = 200
LEARNED_RULES_PER_STRATEGY_MIN_CELLS = 8
# Continuous size/edge mult from cell WR (process 1)
LEARNED_RULES_CONTINUOUS = True
LEARNED_RULES_CONT_BAD_WR = 0.45
LEARNED_RULES_CONT_GOOD_WR = 0.60
LEARNED_RULES_CONT_SIZE_MIN = 0.40
LEARNED_RULES_CONT_SIZE_MAX = 1.25
LEARNED_RULES_CONT_EDGE_TIGHT = 1.25
LEARNED_RULES_CONT_EDGE_SOFT = 0.80
# Walk-forward OOS (process 4)
LEARNED_RULES_OOS_ENABLED = True
LEARNED_RULES_OOS_TRAIN_FRAC = 0.70
LEARNED_RULES_OOS_MIN_EVENTS = 40
LEARNED_RULES_OOS_REQUIRE_TEST_CELL = False
# Skip-reason bandit (process 3)
LEARNED_RULES_SKIP_BANDIT_ENABLED = True
LEARNED_RULES_SKIP_BANDIT_MIN_N = 30
LEARNED_RULES_SKIP_BANDIT_HIGH_CF = 0.58
LEARNED_RULES_SKIP_BANDIT_LOW_CF = 0.48
LEARNED_RULES_SKIP_BANDIT_MAX_SOFTEN = 0.25

# Signal Feed Settings
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"  # ETH/SOL only; BTC = Chainlink
POLYMARKET_RTDS_WS = "wss://ws-live-data.polymarket.com"
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
# After a slippage_band / slippage_exceeded reject, sit out this (bot, market)
# for N seconds so the 1s trader does not spam re-attempts into a whipping
# late-window book (overnight: 5–10 rejects/bot/window). Maker section (~20s)
# also honors the same cooldown.
SLIPPAGE_RETRY_COOLDOWN_SEC = 10.0

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

# PM in-market momentum kill-switch (2026-07-17). The live lane degraded to
# sign(last tick): SCALE=80 in signals/polymarket_prices.py saturates its
# clamp at a 0.19c/step move, ~66x below the median per-minute PM move
# (harness p50 0.126, p97 0.40) — it was pegged at +/-1.000 on 79% of the
# 44%-WR chop-run trades and manufactured model leans of 0.55-0.66 from
# noise. The harness verdict on the RAW quantity: predictive (69.7%
# follow-WR) but NET edge NEGATIVE (-0.80c/share at avg entry 0.688) — the
# market has already priced its own momentum by the time it is measurable.
# House rule: no positive net edge, no live weight. Same treatment as OBI —
# global kill-switch multiplied onto the pm lane for every strategy; keep
# 0.0 unless a reworked pm signal shows positive net edge offline.
SIGNAL_WEIGHT_PM = 0.0

# CVD kill-switch (BUG #27, 2026-07-17). The live lane (net/total over a ~20s
# tape, no volume floor) saturates at +/-0.8-1.0 whenever the thin tape is
# one-sided — sign(recent tape), the same magnitude disease as pm_mom. Live
# ground truth: cvd-driven trades (|cvd| >= 0.8, |drift| < 0.10) ran 53.1% WR
# (+$10.47 over 32 trades) — statistically flat, no net edge. The feed now
# carries a volume floor (CVD_VOLUME_FLOOR below) so thin tapes read weak;
# keep the lane at 0 until the calibrated form measures POSITIVE NET edge in
# the offline harness (house rule: validate-before-weighting).
SIGNAL_WEIGHT_CVD = 0.0
# Quiet-regime damp on the BTC momentum lane (2026-07-19 24h run): momentum-
# driven trades in chop (|drift| < 0.10) ran 47.9% WR / -$74 for momentum-v1
# alone — one candle of quiet-tape noise is not a trend. When the volatility
# regime (signals/volatility_regime.py, computed from the live candle stream)
# reads "quiet", the mom lane's value is multiplied by this before blending.
# Trending/volatile/normal regimes are untouched.
MOM_QUIET_REGIME_DAMP = 0.5
# Strat lane treatment (2026-08 redesign). ``strat`` is NOT a raw market
# signal — it is the bot's analyze() thesis, usually derived from the same
# prices/candles that feed drift/mom. Treating it as an independent additive
# lane double-counts information and let overconfident theses mint edge.
# Modes:
#   confirm — only contribute when sign agrees with drift; magnitude scaled
#   residual — always allow but scale down (legacy-ish)
#   full — old additive behavior (not recommended)
STRAT_LANE_MODE = "confirm"
STRAT_CONFIRM_SCALE = 0.55          # keep this fraction when confirming drift
STRAT_FIGHT_SCALE = 0.0             # scale when strat fights non-trivial drift
STRAT_DRIFT_AGREE_MIN = 0.05        # |drift| above which fight/confirm applies
# Strat-lane confidence cap (BUG #30, 2026-07-20). The per-strategy analyze()
# thesis (EMA-crossover/breakout, z-score fade, trend-follow, etc.) has never
# been offline-validated the way drift/mom were — it was assumed reasonable
# as "differentiation by emphasis". The 24h/279-trade run showed the opposite
# of a working signal: WR fell as the thesis got MORE confident (|strat| >=
# 0.6: 36.1% WR, -$60.15 over 36 trades — the single worst bucket in the run;
# |strat| 0.3-0.6: 55.9% WR; |strat| < 0.3: 46.5%). A maximally confident
# thesis correlates with the strategy pattern-matching a move that's already
# priced in, not with extra information (same shape as KELLY_EDGE_CAP's
# rationale for outsized model-vs-market edges). Clamp the lane's magnitude
# before it enters the blend so overconfident reads fall back into the
# 0.3-0.6 band that actually performed, rather than removing the lane
# outright. A full offline harness validation of the strat lane (same
# treatment as fut/tech/xasset in tools/validate_signals.py) is the
# recommended follow-up before trusting it further.
# Lowered 0.60 -> 0.30 (2026-07-21): after the 290-trade run, live per-lane
# attribution showed strat is anti-predictive at any magnitude >= 0.3
# (|strat| 0.3-0.6 ran 52.7% WR / -$14.81; >= 0.6 ran 46.0% / -$34.05), while
# |strat| < 0.3 was the only profitable band (+$41.23). Clamp to 0.30 so the
# blend only ever sees the magnitude that actually performed.
STRAT_LANE_CONF_CAP = 0.25

# Mean-reversion identity guard (audit 1d): at |drift| >= this floor the
# drift-heavy profile makes the bot a duplicate trend follower. Stand down.
# Stand down only when TWAP is actually locked (honest tanh ~0.70 ≈ Φ 0.85).
# 0.40 was a catch-22 with dual-gate z≥0.35: meanrev never had a fade window.
MEANREV_MIN_FADE_DRIFT = 0.70

# --- NO-side intelligence (2026-08 soak: YES +$245 / NO −$15) ---
# Not a blanket NO ban. Prefer NO only when it is a true market-lag trade
# (cheap relative to signed drift) with real drift conviction. Strategies
# that already print on NO (momentum/meanrev) keep milder extras.
NO_SIDE_ENABLED = True
NO_SIDE_MIN_SIGNED_DRIFT = 0.12
NO_SIDE_MAX_MID = 0.58
NO_SIDE_EDGE_MULT = 1.20            # global min_edge mult on NO
NO_SIDE_UNDERDOG_EDGE_MULT = 1.35   # extra when NO mid in cheap band
# Per-strategy extra min_edge mult on NO (on top of global). 1.0 = no extra.
NO_SIDE_STRATEGY_EDGE_MULT = {
    "sniper": 1.35,
    "hybrid": 1.40,
    "phantom": 1.50,
    "fee_zone_maker": 1.25,
    "late_window_maker": 1.25,
    "momentum": 1.05,
    "mean_reversion": 1.05,
    "mean_reversion_tp": 1.15,
    "mean_reversion_sl": 1.15,
    "lag_residual": 1.10,
    "regime_specialist": 1.20,
    "no_lag": 1.0,              # NO specialist — already NO-only
    "true_maker": 1.20,
}

# --- Cheap underdog band (0.35–0.42): mild leak at 38% WR / −$23 ---
UNDERDOG_BAND_LO = 0.35
UNDERDOG_BAND_HI = 0.42
UNDERDOG_MIN_DRIFT = 0.18
UNDERDOG_EDGE_MULT = 1.40
# Residual-lag check in the mid band, scored with Φ(z) not tanh.
PRICE_QUALITY_MID_LO = 0.50
PRICE_QUALITY_MID_HI = 0.58
PRICE_QUALITY_DRIFT_MIN = 0.15
PRICE_QUALITY_ASK_MAX = 0.99
PRICE_QUALITY_LAG_MIN = 0.04

# --- Maker mid/ask integrity ---
MAKER_MAX_MID_ASK_GAP = 0.08        # |mid − ask| above this → refuse trade
# Tape volume (shares) below which CVD magnitude is damped: cvd =
# net / max(total, floor). A 30-share one-sided tape reads 0.15, not 1.0;
# a 1500-share one-sided tape still reads ~1.0. Calibrate offline before
# re-weighting the lane.
CVD_VOLUME_FLOOR = 200.0

# --- Candidate signal lanes (2026-07-18) — ALL kill-switched at 0 ---
# New lanes computed every tick and logged in trade reasoning, but carrying
# ZERO live weight until tools/validate_signals.py measures POSITIVE NET edge
# for each (house rule: validate-before-weighting — see BUG #23/#26/#27 for
# what shipping an unvalidated lane costs). Each is a global multiplier onto
# its lane for every strategy, same pattern as OBI/PM/CVD above.
SIGNAL_WEIGHT_FUT = 0.0      # Binance perp funding/OI/taker delta (signals/futures_meta.py)
SIGNAL_WEIGHT_TECH = 0.0     # MACD/Bollinger/multi-TF composite (signals/technicals.py)
SIGNAL_WEIGHT_XASSET = 1.0   # ETH/SOL confirm-only (live +0.51¢/share, 79% WR)
# Confirm-only: xasset may not pick a side; it only adds when it agrees with drift.
XASSET_LANE_MODE = "confirm"
XASSET_CONFIRM_SCALE = 1.0
XASSET_FIGHT_SCALE = 0.0
XASSET_DRIFT_AGREE_MIN = 0.05
# --- Expanded candidate lanes (2026-08 audit) — kill-switched until Lab ---
# Logged in cand(...) and available via Signal Lab promote path. Same house
# rule: no live weight without harness + live-shadow net edge.
SIGNAL_WEIGHT_LAG = 0.0      # market-lag residual: moneyness-implied P − mid
SIGNAL_WEIGHT_MS_MOM = 0.0   # multiscale 1m mom (signals/multiscale.py)
SIGNAL_WEIGHT_FLOW_DECAY = 0.0  # time-decayed CVD (signals/flow.py)
# Soft-saturate scale for lag's raw-moneyness map (not time-scaled z).
LAG_MONEYNESS_SCALE = 0.0015   # ~0.15% moneyness → ~0.76
# Spread context (non-directional): widen min_edge when book is wide.
SPREAD_EDGE_MULT_ENABLED = True
SPREAD_EDGE_WIDE = 0.04      # spread fraction of mid above this → tax
SPREAD_EDGE_MULT_MAX = 1.35  # max min_edge multiplier on very wide books
# Macro-release caution (signals/macro_calendar.py) is NON-directional context:
# above this smooth 0..1 caution score, directional takers stand down (same
# philosophy as the session filter — "build the skip, default flat").
MACRO_CAUTION_SKIP = 0.75

# --- Live lane monitor (arena/lane_monitor.py) — the DEMOTION half of the
# lane-promotion pipeline. The harness promotes on backfilled data; this
# demotes on live ground truth. Every trade logs the raw candidate-lane reads
# in its reasoning; the monitor parses them from RESOLVED trades placed after
# a lane's approval and scores the lane's sign against the actual outcome.
# Why it must exist: the 2026-07-19 run approved tech at a harness-measured
# 74-80% follow-WR — live it scored 51.7% over 209 trades (harness numbers
# carry adverse-selection and stale-mid optimism the live tape doesn't).
LANE_MONITOR_MIN_TRADES = 50        # resolved readings before a verdict
LANE_MONITOR_MIN_ACCURACY = 0.53    # live sign-vs-outcome accuracy to stay live
# Demote when live shadow net edge is below this (even if accuracy is OK).
# Matches the house rule: predictive but priced-in lanes must lose weight.
LANE_MONITOR_MIN_NET_EDGE = 0.0     # ≤ 0¢/share after fee → demote at full sample
LANE_MONITOR_DEADBAND = 0.05        # |reading| below this = no directional read
LANE_MONITOR_INTERVAL_SEC = 1800    # check cadence (piggybacks the evolution loop)
# Fast demote: don't wait for the full sample if a newly-live lane is clearly
# anti-predictive (fut post-approval ran ~38% at n=21 while still "collecting").
LANE_MONITOR_FAST_DEMOTE_MIN_TRADES = 20
LANE_MONITOR_FAST_DEMOTE_MAX_ACC = 0.45
LANE_MONITOR_FAST_DEMOTE_MAX_NET_EDGE = -0.02  # strongly negative EV early

# --- Auto-validation scheduler (arena/validation_scheduler.py) ---
# Runs tools/validate_signals.py --propose from inside the arena every
# AUTO_VALIDATE_EVERY_MARKETS 5-min windows (markets are strictly one per
# 5 minutes, so 100 markets ~ 8.3h => ~3 fresh reads/day). The WINDOW stays
# at 300 markets (~25h) because the promotion bar needs n>=200 samples and
# the sparser lanes (fut_oi) only collect ~300-360 samples per 300 markets —
# a shorter window would starve them below the bar. Frequency gives regime
# freshness; window size gives statistical power. Proposals still require
# dashboard approval (Signal Lab) — this only automates the measurement.
AUTO_VALIDATE_ENABLED = True
AUTO_VALIDATE_EVERY_MARKETS = 100   # run cadence, in 5-min market windows
AUTO_VALIDATE_WINDOW_MARKETS = 300  # --markets passed to the harness

# --- Auto-approve promoter (arena/lane_promoter.py) — closed loop ---
# The harness NOMINATES candidate lanes (offline, optimistic); LIVE attribution
# JUDGES them. A pending proposal is auto-approved only once the lane's own
# shadow reads (logged in every directional trade's cand(...) string, pre
# kill-switch) clear a LIVE bar over a real resolved sample — never on the
# harness number alone, which measured tech at 74-80% but scored 51.7% live.
# Bar is intentionally HIGHER than LANE_MONITOR_MIN_ACCURACY (0.53) so a lane
# must earn promotion by a clearer margin than it needs to merely survive —
# hysteresis that stops a borderline lane flapping between approve and demote.
# The toggle is stored in arena_state ('auto_approve_lanes', dashboard-editable);
# this constant is only the boot default. OFF => the promoter still annotates
# each proposal with live evidence for the human, but never flips it.
AUTO_APPROVE_LANES_ENABLED = True
AUTO_APPROVE_MIN_TRADES = 60      # live shadow readings before a promotion verdict
AUTO_APPROVE_MIN_ACCURACY = 0.57  # 0.55→0.57 (audit: wider 4pp gap vs demotion 0.53 for hysteresis)
# Require positive LIVE shadow net edge (¢/share after fee) in addition to
# accuracy — accuracy alone promoted fut at ~55% that later scored ~38% live.
AUTO_APPROVE_MIN_NET_EDGE = 0.005  # +0.5¢/share on shadow follow-the-sign
# Fill-level bar: unique-market / tick shadow cannot promote alone. A lane
# must also show positive net edge on actual resolved fills (the tech
# 80%→−5.5¢ incident). Below this fill count the verdict stays "collecting".
AUTO_APPROVE_MIN_FILLS = 15
AUTO_APPROVE_MAX_ACTIVE = 3       # cap on simultaneously-enabled CANDIDATE lanes
# Core-lane tuner apply toggle — SEPARATE from auto-approve so operators can
# freeze promotions without freezing drift/mom/strat weight nudges (and vice
# versa). Boot default; live value in arena_state ``auto_core_tune``.
AUTO_CORE_TUNE_ENABLED = True

# --- Core-lane auto-tuner (arena/core_lane_tuner.py) — the loop's core half ---
# The candidate-lane loop above tunes fut/tech/xasset (which feed a few bots at
# ~0.10 weight). This tunes the lanes that drive EVERY directional trade —
# drift/mom/strat — PER strategy, on that strategy's own live attribution
# (sign-vs-outcome of the lane reading logged in its trades' reasoning). Because
# these lanes decide 100% of a decision, the tuner is deliberately timid: small
# capped nudges, a per-lane band around the hand-set class default so no lane
# can run away or collapse (drift especially — the one validated lane), a real
# per-(strategy,lane) sample floor, and hysteresis (nudge up only above
# HIGH_ACC, down only below LOW_ACC; the dead band between them holds steady).
# Gated by the SAME auto-approve toggle as the promoter: OFF => compute and
# surface the suggested weights for a human, never apply. Writes a COMPLETE
# per-strategy profile for each tuned lane (a core-lane override zeroes any
# strategy it omits, unlike a candidate lane that defaults to 0).
CORE_TUNE_ENABLED = True
CORE_TUNE_INTERVAL_SEC = 300   # evolution-loop host cadence (was piggybacked on 30m lane monitor)
CORE_TUNE_MIN_TRADES = 40      # per-(strategy,lane) resolved readings before tuning
CORE_TUNE_HIGH_ACC = 0.56      # legacy; UP trigger is EV. Kept as a log/compat alias.
CORE_TUNE_LOW_ACC = 0.48       # legacy; accuracy must not DOWN a lane by itself.
# Live candidate overrides (enabled non-core lanes) share the tuner loop
CANDIDATE_TUNE_WEIGHT_MAX = 0.35
CANDIDATE_TUNE_MIN_TRADES = 30
CANDIDATE_TUNE_BAND = 0.25     # around approved starter weight; floor can hit 0
# Elevated weights that do not clear HIGH_ACC bleed back toward the class
# default (one step/cycle). Prevents mediocre lanes from sitting at the band
# ceiling forever (sentiment strat overnight: 56.7% acc at weight 0.9).
CORE_TUNE_REVERT_BELOW_ACC = 0.56
CORE_TUNE_STEP = 0.05          # per-cycle weight nudge (bounded, one step/lane/strategy)
CORE_TUNE_BAND = 0.20          # max |deviation| of a tuned weight from its class default
CORE_TUNE_WEIGHT_MAX = 0.90    # absolute ceiling on any single lane weight
CORE_TUNE_WEIGHT_MIN = 0.0     # absolute floor (the band around the default binds first)

# --- Regime discovery & conditioning (Layer 3 — arena/regime_map.py) ---
# The toggle is stored in arena_state ('regime_conditioning', dashboard-
# editable via db.get/set_regime_conditioning); this constant is only the
# boot default. Same pattern as AUTO_APPROVE_LANES_ENABLED / CORE_TUNE_ENABLED
# above — OFF means the map is still built and reported, but no downstream
# controller is allowed to act on it.
REGIME_CONDITIONING_ENABLED = True   # dashboard-editable; ON in paper mode
REGIME_MAP_INTERVAL_SEC = 900        # attribution/discovery cadence
REGIME_MIN_SAMPLES = 60              # promote a cell to a named regime
REGIME_SHRINKAGE_K = 40              # empirical-Bayes prior strength
REGIME_RECENCY_HALFLIFE_DAYS = 14    # decay for non-stationarity
REGIME_ALLOC_MIN_WEIGHT = 0.05       # explore floor per active bot
REGIME_ALLOC_MAX_TILT = 0.25         # max deviation from baseline weight
REGIME_HOUR_BLOCK_HOURS = 3          # ET time-of-day granularity

# Sentiment feed master switch (2026-07-18): OFF — no local LLM will be run
# and the keyword/CryptoPanic pipeline isn't worth its noise on 5-min BTC
# markets. When False, SentimentFeed.start() is a no-op: no polling thread,
# no scoring, get_signals() returns {} (every consumer already handles the
# empty dict). Revisit when a hosted-LLM scorer (Claude/Grok) is wired in.
SENTIMENT_FEED_ENABLED = False

# --- BTC drift-from-strike ("price to beat") signal (signals/strike.py) ---
# The dominant fundamental for these markets: where BTC sits vs the window's open
# price. Regime-agnostic (favors whichever side BTC is actually on) and time-
# scaled (more decisive near expiry). Fed into fair value at SIGNAL_WEIGHT_DRIFT.
MARKET_WINDOW_SEC = 300           # 5-min window length
# Prior / fallback full-window fractional vol.
# Recal 2026-08-07: 0.0015 was too tight at ~$65k TWAP tape — small late-window
# wiggles printed as |drift|≥0.3 and directional follow WR collapsed. Prior
# raised; adaptive σ prefers **TWAP** samples (same object as moneyness).
DRIFT_VOL_SCALE = 0.0022          # ~0.22% of price full-window prior
DRIFT_ADAPTIVE_SCALE = True       # False → always use DRIFT_VOL_SCALE prior
DRIFT_VOL_SCALE_MIN = 0.0018      # 0.0010 mapped 5bp TWAP wiggles to |d|≈0.55
DRIFT_VOL_SCALE_MAX = 0.0050      # panic ceiling
DRIFT_ADAPT_EMA_ALPHA = 0.08      # slightly faster than 0.05 (5m product)
DRIFT_ADAPT_MIN_SAMPLES = 20      # warm faster after restart
# Prefer Chainlink TWAP tick series for adaptive σ (falls back to spot 1m).
DRIFT_ADAPT_USE_TWAP = True
# Sample near the TWAP lookback so increments are not overlapping 5s
# averages (those pin σ at the floor and invent 78¢ implied probs).
DRIFT_ADAPT_TWAP_SAMPLE_SEC = 60.0
# Raw moneyness floor (fraction): dual-gate with z-score so $noise ≠ "strong drift".
# Overnight 2026-08-20: 5–6 bp at σ floor 0.0018 cleared |z|≥0.35 and printed
# Φ≈0.64–0.69 on 32–38¢ underdogs (core directional 12/35 = 34% WR).
# 8 bp ≈ $55 at $70k — still below a real 5m move; binds when z is inflated.
DRIFT_MIN_ABS_PCT = 0.00080
# Sniper: 8bp dual-gate is enough inside the 50–58¢ lag pocket (soak +EV);
# outside that band keep the 15bp conviction floor (cheap underdogs).
SNIPER_MIDBAND_LO = 0.50
SNIPER_MIDBAND_HI = 0.58
SNIPER_OUTSIDE_MIN_PCT = 0.0015
# Momentum sits out the last ~80s (soak: late≤80s −$13 even with consecutive bars).
MOMENTUM_LATE_SKIP_SEC = 80
# Meanrev: retrace of TWAP toward this window's strike, not a 4-bar z-score.
MEANREV_PULLBACK_MIN = 0.20
# Floor remaining-window time in σ_rem = scale·√(tr/W) so last-minute noise
# cannot explode z. 60s ⇒ σ_rem never below scale·√(60/300).
DRIFT_TIME_SCALE_MIN_SEC = 60.0
# Dual-gate bar on *raw* z = moneyness / σ_rem (not tanh). 0.35 ≈ 6–8 bp
# mid-window at σ=0.22%. Pre-TWAP-recal this was compared to tanh(z).
DRIFT_MIN_ABS_Z = 0.35
# Sweeper lock floor on tanh(z). 0.32 is only a first filter — a 99¢ buy
# also needs Φ(z) ≥ SWEEPER_MIN_IMPLIED (overnight tanh 0.455 @ 99¢ NO
# lost $5.70; Φ was ~0.67, not a lock).
SWEEPER_MIN_DRIFT = 0.32
SWEEPER_MIN_IMPLIED = 0.97
SWEEPER_MIN_TWAP_CERTAINTY = 0.45
# RE-ENABLED (2026-07-16) after the #23 blow-up was traced to a MISCALCULATED
# strike (mid-window "first sighting"), not a bad signal. Live strike is
# Polymarket's official openPrice (same /api/crypto/crypto-price endpoint the
# website uses for "Price to Beat"). As of 2026-08-07 00:00 UTC both open PTB
# and final settlement are Chainlink **TWAP** values (60s for 5m markets) —
# not a single spot snapshot. See TWAP_* knobs below. Binance is NOT used live
# (basis ~$60–80 / ~0.1% vs Chainlink — enough to flip near-strike drift); if
# openPrice is briefly unavailable, drift stays 0 until the next fetch.
# Offline harnesses may still use Binance klines for ranking. Drift is weighted
# per-strategy inside STRATEGY_SIGNAL_PROFILE (bots/base_bot.py); it is the
# anchor lane of every strategy's model.

# --- Chainlink TWAP resolution (Polymarket 2026-08-07+; 5m window 60s) ------
# Spec: https://docs.polymarket.com/market-data/chainlink-twap
#        @PolymarketDevs — both open PTB and settlement from TWAP feed.
# 5-min markets → **60s TWAP** (was 30s at 2026-08-07 cutover; Polymarket
# announced lengthening the 5m lookback to 60s — override with TWAP_WINDOW_SEC).
# 15-min / 4h remain 60s. RTDS topics: crypto_prices_twap_thirty / _sixty.
# Active topic is chosen from TWAP_WINDOW_SEC via signals.twap.rtds_topic().
TWAP_RESOLUTION_ENABLED = True
TWAP_WINDOW_SEC = int(os.environ.get("TWAP_WINDOW_SEC") or "60")  # 5-min BTC
TWAP_WINDOW_SEC_15M = int(os.environ.get("TWAP_WINDOW_SEC_15M") or "60")
TWAP_RTDS_TOPIC_30 = "crypto_prices_twap_thirty"
TWAP_RTDS_TOPIC_60 = "crypto_prices_twap_sixty"
TWAP_SYMBOL = "btc/usd"
# Use official TWAP (not spot Chainlink) as btc_now for drift moneyness.
TWAP_USE_FOR_DRIFT = True
# Inside the final TWAP_WINDOW_SEC of a market, blend in a settlement nowcast
# built from local ticks over [expiry−W, now] (fill remaining with last price).
TWAP_NOWCAST_ENABLED = True
# Minimum tick coverage of the settlement sub-window before trusting nowcast
# over the rolling RTDS TWAP alone (0–1).
TWAP_NOWCAST_MIN_COVERAGE = 0.40
# TWAP feed stale threshold (lookback windows ≠ publication cadence).
# Slightly looser for 60s lookback; cadence is still sub-second when live.
TWAP_STALE_SEC = float(os.environ.get("TWAP_STALE_SEC") or "20.0")
# When TWAP is unavailable, fall back to spot Chainlink for drift (noisy vs
# true settlement — logged; prefer 0 drift if both missing).
TWAP_FALLBACK_TO_SPOT = True
# Soft vol damp for TWAP-vs-strike z-scores. Was 0.85 (made z *more* sensitive
# while σ was already from spot — double-counted smoothness). With TWAP-based
# adaptive σ, keep mult at 1.0; only damp if σ is still from spot fallback.
TWAP_DRIFT_VOL_MULT = 1.0
TWAP_DRIFT_VOL_MULT_SPOT_FALLBACK = 0.92  # mild damp only when σ from spot 1m
# --- Settlement-window policy (final TWAP_WINDOW_SEC of each market) -------
# Once rem ≤ TWAP_WINDOW_SEC the settlement TWAP is partially observed:
# certainty rises with elapsed fraction; last-tick spot spikes no longer flip
# the outcome. Policy reweights edge floors, size, mom damp — not a hard ban.
TWAP_SETTLEMENT_POLICY = True
# rem in (TWAP_WINDOW, TWAP_WINDOW+lead] = pre-settlement (prepare, mild damp).
# ~1/3 of the averaging window (was 15s for 30s TWAP; 20s for 60s).
TWAP_PRE_SETTLE_LEAD_SEC = int(os.environ.get("TWAP_PRE_SETTLE_LEAD_SEC") or "20")
# Certainty thresholds (0–1 from twap_certainty())
TWAP_SETTLE_CERT_HIGH = 0.55      # "mostly locked" — ease edge, allow size
TWAP_SETTLE_CERT_LOW = 0.25       # noisy partial window — raise edge, cut size
# min_edge multipliers inside settlement window
TWAP_SETTLE_EDGE_MULT_HIGH = 0.92   # high certainty: slightly easier bar
TWAP_SETTLE_EDGE_MULT_LOW = 1.40    # low certainty: much harder bar
TWAP_SETTLE_EDGE_MULT_MID = 1.12    # between low and high
# Kelly bankroll multipliers
TWAP_SETTLE_SIZE_MULT_HIGH = 1.12
TWAP_SETTLE_SIZE_MULT_LOW = 0.80
# Spot mom lane damp (1m candle noise is not settlement)
TWAP_SETTLE_MOM_DAMP = 0.40
TWAP_PRE_SETTLE_MOM_DAMP = 0.70
# Confidence structure boost when high cert (logs / min_conf only)
TWAP_SETTLE_CONF_BOOST = 0.08
# Mean-rev: do not fade against a high-certainty TWAP side in settlement
TWAP_SETTLE_BLOCK_FADE = True
TWAP_SETTLE_BLOCK_FADE_CERT = 0.50
TWAP_SETTLE_BLOCK_FADE_DRIFT = 0.20

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
# Weight of a strategy's analyze() lean inside P_model is now PER-STRATEGY
# (the "strat" key in bots/base_bot.py STRATEGY_SIGNAL_PROFILE — BUG #27
# fidelity redesign; the old flat global 0.15 was too small to differentiate
# anyone). This constant remains only as the DEFAULT_SIGNAL_PROFILE fallback
# reference; nothing multiplies it into the lane anymore.
STRATEGY_SIGNAL_WEIGHT = 0.15
# Sanity clamp on P_model.
MODEL_PROB_MIN = 0.02
MODEL_PROB_MAX = 0.98
# Drift veto: a directional bot never buys the side that CONTRADICTS a drift
# reading of at least this magnitude. Live evidence (2026-07-16 overnight run):
# drift-contradicting trades 26% WR / -$55 vs 52% agreeing. Below the floor
# (drift ~ 0) flow-only trades are allowed — they measured break-even.
DRIFT_VETO_MIN = 0.05
# Continuous flow-only edge scaling (BUG #30, 2026-07-20). The old step
# function only penalized |drift| < 0.10 (full 2x tax below, full trust at or
# above). The 279-trade / 24h run that followed showed the STEP was in the
# wrong place: |drift| < 0.10 ran 33.3% WR / -$49.35 as expected, but the
# 0.10-0.30 "mid" band — released to full trust by the step — was actually
# the single biggest dollar loss (135 trades, 49.6% WR, -$76.32), while only
# |drift| >= 0.30 cleared real predictiveness (79.3% WR, +$25.58). A drift
# reading of 0.12 carries barely more information than 0.05; the old function
# treated it as fully trustworthy. The multiplier now tapers LINEARLY from
# FLOW_ONLY_EDGE_MULT_MAX at drift=0 down to 1.0x (full trust) at
# FLOW_ONLY_DRIFT_FULL_TRUST, so the mid band pays a graduated tax instead of
# a cliff-edge free pass. DRIFT_VETO_MIN (0.05) is unchanged — contradicting
# even a small drift reading is still vetoed outright regardless of this scale.
# Re-tightened to 2.0 after data-gathering window (2026-08 audit): the
# graduated mid-band tax is required; 1.5× left too much mid-drift noise.
FLOW_ONLY_EDGE_MULT_MAX = 1.35     # 2.0→1.35 (audit: noisy cvd/pm lanes kill-switched; graduated tax still applies but gentler)
# Full flow-only tax lifts only once |drift| reaches this. Recal 2026-08-07:
# moderate z (0.15–0.50) was anti-predictive for directionals; require stronger
# conviction before trusting flow/strat residual. Graduated tax below this.
FLOW_ONLY_DRIFT_FULL_TRUST = 0.25   # 0.45→0.25 (audit: moderate z at 0.25± is genuinely predictive now that noisy lanes are gone)
# Adaptive mom saturation scale (1m return → soft_saturate): track live σ so
# high-vol tape does not treat every 0.2% candle as full signal (same disease
# as under-scaled drift). Off → fixed MOM_SCALE_PRIOR.
MOM_ADAPTIVE_SCALE = True
MOM_SCALE_PRIOR = 0.002           # 0.2% 1m → ~0.76 (historical p97)
MOM_SCALE_MIN = 0.0015
MOM_SCALE_MAX = 0.005
MOM_SCALE_VOL_MULT = 1.35         # mom_scale ≈ mult · σ_1m (σ_1m = σ_5m/√5)

# --- Dead-zone gate (2026-07-21) — the single biggest live leak ---
# Over the 290-trade run the 0.42-0.58 price band with |drift| below
# DEAD_ZONE_DRIFT_MIN was 59 trades, 39.0% WR, -$77.83: the model taking a
# low-conviction opinion against a near-coin-flip market. The continuous
# flow-only tax alone (above) did not suppress them. Crucially the SAME price
# band with |drift| >= 0.30 still profited (+$30.10, 65.7% WR) — the validated
# "market lags drift" money — so the gate is drift-CONDITIONAL: a directional
# bot sits flat when the chosen side's MID is in the coin-flip band AND drift
# is flat. Zone bots (sniper/makers) override make_decision and carry their own
# drift gates, so this only affects the directional signal path. Regime-agnostic
# (keys off |drift|, not a side).
DEAD_ZONE_PRICE_LO = 0.42
DEAD_ZONE_PRICE_HI = 0.58
DEAD_ZONE_DRIFT_MIN = 0.15          # 0.20→0.15 (audit: 0.10–0.20 band was profitable at 65.7% WR +$30.10)
# Quiet / range regimes: mid-band "market lags drift" at |drift| 0.10–0.20 was
# a major leak (2026-07-29 soak: 0.50–0.58 band −$35.9 at 48.5% WR under
# low_vol_range). Require stronger drift before allowing coin-flip mids.
DEAD_ZONE_QUIET_DRIFT_MIN = 0.18    # 0.30→0.18 (audit: 0.10–0.20 was profitable in quiet; raised floor still blocks sub-0.10 noise)
DEAD_ZONE_QUIET_REGIMES = (
    "low_vol_range",
    "low_vol_trend",
    "quiet",
)

# --- Extreme-drift market-lag gate (soak 2026-07-27) ---
# |drift| 0.30–0.50 was the money zone (85% WR); |drift| ≥ 0.50 lost (41% WR).
# Do not hard-veto extreme drift — require the market to still LAG (side mid
# at or below the harness "market lags" ceiling). Same spirit as meanrev's
# STRATEGY_MAX_SIDE_PRICE / sniper lag rule.
DRIFT_EXTREME_ABS = 0.50
DRIFT_EXTREME_MAX_SIDE_MID = 0.58
# Audit 2e: deep underdog floor. |drift| >= DRIFT_EXTREME_ABS with side_mid
# below this means the market prices uncertainty, not lag (40-45% WR). Skip.
DRIFT_EXTREME_UNDERDOG_FLOOR = 0.25

# Conviction-scaled trust (2026-07-17 chop-regime leak): trust_eff =
# trust * min(1, |P_model - 0.5| / MODEL_CONVICTION_SCALE). The edge formula
# trust*(P_model - mid) derives its MAGNITUDE from the market's displacement,
# so a near-ignorant model (lean 0.01-0.03) used to book a 3-7c "edge"
# whenever the mid moved away from 0.5 — a structural underdog-fade that ran
# 38.5% WR / -$22 in the 2026-07-17 chop run (YES side 10% WR). Scaling
# trust by the model's own information content kills that trade class while
# leaving the validated market-lags-drift rule (+19.5c/share offline, model
# lean >= 0.10) at full trust. 0.10 = the lean where trust saturates; a
# drift-0.5 reading (lean 0.1125 on the momentum profile) keeps full trust.
# Full trust at lean ≥ 0.10 (validated market-lags-drift band). Re-tightened
# from the 0.06 data-gathering value (2026-08 audit).
MODEL_CONVICTION_SCALE = 0.10
# Hard model-lean floor (BUG #27, 2026-07-17 evening run). Conviction-scaled
# trust DAMPED weak models but still let them trade into large market
# displacement (a trust_eff=0.03 trade is in the log). Below the floor the
# bot has no tradable opinion: skip. RECALIBRATED 0.10 -> 0.05 (2026-07-18):
# 0.10 was measured against the OLD model distribution, where the saturated
# cvd/pm lanes inflated leans; with those lanes killed the same floor
# demanded |drift| >= 0.286 from the drift-pure meanrev profile — while the
# harness validates follow-drift with no magnitude bar (+7.6c/share) and
# puts the ignorance boundary at |drift| ~ 0.15 (its underdog probe, -4.44c/
# share). 0.05 maps the drift-pure profile onto exactly that boundary
# (0.70 * 0.15 * 0.5 = 0.052). The 0.05-0.10 band still trades under DAMPED
# trust (conviction scaling re-engages there) and flow-only trades keep the
# 2x MIN_EDGE bar, so the ignorance-fade class stays suppressed.
MODEL_LEAN_MIN = 0.05
# Book-consistency gate (BUG #27): when the YES and NO book prices disagree
# with each other (|yes + no - 1| beyond this), the data is suspect (stale or
# gapped book) — a directional bot stands down. A REAL cross-book gap is the
# arbitrage bot's two-legged trade; harvesting it one-legged is a coin flip
# minus fees, and Kelly max-sized exactly those trades (19:31/19:34, sums
# 0.84-0.85, 31-34 shares, -$29.15 in two trades). Normal sums cluster
# 0.98-1.02 live.
BOOK_SUM_TOLERANCE = 0.04

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
# Clamp on the edge fed into Kelly SIZING (the trade/skip decision still uses
# the raw edge). Live evidence (2026-07-19 24h run): the 15 biggest bets went
# 8/15 for -$34, and avg loss size exceeded avg win size — an outsized "edge"
# usually means the model maximally disagrees with the market, which is when
# its inputs are most likely stale/wrong, not when it knows the most. Edges
# above the cap size as if they were exactly the cap.
# Hard cap on edge fed into Kelly (trade/skip still uses raw edge). Lowered
# 0.10 → 0.08 after confidence-inversion soak: max-edge trades were the
# worst WR band once everything above the cap max-sized equally.
KELLY_EDGE_CAP = 0.08
# Concave sizing calibration (bots/edge_calibration.py): full Kelly credit
# for edges ≤ this; diminishing returns between here and KELLY_EDGE_CAP.
EDGE_CALIB_FULL_CREDIT = 0.04
EDGE_CALIB_TAPER_SCALE = 0.06
# How long make_decision may reuse the last bankroll read (it runs per-bot
# per-second; the pool changes only on fills/resolutions).
SIZING_BANKROLL_CACHE_SEC = 5.0

# --- Portfolio capital allocation (arena/portfolio.py) ---
# When enabled, each bot Kelly-sizes against bankroll × weight instead of the
# full shared pool (N bots × full bankroll oversubscribed correlated risk).
# Weights sum to 1; rebalance on timer and/or regime change. Editable in the
# dashboard Settings → Portfolio Allocation card.
PORTFOLIO_ALLOCATION_ENABLED = True   # default on; Kelly sizes against bankroll × weight
PORTFOLIO_METHOD = "kelly_portfolio"  # equal | sharpe | expectancy | kelly_portfolio
# Dual-window blend: long lookback stabilizes weights; short window keeps
# regime freshness without letting a lucky 6h FZM streak steal 20% capital.
PORTFOLIO_WINDOW_HOURS = 48.0         # primary (long) lookback
PORTFOLIO_FAST_WINDOW_HOURS = 12.0    # short window for blend
PORTFOLIO_LONG_WEIGHT = 0.65          # blend = long*W + fast*(1-W)
PORTFOLIO_MIN_TRADES = 6              # 5m tape: ~2–3 directional fills/hour
# Dual-window ready: also ready if short window has enough consistent samples
PORTFOLIO_FAST_READY_MIN_TRADES = 4
PORTFOLIO_FAST_READY_ENABLED = True
# When zero veterans, equal-split free capital (don't leave pool dark)
PORTFOLIO_COLD_START_EQUAL = True
# Strategy-family correlation priors (tandem risk) when market-overlap ρ is thin
PORTFOLIO_FAMILY_CORR_PRIOR = 0.75
PORTFOLIO_FAMILY_GROUPS = (
    ("momentum", "hybrid", "phantom"),
    ("mean_reversion", "mean_reversion_sl", "mean_reversion_tp"),
)
PORTFOLIO_MIN_WEIGHT = 0.0            # no forced floor — losers can go to 0
PORTFOLIO_MAX_WEIGHT = 0.50           # hard cap even for proven winners
# Until live edge is proven (n≥EDGE_PROVEN_MIN_N and expectancy>0), no bot
# may take more than UNPROVEN_MAX_WEIGHT of the pool (sweeper overnight soak
# hit ~29% with negative EV).
PORTFOLIO_UNPROVEN_MAX_WEIGHT = 0.20
PORTFOLIO_EDGE_PROVEN_MIN_N = 20
# After n≥NEG_EXP_MIN_N with expectancy<0: strip manual floors and zero auto
# weight so losers cannot keep capital via override floors.
PORTFOLIO_NEG_EXP_MIN_N = 6
PORTFOLIO_NEG_EXP_MAX_WEIGHT = 0.10   # paper-eval floor; 0.0 starves losers to silence
PORTFOLIO_CORR_SHRINK = 0.65          # 0..1: how hard correlation cuts raw score
PORTFOLIO_CORR_MIN_OVERLAP = 8        # shared markets needed to estimate ρ
PORTFOLIO_COLD_START_SCORE = 0.05     # tiny score for bots under sample floor
PORTFOLIO_LOSER_SCORE = 0.0           # ready bots with neg expectancy → zero weight
PORTFOLIO_ARB_FIXED_EQUAL = True      # pin arbitrage at 1/N (market-neutral staple)
# Lock-in (arb + sweeper) stay at 1/N; Core (mom/meanrev/sniper/hybrid) auto-adjusts.
PORTFOLIO_LOCKIN_FIXED_EQUAL = True
# Idle-arb shrink disabled: it pinned arb ~5% at startup. Lock-in stays even.
PORTFOLIO_ARB_DYNAMIC_ENABLED = False
PORTFOLIO_ARB_DYNAMIC_IDLE_HOURS = 6.0
PORTFOLIO_ARB_DYNAMIC_MIN_WEIGHT = 0.04   # unused while DYNAMIC_ENABLED is False
PORTFOLIO_REBALANCE_INTERVAL_SEC = 30 * 60  # 30 min periodic rebalance
PORTFOLIO_REBALANCE_ON_REGIME = True  # also rebalance on regime_detector flip
# Only rebalance on regime *after* the new regime has been held this long
# (avoids thrashing when the detector chatters at quiet-tape boundaries).
PORTFOLIO_REGIME_REBALANCE_MIN_DWELL_SEC = 600.0  # 300→600 (audit: boundary chatter flips every few min; 10min dwell)

# --- Risk Engine (arena/risk_engine.py) ---
# Central pre-trade gates + continuous evaluation: daily loss, drawdown,
# size taper, underperformance pause, historical VaR, kill switch (dashboard /
# API / flag file). Soft paper defaults are real (not 999999) so paper runs
# exercise the same muscle memory as live; override in dashboard or config.
RISK_ENGINE_ENABLED = True
RISK_EVAL_INTERVAL_SEC = 15          # full recompute on evolution-loop host
RISK_HOTPATH_CACHE_SEC = 2.0         # pre_trade / size_mult cache
RISK_PAPER_BOT_DAILY_LOSS = 75.0     # net daily P&L floor per bot (paper)
RISK_PAPER_PORTFOLIO_DAILY_LOSS = 150.0
# None → use LIVE_MAX_DAILY_LOSS_* in live mode, paper defaults above in paper
RISK_BOT_DAILY_LOSS = None
RISK_PORTFOLIO_DAILY_LOSS = None
RISK_BOT_MAX_DRAWDOWN = 0.35         # pause bot at 35% peak-to-trough
RISK_PORTFOLIO_MAX_DRAWDOWN = 0.40
RISK_DRAWDOWN_WINDOW_HOURS = 24.0
RISK_SIZE_REDUCE_DD_FRAC = 0.40      # start tapering earlier (was 0.50)
RISK_SIZE_REDUCE_MIN_MULT = 0.25     # floor mult before hard pause
RISK_UNDERPERFORM_PAUSE_PNL = -30.0  # was −40; catch mom-class bleed sooner
RISK_UNDERPERFORM_WINDOW_HOURS = 12.0
RISK_UNDERPERFORM_MIN_TRADES = 12    # was 15; 5m markets need faster feedback
# Graduated underperform taper (audit 1e): replace binary pause with size
# reduction at intermediate loss thresholds.
RISK_UNDERPERFORM_GRADUATED = True
RISK_UNDERPERFORM_GRADUATED_TIERS = (
    (-20.0, 0.75),   # at −$20 → size ×0.75
    (-30.0, 0.50),   # at −$30 → size ×0.50
    (-40.0, 0.25),   # at −$40 → size ×0.25 (never zero)
)
RISK_VAR_CONFIDENCE = 0.95
RISK_VAR_MIN_TRADES = 20
RISK_VAR_LIMIT_USD = None            # optional portfolio VaR hard reduce
RISK_EVENT_LOG_MAX = 500
# Kill-switch flag file (create to arm, delete to clear). Also settable via
# dashboard / POST /api/risk/kill-switch and arena_state kill_switch.
# Re-bound under LOG_DIR after that path is resolved (see below).
RISK_KILL_SWITCH_FILE = str(Path(__file__).parent / "logs" / "KILL_SWITCH")

# --- Production alerts + health (arena/alerts.py, arena/health.py) ---
# Master switch defaults ON when at least one channel has credentials configured
# (see arena/alerts._default_config). Explicit dashboard Off still wins once saved.
ALERTS_ENABLED = False              # static fallback when no channel credentials
ALERT_DEBOUNCE_SEC = 300            # min seconds between identical alerts
ALERT_HOURLY_REPORT_SEC = 3600      # cadence for hourly performance digests
# Daily EOD: after this America/New_York hour, send previous ET calendar day's
# summary once (default 0 = just after midnight ET + grace for late resolutions).
ALERT_DAILY_REPORT_HOUR_ET = 0
ALERT_DAILY_REPORT_GRACE_MIN = 5    # wait a few minutes for late resolutions
# Deprecated alias — same semantic as HOUR_ET (kept so old env/docs still resolve).
ALERT_DAILY_REPORT_HOUR_UTC = ALERT_DAILY_REPORT_HOUR_ET
# Paper (and pool) capital warnings
ALERT_LOW_BANKROLL_USD = 25.0       # absolute available floor
ALERT_LOW_BANKROLL_FRAC = 0.50      # fraction of bankroll / PAPER_BANKROLL_DEFAULT
# Feed / market-data staleness
ALERT_FEED_STALE_SEC = 90.0
# Skip storm: large skip delta with almost no fills over the check window
ALERT_SKIP_STORM_MIN_SKIPS = 200
ALERT_SKIP_STORM_MAX_TRADES = 2
ALERT_SKIP_STORM_WINDOW_SEC = 600
# Pending trades older than this are "resolver stuck"
ALERT_RESOLVER_STUCK_AGE_MIN = 15.0
ALERT_RESOLVER_STUCK_MIN_COUNT = 2
# Portfolio rebalance digest when any bot weight moves by this much (absolute)
ALERT_PORTFOLIO_REBALANCE_MIN_SHIFT = 0.08
# Core-lane tuner: notify when applied |Δw| ≥ this (one step is 0.05)
ALERT_CORE_LANE_MIN_SHIFT = 0.05
# --- Inbound Telegram commands (arena/telegram_commands.py) ---
# Long-poll loop hosted by the DASHBOARD process (it outlives the arena, so
# `/status` still answers when the arena is the thing that died). Uses the same
# alert_telegram_* credentials; an update from any chat other than the
# configured alert_telegram_chat_id is dropped without a reply.
TELEGRAM_COMMANDS_ENABLED = True
# Control commands (/kill, /pause, /resume, /retire, /deploy). Turn OFF to make
# the bot read-only — the token then only exposes reports, never trading state.
TELEGRAM_COMMANDS_CONTROL_ENABLED = True
# Sender allowlist. Empty tuple = only accept messages whose sender id equals
# the chat id — i.e. the operator's own PRIVATE chat. Without this, pointing
# alert_telegram_chat_id at a GROUP would hand /kill and /retire to every
# current and future member of that group.
TELEGRAM_COMMANDS_ALLOWED_USER_IDS: tuple = ()
# Max age of a CONTROL command, measured from the message's own Telegram
# timestamp. A /kill queued while the laptop slept is an instruction about a
# moment that has passed — refuse it rather than fire it hours late. Reports
# are side-effect free and ignore this.
TELEGRAM_COMMANDS_MAX_AGE_SEC = 300
TELEGRAM_COMMANDS_POLL_TIMEOUT_SEC = 30   # server-side long-poll hold
TELEGRAM_COMMANDS_RATE_LIMIT_SEC = 3.0    # min seconds between same command

ARENA_LOG_STALE_SEC = 300           # health /healthz stale threshold
HEALTH_EVAL_INTERVAL_SEC = 60       # full health recompute on evolution loop

# Live learning bias: the raw-YES-WR learner was anti-predictive (-24pp) and
# double-counted price. Disabled in live decisions (outcomes still recorded)
# pending the edge-calibrated redesign. See spec R5.
LEARNING_ENABLED = False
# Fallback minimum cost-adjusted edge (probability units) to place a trade.
MIN_EDGE_DEFAULT = 0.012  # 2026-07-21 data-gathering: 0.02 -> 0.012 (see base_bot.MIN_EDGE)
# Maps the chosen side's edge -> sizing confidence (~0.10 edge -> 0.45 cap).
EDGE_TO_CONFIDENCE = 4.5
# A bot never buys a side priced above HIGH_PRICE_GUARD (bad risk/reward) or
# below CONSENSUS_GUARD (fighting strong market consensus). Symmetric per side.
HIGH_PRICE_GUARD = 0.72
CONSENSUS_GUARD = 0.35
# Shared-pool concentration cap (BUG #27): max fraction of the GROSS paper
# pool (bankroll + realized P&L, before open-cost deductions) that may be
# committed to one (market, side) across ALL bots. The directional bots read
# identical warm lanes and pile the same side within seconds (20 of 34
# groups had 3+ bots in the 2026-07-17 run) — per-bot Kelly doesn't know the
# pool already holds correlated positions, so hour-22's 4-bot clusters were
# ~4x leverage on single BTC candles. Later bots clamp to the remaining
# headroom or skip. Arbitrage (hedged, own execute()) is exempt. In live
# mode the cap base is LIVE_MAX_POSITION * 2 per market-side.
MARKET_SIDE_EXPOSURE_CAP = 0.30
# Correlation-aware concentration (long-term pile-in control): when counting
# open exposure toward MARKET_SIDE_EXPOSURE_CAP, weight each peer bot's open
# cost by max(corr(self, peer), EXPOSURE_CORR_FLOOR). ρ≈1 bots (momentum/
# phantom/hybrid) almost fully share the same budget slot so tandem fills
# cannot 4× one candle. Floor keeps partially-independent bots from free-riding.
EXPOSURE_CORR_AWARE = True
EXPOSURE_CORR_FLOOR = 0.35
# Hard cap on distinct bots already open on the same (market, side) before
# another directional bot is allowed in (arb exempt). Soft "one thesis" limit.
# Overnight 2026-08-20: unique-market +$3.84 vs multi-bot −$36. Model
# "edges" of 5–23¢ always cleared the 3.5¢ pile-in bar. One directional
# thesis per (market, side).
MARKET_SIDE_MAX_BOTS = 0              # 0 = unlimited (paper-eval sample size)
# Tighter tandem caps when live data says the regime/strategy is toxic.
# 0 = do not re-impose a bot-count cap (paper-eval). Restore 1 before live.
MARKET_SIDE_MAX_BOTS_BAD_REGIME = 0
MARKET_SIDE_MAX_BOTS_CHOP = 0
# Progressive EV pile-in gate (togglable): after peers already hold (market,
# side), a new bot must clear a higher edge bar — unless confidence is very
# high. Not a hard bot-count ban; complementary to MARKET_SIDE_MAX_BOTS.
PILEIN_EV_GATE_ENABLED = False
PILEIN_EV_EDGE_STEP = 0.025       # extra edge required per peer bot open
PILEIN_EV_MIN_EDGE = 0.035        # absolute min edge when ≥1 peer is open
# quality_confidence is a 0–0.95 *structure* score, not P(win). Soak
# 2026-08-19: tandem mom/hybrid/sniper routinely printed 0.82–0.91 and
# skipped this bar — unique-market +$24 vs multi-bot −$12. Sit above the
# 0.95 cap so ordinary structure cannot bypass the extra-edge requirement.
PILEIN_EV_CONF_BYPASS = 0.96
PILEIN_EV_EXEMPT_TYPES = ("arbitrage",)  # market-neutral legs

# --- One directional trade per evaluation (trader tick) ---
# Phase-1: all bots make_decision; phase-2: only the best-edge directional
# buy executes. Structural strategies (arb, sweeper, makers) stay exempt.
ONE_TRADE_PER_TICK = False
# When the one-per-tick ranker is on, hybrid yields if a dedicated directional
# is already pending the same side. Paper-eval default OFF so hybrid still fills.
HYBRID_YIELD_ENABLED = False
ONE_TRADE_PER_TICK_EXEMPT = (
    "arbitrage", "sweeper", "late_window_maker", "fee_zone_maker",
    "btc_maker", "true_maker", "copy_trade",
)
# Optional window lock: after any directional fill on a market, no other
# directional bot may open that window. OFF by default; Settings toggle.
DIRECTIONAL_WINDOW_LOCK = False
DIRECTIONAL_WINDOW_LOCK_EXEMPT = ONE_TRADE_PER_TICK_EXEMPT

# --- Lag-justified edge (all directionals; regime-agnostic) ---
# Require drift-implied fair to beat side mid + fee by min residual before
# buying. Continuous alternative to hard mid caps: high mids still trade when
# |drift| is large enough that lag remains. Sniper/makers already enforce this
# more strictly; this is the shared BaseBot floor.
# --- High-vol favorite tax (soak 2026-08-24: 50–58¢ × high_vol_trend −$15) ---
# Chosen-side MID ≥ 0.52 in high_vol_trend: raise min_edge and cut size.
# Cheap lag (42–50¢) and cheap NO are untouched.
HIGH_VOL_FAVORITE_ENABLED = True
HIGH_VOL_FAVORITE_REGIMES = ("high_vol_trend",)
HIGH_VOL_FAVORITE_MID = 0.52
HIGH_VOL_FAVORITE_STRATEGIES = ("momentum", "sniper", "hybrid")
HIGH_VOL_FAVORITE_EDGE_MULT = 1.50
# Kalshi NO paper tax stub — 1.0 = off (do not tighten dual-gate). Next soak
# can raise this without a rewrite if Kalshi NO stays ~35% WR.
KALSHI_NO_EDGE_MULT = 1.0
HIGH_VOL_FAVORITE_SIZE_MULT = 0.60
# Mom sign fights drift: raise min_edge so Φ-YES at 58¢ with mom=−0.83 dies.
MOM_DRIFT_FIGHT_ENABLED = True
MOM_DRIFT_FIGHT_MOM_ABS = 0.50
MOM_DRIFT_FIGHT_EDGE_MULT = 1.35
MOM_DRIFT_FIGHT_HIGH_VOL_EDGE_MULT = 1.75

LAG_JUSTIFIED_ENABLED = True
LAG_JUSTIFIED_MIN_EDGE = 0.02      # implied − mid − fee must clear this
LAG_JUSTIFIED_EXEMPT = (
    "arbitrage", "sweeper", "late_window_maker", "fee_zone_maker",
    "btc_maker", "true_maker",
)
# Phase-aware drift trust: damp |lane drift contribution| in noisy open/mid
# phases; full trust near settlement when TWAP certainty is high.
DRIFT_PHASE_TRUST_ENABLED = True
DRIFT_PHASE_TRUST = {
    "open": 0.70,
    "mid": 0.85,
    "pre_settle": 0.95,
    "settlement": 1.0,
}
# When strat is zeroed (confirm-mode fight), park weight as uncertainty that
# lowers trust rather than redistributing into drift (2026-08-11 overconf).
STRAT_FIGHT_UNCERTAINTY = True
STRAT_FIGHT_TRUST_DAMP = 0.55  # trust_eff *= 1 − damp * uncertainty_share

# Sticky risk taper: once reduced, stay reduced until DD recovers below
# start_frac * recovery_ratio (stops reduced↔active flapping at the boundary).
RISK_STICKY_TAPER = True
RISK_STICKY_RECOVERY_RATIO = 0.75  # must recover to 75% of taper-start DD

# --- Order execution: limit-first (maker fee = 0 when resting) ---
# "limit" posts a buy limit; "market" keeps the legacy walk-the-asks path.
# Limit price modes for BUYs:
#   passive_mid — min(mid, best_ask − tick): prefer resting maker (fee 0)
#   join_bid    — best_bid (true join; lowest fill rate, pure maker)
#   aggressive  — best_ask (marketable limit; still taker fee when it crosses)
ORDER_STYLE = "limit"
# cap_ask: buy limit = best ask. Immediate fill, taker fee, no book-walk past
# the displayed ask. Honest live-equivalent for 5-min directionals that need
# the fill *now*. join_bid / passive_mid remain available for makers.
LIMIT_PRICE_MODE = "cap_ask"
LIMIT_TICK = 0.01
# Paper must not invent a maker fill when the limit does not cross. Live
# already only logs a trade when the CLOB reports matched. Sweeper / true
# maker pass an explicit limit_price and fill when marketable.
LIMIT_PAPER_ASSUME_MAKER_FILL = False
# Live unique-market scorecard (Signal Lab) + gate tuner.
LIVE_SCORECARD_HOURS = 72
LIVE_SCORECARD_INTERVAL_SEC = 300
GATE_TUNE_ENABLED = True
GATE_TUNE_APPLY = False         # suggest-only until a human/toggle enables apply
GATE_TUNE_MIN_MARKETS = 30
GATE_TUNE_LOOSEN_WR = 0.58
GATE_TUNE_LOOSEN_EDGE = 0.02    # +2¢/share hyp P&L after taker fee
GATE_TUNE_APPLY_COOLDOWN_SEC = 86400  # one step per knob per day if apply is on
# Dual-gate hyp is scored on the 50–58¢ band separately from 90¢ locks.
GATE_TUNE_MIDBAND_LO = 0.50
GATE_TUNE_MIDBAND_HI = 0.58
GATE_TUNE_CHEAP_MAX = 0.62
# Combination / foundational-rule explorer (Signal Lab). Measures pairwise
# lane agreement + named rules on unique-market rows after taker fee.
# Confirm-apply uses only *earned cheap* combos — it does not loosen
# dual-gate / lean floor / sweeper certainty globally.
COMBO_EXPLORE_ENABLED = True
COMBO_EXPLORE_HOURS = 72
COMBO_EXPLORE_INTERVAL_SEC = 300
COMBO_DEADBAND = 0.05
COMBO_MAX_ENTRY = 0.62          # never earn/apply on sweeper-book favorites
COMBO_MIN_MARKETS = 20
COMBO_MIN_ACCURACY = 0.55
COMBO_MIN_NET_EDGE = 0.0
COMBO_CONFIRM_APPLY = False     # suggest-only until a combo earns on cheap fills
COMBO_CONFIRM_MIN_LANES = 2
COMBO_MIDBAND_LO = 0.50
COMBO_MIDBAND_HI = 0.58

# --- Regime-adaptive policy (PLAN 2026-08-05: adapt weights, not starve) ---
# Primary response to a weak regime is reweight lanes + capital routing
# (regime_profiles / regime_router), not hard-skip or deep size cuts.
REGIME_ADAPT_ENABLED = True
REGIME_ADAPT_PRIMARY = "style"   # "style" | "throttle" (legacy)
REGIME_ADAPT_MIN_TRADES = 15
REGIME_ADAPT_BAD_WR = 0.48       # at/below → size toward MIN
REGIME_ADAPT_GOOD_WR = 0.62      # at/above → size toward MAX
REGIME_ADAPT_SIZE_MIN = 0.85     # was 0.35 — keep frequency, mild taper only
REGIME_ADAPT_SIZE_MAX = 1.15
REGIME_ADAPT_CACHE_SEC = 15.0
# Continuous edge tax from blended strategy×regime WR (data-driven)
REGIME_ADAPT_CONT_EDGE_MAX = 1.55   # mult at wr ≪ BAD
REGIME_ADAPT_CONT_EDGE_MIN = 0.95   # mult at wr ≫ GOOD
REGIME_ADAPT_CONT_DRIFT_MAX = 0.10  # extra |drift| floor when wr very bad
REGIME_ADAPT_CONT_MID_MAX = 0.45    # mid-band floor ceiling from live WR
# Hard stand-down: OFF by default; emergency-only path when enabled.
REGIME_HARD_SKIP_ENABLED = False
REGIME_HARD_SKIP_EMERGENCY_ONLY = True
REGIME_HARD_SKIP_MIN_TRADES = 80   # was 20 — avoid thin-sample freezes
REGIME_HARD_SKIP_WR = 0.38
REGIME_HARD_SKIP_CLEAR_WR = 0.48
REGIME_HARD_SKIP_REQUIRE_NEG_PNL = True
# Coin-flip favorite band: mid in [0.50, 0.58] needs stronger drift/lag
# (2026-08: 229 trades, 50% WR, −$37; with low_vol_trend −$49).
MID_COINFLIP_LO = 0.50
MID_COINFLIP_HI = 0.58
MID_COINFLIP_DRIFT_MIN = 0.40
MID_COINFLIP_DRIFT_MIN_BAD_REGIME = 0.45

# --- Decision-event log (counterfactual learning) ---
# Hot path only enqueues; a background flusher batch-inserts. Non-buy actions
# are throttled per (bot, market) so 1s re-evals do not flood SQLite. Buys
# always log. Resolved against market outcomes for lane/strategy fine-tuning
# beyond the trade-only sample.
DECISION_LOG_ENABLED = True
DECISION_LOG_MIN_INTERVAL_SEC = 20.0   # throttle non-buy per (bot, market)
DECISION_LOG_FLUSH_SEC = 2.0
DECISION_LOG_QUEUE_MAX = 8000
DECISION_ROLLUP_INTERVAL_SEC = 900     # offline rollup cadence (evolution loop)
# When True, core tuner + lane promoter prefer decision_events (incl. skips)
# over trade-only reasoning parses once enough resolved decisions exist.
DECISION_LEARN_FROM_ALL = True
DECISION_LEARN_MIN_RESOLVED = 30       # floor before replacing trade-only path
# Hybrid meta-learner counterfactuals (bots/meta_learner.py): score sub-votes
# from resolved decision_events skips (would-be trades), not only filled buys.
# CF step is scaled by HYBRID_META_CF_ETA_SCALE so one skip ≠ one real trade.
HYBRID_META_CF_ENABLED = False         # fills only — skip CF overfit (2.5× mom)
HYBRID_META_MAX_MULT = 1.20            # cap online multiplier until fill n is large
HYBRID_META_CF_ETA_SCALE = 0.25        # CF Hedge step = eta * this
HYBRID_META_CF_MAX_PER_CYCLE = 200     # bound per maybe_update pass

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
COPYTRADING_ENABLED = False
COPYTRADING_MAX_WALLETS_TO_TRACK = 10
COPYTRADING_POSITION_SIZE_FRACTION = 0.5  # Copy 50% of whale's position size
COPYTRADING_DAILY_LOSS_LIMIT = 50.0     # Max USDC in realized losses per calendar day (wins are unlimited)
COPYTRADING_MAX_TRADES_PER_CYCLE = 5    # Max trades to execute per arena loop cycle
COPYTRADING_MIN_PRICE = 0.40            # Skip trades where whale's entry price < this
COPYTRADING_MAX_PRICE = 0.65            # Skip trades where whale's entry price > this (expensive bets need 65%+ WR to break even)
COPYTRADING_COPY_NO_BETS = False        # Copy NO bets — data shows NO side loses money, skip by default
COPYTRADING_BLOCKED_HOURS_UTC = [22]    # UTC hours to skip entirely (22:00 = -$76 in data)

# Dashboard Settings (env overrides match bin/arena / docker-compose)
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT") or "8501")
DASHBOARD_HOST = os.environ.get("DASHBOARD_HOST") or "0.0.0.0"

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
# CLOB book GET timeout (warmer path). Keep short so a hung CLOB cannot stall
# the whole warm cycle; fail soft to last snapshot / invalid book.
BOOK_FETCH_TIMEOUT_SEC = 2.0
# Kill-switched feeds (CVD/PM) refresh on this slower cadence while their
# global weights are 0 — frees the 1s cycle for books. When a lane is live
# via override, warmer refreshes them every cycle again.
SIGNAL_SLOW_REFRESH_SEC = 10.0
# Trader skips the tick when warm data is older than this (or missing).
# Never blocks the 1s tick on a cold refresh_price (15s timeout trap).
WARM_MAX_AGE_SEC = 3.0
# Shared-pool exposure headroom cache (invalidate on successful place).
EXPOSURE_CACHE_TTL_SEC = 1.5

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

# --- Live ops ---
# When any bot is live, require at least one alert channel (Telegram/Discord
# webhook env or configured credentials). Fail-closed at startup so unattended
# live never runs blind.
LIVE_REQUIRE_ALERTS = True

# --- Learned-rules skip bandit safety ---
# Never soften dead-zone purely on counterfactual WR; require positive $ PnL
# of the counterfactuals AND a non-flat drift band. Softening the largest
# historical $ leak on WR alone re-opens BUG #31.
LEARNED_RULES_SOFTEN_REQUIRE_PNL = True
LEARNED_RULES_SOFTEN_MIN_CF_PNL = 0.0
LEARNED_RULES_NEVER_SOFTEN = (
    "dead_zone", "drift_dual_gate", "price_quality",
    "strike_unconfirmed", "twap_coverage",
)
CORE_TUNE_NEVER_CUT_DRIFT = True
PORTFOLIO_EXPLORE_FLOOR = 0.05

# Polymarket enforces a per-order minimum of 5 shares. Bet sizing floors the
# spend so a trade always clears this (5 shares × price × buffer) — otherwise
# small-edge bets get rejected 'below_min_size' and never fill.
POLYMARKET_MIN_SHARES = 5

# --- Backtesting (backtest/ package — offline replay only) ---
# Historical order-book DEPTH is not archived by Polymarket, so backtest fills
# walk a SYNTHETIC ask ladder anchored on the recorded PM mid: best ask =
# mid + BACKTEST_HALF_SPREAD, then (offset, shares) tiers below. Tune these to
# stress liquidity assumptions; results are an optimistic upper bound either
# way (same caveat as the Signal Lab harness's stale-mid net-EV numbers).
BACKTEST_HALF_SPREAD = 0.01
BACKTEST_BOOK_DEPTH = [(0.00, 400.0), (0.01, 600.0), (0.02, 1000.0)]
BACKTEST_BANKROLL = PAPER_BANKROLL_DEFAULT   # starting virtual pool per run
BACKTEST_TICK_SEC = 60      # decision-tick spacing inside each 5-min window
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

# Logging — override with ARENA_LOG_DIR for Docker / non-default layouts.
LOG_DIR = Path(os.environ.get("ARENA_LOG_DIR") or (Path(__file__).parent / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Kill-switch lives next to the logs so a mounted data volume covers both.
RISK_KILL_SWITCH_FILE = str(
    Path(os.environ.get("ARENA_KILL_SWITCH_FILE") or (LOG_DIR / "KILL_SWITCH"))
)


# ---------------------------------------------------------------------------
# Environment overrides (opt-in) — operational knobs only
# ---------------------------------------------------------------------------
# A curated set of NON-safety knobs can be overridden from the environment so an
# operator can tune them without editing source (matches the DASHBOARD_* pattern
# from slice D). Deliberately EXCLUDED: TRADING_MODE (must start paper — flip it
# via the dashboard, never an env var) and the live risk caps / guard thresholds
# (those belong in reviewed code, not ambient environment). An unset var leaves
# the literal default above untouched; a malformed value fails fast below.
def _env_num(name: str, current, cast):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return current
    try:
        return cast(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid environment override {name}={raw!r}: expected "
            f"{cast.__name__}"
        ) from exc


PAPER_BANKROLL_DEFAULT = _env_num("ARENA_PAPER_BANKROLL", PAPER_BANKROLL_DEFAULT, float)
KELLY_FRACTION = _env_num("ARENA_KELLY_FRACTION", KELLY_FRACTION, float)
TRADE_LOOP_INTERVAL_SEC = _env_num("ARENA_TRADE_LOOP_INTERVAL_SEC", TRADE_LOOP_INTERVAL_SEC, float)
MARKET_DATA_INTERVAL_SEC = _env_num("ARENA_MARKET_DATA_INTERVAL_SEC", MARKET_DATA_INTERVAL_SEC, float)
HTTP_MAX_RETRIES = _env_num("ARENA_HTTP_MAX_RETRIES", HTTP_MAX_RETRIES, int)


# ---------------------------------------------------------------------------
# Fail-fast configuration validation (pydantic)
# ---------------------------------------------------------------------------
# Validate the safety-critical invariants and cross-field relationships at
# IMPORT time so a bad edit or env override crashes the arena at startup with a
# clear message — never mid-session against real (or simulated) money. This does
# NOT change the config.X import surface: every constant above is still a plain
# module global; this only asserts they are self-consistent.
class _ConfigInvariants(BaseModel):
    trading_mode: str
    taker_fee_rate: float = Field(gt=0, lt=1)
    kelly_fraction: float = Field(gt=0, le=1)
    model_lean_min: float = Field(ge=0, le=0.5)
    model_conviction_scale: float = Field(gt=0)
    book_sum_tolerance: float = Field(ge=0, lt=0.5)
    consensus_guard: float = Field(gt=0, lt=1)
    high_price_guard: float = Field(gt=0, lt=1)
    dead_zone_lo: float = Field(gt=0, lt=1)
    dead_zone_hi: float = Field(gt=0, lt=1)
    market_side_exposure_cap: float = Field(gt=0, le=1)
    paper_bankroll: float = Field(gt=0)
    live_max_position: float = Field(gt=0)
    evolution_window_hours: float = Field(gt=0)
    trade_loop_interval_sec: float = Field(gt=0)
    market_data_interval_sec: float = Field(gt=0)
    http_max_retries: int = Field(ge=0)
    regime_alloc_min_weight: float = Field(gt=0, lt=1)
    regime_alloc_max_tilt: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _relationships(self):
        if self.trading_mode not in ("paper", "live"):
            raise ValueError(f"trading_mode must be 'paper' or 'live', got {self.trading_mode!r}")
        if not (self.regime_alloc_min_weight < self.regime_alloc_max_tilt):
            raise ValueError(
                f"regime_alloc_min_weight ({self.regime_alloc_min_weight}) must be "
                f"below regime_alloc_max_tilt ({self.regime_alloc_max_tilt})"
            )
        if not (self.consensus_guard < self.high_price_guard):
            raise ValueError(
                f"consensus_guard ({self.consensus_guard}) must be below "
                f"high_price_guard ({self.high_price_guard})"
            )
        if not (self.dead_zone_lo < self.dead_zone_hi):
            raise ValueError(
                f"dead_zone_lo ({self.dead_zone_lo}) must be below "
                f"dead_zone_hi ({self.dead_zone_hi})"
            )
        return self


def _validate_config() -> None:
    """Raise RuntimeError with a clear message if the config is inconsistent."""
    try:
        _ConfigInvariants(
            trading_mode=TRADING_MODE,
            taker_fee_rate=POLYMARKET_TAKER_FEE_RATE,
            kelly_fraction=KELLY_FRACTION,
            model_lean_min=MODEL_LEAN_MIN,
            model_conviction_scale=MODEL_CONVICTION_SCALE,
            book_sum_tolerance=BOOK_SUM_TOLERANCE,
            consensus_guard=CONSENSUS_GUARD,
            high_price_guard=HIGH_PRICE_GUARD,
            dead_zone_lo=DEAD_ZONE_PRICE_LO,
            dead_zone_hi=DEAD_ZONE_PRICE_HI,
            market_side_exposure_cap=MARKET_SIDE_EXPOSURE_CAP,
            paper_bankroll=PAPER_BANKROLL_DEFAULT,
            live_max_position=LIVE_MAX_POSITION,
            evolution_window_hours=EVOLUTION_WINDOW_HOURS,
            trade_loop_interval_sec=TRADE_LOOP_INTERVAL_SEC,
            market_data_interval_sec=MARKET_DATA_INTERVAL_SEC,
            http_max_retries=HTTP_MAX_RETRIES,
            regime_alloc_min_weight=REGIME_ALLOC_MIN_WEIGHT,
            regime_alloc_max_tilt=REGIME_ALLOC_MAX_TILT,
        )
    except Exception as exc:  # pydantic.ValidationError or ValueError
        raise RuntimeError(f"Invalid arena configuration: {exc}") from exc


_validate_config()


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

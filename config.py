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

# Paper mode is a full simulation against real Polymarket order books (no order
# is submitted). All paper bots share ONE virtual USDC bankroll, set by the user
# in the dashboard Settings tab (arena_state key 'paper_bankroll'); this default
# is used until they set one. Live mode uses the real wallet USDC balance.
PAPER_BANKROLL_DEFAULT = 200.0

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
# Directed evolution (BUG #31): with the parent now chosen as the BEST-ranked
# survivor (not random) and lane weights auto-tuned per strategy by the
# core-lane tuner, mutation should EXPLOIT the proven config, not wander off it
# — a tighter jiggle keeps the mutant near a configuration that actually earned
# its survival instead of re-rolling the dice at 15%.
MUTATION_RATE_DIRECTED = 0.07
NUM_BOTS = 4
SURVIVORS_PER_CYCLE = 1  # Top 1 survives, bottom 3 replaced
# Judgment WINDOW is decoupled from the 2h cycle CADENCE (2026-07-19): judging
# on the 2h window with a 20-trade floor made every bot permanently IMMUNE
# (bots average 5-12 trades per 2h), so zero evolutions fired in the whole
# 24h v5 run while momentum-v1 bled -$86. The window is what a bot is judged
# ON; the interval is only how often the judgment runs.
EVOLUTION_WINDOW_HOURS = 24
# Raised 15 -> 30 (2026-07-21): a 5-min-market window of 15-20 resolved trades
# is dominated by noise — cycle 6 killed sniper-v1 on a 17-trade / -$8.49 dip
# one cycle after it survived at 61% WR, and mutated survivors were "judged" on
# 1-2 trades. Empirically the per-bucket WR/P&L numbers in the run only
# stabilized past ~30 samples, so a bot needs at least that many resolved this
# window before it can be replaced.
MIN_TRADES_FOR_JUDGMENT = 30   # Fewer resolved trades in the window = immune
# Survival bar is the BREAK-EVEN GAP (win_rate - avg_entry_price), not a flat
# WR threshold: 65% WR bought at 70c loses money while 55% bought at 45c
# prints. A bot survives if its gap clears this floor OR its window P&L is
# positive (good sizing can rescue a thin gap). The old MIN_WIN_RATE=0.65
# would have culled every bot in the v5 run including the profitable ones
# (best WR was 63.3%).
EVOLUTION_BE_GAP_MIN = 0.03    # survive if WR beats avg entry by >= 3c

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
STRAT_LANE_CONF_CAP = 0.30
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
SIGNAL_WEIGHT_XASSET = 0.0   # ETH/SOL cross-asset confirmation (signals/cross_asset.py)
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
LANE_MONITOR_DEADBAND = 0.05        # |reading| below this = no directional read
LANE_MONITOR_INTERVAL_SEC = 1800    # check cadence (piggybacks the evolution loop)

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
AUTO_APPROVE_MIN_ACCURACY = 0.55  # live sign-vs-outcome accuracy to auto-promote
AUTO_APPROVE_MAX_ACTIVE = 3       # cap on simultaneously-enabled CANDIDATE lanes

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
CORE_TUNE_MIN_TRADES = 40      # per-(strategy,lane) resolved readings before tuning
CORE_TUNE_HIGH_ACC = 0.56      # lane sign-accuracy above this => nudge weight UP
CORE_TUNE_LOW_ACC = 0.48       # below this => nudge weight DOWN (toward the band floor)
CORE_TUNE_STEP = 0.05          # per-cycle weight nudge (bounded, one step/lane/strategy)
CORE_TUNE_BAND = 0.20          # max |deviation| of a tuned weight from its class default
CORE_TUNE_WEIGHT_MAX = 0.90    # absolute ceiling on any single lane weight
CORE_TUNE_WEIGHT_MIN = 0.0     # absolute floor (the band around the default binds first)

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
# 2026-07-21 (data-gathering): loosened 2.0 -> 1.5. The full 2.0x tax + the
# fee-net MIN_EDGE floors + conviction scaling stacked into a ~6.5pt model-vs-ask
# bar that produced ~63k no_edge skips per ~12 trades (run starved of evaluation
# data). This partially reopens the moderate-drift band for measurement; the
# drift-veto, dead-zone, consensus and book-sum guards are unchanged. Revert to
# 2.0 once enough trades accumulate to judge per-drift-band P&L live.
FLOW_ONLY_EDGE_MULT_MAX = 1.5
FLOW_ONLY_DRIFT_FULL_TRUST = 0.30

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
DEAD_ZONE_DRIFT_MIN = 0.10

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
# 2026-07-21 (data-gathering): lowered 0.10 -> 0.06 so trust_eff saturates at a
# moderate lean (0.06) instead of 0.10, giving moderate-conviction models real
# edge instead of near-zero. Part of the loosening to un-starve the dataset (see
# FLOW_ONLY_EDGE_MULT_MAX note); revert to 0.10 after the eval window.
MODEL_CONVICTION_SCALE = 0.06
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
KELLY_EDGE_CAP = 0.10
# How long make_decision may reuse the last bankroll read (it runs per-bot
# per-second; the pool changes only on fills/resolutions).
SIZING_BANKROLL_CACHE_SEC = 5.0
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
MARKET_SIDE_EXPOSURE_CAP = 0.10

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

    @model_validator(mode="after")
    def _relationships(self):
        if self.trading_mode not in ("paper", "live"):
            raise ValueError(f"trading_mode must be 'paper' or 'live', got {self.trading_mode!r}")
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

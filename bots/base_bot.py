"""Abstract base class all arena bots inherit from."""

import random
import copy
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import db
import learning
import polymarket_fills

logger = logging.getLogger(__name__)

# Bankroll read for Kelly sizing, cached off the 1s hot path (the pool only
# changes on fills/resolutions). Shared across bots — the pool is shared too.
_bankroll_cache: tuple = (0.0, 0.0)  # (ts, value)
_kelly_cache: tuple = (0.0, 0.0)     # (ts, value)


def _kelly_fraction() -> float:
    """Kelly fraction for sizing, live-editable in dashboard Settings.

    Read from the DB (the dashboard runs in a separate process, so a module
    constant would never see edits) with the same short hot-path cache as the
    bankroll.
    """
    global _kelly_cache
    now = time.time()
    if (now - _kelly_cache[0]) < getattr(config, "SIZING_BANKROLL_CACHE_SEC", 5.0):
        return _kelly_cache[1]
    value = db.get_kelly_fraction()
    _kelly_cache = (now, value)
    return value


def _sizing_bankroll(mode: str) -> float:
    """Current bankroll for bet sizing (cached, config.SIZING_BANKROLL_CACHE_SEC).

    Paper: the shared virtual pool's available cash. Live: a notional bankroll
    consistent with the per-trade cap (wallet reads are too slow for the 1s
    tick; LIVE_MAX_POSITION already hard-caps exposure).
    """
    if mode == "live":
        pct = max(config.MAX_POSITION_PCT_OF_BALANCE, 0.01)
        return config.LIVE_MAX_POSITION / pct
    global _bankroll_cache
    now = time.time()
    if (now - _bankroll_cache[0]) < getattr(config, "SIZING_BANKROLL_CACHE_SEC", 5.0):
        return _bankroll_cache[1]
    value = max(0.0, db.get_paper_available())
    _bankroll_cache = (now, value)
    return value


class BaseBot(ABC):
    name: str
    strategy_type: str
    strategy_params: dict
    generation: int
    lineage: str

    # Exit strategy: None = hold to resolution (default)
    # "stop_loss" = exit when position is down stop_loss_pct
    # "take_profit" = exit when position is up take_profit_pct
    exit_strategy: str = None
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0

    # Each strategy type gets different parameters for differentiation.
    # This creates real competition for evolution to select from.
    STRATEGY_PRIORS = {
        "momentum": 0.52,       # slight YES bias — momentum tends bullish
        "mean_reversion": 0.48, # slight NO bias — mean reversion bets against crowd
        "mean_reversion_sl": 0.48,
        "mean_reversion_tp": 0.48,
        "sniper": 0.50,         # neutral — sniper uses its own rules
        "phantom": 0.52,        # trend-following
        "sentiment": 0.50,      # neutral
        "hybrid": 0.50,         # neutral
    }
    # Per-strategy MODEL weight profile. Each lane arrives normalized to
    # [-1, 1] in YES-frame and the weighted sum maps to a model probability:
    #   P_model = 0.5 + 0.5 * sum(w_lane * lane)
    # Differentiation is by EMPHASIS, never by a hardcoded direction — every
    # weight is >= 0 and every lane is regime-agnostic, so no strategy carries
    # a baked-in YES/NO bias (see memory: regime-agnostic-signals).
    #   momentum / phantom  — trend followers: drift anchor + heavy flow/momentum
    #   mean_reversion*     — fundamentals-only: near-pure drift; by ignoring
    #                         momentum/flow it naturally fades price moves that
    #                         BTC's actual position doesn't back
    #   sentiment           — order-flow reader: CVD (executed aggression) heavy
    #   hybrid              — balanced blend of everything
    # Fidelity redesign (BUG #27): with the pm/obi/cvd lanes killed pending
    # offline validation, the LIVE inputs are drift, mom (BTC candle trend)
    # and strat (this strategy's own analyze() thesis — now carrying a
    # PER-STRATEGY weight here instead of the old flat global 0.15, which was
    # too small to differentiate anyone). Live weights sum to ~1.0 per
    # strategy; dead lanes stay listed at their revival weights' position (0)
    # so re-enabling a validated signal is a one-line profile edit.
    #   momentum — trades the BTC short-term trend (mom lane + trend analyze)
    #   phantom  — EMA-crossover/breakout swing: analyze()-dominant
    #   meanrev  — fundamentals + fade: drift anchor + z-score reversion
    #              thesis => "buy the dip in the winning direction" (the two
    #              agree only when price overextends AGAINST the drift side)
    #   sentiment— in-market flow reader (raw pm/cvd via analyze; its lanes
    #              stay killed until validated)
    #   hybrid   — balanced ensemble of the sub-strategies
    STRATEGY_SIGNAL_PROFILE = {
        "momentum":          {"drift": 0.25, "mom": 0.45, "pm": 0.00, "cvd": 0.00, "obi": 0.00, "strat": 0.30},
        "phantom":           {"drift": 0.20, "mom": 0.30, "pm": 0.00, "cvd": 0.00, "obi": 0.00, "strat": 0.50},
        "mean_reversion":    {"drift": 0.70, "mom": 0.00, "pm": 0.00, "cvd": 0.00, "obi": 0.00, "strat": 0.30},
        "mean_reversion_sl": {"drift": 0.70, "mom": 0.00, "pm": 0.00, "cvd": 0.00, "obi": 0.00, "strat": 0.30},
        "mean_reversion_tp": {"drift": 0.70, "mom": 0.00, "pm": 0.00, "cvd": 0.00, "obi": 0.00, "strat": 0.30},
        "sentiment":         {"drift": 0.30, "mom": 0.00, "pm": 0.00, "cvd": 0.00, "obi": 0.00, "strat": 0.70},
        "hybrid":            {"drift": 0.40, "mom": 0.20, "pm": 0.00, "cvd": 0.00, "obi": 0.00, "strat": 0.40},
        "sniper":            {"drift": 0.50, "mom": 0.10, "pm": 0.00, "cvd": 0.00, "obi": 0.00, "strat": 0.15},
    }
    DEFAULT_SIGNAL_PROFILE = {"drift": 0.50, "mom": 0.10, "pm": 0.00, "cvd": 0.00, "obi": 0.00, "strat": 0.15}
    # How far fair value moves from the market mid toward the bot's own model.
    # fair = mid + trust * (P_model - mid): the bot only sees edge when its
    # model DISAGREES with the price — the honest replacement for the additive
    # tilt/alpha stack that manufactured edge by construction.
    STRATEGY_MODEL_TRUST = {
        "momentum": 0.50,
        "mean_reversion": 0.60,
        "mean_reversion_sl": 0.60,
        "mean_reversion_tp": 0.60,
        "sniper": 0.50,
        "phantom": 0.50,
        "sentiment": 0.50,
        "hybrid": 0.50,
    }
    # Minimum confidence to place a trade
    # (MIN_TRADE_CONFIDENCE removed 2026-07-17 — it was dead code: defined but
    # never read since the two-sided rewrite; the MIN_EDGE gate below is the
    # real trade filter.)
    # Minimum cost-adjusted edge (probability units) to place a trade. Two-sided
    # selection buys the side with the larger positive edge above this floor.
    MIN_EDGE = {
        "momentum": 0.015,
        "mean_reversion": 0.02,
        "mean_reversion_sl": 0.02,
        "mean_reversion_tp": 0.02,
        "sniper": 0.02,
        "phantom": 0.015,
        "sentiment": 0.02,
        "hybrid": 0.02,
    }

    def __init__(self, name, strategy_type, params, generation=0, lineage=None):
        self.name = name
        self.strategy_type = strategy_type
        self.strategy_params = params
        self.generation = generation
        self.lineage = lineage or name
        self._paused = False
        self.trading_mode = "paper"
        # (ts, total_resolved) cache for the learning-weight ramp — refreshed
        # off the 1s hot path (see make_decision). Resolved count changes only
        # when a trade settles (~60s), so a short TTL is plenty.
        self._perf_cache = None

    @abstractmethod
    def analyze(self, market: dict, signals: dict) -> dict:
        """Analyze market + signals and return a trade signal.

        Returns:
            {
                "action": "buy" | "sell" | "hold",
                "side": "yes" | "no",
                "confidence": 0.0-1.0,
                "reasoning": "why this trade",
                "suggested_amount": float,
            }
        """
        pass

    def _model_prob_yes(self, lanes: dict) -> float:
        """Model probability of YES from normalized signal lanes.

        ``lanes`` maps lane name -> value in [-1, 1] (YES-frame). Weighted by
        this strategy's profile and mapped to a probability. Lanes not in the
        profile (e.g. ``strat``/``learn``) carry their weight in the value.
        """
        prof = self.STRATEGY_SIGNAL_PROFILE.get(
            self.strategy_type, self.DEFAULT_SIGNAL_PROFILE)
        s = 0.0
        for k, v in lanes.items():
            s += prof.get(k, 1.0) * v
        return max(config.MODEL_PROB_MIN,
                   min(config.MODEL_PROB_MAX, 0.5 + 0.5 * s))

    def _compute_fair_yes(self, yes_mid: float, model_prob: float,
                          trust: float) -> float:
        """Fair YES probability: market mid pulled toward the bot's model.

        fair = mid + trust * (P_model - mid). If the market already prices the
        model's view, fair == mid and there is NO edge — a signal the market
        has absorbed earns nothing (the flaw in the old additive stack, where
        edge equalled the bonus terms by construction).
        """
        fair = yes_mid + trust * (model_prob - yes_mid)
        return max(0.02, min(0.98, fair))

    def _side_net_edges(self, model_prob: float, trust_eff: float,
                        yes_price: float, no_price: float) -> tuple:
        """Cost-adjusted edge per side, each anchored on its OWN book price.

        edge_side = trust_eff * (P_model_side - side_price) - taker_fee. The
        old form anchored fair on the YES mid but paid the NO book, so any
        cross-book gap (stale/inconsistent books, yes+no != 1) landed in the
        NO edge as phantom directional signal with zero model input — and
        Kelly max-sized exactly those trades (BUG #27). Per-side anchoring
        makes edge purely model-vs-that-side's-price; real cross-book gaps
        belong to the arbitrage bot's two-legged trade.
        """
        edge_yes = (trust_eff * (model_prob - yes_price)
                    - polymarket_fills.taker_fee(1.0, yes_price))
        edge_no = (trust_eff * ((1.0 - model_prob) - no_price)
                   - polymarket_fills.taker_fee(1.0, no_price))
        return edge_yes, edge_no

    def make_decision(self, market: dict, signals: dict) -> dict:
        """Make a trading decision using market price edge + strategy + learning.

        Signal hierarchy:
        1. Market price edge (strongest — when price is far from 50c, follow it)
        2. BTC momentum (if price is moving, lean that direction)
        3. Strategy analysis (adds differentiation between bots)
        4. Learned bias (accumulates over time, adjusts everything)

        Skips trades when confidence is too low (no edge = no bet).
        """
        # `or 0.5`, not a .get default: _normalize sets current_price=None
        # explicitly (key present), so a default arg wouldn't fire. A book that
        # is momentarily unavailable leaves it None — coalesce to neutral 0.5.
        market_price = market.get("current_price") or 0.5

        # --- Lane: BTC momentum (normalized to [-1, 1]) ---
        prices = signals.get("prices", [])
        btc_latest = signals.get("latest", 0)
        price_momentum = 0.0
        if len(prices) >= 2 and prices[-1] > 0:
            price_momentum = (prices[-1] - prices[-2]) / prices[-2]
        elif btc_latest > 0 and len(prices) >= 1 and prices[-1] > 0:
            # Use live price vs last closed candle
            price_momentum = (btc_latest - prices[-1]) / prices[-1]
        # No candles at all -> 0. (The old fallback leaked the market price in
        # as "momentum", i.e. favorite-following in disguise.)
        # Saturation at a 0.2% one-candle move (~p97 of real BTC 1-min moves;
        # median is 0.022%). The first normalization saturated at 0.05% — BELOW
        # the median — so the lane sat at +/-0.5..1.0 of pure noise and outvoted
        # the time-damped drift early in the window (26% WR on the 34 trades
        # that contradicted drift, -$55 — the whole loss of that run).
        momentum_signal = max(-1.0, min(1.0, price_momentum * 500))

        # --- Lane: strategy thesis from analyze() ---
        raw_signal = self.analyze(market, signals)
        strategy_signal = 0.0
        if raw_signal["action"] != "hold":
            strategy_yes = 1.0 if raw_signal["side"] == "yes" else -1.0
            strategy_signal = strategy_yes * raw_signal["confidence"]

        # --- Signal 4: Learning bias ---
        of_data = signals.get("orderflow", {})
        volume = of_data.get("volume_24h")
        time_rem = market.get("time_remaining_seconds")
        
        features = learning.extract_features(
            market_price, price_momentum, 
            volume=volume, time_rem=time_rem
        )
        prior = self.STRATEGY_PRIORS.get(self.strategy_type, 0.5)
        learned_yes_bias = learning.get_learned_bias(self.name, features, prior)
        # Convert from 0-1 to -0.5 to +0.5
        learning_signal = (learned_yes_bias - 0.5)

        # Dynamic learning weight: ramps up as bot accumulates data
        # Capped at 0.30 (was 0.60) — stale inherited data was making all bots identical
        # Cached off the hot path: the resolved-trade count changes only when a
        # trade settles, so re-querying it every 1s tick was pure waste.
        now_ts = time.time()
        ttl = getattr(config, "HOTPATH_CACHE_TTL_SEC", 30)
        if self._perf_cache is not None and (now_ts - self._perf_cache[0]) < ttl:
            total_resolved = self._perf_cache[1]
        else:
            total_resolved = db.get_bot_performance(
                self.name, hours=168
            ).get("total_trades", 0)
            self._perf_cache = (now_ts, total_resolved)
        # Live learning disabled (spec R5): the raw-YES-WR bias was
        # anti-predictive. Outcomes are still recorded for the redesign, but the
        # bias contributes 0 to live decisions until the edge-calibrated learner
        # replaces it.
        if config.LEARNING_ENABLED:
            learning_weight = min(0.30, 0.05 + total_resolved * 0.005)
        else:
            learning_weight = 0.0

        # --- Lane: Polymarket in-market price momentum (normalized) ---
        # Rate of change of the YES price on Polymarket itself (from price history API).
        # Distinct from BTC spot momentum — this captures how *traders in this market*
        # are actually positioning, which can lead or lag BTC spot price.
        # A 0.15 YES-price move saturates the lane.
        pm_momentum_raw = float(signals.get("pm_momentum", 0.0) or 0.0)
        pm_momentum_signal = max(-1.0, min(1.0, pm_momentum_raw / 0.15))
        # Global kill-switch (see config comment): the live lane saturates at
        # a 0.19c/step move -> sign(last tick), and the raw quantity measured
        # NET-NEGATIVE after the price in the offline harness (-0.80c/share).
        pm_momentum_signal *= config.SIGNAL_WEIGHT_PM

        # --- Lane: Order flow (OBI + CVD), already in [-1, 1] ---
        # CVD = executed aggression (validated edge); OBI = resting depth,
        # globally killed via config.SIGNAL_WEIGHT_OBI until validated offline.
        obi_signal = max(-1.0, min(1.0, float(signals.get("obi", 0.0) or 0.0)))
        obi_signal *= config.SIGNAL_WEIGHT_OBI
        cvd_signal = max(-1.0, min(1.0, float(signals.get("cvd", 0.0) or 0.0)))
        # Global kill-switch (BUG #27): live cvd-driven trades measured
        # statistically flat (53.1% WR); the feed now has a volume floor but
        # stays at weight 0 until the calibrated form validates offline.
        cvd_signal *= config.SIGNAL_WEIGHT_CVD

        # --- Lane: BTC drift from the window's "price to beat" (strike) ---
        # The fundamental anchor: where BTC sits vs the window open price.
        # Already bounded [-1, 1] and time-scaled (signals/strike.py). Regime-
        # agnostic: >0 favors YES, <0 favors NO. Because it is time-damped, the
        # model has little conviction early in the window — so with the honest
        # blend below, bots naturally sit out the noisy first minute instead of
        # spending their one trade per market there (the -$79 early-window leak).
        drift_signal_val = max(-1.0, min(1.0, float(signals.get("btc_drift", 0.0) or 0.0)))

        # --- Model probability, then fair value as market-vs-model blend ---
        # Edge appears ONLY where the model disagrees with the market price
        # ("follow drift only when the market lags" was the top rule in the
        # offline net-edge harness). Weighted per-strategy for real
        # differentiation; strat/learn lanes carry their weight in the value.
        lanes = {
            "drift": drift_signal_val,
            "mom": momentum_signal,
            "pm": pm_momentum_signal,
            "cvd": cvd_signal,
            "obi": obi_signal,
            "strat": strategy_signal,
            "learn": learning_signal * 2.0 * learning_weight,
        }
        model_prob = self._model_prob_yes(lanes)

        # --- Hard model-lean floor: no opinion, no trade (BUG #27) ---
        # Conviction-scaled trust damps a weak model but its residual edge
        # still scales with MARKET displacement, so near-ignorant models kept
        # clearing MIN_EDGE against displaced prices (lean < 0.10: 28.6% WR /
        # -$78.74 live; lean >= 0.10: 73% WR / +$96.12). Below the floor the
        # model has nothing tradable to say — skip outright.
        model_lean = abs(model_prob - 0.5)
        if model_lean < config.MODEL_LEAN_MIN:
            return {
                "action": "skip",
                "side": "yes",
                "confidence": 0.0,
                "reasoning": (
                    f"Model lean too weak: |{model_prob:.3f}-0.5|="
                    f"{model_lean:.3f} < {config.MODEL_LEAN_MIN:.2f}"
                ),
                "suggested_amount": 0,
                "features": features,
            }

        trust = self.STRATEGY_MODEL_TRUST.get(self.strategy_type, 0.5)
        # --- Conviction-scaled trust: the model's say is proportional to how
        # much it actually knows. edge = trust*(P_model - mid) takes its
        # magnitude from the MARKET's displacement, so a near-ignorant model
        # (P_model ~ 0.5) could book a phantom edge on any market move away
        # from 0.5 and systematically fade it (2026-07-17 chop run: underdog
        # buys 38.5% WR, YES side 10% WR). A decisive model (lean >= the
        # scale) keeps full trust — the validated market-lags-drift trade is
        # untouched.
        conviction = min(1.0, abs(model_prob - 0.5) / config.MODEL_CONVICTION_SCALE)
        trust_eff = trust * conviction
        fair_yes = self._compute_fair_yes(market_price, model_prob, trust_eff)

        # --- Per-side evaluation: each side scored on its OWN price + fee ---
        # Binary outcomes must sum to 1, so both sides share the one fair
        # probability (fair_yes / 1-fair_yes) — but each side is evaluated
        # INDEPENDENTLY: its own book price, its own taker fee, and therefore its
        # own net edge and its own confidence. The bot takes whichever side wins
        # its own evaluation. Fully symmetric / regime-agnostic — no side is
        # favored by a constant, only by the signals (the drift lane above being
        # the one that actually reads which way BTC is going).
        yes_price = market_price
        no_price = market.get("no_price")
        if no_price is None:
            no_price = round(1.0 - yes_price, 4)

        # --- Book-consistency gate (BUG #27) ---
        # YES and NO books disagreeing about the same event = suspect data
        # (stale/gapped book). A real gap is the arb bot's two-legged trade;
        # one-legged it is a coin flip minus fees — and Kelly max-sized
        # exactly those (sums 0.84-0.85 -> "13c edges" -> -$29.15 in 2 trades).
        book_sum = yes_price + no_price
        if abs(book_sum - 1.0) > config.BOOK_SUM_TOLERANCE:
            return {
                "action": "skip",
                "side": "yes",
                "confidence": 0.0,
                "reasoning": (
                    f"Book inconsistency: yes={yes_price:.2f}+no={no_price:.2f}"
                    f"={book_sum:.2f} outside 1±{config.BOOK_SUM_TOLERANCE:.2f}"
                ),
                "suggested_amount": 0,
                "features": features,
            }

        edge_yes, edge_no = self._side_net_edges(model_prob, trust_eff,
                                                 yes_price, no_price)

        # --- Model-lean eligibility: never fade the market on IGNORANCE ---
        # With no information the model sits at 0.5 and the blend would pull
        # fair toward 0.5, making the non-favorite side look "cheap" on every
        # market — a pure contrarian leak. A side is only tradable when the
        # model ACTIVELY leans toward it (its lanes point that way), so the
        # trade thesis is "market lags my information", never "market is more
        # confident than my nothing".
        if model_prob <= 0.5:
            edge_yes = float("-inf")
        if model_prob >= 0.5:
            edge_no = float("-inf")

        # --- Drift veto: never trade AGAINST a non-trivial drift reading ---
        # Drift is the validated fundamental (where BTC actually sits vs the
        # strike). Momentum/flow lanes may modulate conviction but must not
        # outvote it: live, trades contradicting a non-zero drift ran 26% WR
        # (vs 52% agreeing). Symmetric and regime-agnostic — the veto follows
        # whichever side BTC is on. Flow-only trades (|drift| below the floor)
        # remain allowed.
        veto = getattr(config, "DRIFT_VETO_MIN", 0.05)
        if drift_signal_val >= veto:
            edge_no = float("-inf")
        elif drift_signal_val <= -veto:
            edge_yes = float("-inf")

        conf_yes = min(0.95, max(0.0, edge_yes) * config.EDGE_TO_CONFIDENCE)
        conf_no = min(0.95, max(0.0, edge_no) * config.EDGE_TO_CONFIDENCE)
        if edge_yes >= edge_no:
            side, side_price, chosen_edge, confidence = "yes", yes_price, edge_yes, conf_yes
        else:
            side, side_price, chosen_edge, confidence = "no", no_price, edge_no, conf_no

        # --- Minimum-edge gate (no edge = no bet) — SAME bar on both sides ---
        # Information-scaled: with drift flat the model's disagreement with the
        # market rests entirely on the noisy flow/momentum lanes, so a
        # flow-only claim must clear a HIGHER bar (overnight run: flow-only
        # cheap-side trades by the trend bots ran 29% WR in the 0.30-0.42
        # bucket; drift-backed trades in the same bucket were profitable).
        min_edge = self.MIN_EDGE.get(self.strategy_type, config.MIN_EDGE_DEFAULT)
        if abs(drift_signal_val) < getattr(config, "DRIFT_VETO_MIN", 0.05):
            min_edge *= getattr(config, "FLOW_ONLY_EDGE_MULT", 2.0)
        if chosen_edge < min_edge:
            return {
                "action": "skip",
                "side": side,
                "confidence": confidence,
                "reasoning": (
                    f"No edge: {side} edge={chosen_edge:+.3f} < {min_edge:.3f} "
                    f"| fair={fair_yes:.2f} yes={yes_price:.2f} no={no_price:.2f}"
                ),
                "suggested_amount": 0,
                "features": features,
            }

        # --- Symmetric guards (keyed on the chosen side's price) ---
        if side_price > config.HIGH_PRICE_GUARD:
            return {
                "action": "skip",
                "side": side,
                "confidence": confidence,
                "reasoning": (
                    f"High-price guard: {side} price={side_price:.2f} "
                    f">{config.HIGH_PRICE_GUARD:.2f}, bad risk/reward"
                ),
                "suggested_amount": 0,
                "features": features,
            }
        if side_price < config.CONSENSUS_GUARD:
            return {
                "action": "skip",
                "side": side,
                "confidence": confidence,
                "reasoning": (
                    f"Consensus guard: {side} price={side_price:.2f} "
                    f"<{config.CONSENSUS_GUARD:.2f}, fighting consensus"
                ),
                "suggested_amount": 0,
                "features": features,
            }

        # --- Late-window conviction boost ---
        # BTC direction increasingly locked in during the final 60s of a market.
        # Boost confidence to better reflect signal certainty at market close.
        time_rem = market.get("time_remaining_seconds")
        if time_rem is not None and time_rem < 60:
            confidence = min(0.95, confidence * 1.25)

        # --- Bet sizing: pure fractional Kelly, SHARES-FIRST ---
        # Binary-market Kelly: buying a side at price c with true probability p
        # grows fastest at bankroll fraction f* = (p - c)/(1 - c); with the
        # fee-adjusted edge already computed, f* = edge/(1 - price). We bet
        # the Kelly fraction (live-editable in dashboard Settings; full Kelly
        # over-bets estimation error) of the LIVE bankroll — no per-trade or
        # %-of-balance caps (removed 2026-07-17 to run pure Kelly sizing; the
        # venue's shared-pool gate still prevents spending cash the pool
        # lacks). Size scales with edge, odds, AND bankroll — the old formula
        # (flat % of max_pos by confidence) sized wins and losses almost
        # identically ($3.83 vs $3.76).
        price = max(side_price, 0.01)
        bankroll = _sizing_bankroll(self.trading_mode)
        kelly_f = max(0.0, chosen_edge) / max(1.0 - price, 0.05)
        kelly_usd = kelly_f * _kelly_fraction() * bankroll
        # SHARES-FIRST: derive the exact share count, then the USD from it.
        # Sizing USD-first and dividing by price rounds away PnL at low prices.
        # Floor to clear Polymarket's 5-share minimum (× buffer for slippage).
        target_shares = max(kelly_usd / price, config.POLYMARKET_MIN_SHARES * 1.15)
        target_shares = round(target_shares, 4)
        amount = target_shares * price

        reasoning = (
            f"fair={fair_yes:.2f} model={model_prob:.2f} "
            f"trust={trust:.2f}x{conviction:.2f}={trust_eff:.2f} "
            f"yes={yes_price:.2f} no={no_price:.2f} "
            f"=> {side} edge={chosen_edge:+.3f} (eY={edge_yes:+.3f} eN={edge_no:+.3f}) "
            f"drift={drift_signal_val:+.3f} mom={momentum_signal:+.3f} pm={pm_momentum_signal:+.3f} "
            f"of(obi={obi_signal:+.3f} cvd={cvd_signal:+.3f}) "
            f"strat={strategy_signal:+.3f} "
            f"{target_shares:.2f}sh conf={confidence:.2f}"
        )

        return {
            "action": "buy",
            "side": side,
            "confidence": confidence,
            "reasoning": reasoning,
            "suggested_amount": amount,
            "target_shares": target_shares,
            # Price the decision expects to pay. execute() turns this into a
            # slippage limit so an adverse book move between decision and fill
            # rejects the trade instead of filling worse (config.MAX_FILL_SLIPPAGE).
            "entry_price": round(price, 4),
            "features": features,
        }

    def execute(self, signal: dict, market: dict) -> dict:
        """Place a trade via the venue engine (paper sim or live Polymarket)."""
        if self._paused:
            logger.info(f"[{self.name}] Paused, skipping trade")
            return {"success": False, "reason": "bot_paused"}

        # Per-bot mode: fresh read from DB so dashboard toggles take effect immediately
        self.trading_mode = db.get_bot_mode(self.name)
        mode = self.trading_mode

        # Check risk limits
        daily_loss = db.get_bot_daily_loss(self.name, mode)
        max_daily = config.get_max_daily_loss_per_bot()
        if daily_loss >= max_daily:
            self._paused = True
            logger.warning(f"[{self.name}] Daily loss limit hit (${daily_loss:.2f}), pausing")
            return {"success": False, "reason": "daily_loss_limit"}

        total_daily = db.get_total_daily_loss(mode)
        max_total = config.get_max_daily_loss_total()
        if total_daily >= max_total:
            logger.warning(f"[{self.name}] Total arena daily loss limit hit (${total_daily:.2f})")
            return {"success": False, "reason": "arena_loss_limit"}

        # Pure Kelly sizing: paper amounts are uncapped (the shared-pool gate
        # in venues/paper.py still refuses to overspend the pool). LIVE keeps
        # the hard per-trade safety cap — real money.
        amount = signal.get("suggested_amount", 0.0)
        if mode == "live":
            amount = min(amount, config.LIVE_MAX_POSITION)

        # Shared-pool concentration cap (BUG #27): per-bot Kelly can't see the
        # correlated positions the OTHER bots just opened on the same (market,
        # side) — clamp to the pool's remaining headroom, or stand down.
        # Arbitrage is exempt (overrides execute(); its legs are hedged).
        market_id = market.get("condition_id") or market.get("id")
        headroom = self._exposure_headroom(market_id, signal.get("side"), mode)
        if headroom is not None and amount > headroom:
            min_viable = config.POLYMARKET_MIN_SHARES * max(
                signal.get("entry_price") or 0.5, 0.05)
            if headroom < min_viable:
                logger.info(
                    f"[{self.name}] Exposure cap: (market, {signal.get('side')}) "
                    f"pool headroom ${headroom:.2f} < min viable ${min_viable:.2f}, skipping")
                return {"success": False, "reason": "exposure_cap"}
            logger.info(
                f"[{self.name}] Exposure cap: clamping ${amount:.2f} -> ${headroom:.2f}")
            amount = headroom

        try:
            return self._place_via_engine(signal, market, amount, mode)
        except Exception as e:
            logger.error(f"[{self.name}] Trade exception: {e}")
            return {"success": False, "reason": str(e)}

    def _exposure_headroom(self, market_id, side, mode) -> float:
        """Remaining shared-pool budget for this (market, side), or None when
        it can't be computed (missing ids — fail open, other guards still
        apply). Cap base: gross paper pool in paper mode; a fixed
        2x LIVE_MAX_POSITION per market-side in live mode."""
        if not market_id or side not in ("yes", "no"):
            return None
        if mode == "live":
            cap_usd = 2.0 * config.LIVE_MAX_POSITION
        else:
            cap_usd = config.MARKET_SIDE_EXPOSURE_CAP * db.get_paper_pool_gross()
        return cap_usd - db.get_open_exposure(market_id, side, mode)

    def _place_via_engine(self, signal, market, amount, mode) -> dict:
        """Delegate order placement to the paper or live venue engine.

        Paper → local simulated fill (``venues.paper``); live → Polymarket CLOB
        (``venues.live``). Adapts the engine's ``TradeResult`` to the legacy
        dict shape callers (trader.py, maker bots) expect.
        """
        from venues import get_engine

        # Slippage limit: reject a fill that drifts more than MAX_FILL_SLIPPAGE
        # above the price the decision expected. Only applied when the signal
        # carries an expected ``entry_price`` (all buy signals now do).
        expected = signal.get("entry_price")
        limit_price = (
            expected + config.MAX_FILL_SLIPPAGE if expected is not None else None
        )

        res = get_engine(mode).place(
            bot_name=self.name,
            side=signal["side"],
            amount=amount,
            market=market,
            mode=mode,
            confidence=signal.get("confidence"),
            reasoning=signal.get("reasoning"),
            features=signal.get("features"),
            limit_price=limit_price,
        )
        return {
            "success": res.success,
            "trade_id": res.trade_id,
            "reason": res.reason,
            "fill_source": res.fill_source,
        }

    def get_performance(self, hours=12) -> dict:
        """Get bot performance stats."""
        perf = db.get_bot_performance(self.name, hours)
        perf["name"] = self.name
        perf["strategy_type"] = self.strategy_type
        perf["generation"] = self.generation
        perf["paused"] = self._paused
        return perf

    def export_params(self) -> dict:
        return {
            "name": self.name,
            "strategy_type": self.strategy_type,
            "generation": self.generation,
            "lineage": self.lineage,
            "params": copy.deepcopy(self.strategy_params),
        }

    def mutate(self, winning_params: dict, mutation_rate: float = None) -> dict:
        """Create mutated params from winning bot's params."""
        rate = mutation_rate or config.MUTATION_RATE
        new_params = copy.deepcopy(winning_params)

        numeric_keys = [k for k, v in new_params.items() if isinstance(v, (int, float))]
        num_mutations = min(random.randint(2, 3), len(numeric_keys))
        keys_to_mutate = random.sample(numeric_keys, num_mutations) if numeric_keys else []

        for key in keys_to_mutate:
            val = new_params[key]
            delta = val * random.uniform(-rate, rate)
            new_val = val + delta
            if isinstance(val, int):
                new_params[key] = max(1, int(new_val))
            else:
                new_params[key] = max(0.01, round(new_val, 4))

        return new_params

    def reset_daily(self):
        """Reset daily pause state."""
        self._paused = False

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
    STRATEGY_SIGNAL_PROFILE = {
        "momentum":          {"drift": 0.45, "mom": 0.25, "pm": 0.15, "cvd": 0.15, "obi": 0.10},
        "phantom":           {"drift": 0.40, "mom": 0.30, "pm": 0.20, "cvd": 0.10, "obi": 0.10},
        "mean_reversion":    {"drift": 0.65, "mom": 0.00, "pm": 0.00, "cvd": 0.05, "obi": 0.00},
        "mean_reversion_sl": {"drift": 0.65, "mom": 0.00, "pm": 0.00, "cvd": 0.05, "obi": 0.00},
        "mean_reversion_tp": {"drift": 0.65, "mom": 0.00, "pm": 0.00, "cvd": 0.05, "obi": 0.00},
        "sentiment":         {"drift": 0.35, "mom": 0.00, "pm": 0.10, "cvd": 0.35, "obi": 0.15},
        "hybrid":            {"drift": 0.50, "mom": 0.10, "pm": 0.10, "cvd": 0.15, "obi": 0.05},
        "sniper":            {"drift": 0.50, "mom": 0.10, "pm": 0.10, "cvd": 0.15, "obi": 0.05},
    }
    DEFAULT_SIGNAL_PROFILE = {"drift": 0.50, "mom": 0.10, "pm": 0.10, "cvd": 0.15, "obi": 0.05}
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
    # v6.3: Simmer BTC markets price near 47-55¢ most of the time. At 52¢ the
    # combined signal peaks at ~0.08 even with strong BTC momentum — so any
    # threshold above 0.08 means the bot NEVER trades. Lowered aggressively to
    # let all bots trade and accumulate learning data. "If we never trade we
    # never get rich." — higher WR thresholds are moot if we place zero trades.
    MIN_TRADE_CONFIDENCE = {
        "momentum": 0.05,       # was 0.30 — old data was from more extreme prices
        "mean_reversion": 0.03,
        "mean_reversion_sl": 0.03,
        "mean_reversion_tp": 0.03,
        "sniper": 0.10,         # sniper has its own decision logic
        "phantom": 0.04,
        "sentiment": 0.03,
        "hybrid": 0.03,
    }
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

    def _side_net_edges(self, fair_yes: float, yes_price: float,
                        no_price: float) -> tuple:
        """Cost-adjusted edge on each side: prob - price - per-share fee.

        Fee is the canonical taker fee for one share at that side's price.
        """
        edge_yes = fair_yes - yes_price - polymarket_fills.taker_fee(1.0, yes_price)
        edge_no = (1.0 - fair_yes) - no_price - polymarket_fills.taker_fee(1.0, no_price)
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
        # 0.05% 1-candle move saturates the lane.
        momentum_signal = max(-1.0, min(1.0, price_momentum * 2000))

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

        # --- Lane: Order flow (OBI + CVD), already in [-1, 1] ---
        # CVD = executed aggression (validated edge); OBI = resting depth,
        # globally killed via config.SIGNAL_WEIGHT_OBI until validated offline.
        obi_signal = max(-1.0, min(1.0, float(signals.get("obi", 0.0) or 0.0)))
        obi_signal *= config.SIGNAL_WEIGHT_OBI
        cvd_signal = max(-1.0, min(1.0, float(signals.get("cvd", 0.0) or 0.0)))

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
            "strat": strategy_signal * config.STRATEGY_SIGNAL_WEIGHT,
            "learn": learning_signal * 2.0 * learning_weight,
        }
        model_prob = self._model_prob_yes(lanes)
        trust = self.STRATEGY_MODEL_TRUST.get(self.strategy_type, 0.5)
        fair_yes = self._compute_fair_yes(market_price, model_prob, trust)

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
        edge_yes, edge_no = self._side_net_edges(fair_yes, yes_price, no_price)

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

        conf_yes = min(0.95, max(0.0, edge_yes) * config.EDGE_TO_CONFIDENCE)
        conf_no = min(0.95, max(0.0, edge_no) * config.EDGE_TO_CONFIDENCE)
        if edge_yes >= edge_no:
            side, side_price, chosen_edge, confidence = "yes", yes_price, edge_yes, conf_yes
        else:
            side, side_price, chosen_edge, confidence = "no", no_price, edge_no, conf_no

        # --- Minimum-edge gate (no edge = no bet) — SAME bar on both sides ---
        min_edge = self.MIN_EDGE.get(self.strategy_type, config.MIN_EDGE_DEFAULT)
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

        # --- Bet sizing: SHARES-FIRST, proportional to edge strength ---
        # Decide the exact SHARE count from edge, THEN derive USD (USD = shares ×
        # price). Sizing in USD first and dividing by price silently destroys PnL
        # at low prices via rounding — the research credits the shares-first flip
        # with turning a bot from negative to profitable. Confidence for sizing
        # is capped at 0.45 (conf 0.30-0.50 is the 67.9% WR sweet spot; >0.50
        # sizes bigger but wins less).
        bet_conf = min(confidence, 0.45)
        max_pos = config.get_max_position()
        price = max(side_price, 0.01)
        max_shares = max_pos / price  # most shares max_pos can buy at this price
        if bet_conf > 0.2:
            # Moderate-to-strong edge
            target_shares = max_shares * (0.05 + bet_conf * 0.10)
        else:
            # Weak edge — small bet (still generates learning data)
            target_shares = max_shares * 0.03

        # Floor to clear Polymarket's 5-share minimum (× buffer for slippage),
        # cap at what max_pos can buy, then round to a clean share count and
        # derive the USD spend from it. Never USD → shares.
        target_shares = min(
            max(target_shares, config.POLYMARKET_MIN_SHARES * 1.15), max_shares
        )
        target_shares = round(target_shares, 4)
        amount = min(target_shares * price, max_pos)

        reasoning = (
            f"fair={fair_yes:.2f} model={model_prob:.2f} trust={trust:.2f} "
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
        # Use per-bot mode for position limits (global config.TRADING_MODE is always "paper")
        max_pos = config.LIVE_MAX_POSITION if mode == "live" else config.PAPER_MAX_POSITION

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

        amount = min(signal.get("suggested_amount", max_pos * 0.5), max_pos)

        try:
            return self._place_via_engine(signal, market, amount, mode)
        except Exception as e:
            logger.error(f"[{self.name}] Trade exception: {e}")
            return {"success": False, "reason": str(e)}

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

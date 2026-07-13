"""Abstract base class all arena bots inherit from."""

import random
import copy
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import db
import learning

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
    # How aggressively each strategy trusts the market price signal
    MARKET_PRICE_AGGRESSION = {
        "momentum": 1.2,        # follows market price strongly
        "mean_reversion": 0.95, # nearly follows market (contrarian was -$16 loser)
        "mean_reversion_sl": 0.95,
        "mean_reversion_tp": 0.95,
        "sniper": 1.0,          # sniper overrides make_decision entirely
        "phantom": 1.1,         # trend-following
        "sentiment": 1.0,       # neutral
        "hybrid": 1.0,          # neutral (was 0.9, contrarian loses)
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

    def __init__(self, name, strategy_type, params, generation=0, lineage=None):
        self.name = name
        self.strategy_type = strategy_type
        self.strategy_params = params
        self.generation = generation
        self.lineage = lineage or name
        self._paused = False
        self.trading_mode = "paper"

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

    def make_decision(self, market: dict, signals: dict) -> dict:
        """Make a trading decision using market price edge + strategy + learning.

        Signal hierarchy:
        1. Market price edge (strongest — when price is far from 50c, follow it)
        2. BTC momentum (if price is moving, lean that direction)
        3. Strategy analysis (adds differentiation between bots)
        4. Learned bias (accumulates over time, adjusts everything)

        Skips trades when confidence is too low (no edge = no bet).
        """
        market_price = market.get("current_price", 0.5)

        # --- Signal 1: Market price edge ---
        # When YES is priced high, YES usually wins. The further from 50c, the stronger.
        aggression = self.MARKET_PRICE_AGGRESSION.get(self.strategy_type, 1.0)
        price_edge = (market_price - 0.5) * aggression
        # price_edge > 0 means lean YES, < 0 means lean NO

        # --- Signal 2: BTC momentum ---
        prices = signals.get("prices", [])
        btc_latest = signals.get("latest", 0)
        price_momentum = 0.0
        if len(prices) >= 2 and prices[-1] > 0:
            price_momentum = (prices[-1] - prices[-2]) / prices[-2]
        elif btc_latest > 0 and len(prices) >= 1 and prices[-1] > 0:
            # Use live price vs last closed candle
            price_momentum = (btc_latest - prices[-1]) / prices[-1]
        elif btc_latest > 0 and len(prices) == 0:
            # No candles yet — use market price direction as weak proxy
            # Market price > 0.5 suggests BTC trending up in this window
            price_momentum = (market_price - 0.5) * 0.005
        # Momentum signal: BTC going up → lean YES
        momentum_signal = max(-0.15, min(0.15, price_momentum * 30))

        # --- Signal 3: Strategy analysis ---
        raw_signal = self.analyze(market, signals)
        strategy_signal = 0.0
        if raw_signal["action"] != "hold":
            strategy_yes = 1.0 if raw_signal["side"] == "yes" else -1.0
            strategy_signal = strategy_yes * raw_signal["confidence"] * 0.15

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
        perf = db.get_bot_performance(self.name, hours=168)
        total_resolved = perf.get("total_trades", 0)
        learning_weight = min(0.30, 0.05 + total_resolved * 0.005)

        # --- Signal 4b: Polymarket in-market price momentum ---
        # Rate of change of the YES price on Polymarket itself (from price history API).
        # Distinct from BTC spot momentum — this captures how *traders in this market*
        # are actually positioning, which can lead or lag BTC spot price.
        pm_momentum_raw = signals.get("pm_momentum", 0.0)
        pm_momentum_signal = max(-0.15, min(0.15, float(pm_momentum_raw)))

        # --- Combine all signals ---
        # Rebalanced: PM momentum takes 10% from BTC momentum (most similar signal)
        combined = (
            price_edge * 0.50 +           # Market price is primary signal
            momentum_signal * 0.15 +       # BTC spot momentum (Binance candles)
            pm_momentum_signal * 0.10 +    # Polymarket YES price momentum (in-market)
            strategy_signal * 0.15 +       # Strategy adds differentiation
            learning_signal * learning_weight  # Learning grows over time
        )
        # combined > 0 → YES, < 0 → NO

        side = "yes" if combined > 0 else "no"
        confidence = min(0.95, abs(combined) * 2)

        # --- NO bet ban ---
        # Data: NO bets lose at EVERY confidence level (44% WR, -$132 all-time).
        # YES-only strategy is strictly better.
        if side == "no":
            return {
                "action": "skip",
                "side": side,
                "confidence": confidence,
                "reasoning": f"NO ban: NO bets 44% WR all-time | price={market_price:.2f}",
                "suggested_amount": 0,
                "features": features,
            }

        # --- High-price YES guard ---
        # Data: YES at >72c has bad risk/reward (pay 75c, profit 25c on win,
        # lose 75c on loss). conf 0.50+ is 59% WR but net -$23 from this.
        if market_price > 0.72:
            return {
                "action": "skip",
                "side": side,
                "confidence": confidence,
                "reasoning": f"High-price guard: price={market_price:.2f} >72c, bad risk/reward for YES",
                "suggested_amount": 0,
                "features": features,
            }

        # --- Market consensus guard ---
        # Data shows: betting against strong market consensus is 0-10% WR.
        if market_price < 0.35 and side == "yes":
            return {
                "action": "skip",
                "side": side,
                "confidence": confidence,
                "reasoning": f"Market consensus guard: price={market_price:.2f} too low to bet YES",
                "suggested_amount": 0,
                "features": features,
            }

        min_conf = self.MIN_TRADE_CONFIDENCE.get(self.strategy_type, 0.03)

        # --- Skip low-confidence trades (no edge = no bet) ---
        if confidence < min_conf:
            return {
                "action": "skip",
                "side": side,
                "confidence": confidence,
                "reasoning": f"No edge: conf={confidence:.3f} < {min_conf:.3f} | price={market_price:.2f}",
                "suggested_amount": 0,
                "features": features,
            }

        # --- Late-window conviction boost ---
        # BTC direction increasingly locked in during the final 60s of a market.
        # Boost confidence to better reflect signal certainty at market close.
        time_rem = market.get("time_remaining_seconds")
        if time_rem is not None and time_rem < 60:
            confidence = min(0.95, confidence * 1.25)

        # --- Bet sizing: proportional to edge strength ---
        # Data shows: conf 0.30-0.50 is the sweet spot (67.9% WR, +$48).
        # conf >0.50 drops to 48.6% WR but bets are bigger → big losses.
        # Cap bet-sizing confidence at 0.45 to stay in the profitable zone.
        bet_conf = min(confidence, 0.45)
        max_pos = config.get_max_position()
        if bet_conf > 0.2:
            # Moderate-to-strong edge
            amount = max_pos * (0.05 + bet_conf * 0.10)
        else:
            # Weak edge — small bet (still generates learning data)
            amount = max_pos * 0.03

        # Floor the spend so the fill clears Polymarket's 5-share minimum. A
        # $1.50 bet buys <5 shares above ~30c and gets rejected 'below_min_size';
        # size up to 5 shares' worth (× buffer for slippage), capped at max_pos.
        min_notional = config.POLYMARKET_MIN_SHARES * market_price * 1.15
        amount = min(max(amount, min_notional), max_pos)

        reasoning = (
            f"price={market_price:.2f} edge={price_edge:+.3f} "
            f"mom={momentum_signal:+.3f} pm={pm_momentum_signal:+.3f} "
            f"strat={strategy_signal:+.3f} "
            f"learn={learning_signal:+.3f}(w={learning_weight:.0%}) "
            f"=> {side} conf={confidence:.2f}"
        )

        return {
            "action": "buy",
            "side": side,
            "confidence": confidence,
            "reasoning": reasoning,
            "suggested_amount": amount,
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

        res = get_engine(mode).place(
            bot_name=self.name,
            side=signal["side"],
            amount=amount,
            market=market,
            mode=mode,
            confidence=signal.get("confidence"),
            reasoning=signal.get("reasoning"),
            features=signal.get("features"),
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

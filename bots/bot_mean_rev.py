"""Bot 2: Mean Reversion strategy."""

import math
from bots.base_bot import BaseBot, strategy_decision
from signals.curves import smooth_ramp
from signals.lab import SignalView

DEFAULT_PARAMS = {
    # lookback 20->10 (20 rarely had enough 1-min candles in a 5-min window, so
    # the bot never fired); z-score threshold 0.6->0.4 and RSI is now a
    # confidence modifier, not a hard AND-gate — so mean-reversion emits a
    # frequent, distinct (contrarian) lean instead of holding ~always.
    "lookback_candles": 10,
    "bb_std_dev": 2.0,         # Bollinger Band width
    "rsi_period": 14,
    "rsi_oversold": 40,
    "rsi_overbought": 60,
    "reversion_threshold": 0.4, # z-score threshold to fade
    # Drift-agreement gate (BUG #28): the fade may only fire toward the side
    # a signed btc_drift of at least this magnitude already favors. Ungated,
    # the z-fade was a pure contrarian knife-catcher — 10 of 11 live trades
    # fired with drift 0.00-0.08 and ALL lost (-$55.30; the documented
    # "contrarian loses in 5-min markets" death class). Gated, the identity
    # becomes "buy the dip in the WINNING direction": drift picks the side,
    # the z-score times the pullback entry.
    "min_drift": 0.10,
    # Regime conditioning: pure mean-reversion is a RANGING-market thesis —
    # "contrarian loses in 5-min markets" is the documented death class, and
    # fading a genuine trend is exactly how. Confidence is damped by up to
    # this fraction as trend_score rises past ~0.35 (full damp by ~0.75);
    # clearly-ranging tape (or no regime reading) fades nothing.
    "trending_conf_damp": 0.60,
    "position_size_pct": 0.05,
    "min_confidence": 0.55,
}


class MeanRevBot(BaseBot):
    def __init__(self, name="meanrev-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="mean_reversion",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def _calc_rsi(self, prices, period):
        if len(prices) < period + 1:
            return 50  # neutral
        gains, losses = [], []
        for i in range(1, len(prices)):
            delta = prices[i] - prices[i-1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))

        gains = gains[-period:]
        losses = losses[-period:]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_zscore(self, prices, lookback):
        if len(prices) < lookback:
            return 0
        window = prices[-lookback:]
        mean = sum(window) / len(window)
        variance = sum((p - mean) ** 2 for p in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 1
        return (prices[-1] - mean) / std

    def analyze(self, market: dict, signals: dict) -> dict:
        """Bet against overextended moves."""
        sv = SignalView.of(signals)
        prices = sv.prices
        lookback = self.strategy_params["lookback_candles"]

        if len(prices) < lookback:
            return strategy_decision("hold", reasoning="insufficient data")

        # Z-score: how far price is from recent mean
        zscore = self._calc_zscore(prices, lookback)

        # RSI: momentum oscillator
        rsi = self._calc_rsi(prices, self.strategy_params["rsi_period"])

        threshold = self.strategy_params["reversion_threshold"]
        import config
        amount = config.get_max_position() * self.strategy_params["position_size_pct"]

        # Drift-agreement gate: the fade side must be the side BTC's actual
        # position vs the strike already favors (see DEFAULT_PARAMS comment).
        drift = sv.btc_drift
        min_drift = self.strategy_params.get("min_drift", 0.10)
        fade_no_ok = drift <= -min_drift    # fade an up-move only in a DOWN window
        fade_yes_ok = drift >= min_drift    # fade a down-move only in an UP window

        # Regime conditioning: the z-fade is a RANGING thesis. On trending
        # tape the "overextension" is usually the trend itself — damp the
        # confidence smoothly with trend_score (no damp when clearly ranging
        # or when the regime feed has no reading).
        regime = self.regime_context(signals)
        damp = self.strategy_params.get("trending_conf_damp", 0.60)
        regime_factor = 1.0
        if regime["known"] and not regime["ranging"]:
            regime_factor = 1.0 - damp * smooth_ramp(
                regime["trend_score"], 0.35, 0.75)

        contributing = {"zscore": zscore, "rsi": rsi, "drift": drift,
                        "regime": regime["label"],
                        "regime_factor": regime_factor}

        # Overextended UP → fade → bet NO (expect reversion down). RSI is a
        # confidence booster (stronger when also overbought), not a hard gate.
        if zscore > threshold and not fade_no_ok:
            return strategy_decision(
                "hold", signals=contributing,
                reasoning=f"Fade NO not drift-backed: z={zscore:.2f}, drift={drift:+.3f}")
        if zscore < -threshold and not fade_yes_ok:
            return strategy_decision(
                "hold", signals=contributing,
                reasoning=f"Fade YES not drift-backed: z={zscore:.2f}, drift={drift:+.3f}")

        if zscore > threshold:
            rsi_boost = max(0.0, rsi - self.strategy_params["rsi_overbought"]) * 0.005
            confidence = min(0.95, (0.35 + abs(zscore) * 0.15 + rsi_boost)
                             * regime_factor)
            return strategy_decision(
                "buy", "no",
                edge=min(0.10, (abs(zscore) - threshold) * 0.02 * regime_factor),
                confidence=confidence,
                reasoning=(f"Mean reversion SHORT: z={zscore:.2f}, RSI={rsi:.1f} "
                           f"(fade up, regime={regime['label']}x{regime_factor:.2f})"),
                signals=contributing,
                suggested_amount=amount,
            )

        # Overextended DOWN → fade → bet YES (expect reversion up)
        if zscore < -threshold:
            rsi_boost = max(0.0, self.strategy_params["rsi_oversold"] - rsi) * 0.005
            confidence = min(0.95, (0.35 + abs(zscore) * 0.15 + rsi_boost)
                             * regime_factor)
            return strategy_decision(
                "buy", "yes",
                edge=min(0.10, (abs(zscore) - threshold) * 0.02 * regime_factor),
                confidence=confidence,
                reasoning=(f"Mean reversion LONG: z={zscore:.2f}, RSI={rsi:.1f} "
                           f"(fade down, regime={regime['label']}x{regime_factor:.2f})"),
                signals=contributing,
                suggested_amount=amount,
            )

        return strategy_decision(
            "hold", signals=contributing,
            reasoning=f"No reversion signal: z={zscore:.2f}, RSI={rsi:.1f}")

"""Bot 1: Momentum / Trend Following strategy."""

import config
from bots.base_bot import BaseBot, strategy_decision
from signals.lab import SignalView

DEFAULT_PARAMS = {
    "lookback_candles": 5,
    # 0.03% move to trigger (was 0.2% — fired in ~6% of ticks, so the strategy
    # lane was silent and all bots were clones). BTC 1-min candles routinely move
    # >0.03%, so momentum now emits a directional lean whenever there is a real
    # trend; the strategy lane is capped at +/-0.043 of fair value, so a frequent
    # lean nudges rather than dominates.
    "momentum_threshold": 0.0003,
    "position_size_pct": 0.05,    # 5% of max position
    "min_confidence": 0.55,
    "trend_strength_weight": 0.7,
    "volume_weight": 0.3,
    # Regime conditioning: confidence scales by (1 + w * (2*trend_score - 1)).
    # Trend-following earns MORE trust on trending tape and LESS in chop
    # (2026-07-19 live: momentum-driven trades in chop ran 47.9% WR / -$74;
    # the lab's quiet-regime mom-lane damp attacks the same leak — this
    # conditions the strategy's own thesis the same way). Neutral (x1.0) when
    # the regime feed has no reading.
    "regime_conf_weight": 0.35,
}


class MomentumBot(BaseBot):
    def __init__(self, name="momentum-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="momentum",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def analyze(self, market: dict, signals: dict) -> dict:
        """Trade in the direction of short-term price momentum."""
        sv = SignalView.of(signals)
        prices = sv.prices
        if len(prices) < self.strategy_params["lookback_candles"]:
            return strategy_decision("hold", reasoning="insufficient price data")

        lookback = self.strategy_params["lookback_candles"]
        recent = prices[-lookback:]
        oldest = recent[0]
        newest = recent[-1]

        if oldest == 0:
            return strategy_decision("hold", reasoning="zero price")

        pct_change = (newest - oldest) / oldest
        threshold = self.strategy_params["momentum_threshold"]

        # Calculate trend strength (consecutive moves in same direction)
        consecutive = 0
        for i in range(1, len(recent)):
            if pct_change > 0 and recent[i] > recent[i-1]:
                consecutive += 1
            elif pct_change < 0 and recent[i] < recent[i-1]:
                consecutive += 1

        trend_strength = consecutive / (len(recent) - 1) if len(recent) > 1 else 0

        # Volume signal (if available)
        volumes = sv.volumes
        vol_signal = 0.5
        if len(volumes) >= lookback:
            recent_vol = sum(volumes[-lookback:])
            prev_vol = sum(volumes[-lookback*2:-lookback]) if len(volumes) >= lookback*2 else recent_vol
            vol_signal = min(1.0, recent_vol / max(prev_vol, 1)) * 0.5 + 0.25

        # Combine signals
        tw = self.strategy_params["trend_strength_weight"]
        vw = self.strategy_params["volume_weight"]
        confidence = (trend_strength * tw + vol_signal * vw)

        # Regime conditioning: trend-following deserves more say on trending
        # tape, less in chop. Smoothly scaled by trend_score; neutral when the
        # regime feed is silent (regime_context sets trend_score=0.5 then).
        regime = self.regime_context(signals)
        rw = self.strategy_params.get("regime_conf_weight", 0.35)
        regime_factor = 1.0 + rw * (2.0 * regime["trend_score"] - 1.0)
        confidence *= regime_factor

        contributing = {
            "pct_change": pct_change,
            "trend_strength": trend_strength,
            "vol_signal": vol_signal,
            "regime": regime["label"],
            "regime_factor": regime_factor,
        }

        if abs(pct_change) < threshold:
            return strategy_decision(
                "hold", confidence=confidence, signals=contributing,
                reasoning=f"momentum {pct_change:.4f} below threshold {threshold}")

        side = "yes" if pct_change > 0 else "no"
        amount = config.get_max_position() * self.strategy_params["position_size_pct"]
        # Strategy edge estimate: how far the move clears the trigger, scaled
        # by trend quality — a thesis-strength proxy in probability units.
        edge = min(0.10, (abs(pct_change) - threshold) * 50.0 * max(trend_strength, 0.2))

        return strategy_decision(
            "buy", side,
            edge=edge,
            confidence=min(confidence, 0.95),
            reasoning=(f"Momentum {pct_change:.4f} ({lookback} candles), "
                       f"trend_str={trend_strength:.2f}, vol={vol_signal:.2f}, "
                       f"regime={regime['label']}x{regime_factor:.2f}"),
            signals=contributing,
            suggested_amount=amount,
        )

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
    "momentum_threshold": 0.0012,
    "position_size_pct": 0.05,    # 5% of max position
    "min_confidence": 0.55,
    "trend_strength_weight": 0.7,
    "volume_weight": 0.3,
    # Hold-to-resolution: candle momentum must not fight btc_drift (PTB side).
    "min_drift_align": 0.05,
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
        """Trade in the direction of short-term price momentum.

        Hold-to-resolution: intermediate candle momentum only helps when it
        agrees with BTC's side of the Price-to-Beat (``btc_drift``). A move
        against the strike is noise, not a binary edge.
        """
        sv = SignalView.of(signals)
        # Settlement-object candles: TWAP on Polymarket, BRTI on Kalshi.
        from signals.tape import candle_prices
        prices = candle_prices(market, signals if isinstance(signals, dict) else {},
                               sample_sec=60.0)
        if len(prices) < self.strategy_params["lookback_candles"]:
            prices = list(sv.prices)
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

        # Volume signal (Chainlink BTC volumes are empty — do not invent 0.5)
        volumes = sv.volumes or []
        has_vol = (
            len(volumes) >= lookback
            and any(v and v > 0 for v in volumes[-lookback:])
        )
        vol_signal = 0.0
        if has_vol:
            recent_vol = sum(volumes[-lookback:])
            prev_vol = (
                sum(volumes[-lookback * 2:-lookback])
                if len(volumes) >= lookback * 2 else recent_vol
            )
            vol_signal = min(1.0, recent_vol / max(prev_vol, 1)) * 0.5 + 0.25
            tw = self.strategy_params["trend_strength_weight"]
            vw = self.strategy_params["volume_weight"]
            confidence = trend_strength * tw + vol_signal * vw
        else:
            confidence = trend_strength

        # Regime conditioning: trend-following deserves more say on trending
        # tape, less in chop. Smoothly scaled by trend_score; neutral when the
        # regime feed is silent (regime_context sets trend_score=0.5 then).
        regime = self.regime_context(signals)
        rw = self.strategy_params.get("regime_conf_weight", 0.35)
        regime_factor = 1.0 + rw * (2.0 * regime["trend_score"] - 1.0)
        confidence *= regime_factor
        # Extra chop damp (hold-to-resolution): chop momentum is mean-reverting
        if regime.get("choppy") or (regime.get("label") or "").endswith("chop"):
            confidence *= 0.55
        elif regime.get("ranging"):
            confidence *= 0.70

        # Drift alignment gate: only lean toward the side strike already favors
        # (or flat drift with strong multi-candle trend). Prevents fading PTB.
        # btc_drift is TWAP moneyness; 1m spot mom is damped in settlement by
        # SignalLab (settlement_policy.mom_damp).
        drift = float(sv.btc_drift or 0.0)
        min_align = float(self.strategy_params.get("min_drift_align", 0.05))
        try:
            pol = sv.settlement_policy or {}
            if pol.get("block_fade") or (
                sv.in_settlement_window
                and float(pol.get("certainty") or 0) >= 0.55
            ):
                min_align = max(min_align, 0.08)
        except Exception:
            pol = {}

        contributing = {
            "pct_change": pct_change,
            "trend_strength": trend_strength,
            "vol_signal": vol_signal,
            "drift": drift,
            "regime": regime["label"],
            "regime_factor": regime_factor,
            "market_phase": getattr(sv, "market_phase", "unknown"),
        }

        window = float(market.get("window_sec") or getattr(config, "MARKET_WINDOW_SEC", 300) or 300)
        try:
            late_sec = float(getattr(config, "MOMENTUM_LATE_SKIP_SEC", 80))
        except (TypeError, ValueError):
            late_sec = 80.0
        if late_sec < 0:
            late_sec = 80.0
        try:
            from exchanges import KALSHI, exchange_of as _ex_of
            if _ex_of(market) == KALSHI:
                late_sec = float(getattr(config, "KALSHI_MOMENTUM_LATE_SKIP_SEC", 120) or 120)
                window = float(market.get("window_sec") or getattr(config, "KALSHI_WINDOW_SEC", 900) or 900)
        except Exception:
            pass
        tr = market.get("time_remaining_seconds")
        try:
            remaining = float(tr) if tr is not None else None
            age = max(0.0, window - remaining) if remaining is not None else float(
                market.get("window_age_seconds") or 0.0
            )
        except (TypeError, ValueError):
            remaining = None
            age = 0.0
        if remaining is None and age > 0:
            remaining = max(0.0, window - age)
        if late_sec > 0 and remaining is not None and remaining <= late_sec:
            return strategy_decision(
                "hold", confidence=confidence, signals=contributing,
                reasoning=(
                    f"momentum late-window remaining={remaining:.0f}s "
                    f"(sit out last {late_sec:.0f}s)"
                ),
            )

        if abs(pct_change) < threshold:
            return strategy_decision(
                "hold", confidence=confidence, signals=contributing,
                reasoning=f"momentum {pct_change:.4f} below threshold {threshold}")

        side = "yes" if pct_change > 0 else "no"
        # Require drift not to contradict (signed drift toward chosen side)
        signed_drift = drift if side == "yes" else -drift
        if signed_drift < -min_align:
            return strategy_decision(
                "hold", confidence=confidence, signals=contributing,
                reasoning=(f"momentum {side} fights drift={drift:+.3f} "
                           f"(hold-to-resolution: PTB wins)"))
        if abs(drift) >= min_align and signed_drift < min_align:
            # Drift flat-to-weak vs move — require stronger trend_strength
            if trend_strength < 0.6:
                return strategy_decision(
                    "hold", confidence=confidence, signals=contributing,
                    reasoning=(f"momentum weak vs PTB: drift={drift:+.3f} "
                               f"trend_str={trend_strength:.2f}"))

        amount = config.get_max_position() * self.strategy_params["position_size_pct"]
        # Strategy edge estimate: how far the move clears the trigger, scaled
        # by trend quality — a thesis-strength proxy in probability units.
        edge = min(0.10, (abs(pct_change) - threshold) * 50.0 * max(trend_strength, 0.2))
        # Boost edge slightly when drift agrees strongly (aligned thesis)
        if signed_drift >= 0.15:
            edge = min(0.10, edge * 1.15)

        return strategy_decision(
            "buy", side,
            edge=edge,
            confidence=min(confidence, 0.95),
            reasoning=(f"Momentum {pct_change:.4f} ({lookback} candles), "
                       f"trend_str={trend_strength:.2f}, vol={vol_signal:.2f}, "
                       f"drift={drift:+.3f}, "
                       f"regime={regime['label']}x{regime_factor:.2f}"),
            signals=contributing,
            suggested_amount=amount,
        )

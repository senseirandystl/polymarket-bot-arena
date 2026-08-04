"""Phantom Swing — short-horizon EMA breakout with strict drift confirmation.

2026-08 redesign (prior phantom: 50% WR, −$5, toxic NO book):
  * Faster EMAs (5/13) + 6-bar breakout for 5-min markets (less lag).
  * Drift is primary: breakout only *times* an already-signed PTB lean.
  * NO side requires stronger drift + lag mid (same family as NO-side gates).
  * Chop/range regimes heavily damp or block; quiet-trend uses higher bars.
  * Strat confidence kept modest so confirm-mode blend doesn't over-weight.
"""

from __future__ import annotations

import config
from bots.base_bot import BaseBot, strategy_decision
from bots.edge_calibration import quality_confidence
from signals.lab import SignalView

DEFAULT_PARAMS = {
    "ema_fast": 5,
    "ema_slow": 13,
    "atr_period": 8,
    "breakout_lookback": 6,
    # BTC 1-min |move| ~ p50 0.022%; skip only dead / chaos tape.
    "min_atr_pct": 0.00015,
    "max_atr_pct": 0.008,
    "position_size_pct": 0.05,
    "min_confidence": 0.22,
    # Drift alignment (YES-frame). NO uses no_min_drift_align (stricter).
    "min_drift_align": 0.12,
    "no_min_drift_align": 0.18,
    "no_max_side_mid": 0.55,
    "yes_max_side_mid": 0.62,
    "regime_conf_weight": 0.25,
    "chop_block": True,          # hard hold in chop labels
}


class PhantomBot(BaseBot):
    def __init__(self, name="phantom-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="phantom",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def _calc_ema(self, prices, period):
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0.0
        alpha = 2 / (period + 1)
        ema = prices[0]
        for px in prices[1:]:
            ema = (px * alpha) + (ema * (1 - alpha))
        return ema

    def _calc_atr(self, prices, period):
        if len(prices) < period + 1:
            return 0.0
        diffs = [abs(prices[i] - prices[i - 1])
                 for i in range(len(prices) - period, len(prices))]
        return sum(diffs) / period

    def analyze(self, market: dict, signals: dict) -> dict:
        sv = SignalView.of(signals)
        prices = sv.prices
        p = self.strategy_params

        need = int(p["ema_slow"]) + int(p["breakout_lookback"])
        if len(prices) < need:
            return strategy_decision("hold", reasoning="insufficient data")

        current_price = sv.latest or prices[-1]
        ema_fast = self._calc_ema(prices, int(p["ema_fast"]))
        ema_slow = self._calc_ema(prices, int(p["ema_slow"]))
        recent_window = prices[-int(p["breakout_lookback"]):]
        recent_high = max(recent_window)
        recent_low = min(recent_window)
        atr = self._calc_atr(prices, int(p["atr_period"]))
        atr_pct = atr / current_price if current_price > 0 else 0.0

        if not (float(p["min_atr_pct"]) <= atr_pct <= float(p["max_atr_pct"])):
            return strategy_decision(
                "hold", signals={"atr_pct": atr_pct},
                reasoning=f"phantom: vol out of bounds ({atr_pct:.4%})")

        regime = self.regime_context(signals)
        label = (regime.get("label") or "") or ""
        if p.get("chop_block") and (
            regime.get("choppy")
            or regime.get("chop")
            or label.endswith("chop")
            or label in ("choppy", "high_vol_chop", "volatile")
        ):
            return strategy_decision(
                "hold",
                reasoning=f"phantom: block in chop regime={label}")

        rw = float(p.get("regime_conf_weight", 0.25))
        regime_factor = 1.0 + rw * (2.0 * float(regime.get("trend_score") or 0.5) - 1.0)
        if regime.get("ranging") or label == "low_vol_range":
            regime_factor *= 0.55
        if label == "low_vol_trend":
            # Quiet "trend" was a live leak for breakout styles.
            regime_factor *= 0.60

        drift = float(sv.btc_drift or 0.0)
        yes_mid = market.get("current_price") or 0.5
        no_mid = market.get("no_price")
        if no_mid is None:
            no_mid = round(1.0 - float(yes_mid), 4)

        contributing = {
            "ema_fast": ema_fast, "ema_slow": ema_slow, "atr_pct": atr_pct,
            "recent_high": recent_high, "recent_low": recent_low,
            "drift": drift, "regime": label, "regime_factor": regime_factor,
        }

        # Long: EMA stack + breakout + drift above PTB + lagging YES mid
        if (ema_fast > ema_slow and current_price > ema_fast
                and current_price > recent_high):
            min_align = float(p.get("min_drift_align", 0.12))
            if label == "low_vol_trend":
                min_align += 0.05
            if drift < min_align:
                return strategy_decision(
                    "hold", signals=contributing,
                    reasoning=(f"phantom LONG blocked: drift={drift:+.3f}"
                               f"<{min_align:.2f}"))
            yes_max = float(p.get("yes_max_side_mid", 0.62))
            if float(yes_mid) > yes_max:
                return strategy_decision(
                    "hold", signals=contributing,
                    reasoning=(f"phantom LONG blocked: yes mid={float(yes_mid):.2f}"
                               f">{yes_max:.2f} (no lag)"))
            trend_strength = (ema_fast - ema_slow) / current_price
            edge = min(0.08, trend_strength * 15.0 * regime_factor)
            conf = quality_confidence(
                edge=edge, abs_drift=abs(drift),
                side_mid=float(yes_mid), side="yes", regime_label=label,
            )
            conf = min(conf, 0.55)  # never claim high conf from breakout alone
            if conf < float(p.get("min_confidence", 0.22)):
                return strategy_decision(
                    "hold", signals=contributing,
                    reasoning=f"phantom LONG conf {conf:.3f} low")
            return strategy_decision(
                "buy", "yes",
                edge=edge,
                confidence=conf,
                reasoning=(
                    f"phantom LONG: trend={trend_strength:.4%}, breakout "
                    f"above {recent_high:.0f}, drift={drift:+.3f}, "
                    f"regime={label}x{regime_factor:.2f}"
                ),
                signals={**contributing, "trend_strength": trend_strength},
                suggested_amount=config.get_max_position() * float(
                    p["position_size_pct"]),
            )

        # Short: stricter NO path
        if (ema_fast < ema_slow and current_price < ema_fast
                and current_price < recent_low):
            min_align = float(p.get("no_min_drift_align", 0.18))
            if label == "low_vol_trend":
                min_align += 0.06
            if drift > -min_align:
                return strategy_decision(
                    "hold", signals=contributing,
                    reasoning=(f"phantom SHORT blocked: drift={drift:+.3f}"
                               f">{-min_align:.2f}"))
            no_max = float(p.get("no_max_side_mid", 0.55))
            if float(no_mid) > no_max:
                return strategy_decision(
                    "hold", signals=contributing,
                    reasoning=(f"phantom SHORT blocked: no mid={float(no_mid):.2f}"
                               f">{no_max:.2f}"))
            trend_strength = (ema_slow - ema_fast) / current_price
            edge = min(0.07, trend_strength * 12.0 * regime_factor)
            conf = quality_confidence(
                edge=edge, abs_drift=abs(drift),
                side_mid=float(no_mid), side="no", regime_label=label,
            )
            conf = min(conf, 0.50)
            if conf < float(p.get("min_confidence", 0.22)):
                return strategy_decision(
                    "hold", signals=contributing,
                    reasoning=f"phantom SHORT conf {conf:.3f} low")
            return strategy_decision(
                "buy", "no",
                edge=edge,
                confidence=conf,
                reasoning=(
                    f"phantom SHORT: trend={trend_strength:.4%}, breakdown "
                    f"below {recent_low:.0f}, drift={drift:+.3f}, "
                    f"regime={label}x{regime_factor:.2f}"
                ),
                signals={**contributing, "trend_strength": trend_strength},
                suggested_amount=config.get_max_position() * float(
                    p["position_size_pct"]),
            )

        return strategy_decision(
            "hold", signals=contributing,
            reasoning=(
                f"phantom: no signal (ema_f={ema_fast:.0f}, "
                f"ema_s={ema_slow:.0f}, high={recent_high:.0f}, "
                f"low={recent_low:.0f})"
            ),
        )

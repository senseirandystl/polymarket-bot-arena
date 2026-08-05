"""Regime specialist — only trades in allowed market regimes.

Uses the full BaseBot model blend (drift/mom/strat) but hard-stands-down
outside configured regimes. Default allow-list favors high_vol_trend and
normal — chop/range requires stronger evidence elsewhere.
"""

from __future__ import annotations

from bots.base_bot import BaseBot, strategy_decision
from signals.lab import SignalView

DEFAULT_PARAMS = {
    # Robust detector ids + legacy labels.
    "allowed_regimes": (
        "high_vol_trend",
        "normal",
        "trending",
        "low_vol_trend",
    ),
    "min_regime_confidence": 0.35,
    "position_size_pct": 0.06,
    "min_confidence": 0.15,
    # When regime unknown: sit flat (set-and-forget safety).
    "trade_unknown": False,
}


class RegimeSpecialistBot(BaseBot):
    """Directional bot gated to productive regimes (menu only)."""

    def __init__(self, name="regime-specialist-v1", params=None,
                 generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="regime_specialist",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def analyze(self, market, signals):
        """Lean with signed drift when regime is allowed; else hold."""
        p = self.strategy_params
        sv = SignalView.of(signals)
        ctx = self.regime_context(signals)
        label = (ctx.get("label") or "unknown").lower()
        allowed = {
            str(x).lower()
            for x in (p.get("allowed_regimes") or ())
        }
        conf = float(ctx.get("confidence") or 0.0)
        min_c = float(p.get("min_regime_confidence", 0.35))

        if label == "unknown" or not ctx.get("known", True):
            if not p.get("trade_unknown", False):
                return strategy_decision(
                    "hold",
                    reasoning="regime specialist: unknown regime — flat",
                    signals={"regime": label},
                )
        elif label not in allowed:
            return strategy_decision(
                "hold",
                reasoning=f"regime specialist: {label} not in allow-list",
                signals={"regime": label},
            )
        elif conf < min_c and label not in ("normal", "trending"):
            return strategy_decision(
                "hold",
                reasoning=f"regime specialist: low conf {conf:.2f} in {label}",
                signals={"regime": label, "regime_conf": conf},
            )

        drift = float(sv.btc_drift or 0.0)
        if abs(drift) < 0.08:
            return strategy_decision(
                "hold",
                reasoning=f"regime specialist: flat drift {drift:+.3f}",
                signals={"regime": label, "drift": drift},
            )

        side = "yes" if drift > 0 else "no"
        confidence = min(0.85, 0.25 + abs(drift) * 0.6)
        return strategy_decision(
            "buy", side,
            confidence=confidence,
            edge=abs(drift) * 0.15,
            reasoning=f"regime specialist: {label} drift={drift:+.3f}",
            signals={"regime": label, "drift": drift, "regime_conf": conf},
        )

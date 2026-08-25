"""Lag residual bot — pure "market lags drift" specialist.

Collapses the sniper/maker lag thesis into one clean directional policy:

1. Read signed ``btc_drift`` (YES-frame).
2. Implied P = 0.5 + 0.5 * drift.
3. Residual = implied − side mid (positive → market underprices that side).
4. Trade only when residual ≥ min_residual, |drift| ≥ min_drift, mid ≤ max_mid.

Uses BaseBot.make_decision for guards/Kelly when analyze() returns buy —
but overrides make_decision for a pure lag path (no noisy strat blend).
"""

from __future__ import annotations

import config
import polymarket_fills
from bots.base_bot import BaseBot, data_quality_skip, strategy_decision
from bots.edge_calibration import quality_confidence
from signals.lab import SignalView

DEFAULT_PARAMS = {
    "min_drift": 0.12,
    "min_residual": 0.04,      # implied − mid (probability units)
    "min_edge": 0.018,
    "max_side_mid": 0.58,
    "min_side_mid": 0.30,
    "position_size_pct": 0.07,
    "min_confidence": 0.12,
}


class LagResidualBot(BaseBot):
    """Single-policy market-lags-drift hunter (menu only; not default slate)."""

    def __init__(self, name="lag-residual-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="lag_residual",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def analyze(self, market, signals):
        return strategy_decision("hold", reasoning="lag_residual: pure make_decision path")

    def make_decision(self, market, signals):
        _dq = data_quality_skip(signals)
        if _dq is not None:
            return _dq
        p = self.strategy_params
        sv = SignalView.of(signals)

        yes_mid = market.get("current_price") or 0.5
        no_mid = market.get("no_price")
        if no_mid is None:
            no_mid = round(1.0 - yes_mid, 4)
        yes_ask = market.get("yes_ask") or yes_mid
        no_ask = market.get("no_ask") or no_mid

        drift = float(sv.btc_drift or 0.0)
        min_drift = float(p.get("min_drift", 0.12))
        try:
            from arena.regime_adapt import adjustments as _regime_adj
            radj = _regime_adj(
                self.regime_context(signals).get("label"),
                strategy_type="lag_residual",
            )
            if getattr(radj, "block_directional", False):
                return strategy_decision(
                    "skip",
                    reasoning=f"lag: regime hard-skip {radj.label}",
                )
            min_drift += float(getattr(radj, "extra_drift_floor", 0.0) or 0.0)
        except Exception:
            radj = None

        if abs(drift) < min_drift:
            return strategy_decision(
                "skip",
                reasoning=f"lag: weak drift {drift:+.3f} < {min_drift:.2f}",
            )

        implied_yes = 0.5 + 0.5 * drift
        residual_yes = implied_yes - yes_mid
        residual_no = (1.0 - implied_yes) - no_mid
        min_res = float(p.get("min_residual", 0.04))
        max_mid = float(p.get("max_side_mid", 0.58))
        min_mid = float(p.get("min_side_mid", 0.30))
        min_edge = float(p.get("min_edge", 0.018))

        candidates = []
        if residual_yes >= min_res and min_mid <= yes_mid <= max_mid and drift > 0:
            fee = polymarket_fills.fee_per_share(yes_ask, is_maker=False)
            edge = residual_yes - fee - (yes_ask - yes_mid)
            candidates.append(("yes", edge, yes_mid, yes_ask, residual_yes))
        if residual_no >= min_res and min_mid <= no_mid <= max_mid and drift < 0:
            fee = polymarket_fills.fee_per_share(no_ask, is_maker=False)
            edge = residual_no - fee - (no_ask - no_mid)
            candidates.append(("no", edge, no_mid, no_ask, residual_no))

        if not candidates:
            return strategy_decision(
                "skip",
                reasoning=(
                    f"lag: no residual edge drift={drift:+.3f} "
                    f"rY={residual_yes:+.3f} rN={residual_no:+.3f}"
                ),
            )

        side, edge, mid, ask, residual = max(candidates, key=lambda c: c[1])
        if edge < min_edge:
            return strategy_decision(
                "skip", side=side,
                reasoning=f"lag: edge {edge:+.3f} < {min_edge:.3f}",
            )

        conf = quality_confidence(
            edge=edge, abs_drift=abs(drift), side_mid=mid, side=side,
            regime_label=self.regime_context(signals).get("label"),
        )
        # Shares-first Kelly-ish size via position_size_pct of bankroll fallback.
        try:
            bankroll = float(config.PAPER_BANKROLL_DEFAULT)
            from bots.base_bot import _sizing_bankroll
            bankroll = _sizing_bankroll(self.trading_mode)
        except Exception:
            bankroll = float(getattr(config, "PAPER_BANKROLL_DEFAULT", 200.0))
        pct = float(p.get("position_size_pct", 0.07))
        amount = max(bankroll * pct, config.POLYMARKET_MIN_SHARES * ask * 1.15)
        shares = amount / max(ask, 0.01)

        return strategy_decision(
            "buy", side,
            edge=edge,
            confidence=conf,
            reasoning=(
                f"lag residual: drift={drift:+.3f} residual={residual:+.3f} "
                f"mid={mid:.2f} ask={ask:.2f} edge={edge:+.3f}"
            ),
            signals={"drift": drift, "lag": residual, "implied": implied_yes},
            suggested_amount=amount,
            entry_price=round(ask, 4),
            target_shares=round(shares, 4),
        )

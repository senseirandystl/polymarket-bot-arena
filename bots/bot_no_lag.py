"""NO-lag specialist — only buys NO when drift and residual agree.

YES historically cleaner; NO needs stricter gates. This bot:
  * only considers side=NO
  * requires signed drift toward NO (≥ min_signed_drift)
  * requires mid ≤ max_mid (market still lags)
  * residual (implied_NO − no_mid) ≥ min_residual
"""

from __future__ import annotations

import config
import polymarket_fills
from bots.base_bot import BaseBot, data_quality_skip, strategy_decision, implied_side_prob
from bots.edge_calibration import quality_confidence
from signals.lab import SignalView

DEFAULT_PARAMS = {
    "min_signed_drift": 0.15,   # drift must be negative (NO)
    "min_residual": 0.05,
    "min_edge": 0.022,
    "max_side_mid": 0.55,
    "min_side_mid": 0.32,
    "position_size_pct": 0.05,
    "min_confidence": 0.15,
}


class NoLagBot(BaseBot):
    """Strict NO-only market-lag specialist (menu only)."""

    def __init__(self, name="no-lag-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="no_lag",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def analyze(self, market, signals):
        return strategy_decision("hold", reasoning="no_lag: pure make_decision path")

    def make_decision(self, market, signals):
        _dq = data_quality_skip(signals)
        if _dq is not None:
            return _dq
        p = self.strategy_params
        sv = SignalView.of(signals)

        no_mid = market.get("no_price")
        yes_mid = market.get("current_price") or 0.5
        if no_mid is None:
            no_mid = round(1.0 - yes_mid, 4)
        no_ask = market.get("no_ask") or no_mid

        drift = float(sv.btc_drift or 0.0)
        # NO needs negative drift (BTC below strike).
        signed_toward_no = -drift
        min_sd = float(p.get("min_signed_drift", 0.15))
        if signed_toward_no < min_sd:
            return strategy_decision(
                "skip", side="no",
                reasoning=(
                    f"no_lag: need drift toward NO ≥ {min_sd:.2f}, "
                    f"got {signed_toward_no:+.3f} (drift={drift:+.3f})"
                ),
            )

        max_mid = float(p.get("max_side_mid", 0.55))
        min_mid = float(p.get("min_side_mid", 0.32))
        if not (min_mid <= no_mid <= max_mid):
            return strategy_decision(
                "skip", side="no",
                reasoning=f"no_lag: NO mid {no_mid:.2f} outside [{min_mid},{max_mid}]",
            )

        # Same Phi(z) / btc_implied_yes path as sniper — never 0.5+0.5*tanh.
        implied_no = implied_side_prob(
            side="no", signals=signals, signed_lane=drift,
        )
        residual = implied_no - no_mid
        min_res = float(p.get("min_residual", 0.05))
        if residual < min_res:
            return strategy_decision(
                "skip", side="no",
                reasoning=f"no_lag: residual {residual:+.3f} < {min_res:.2f}",
            )

        fee = polymarket_fills.fee_per_share(no_ask, is_maker=False)
        edge = residual - fee - max(0.0, no_ask - no_mid)
        min_edge = float(p.get("min_edge", 0.022))
        if edge < min_edge:
            return strategy_decision(
                "skip", side="no",
                reasoning=f"no_lag: edge {edge:+.3f} < {min_edge:.3f}",
            )

        conf = quality_confidence(
            edge=edge, abs_drift=abs(drift), side_mid=no_mid, side="no",
            regime_label=self.regime_context(signals).get("label"),
        )
        try:
            from bots.base_bot import _sizing_bankroll
            bankroll = _sizing_bankroll(self.trading_mode)
        except Exception:
            bankroll = float(getattr(config, "PAPER_BANKROLL_DEFAULT", 200.0))
        amount = max(
            bankroll * float(p.get("position_size_pct", 0.05)),
            config.POLYMARKET_MIN_SHARES * no_ask * 1.15,
        )
        shares = amount / max(no_ask, 0.01)
        return strategy_decision(
            "buy", "no",
            edge=edge,
            confidence=conf,
            reasoning=(
                f"no_lag: drift={drift:+.3f} residual={residual:+.3f} "
                f"mid={no_mid:.2f} ask={no_ask:.2f} edge={edge:+.3f}"
            ),
            signals={"drift": drift, "lag": residual, "side": "no"},
            suggested_amount=amount,
            entry_price=round(no_ask, 4),
            target_shares=round(shares, 4),
        )

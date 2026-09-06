"""Cross-venue lag prototype — Polymarket BTC 5m vs Kalshi BTC 15m.

Thesis: when the same underlying BTC drift disagrees with relative mids across
venues (PM 5m vs Kalshi 15m), the lagging venue may reprice. Conservative,
paper-oriented gates. Menu-only — NOT in DEFAULT_INDICES.

Requires ``signals["cross_venue"]`` (attached by the trader from discovery's
per-exchange snapshot). Skips cleanly when peer data is missing.
"""

from __future__ import annotations

import config
from bots.base_bot import (
    BaseBot,
    data_quality_skip,
    strategy_decision,
    implied_side_prob,
)
from bots.edge_calibration import quality_confidence
from signals.lab import SignalView

DEFAULT_PARAMS = {
    # Minimum absolute mid gap between local and peer (probability units).
    "min_mid_gap": 0.04,
    # Local mid must still lag honest Phi(z) by this residual.
    "min_residual": 0.03,
    "min_edge": 0.025,
    "min_drift": 0.12,
    "max_side_mid": 0.62,
    "min_side_mid": 0.38,
    "position_size_pct": 0.04,
    "min_confidence": 0.25,
    # Prefer trading the shorter window (usually Polymarket 5m) when it lags.
    "prefer_short_window": True,
}


class CrossVenueLagBot(BaseBot):
    """Lead-lag / relative mispricing across PM 5m and Kalshi 15m."""

    def __init__(
        self,
        name="cross-venue-lag-v1",
        params=None,
        generation=0,
        lineage=None,
    ):
        super().__init__(
            name=name,
            strategy_type="cross_venue_lag",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def analyze(self, market, signals):
        return strategy_decision(
            "hold", reasoning="cross_venue_lag: pure make_decision path"
        )

    def make_decision(self, market, signals):
        p = self.strategy_params
        _dq = data_quality_skip(signals)
        if _dq is not None:
            return _dq

        cv = (signals or {}).get("cross_venue") or {}
        peer_mid = cv.get("peer_yes_mid")
        local_mid = cv.get("local_yes_mid")
        if peer_mid is None or local_mid is None:
            # Fallback to market mid when trader attached only peer.
            local_mid = market.get("current_price")
            if peer_mid is None or local_mid is None:
                return strategy_decision(
                    "skip",
                    reasoning="cross_venue: missing peer/local mid",
                    skip_reason="no_cross_venue",
                )
        try:
            peer_mid = float(peer_mid)
            local_mid = float(local_mid)
        except (TypeError, ValueError):
            return strategy_decision(
                "skip",
                reasoning="cross_venue: non-numeric mids",
                skip_reason="no_cross_venue",
            )

        gap = local_mid - peer_mid
        min_gap = float(p.get("min_mid_gap", 0.04))
        if abs(gap) < min_gap:
            return strategy_decision(
                "skip",
                reasoning=f"cross_venue: mid gap {gap:+.3f} < {min_gap:.3f}",
                skip_reason="cross_gap_thin",
            )

        sv = SignalView.of(signals)
        drift = float(sv.btc_drift or 0.0)
        min_drift = float(p.get("min_drift", 0.12))
        if abs(drift) < min_drift:
            return strategy_decision(
                "skip",
                reasoning=f"cross_venue: weak drift {drift:+.3f}",
                skip_reason="weak_drift",
            )

        # Prefer the shorter window when configured (PM 5m typically).
        local_w = float(cv.get("local_window_sec") or market.get("window_sec") or 300)
        peer_w = float(cv.get("peer_window_sec") or 900)
        if bool(p.get("prefer_short_window", True)) and local_w > peer_w + 1e-9:
            return strategy_decision(
                "skip",
                reasoning=(
                    f"cross_venue: local window {local_w:.0f}s > peer {peer_w:.0f}s "
                    f"— prefer short-window venue"
                ),
                skip_reason="prefer_short_window",
            )

        # Side: buy local YES when local mid is cheap vs peer AND drift leans YES;
        # buy local NO when local mid is rich vs peer AND drift leans NO.
        if gap < -min_gap and drift > 0:
            side = "yes"
            side_mid = local_mid
        elif gap > min_gap and drift < 0:
            side = "no"
            no_mid = market.get("no_price")
            if no_mid is None:
                no_mid = round(1.0 - local_mid, 4)
            side_mid = float(no_mid)
        else:
            return strategy_decision(
                "skip",
                reasoning=(
                    f"cross_venue: gap={gap:+.3f} drift={drift:+.3f} "
                    f"disagree / no clear lag trade"
                ),
                skip_reason="cross_disagree",
            )

        max_mid = float(p.get("max_side_mid", 0.62))
        min_mid = float(p.get("min_side_mid", 0.38))
        if not (min_mid <= side_mid <= max_mid):
            return strategy_decision(
                "skip",
                side=side,
                reasoning=f"cross_venue: mid {side_mid:.2f} outside band",
                skip_reason="mid_band",
            )

        implied = implied_side_prob(
            side=side, signals=signals, signed_lane=drift,
        )
        residual = implied - side_mid
        min_res = float(p.get("min_residual", 0.03))
        if residual < min_res:
            return strategy_decision(
                "skip",
                side=side,
                reasoning=(
                    f"cross_venue: residual {residual:+.3f} < {min_res:.3f} "
                    f"(implied={implied:.2f} mid={side_mid:.2f})"
                ),
                skip_reason="thin_residual",
            )

        ask = market.get(f"{side}_ask") or side_mid
        try:
            ask = float(ask)
        except (TypeError, ValueError):
            ask = side_mid
        import polymarket_fills
        fee = polymarket_fills.fee_per_share(ask, is_maker=False)
        edge = residual - fee - max(0.0, ask - side_mid)
        min_edge = float(p.get("min_edge", 0.025))
        if edge < min_edge:
            return strategy_decision(
                "skip",
                side=side,
                reasoning=f"cross_venue: edge {edge:+.3f} < {min_edge:.3f}",
                skip_reason="no_edge",
            )

        conf = quality_confidence(
            edge=edge,
            abs_drift=abs(drift),
            side_mid=side_mid,
            side=side,
            regime_label=self.regime_context(signals).get("label"),
        )
        if conf < float(p.get("min_confidence", 0.25)):
            return strategy_decision(
                "skip",
                side=side,
                reasoning=f"cross_venue: conf {conf:.2f} too low",
                skip_reason="low_confidence",
            )

        try:
            from bots.base_bot import _sizing_bankroll
            bankroll = _sizing_bankroll(self.trading_mode)
        except Exception:
            bankroll = float(getattr(config, "PAPER_BANKROLL_DEFAULT", 200.0))
        amount = max(
            bankroll * float(p.get("position_size_pct", 0.04)),
            getattr(config, "POLYMARKET_MIN_SHARES", 5) * ask * 1.15,
        )
        return strategy_decision(
            "buy",
            side,
            edge=edge,
            confidence=conf,
            reasoning=(
                f"cross_venue: gap={gap:+.3f} drift={drift:+.3f} "
                f"implied={implied:.2f} mid={side_mid:.2f} edge={edge:+.3f} "
                f"peer={cv.get('peer_exchange')}@{peer_mid:.2f}"
            ),
            signals={
                "cross_gap": gap,
                "peer_mid": peer_mid,
                "local_mid": local_mid,
                "drift": drift,
                "implied": implied,
            },
            suggested_amount=amount,
            entry_price=round(ask, 4),
        )

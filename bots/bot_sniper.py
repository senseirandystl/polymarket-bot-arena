"""Sniper bot — drift-vs-price lag hunter (v3).

Previous versions used hand-tuned YES/NO price buckets from unverified
third-party WR tables. Those buckets bit us live (expensive YES snipes,
near-flat P&L). v3 drops zone tables entirely.

Thesis
------
BTC 5-minute markets resolve from BTC vs the window-open strike. The
validated edge is **"follow drift only when the market lags"** (harness +
live soak). The sniper does only that:

1. Read signed ``btc_drift`` (YES-frame, in [-1, 1]).
2. Convert to a drift-implied probability: ``p = 0.5 + 0.5 * signed_drift``.
3. Score BOTH sides: ``edge = p_side - side_mid - fee`` (maker fee when
   limit-first passive mode is on).
4. Trade only when:
   * |drift| ≥ min_drift (real conviction),
   * edge ≥ min_edge,
   * the chosen side's MID still **lags** (≤ max_side_mid, default 0.58),
   * model leans the same way as drift (no fade).

No arbitrary cheap/strong price buckets. Optional late-window confidence
ramp (direction locks in near expiry). Sizing uses fractional Kelly on the
fee-adjusted edge, same as the directional stack.
"""

from __future__ import annotations

import config
import learning
import polymarket_fills
from bots.base_bot import BaseBot, strategy_decision
from signals.curves import smooth_ramp
from signals.lab import SignalView

DEFAULT_PARAMS = {
    # Minimum |signed drift| toward the chosen side (quiet regime gets a bump).
    "min_drift": 0.15,
    "quiet_drift_bump": 0.05,
    # Net edge floor after fee (probability units).
    "min_edge": 0.02,
    # Market-lag ceiling: never snipe a side already priced above this mid.
    # Matches the harness "follow drift when side ≤ 58¢" rule.
    "max_side_mid": 0.58,
    # Optional absolute floor so we don't buy deep longshots on noise.
    "min_side_mid": 0.30,
    # Extreme drift must still lag (same as base_bot gate).
    "extreme_drift_abs": 0.50,
    "position_size_pct": 0.08,  # fallback if Kelly path unavailable
    "min_confidence": 0.10,
}


class SniperBot(BaseBot):
    def __init__(self, name="sniper-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="sniper",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def analyze(self, market, signals):
        """Sniper does not use the strat lane blend — pure drift-vs-price."""
        return strategy_decision("hold", reasoning="sniper: no signal")

    def make_decision(self, market, signals):
        """Drift-implied fair vs mid/ask — snipe only when the market lags."""
        p = self.strategy_params
        sv = SignalView.of(signals)

        yes_mid = market.get("current_price") or 0.5
        no_mid = market.get("no_price")
        if no_mid is None:
            no_mid = round(1.0 - yes_mid, 4)

        # Executable costs (asks) for entry_price / fill; guards use mids.
        yes_ask = market.get("yes_ask") or yes_mid
        no_ask = market.get("no_ask") or no_mid

        drift = float(sv.btc_drift or 0.0)
        min_drift = float(p.get("min_drift", 0.15))
        regime = self.regime_context(signals)
        quiet = (
            regime.get("legacy") == "quiet"
            or regime.get("label") in ("low_vol_range", "low_vol_trend", "quiet")
            or (regime.get("known") and regime.get("vol_score", 0.5) < 0.35)
        )
        if quiet:
            min_drift += float(p.get("quiet_drift_bump", 0.05))

        min_edge = float(p.get("min_edge", 0.02))
        max_mid = float(p.get("max_side_mid", 0.58))
        min_mid = float(p.get("min_side_mid", 0.30))
        ext_abs = float(p.get("extreme_drift_abs",
                              getattr(config, "DRIFT_EXTREME_ABS", 0.50)))

        is_maker = (
            getattr(config, "ORDER_STYLE", "limit") == "limit"
            and getattr(config, "LIMIT_PRICE_MODE", "passive_mid")
            in ("passive_mid", "join_bid")
        )

        def _edge(side_mid: float, signed_drift: float) -> float:
            implied = 0.5 + 0.5 * signed_drift
            fee = polymarket_fills.fee_per_share(side_mid, is_maker=is_maker)
            return implied - side_mid - fee

        yes_edge = _edge(yes_mid, drift)
        no_edge = _edge(no_mid, -drift)

        prices = sv.prices
        btc_momentum = 0.0
        if len(prices) >= 2 and prices[-1] > 0:
            btc_momentum = (prices[-1] - prices[-2]) / prices[-2]
        of_data = sv.orderflow
        features = learning.extract_features(
            yes_mid, btc_momentum,
            volume=of_data.get("volume_24h"),
            time_rem=market.get("time_remaining_seconds"),
        )
        contributing = {
            "drift": drift, "yes_edge": yes_edge, "no_edge": no_edge,
            "regime": regime.get("label"), "min_drift": min_drift,
        }

        # Eligibility per side: drift magnitude + lag + edge + not deep junk.
        def _ok(signed_d: float, mid: float, edge: float) -> bool:
            if abs(signed_d) < min_drift:
                return False
            if mid > max_mid or mid < min_mid:
                return False
            if abs(signed_d) >= ext_abs and mid > max_mid:
                return False
            return edge >= min_edge

        yes_ok = drift >= min_drift and _ok(drift, yes_mid, yes_edge)
        no_ok = (-drift) >= min_drift and _ok(-drift, no_mid, no_edge)

        if yes_ok and (not no_ok or yes_edge >= no_edge):
            side, side_mid, side_ask, side_edge = "yes", yes_mid, yes_ask, yes_edge
            signed = drift
        elif no_ok:
            side, side_mid, side_ask, side_edge = "no", no_mid, no_ask, no_edge
            signed = -drift
        else:
            return strategy_decision(
                "skip",
                reasoning=(
                    f"sniper: no lag edge (drift={drift:+.3f} "
                    f"yes_mid={yes_mid:.2f} eY={yes_edge:+.3f} "
                    f"no_mid={no_mid:.2f} eN={no_edge:+.3f} "
                    f"min_d={min_drift:.2f})"
                ),
                signals=contributing, features=features,
            )

        # Confidence from edge magnitude + drift conviction (no zone Gaussian).
        confidence = min(0.95, max(0.0, side_edge) * 3.0 + 0.15 * abs(signed))
        time_rem = market.get("time_remaining_seconds")
        late = 0.0
        if time_rem is not None and time_rem > 0:
            late = smooth_ramp(-float(time_rem), -90.0, -30.0)
            if late > 0.05:
                confidence = min(0.95, confidence * (1.0 + 0.25 * late))

        min_conf = float(p.get("min_confidence", 0.10))
        if confidence < min_conf:
            return strategy_decision(
                "skip", side, confidence=confidence,
                reasoning=f"sniper: conf {confidence:.2f} < {min_conf}",
                signals=contributing, features=features,
            )

        # Fractional Kelly on fee-adjusted edge (shares-first), with portfolio
        # + risk + regime mults — same stack as BaseBot.make_decision.
        price = max(float(side_ask), 0.01)
        try:
            from bots.base_bot import (
                _sizing_bankroll, _portfolio_weight, _risk_size_mult,
                _kelly_fraction,
            )
            from arena.regime_adapt import size_multiplier as regime_mult
            bankroll = (
                _sizing_bankroll(self.trading_mode)
                * _portfolio_weight(self.name)
                * _risk_size_mult(self.name)
                * regime_mult(regime.get("label"))
            )
            sizing_edge = min(max(0.0, side_edge),
                              getattr(config, "KELLY_EDGE_CAP", 0.10))
            kelly_f = sizing_edge / max(1.0 - price, 0.05)
            kelly_usd = kelly_f * _kelly_fraction() * bankroll
            target_shares = max(
                kelly_usd / price, config.POLYMARKET_MIN_SHARES * 1.15)
            target_shares = round(target_shares, 4)
            amount = target_shares * price
        except Exception:
            max_pos = config.get_max_position()
            pct = float(p.get("position_size_pct", 0.08)) * (1.0 + 0.2 * late)
            amount = min(max_pos * pct * (0.5 + confidence), max_pos)
            target_shares = None

        reasoning = (
            f"sniper: drift={drift:+.3f} → {side} mid={side_mid:.2f} "
            f"ask={side_ask:.2f} edge={side_edge:+.3f} "
            f"implied={0.5 + 0.5 * signed:.2f} lag≤{max_mid:.2f} "
            f"reg={regime.get('label', '?')} conf={confidence:.2f}"
        )
        out = strategy_decision(
            "buy", side,
            edge=side_edge,
            confidence=confidence,
            reasoning=reasoning,
            signals=contributing,
            suggested_amount=amount,
            entry_price=round(float(side_ask), 4),
            features=features,
        )
        if target_shares is not None:
            out["target_shares"] = target_shares
        return out

"""Sniper bot — drift-vs-price lag hunter (v3, TWAP-aware).

Previous versions used hand-tuned YES/NO price buckets from unverified
third-party WR tables. Those buckets bit us live (expensive YES snipes,
near-flat P&L). v3 drops zone tables entirely.

Thesis
------
BTC 5-minute markets resolve from **Chainlink TWAP** at close vs TWAP at
open (``TWAP_WINDOW_SEC`` lookback — 60s for 5m). Instant spot at expiry no
longer decides the outcome — a last-second spike is diluted across the
averaging window. ``btc_drift`` is already moneyness on the TWAP path
(RTDS TWAP / settlement nowcast). The validated edge remains **"follow
drift only when the market lags"** (harness + live soak). The sniper does
only that:

1. Read signed ``btc_drift`` (YES-frame, in [-1, 1]; TWAP-based).
2. Convert to a drift-implied probability: ``p = 0.5 + 0.5 * signed_drift``.
3. Score BOTH sides: ``edge = p_side - side_mid - fee`` (maker fee when
   limit-first passive mode is on).
4. Trade only when:
   * |drift| ≥ min_drift (real conviction),
   * edge ≥ min_edge,
   * the chosen side's MID still **lags** (≤ max_side_mid, default 0.58),
   * model leans the same way as drift (no fade).

No arbitrary cheap/strong price buckets. Inside the final TWAP window,
certainty can slightly boost confidence (partially observed settlement).
Sizing uses fractional Kelly on the fee-adjusted edge, same as the
directional stack.
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
        try:
            d_pct = float(sv.btc_drift_pct or 0.0)
        except Exception:
            d_pct = float(signals.get("btc_drift_pct") or 0.0)
        min_drift = float(p.get("min_drift", 0.15))
        # Dual gate floors (shared with BaseBot) — sniper is pure lag hunter.
        min_pct = float(getattr(config, "DRIFT_MIN_ABS_PCT", 0.00030) or 0.0)
        min_z = float(getattr(config, "DRIFT_MIN_ABS_Z", 0.35) or 0.0)
        # Sniper needs stronger z than generic directional (was free at 0.15).
        min_drift = max(min_drift, min_z, 0.40)
        if abs(d_pct) < min_pct or abs(drift) < min_drift:
            return strategy_decision(
                "skip",
                reasoning=(
                    f"sniper: dual-gate d_pct={d_pct:+.5f}"
                    f" (need |≥{min_pct:.5f}) d_z={drift:+.3f}"
                    f" (need |≥{min_drift:.2f})"
                ),
            )
        regime = self.regime_context(signals)
        # Data-driven hard stand-down when live regime is toxic.
        try:
            from arena.regime_adapt import adjustments as _regime_adj
            _radj = _regime_adj(regime.get("label"), strategy_type="sniper")
            if getattr(_radj, "block_directional", False):
                return strategy_decision(
                    "skip",
                    reasoning=(
                        f"sniper: regime hard-skip {_radj.label} "
                        f"({_radj.reason})"
                    ),
                )
            if getattr(_radj, "block_strategy", False):
                return strategy_decision(
                    "skip",
                    reasoning=(
                        f"sniper: regime style-skip {_radj.label} "
                        f"({_radj.reason})"
                    ),
                )
            min_drift += float(getattr(_radj, "extra_drift_floor", 0.0) or 0.0)
            # Continuous strategy×regime edge tax (data-driven).
            min_edge_tax = float(getattr(_radj, "edge_mult", 1.0) or 1.0)
        except Exception:
            _radj = None
            min_edge_tax = 1.0
        quiet = (
            regime.get("legacy") == "quiet"
            or regime.get("label") in ("low_vol_range", "low_vol_trend", "quiet")
            or (regime.get("known") and regime.get("vol_score", 0.5) < 0.35)
        )
        if quiet:
            min_drift += float(p.get("quiet_drift_bump", 0.05))

        min_edge = float(p.get("min_edge", 0.02)) * float(min_edge_tax or 1.0)
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

        of_data = sv.orderflow
        prices = sv.prices
        btc_momentum = 0.0
        if len(prices) >= 2 and prices[-1] > 0:
            btc_momentum = (prices[-1] - prices[-2]) / prices[-2]
        features = learning.extract_features(
            yes_mid, btc_momentum,
            volume=of_data.get("volume_24h"),
            time_rem=market.get("time_remaining_seconds"),
        )
        contributing = {
            "drift": drift, "yes_edge": yes_edge, "no_edge": no_edge,
            "regime": regime.get("label"), "min_drift": min_drift,
            "btc_momentum": btc_momentum,
        }

        # Ask-quality: refuse when mid still "lags" but the executable ask has
        # already gapped away (2026-07-29: mid 0.54 / ask 0.75 losses). The
        # edge math can look fine on mid while fill risk/reward is trash.
        max_spread = float(p.get(
            "max_ask_mid_spread",
            getattr(config, "SNIPER_MAX_ASK_MID_SPREAD", 0.08),
        ))

        # Coin-flip mid band needs stronger signed drift (same spirit as base).
        cf_lo = float(getattr(config, "MID_COINFLIP_LO", 0.50))
        cf_hi = float(getattr(config, "MID_COINFLIP_HI", 0.58))
        cf_drift = float(getattr(config, "MID_COINFLIP_DRIFT_MIN", 0.28))
        if _radj is not None and getattr(_radj, "mid_band_drift_min", None):
            cf_drift = max(cf_drift, float(_radj.mid_band_drift_min))

        # Eligibility per side: drift magnitude + lag + edge + not deep junk.
        def _ok(signed_d: float, mid: float, edge: float, ask: float) -> bool:
            need = min_drift
            if cf_lo <= float(mid) <= cf_hi:
                need = max(need, cf_drift)
            if abs(signed_d) < need:
                return False
            if mid > max_mid or mid < min_mid:
                return False
            if abs(signed_d) >= ext_abs and mid > max_mid:
                return False
            if edge < min_edge:
                return False
            if (float(ask) - float(mid)) > max_spread:
                return False
            return True

        # NO needs a stricter drift floor + lag ceiling (2026-08 soak: sniper
        # NO −$12 vs YES +$51). Intelligent lag hunt, not a mirror of YES.
        no_min_drift = min_drift + float(p.get("no_extra_drift", 0.05))
        no_max_mid = min(max_mid, float(p.get("no_max_side_mid", 0.52)))
        no_min_edge = min_edge * float(p.get("no_edge_mult", 1.30))

        # Momentum non-contradiction (especially for NO — rising BTC tape
        # against a NO lag snipe was a live loser class).
        mom_contra = float(p.get("mom_contradict", 0.0008))
        yes_mom_ok = btc_momentum >= -mom_contra
        no_mom_ok = btc_momentum <= mom_contra

        yes_ok = (
            drift >= min_drift
            and yes_mom_ok
            and _ok(drift, yes_mid, yes_edge, yes_ask)
        )
        no_ok = (
            (-drift) >= no_min_drift
            and no_mid <= no_max_mid
            and no_edge >= no_min_edge
            and no_mom_ok
            and _ok(-drift, no_mid, no_edge, no_ask)
        )

        if yes_ok and (not no_ok or yes_edge >= no_edge):
            side, side_mid, side_ask, side_edge = "yes", yes_mid, yes_ask, yes_edge
            signed = drift
        elif no_ok:
            side, side_mid, side_ask, side_edge = "no", no_mid, no_ask, no_edge
            signed = -drift
        else:
            # Distinguish ask-quality skips for telemetry when lag edge existed
            # on mid but ask gap killed it.
            y_spread = float(yes_ask) - float(yes_mid)
            n_spread = float(no_ask) - float(no_mid)
            ask_gap = (
                (drift >= min_drift and yes_edge >= min_edge
                 and y_spread > max_spread)
                or ((-drift) >= min_drift and no_edge >= min_edge
                    and n_spread > max_spread)
            )
            why = (
                f"sniper: ask gap (yes {y_spread:.2f}/no {n_spread:.2f}"
                f">{max_spread:.2f})"
                if ask_gap else
                f"sniper: no lag edge (drift={drift:+.3f} "
                f"yes_mid={yes_mid:.2f} eY={yes_edge:+.3f} "
                f"no_mid={no_mid:.2f} eN={no_edge:+.3f} "
                f"min_d={min_drift:.2f})"
            )
            return strategy_decision(
                "skip",
                reasoning=why,
                signals=contributing, features=features,
            )

        # Data-driven side skip / continuous side edge tax
        if _radj is not None:
            _bs = getattr(_radj, "block_side", None)
            if _bs and side == str(_bs).lower():
                return strategy_decision(
                    "skip", side=side,
                    reasoning=(
                        f"sniper: regime side-skip {side} in "
                        f"{_radj.label} ({_radj.reason})"
                    ),
                    signals=contributing, features=features,
                )
            try:
                if hasattr(_radj, "side_edge_for"):
                    min_edge *= float(_radj.side_edge_for(side) or 1.0)
                else:
                    _sm = getattr(_radj, "side_edge_mult", None) or {}
                    min_edge *= float(_sm.get(side, 1.0) or 1.0)
            except Exception:
                pass
            if side_edge < min_edge:
                return strategy_decision(
                    "skip", side=side,
                    reasoning=(
                        f"sniper: side edge tax e={side_edge:+.3f}"
                        f"<{min_edge:.3f} ({_radj.reason})"
                    ),
                    signals=contributing, features=features,
                )

        # Structure confidence (not edge × constant — inversion fix 2026-08).
        try:
            from bots.edge_calibration import quality_confidence
            confidence = quality_confidence(
                edge=float(side_edge),
                abs_drift=abs(float(signed)),
                side_mid=float(side_mid),
                side=side,
                regime_label=regime.get("label"),
            )
        except Exception:
            confidence = min(0.85, 0.25 + 0.4 * abs(signed) + min(0.2, side_edge * 2))
        # TWAP settlement policy: conf / size from certainty, not last-tick.
        pol = sv.settlement_policy or {}
        twap_cert = float(pol.get("certainty") or sv.twap_certainty or 0.0)
        if pol.get("policy_active") and float(pol.get("conf_boost") or 0) > 0:
            confidence = min(0.95, confidence + float(pol["conf_boost"]))
        elif sv.in_settlement_window and twap_cert > 0:
            confidence = min(0.95, confidence + 0.08 * twap_cert)
        contributing["twap_certainty"] = twap_cert
        contributing["resolution_source"] = sv.resolution_source
        contributing["market_phase"] = sv.market_phase
        contributing["settlement_edge_mult"] = float(pol.get("edge_mult") or 1.0)
        # Inside noisy settlement (low cert): demand more edge before sniping
        if pol.get("policy_active"):
            min_edge *= float(pol.get("edge_mult") or 1.0)
            # Re-check edge after tax (side already chosen)
            if side_edge < min_edge:
                return strategy_decision(
                    "skip",
                    reasoning=(
                        f"sniper: TWAP settle edge tax "
                        f"e={side_edge:+.3f}<{min_edge:.3f} "
                        f"phase={pol.get('phase')} cert={twap_cert:.2f}"
                    ),
                    signals=contributing, features=features,
                )
        time_rem = market.get("time_remaining_seconds")
        late = 0.0
        late_size = 1.0
        if pol.get("policy_active"):
            late_size = float(pol.get("size_mult") or 1.0)
        elif time_rem is not None and time_rem > 0 and abs(signed) >= 0.20:
            # Soft ramp into settlement TWAP window (not to expiry print)
            settle_w = float(getattr(config, "TWAP_WINDOW_SEC", 60) or 60)
            late = smooth_ramp(-float(time_rem), -90.0 - settle_w * 0.5, -settle_w)
            late_size = 1.0 + 0.10 * late

        min_conf = float(p.get("min_confidence", 0.10))
        if confidence < min_conf:
            return strategy_decision(
                "skip", side, confidence=confidence,
                reasoning=f"sniper: conf {confidence:.2f} < {min_conf}",
                signals=contributing, features=features,
            )

        # Learned skip/go (same decision_events mining as directional bots)
        _learn_size = 1.0
        try:
            from arena.learned_rules import evaluate as _learned_eval
            _lr = _learned_eval(
                regime=regime.get("label"),
                side_price=side_mid,
                drift=drift,
                side=side,
                strategy_type=self.strategy_type,
            )
            if _lr.get("action") == "skip":
                return strategy_decision(
                    "skip", side, confidence=confidence,
                    reasoning=_lr.get("reason") or "sniper: learned_skip",
                    signals=contributing, features=features,
                )
            _learn_size = float(_lr.get("size_mult") or 1.0)
        except Exception:
            pass

        # Fractional Kelly on fee-adjusted edge (shares-first), with portfolio
        # + risk + regime mults — same stack as BaseBot.make_decision.
        price = max(float(side_ask), 0.01)
        try:
            from bots.base_bot import (
                _sizing_bankroll, _portfolio_weight, _risk_size_mult,
                _kelly_fraction,
            )
            from arena.regime_adapt import adjustments as regime_adj
            from bots.edge_calibration import calibrated_sizing_edge
            _ra = regime_adj(regime.get("label"), strategy_type="sniper")
            bankroll = (
                _sizing_bankroll(self.trading_mode)
                * _portfolio_weight(self.name)
                * _risk_size_mult(self.name)
                * float(_ra.size_mult)
                * _learn_size
                * late_size
            )
            sizing_edge = calibrated_sizing_edge(float(side_edge))
            kelly_f = sizing_edge / max(1.0 - price, 0.05)
            kelly_usd = kelly_f * _kelly_fraction() * bankroll
            target_shares = max(
                kelly_usd / price, config.POLYMARKET_MIN_SHARES * 1.15)
            target_shares = round(target_shares, 4)
            amount = target_shares * price
        except Exception:
            max_pos = config.get_max_position()
            pct = float(p.get("position_size_pct", 0.08)) * late_size
            amount = min(max_pos * pct, max_pos)
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

"""Sweeper bot — buy effectively-decided outcomes still priced under $1.

Inspired by 0x_Punisher's sweeper V2 thesis (Polymarket):

  A decided outcome is worth $1. Markets often still sell the winning token
  at 97–99.9¢ for a short window. At those extremes Polymarket's fee curve
  (rate · p · (1−p)) nearly vanishes, so thin gross edges can clear fees.
  Do NOT chase mid-book "sweeps" at 90–95¢ — fee rises and flip risk returns.

BTC 5-minute adaptation (TWAP resolution, 2026-08-07+; 60s lookback)
--------------------------------------------------------------------
"Decided" is not a single settlement print — both open and close are
Chainlink TWAPs (``TWAP_WINDOW_SEC``, 60s for 5m). Instant late spikes no
longer flip the outcome; the averaging window starts ~60s before expiry.
"Decided" means:

  * late enough (inside entry_window_sec = pre_settle + TWAP window)
  * strong signed ``btc_drift`` (TWAP moneyness) toward one side
  * optional ``twap_certainty`` floor once the settlement window is open
  * book already confirming that side at extreme prices

Edge is settlement-style, not model-blend::

    net_edge = 1.0 − ask − taker_fee_per_share(ask)

Overrides ``make_decision`` so the global HIGH_PRICE_GUARD (0.72) does not
block the only zone this strategy is allowed to trade. Menu-only specialist;
evolution-exempt (structural fee-curve edge, not directional GA fitness).
"""

from __future__ import annotations

import config
import learning
import polymarket_fills
from bots.base_bot import BaseBot, strategy_decision, data_quality_skip
from signals.lab import SignalView

DEFAULT_PARAMS = {
    # pre_settle lead + full TWAP settlement window (config-driven; 20+60=80).
    # Overnight soak 2026-08-07: early 98¢ entries with moderate cert blew
    # the book — require the averaging window / late certainty.
    "entry_window_sec": 80,
    # Fundamental certainty proxy (YES-frame TWAP-drift magnitude toward side).
    "min_drift": 0.32,
    # Once inside the TWAP settlement window, require this certainty (0–1).
    "min_twap_certainty": 0.45,
    # Outside settlement (pre_settle only): need even stronger drift.
    "pre_settle_extra_drift": 0.10,
    # Fee-curve extreme only. Do not lower min_price toward 0.90.
    "min_price": 0.97,
    "max_price": 0.999,
    # Net settlement edge after taker fee (probability / dollars-per-share).
    "min_edge": 0.003,
    # Refuse fantasy books where mid says 99¢ but ask is gapped away.
    "max_ask_mid_spread": 0.015,
    # Tape must not contradict the decided side.
    "mom_contradict": 0.0010,
    # Sizing: thin edge → modest bankroll fraction + hard USD cap.
    "position_size_pct": 0.10,
    "max_trade_usd": 15.0,
    "min_confidence": 0.20,
}


class SweeperBot(BaseBot):
    """Certainty sweeper — buy locked outcomes still offered under $1."""

    def __init__(self, name="sweeper-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="sweeper",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def analyze(self, market, signals):
        return strategy_decision("hold", reasoning="sweeper: pure make_decision path")

    @staticmethod
    def _settlement_edge(ask: float) -> float:
        """Locked-outcome edge: worth $1, pay ask + fee."""
        fee = polymarket_fills.fee_per_share(float(ask), is_maker=False)
        return 1.0 - float(ask) - fee

    def make_decision(self, market, signals):
        p = self.strategy_params
        _dq = data_quality_skip(signals)
        if _dq is not None:
            return _dq
        sv = SignalView.of(signals)

        time_rem = market.get("time_remaining_seconds")
        # Floor to config horizon (pre_settle + TWAP window) so a stale
        # evolved param cannot re-open early 98¢ entries after a window cutover.
        try:
            from signals.twap import settlement_entry_horizon_sec
            horizon = int(settlement_entry_horizon_sec())
        except Exception:
            horizon = 80
        entry_window = max(int(p.get("entry_window_sec", horizon) or horizon), horizon)
        if time_rem is None or float(time_rem) > entry_window:
            return strategy_decision(
                "skip",
                reasoning=(
                    f"sweeper: waiting (rem={time_rem}s, window={entry_window}s)"
                ),
                skip_reason="sweeper_window",
            )

        yes_mid = float(market.get("current_price") or 0.5)
        no_mid = market.get("no_price")
        if no_mid is None:
            no_mid = round(1.0 - yes_mid, 4)
        else:
            no_mid = float(no_mid)
        yes_ask = float(market.get("yes_ask") or yes_mid)
        no_ask = float(market.get("no_ask") or no_mid)

        drift = float(sv.btc_drift or 0.0)
        min_drift = float(p.get("min_drift", 0.32))
        # DB/evolved 0.65 was calibrated when 5 bp printed as tanh 0.75.
        if min_drift >= 0.55:
            min_drift = float(getattr(config, "SWEEPER_MIN_DRIFT", 0.32))
        min_price = float(p.get("min_price", 0.97))
        max_price = float(p.get("max_price", 0.999))
        min_edge = float(p.get("min_edge", 0.003))
        max_spread = float(p.get("max_ask_mid_spread", 0.015))
        mom_contra = float(p.get("mom_contradict", 0.0010))
        min_twap_cert = float(p.get("min_twap_certainty", 0.45))
        if min_twap_cert >= 0.54:
            min_twap_cert = float(getattr(config, "SWEEPER_MIN_TWAP_CERTAINTY", 0.45))
        pol = sv.settlement_policy or {}
        twap_cert = float(pol.get("certainty") or sv.twap_certainty or 0.0)
        in_twap_win = bool(sv.in_settlement_window)
        phase = sv.market_phase or pol.get("phase") or "unknown"

        # Soft regime stand-down only when hard-skip is actually enabled.
        try:
            from arena.regime_adapt import adjustments as _regime_adj
            radj = _regime_adj(
                self.regime_context(signals).get("label"),
                strategy_type="sweeper",
            )
            if getattr(radj, "block_directional", False):
                return strategy_decision(
                    "skip",
                    reasoning=f"sweeper: regime hard-skip {radj.label}",
                    skip_reason="regime_hard_skip",
                )
        except Exception:
            pass

        # Prefer settlement window; pre_settle needs stronger drift.
        if phase == "pre_settle":
            min_drift += float(p.get("pre_settle_extra_drift", 0.10))
        # Inside the TWAP averaging window, require partial-settlement certainty
        # so a single tick spike can't look like a free lock. Coverage outage
        # is skipped earlier via data_quality_skip.
        if (
            in_twap_win
            and twap_cert < min_twap_cert
        ):
            return strategy_decision(
                "skip",
                reasoning=(
                    f"sweeper: TWAP not locked (cert={twap_cert:.2f}"
                    f"<{min_twap_cert:.2f} rem={time_rem}s phase={phase})"
                ),
                skip_reason="twap_certainty",
            )
        # Outside the settlement window still require strong cert when phase
        # is pre_settle (partial TWAP coverage) — soak flip losses were here.
        if phase == "pre_settle" and twap_cert < min_twap_cert * 0.85:
            return strategy_decision(
                "skip",
                reasoning=(
                    f"sweeper: pre_settle cert weak (cert={twap_cert:.2f}"
                    f"<{min_twap_cert * 0.85:.2f} rem={time_rem}s)"
                ),
                skip_reason="twap_certainty",
            )

        prices = sv.prices
        btc_momentum = 0.0
        if len(prices) >= 2 and prices[-1] > 0:
            btc_momentum = (prices[-1] - prices[-2]) / prices[-2]

        features = learning.extract_features(
            yes_mid, btc_momentum,
            volume=(sv.orderflow or {}).get("volume_24h")
            if isinstance(sv.orderflow, dict) else None,
            time_rem=time_rem,
        )

        def _candidate(side: str, signed_d: float, mid: float, ask: float):
            if signed_d < min_drift:
                return None
            # 1 − ask − fee is only EV if the outcome is actually decided.
            # tanh 0.45 @ 99¢ is Φ≈0.67 — overnight 02:34 lost $5.70.
            from bots.base_bot import implied_side_prob as _imp
            implied = _imp(
                side=side, signals=signals, signed_lane=drift,
            )
            need_imp = float(
                p.get(
                    "min_implied",
                    getattr(config, "SWEEPER_MIN_IMPLIED", 0.97),
                )
            )
            if implied + 1e-12 < need_imp:
                return None
            if not (min_price <= mid <= max_price):
                return None
            if not (min_price <= ask <= max_price):
                return None
            if (ask - mid) > max_spread:
                return None
            # Momentum non-contradiction: YES needs non-crashing tape; NO
            # needs non-ripping tape.
            if side == "yes" and btc_momentum < -mom_contra:
                return None
            if side == "no" and btc_momentum > mom_contra:
                return None
            edge = self._settlement_edge(ask)
            if edge < min_edge:
                return None
            return {
                "side": side,
                "mid": mid,
                "ask": ask,
                "edge": edge,
                "signed_d": signed_d,
            }

        yes_c = _candidate("yes", drift, yes_mid, yes_ask)
        no_c = _candidate("no", -drift, no_mid, no_ask)

        contributing = {
            "drift": drift,
            "btc_momentum": btc_momentum,
            "time_rem": time_rem,
            "yes_mid": yes_mid,
            "no_mid": no_mid,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "yes_edge": self._settlement_edge(yes_ask) if yes_ask else 0.0,
            "no_edge": self._settlement_edge(no_ask) if no_ask else 0.0,
            "twap_certainty": twap_cert,
            "in_settlement_window": in_twap_win,
            "market_phase": phase,
            "resolution_source": sv.resolution_source,
        }

        if yes_c and (not no_c or yes_c["edge"] >= no_c["edge"]):
            pick = yes_c
        elif no_c:
            pick = no_c
        else:
            # Diagnostic skip reasons for soak telemetry.
            y_spread = yes_ask - yes_mid
            n_spread = no_ask - no_mid
            why = (
                f"sweeper: no lock (drift={drift:+.3f} rem={time_rem}s "
                f"yes={yes_mid:.3f}/{yes_ask:.3f} eY={contributing['yes_edge']:+.4f} "
                f"no={no_mid:.3f}/{no_ask:.3f} eN={contributing['no_edge']:+.4f} "
                f"band=[{min_price:.3f},{max_price:.3f}] min_d={min_drift:.2f})"
            )
            if (
                abs(drift) >= min_drift
                and (
                    y_spread > max_spread
                    or n_spread > max_spread
                )
            ):
                why = (
                    f"sweeper: ask gap (yes {y_spread:.3f}/no {n_spread:.3f}"
                    f">{max_spread:.3f})"
                )
            skip_r = "ask_quality" if "ask gap" in why else "no_lock"
            return strategy_decision(
                "skip",
                reasoning=why,
                signals=contributing,
                features=features,
                skip_reason=skip_r,
            )

        side = pick["side"]
        side_mid = pick["mid"]
        side_ask = pick["ask"]
        side_edge = pick["edge"]
        signed = pick["signed_d"]

        # Structure confidence: late + extreme TWAP-drift + deep in band +
        # settlement certainty. Edge is tiny by design — do not scale conf
        # off edge size.
        time_frac = 1.0 - min(1.0, float(time_rem) / max(entry_window, 1))
        price_depth = (side_mid - min_price) / max(max_price - min_price, 1e-6)
        conf = min(
            0.95,
            0.30
            + 0.30 * min(1.0, abs(signed))
            + 0.15 * time_frac
            + 0.10 * max(0.0, min(1.0, price_depth))
            + 0.15 * min(1.0, twap_cert if in_twap_win else time_frac),
        )
        min_conf = float(p.get("min_confidence", 0.20))
        if conf < min_conf:
            return strategy_decision(
                "skip", side,
                confidence=conf,
                reasoning=f"sweeper: conf {conf:.2f} < {min_conf}",
                signals=contributing,
                features=features,
                skip_reason="low_confidence",
            )

        from signals.prob import live_side_prob, directional_net_edge
        p_side, _psrc = live_side_prob(
            side=side, signals=signals, strategy_type=self.strategy_type,
            signed_lane=drift,
        )
        try:
            import db as _db
            cutoff = _db.et_day_start_utc(0)
            with _db.get_conn() as _conn:
                n_loss = int((_conn.execute(
                    """SELECT COUNT(*) FROM trades
                       WHERE bot_name=? AND outcome='loss'
                         AND created_at>=?""",
                    (self.name, cutoff),
                ).fetchone() or [0])[0] or 0)
        except Exception:
            n_loss = 0
        if n_loss > 0 and p_side < 0.99:
            return strategy_decision(
                "skip", side,
                reasoning=f"sweeper: post-flip p={p_side:.3f}<0.99",
                signals=contributing,
                features=features,
                skip_reason="sweeper_post_flip",
            )

        # Size: Kelly on (p−ask−fee), not assume P=1. Cap USD modest.
        price = max(float(side_ask), 0.01)
        try:
            from bots.base_bot import _sizing_bankroll, _portfolio_weight, _risk_size_mult
            bankroll = (
                _sizing_bankroll(self.trading_mode)
                * _portfolio_weight(self.name)
                * _risk_size_mult(self.name)
            )
        except Exception:
            bankroll = float(getattr(config, "PAPER_BANKROLL_DEFAULT", 200.0))

        size_edge = max(0.0, directional_net_edge(
            p_side, side_ask, exchange=market.get("exchange"),
        ))
        f_star = size_edge / max(1e-6, 1.0 - price)
        kf = float(getattr(config, "KELLY_FRACTION", 0.25) or 0.25)
        max_usd = float(p.get("max_trade_usd", 15.0))
        amount = min(max_usd, kf * f_star * bankroll)
        # Venue minimum shares (slightly above floor for tick/fee rounding).
        min_cost = config.POLYMARKET_MIN_SHARES * price * 1.15
        amount = max(amount, min_cost)
        target_shares = round(amount / price, 4)
        amount = target_shares * price

        reasoning = (
            f"sweeper: lock drift={drift:+.3f} → {side} mid={side_mid:.3f} "
            f"ask={side_ask:.3f} edge={side_edge:+.4f} rem={time_rem}s "
            f"fee={polymarket_fills.fee_per_share(side_ask, is_maker=False):.4f} "
            f"conf={conf:.2f}"
        )
        return strategy_decision(
            "buy", side,
            edge=side_edge,
            confidence=conf,
            reasoning=reasoning,
            signals=contributing,
            suggested_amount=amount,
            entry_price=round(price, 4),
            features=features,
            target_shares=target_shares,
        )

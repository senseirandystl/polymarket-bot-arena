"""FeeZoneMaker — drift-backed taker entries in the fee-friction price band.

REALITY: venues still fill as *takers* (paper walks asks; live market orders).
The zero-fee maker advantage is aspirational until true limit posting exists.
So this bot is a **selective directional** entry in the mid-high fee band,
not a spread-capture MM.

2026-08 redesign (after soak):
  * Hard mid/ask consistency gate (fantasy fills at ask≪mid rejected).
  * Side chosen by *signed drift first*, then zone membership — not
    "whichever side is in zone" alone (that bought expensive favorites).
  * Edge priced on ask; lag rule: mid must still lag implied_P enough.
  * Size via calibrated Kelly (not flat %); conf from structure not zone depth.
  * Tighter zone [0.58, 0.78] for better BE after fees.
"""

from __future__ import annotations

import config
import learning
import polymarket_fills
from bots.base_bot import BaseBot, strategy_decision
from bots.edge_calibration import quality_confidence
from bots.maker_utils import maker_kelly_amount, mid_ask_gap_ok, resolve_side_exec
from signals.lab import SignalView


def taker_fee(price: float) -> float:
    """Canonical Polymarket taker fee for ONE share at ``price`` (USDC)."""
    return polymarket_fills.taker_fee(1.0, price)


DEFAULT_PARAMS = {
    # Tighter than the old 0.56–0.86 band: extremes either thin margin or
    # underdog fights. Fee still ≥ ~100 bps through most of this range.
    "min_price_zone": 0.58,
    "max_price_zone": 0.78,
    "min_fee_bps": 90,
    "min_drift": 0.18,          # signed drift toward quoted side
    "min_edge": 0.025,          # implied_P − ask − fee
    "max_mid_vs_implied": 0.02, # mid must lag implied by at least this
    "spread_ticks": 2,
    "position_size_pct": 0.06,  # Kelly cap as fraction of get_max_position()
    "lookback_candles": 5,
    "min_confidence": 0.20,     # structure conf floor (not zone-depth conf)
    "max_inventory_usd": 20.0,
    "mom_contradict": 0.0012,   # hard BTC move against side → stand down
}


class FeeZoneMakerBot(BaseBot):
    """Drift-first selective entries when the lagging side sits in the fee zone."""

    strategy_type = "fee_zone_maker"

    def __init__(self, name="fee-zone-maker-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="fee_zone_maker",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def analyze(self, market: dict, signals: dict) -> dict:
        p = self.strategy_params
        market_price = market.get("current_price") or 0.5
        time_rem = market.get("time_remaining_seconds")

        tick = 0.01
        half_spread = p["spread_ticks"] * tick
        maker_bid = round(max(0.01, market_price - half_spread), 2)
        maker_ask = round(min(0.99, market_price + half_spread), 2)
        maker_mid = market_price

        def _hold(reason, signals_out=None):
            return strategy_decision(
                "hold", reasoning=reason, signals=signals_out or {},
                maker_bid=maker_bid, maker_ask=maker_ask,
                maker_mid=maker_mid, maker_side="both",
            )

        sv = SignalView.of(signals)
        drift = float(sv.btc_drift or 0.0)
        min_drift = float(p.get("min_drift", 0.18))
        if abs(drift) < min_drift:
            return _hold(f"fzm: |drift|={abs(drift):.3f} < {min_drift:.2f}")

        # Drift picks the side; zone membership is a filter, not the selector.
        no_price = market.get("no_price")
        if no_price is None:
            no_price = round(1.0 - market_price, 4)
        if drift > 0:
            side, side_mid = "yes", float(market_price)
        else:
            side, side_mid = "no", float(no_price)

        min_zone = float(p["min_price_zone"])
        max_zone = float(p["max_price_zone"])
        if not (min_zone <= side_mid <= max_zone):
            return _hold(
                f"fzm: {side} mid={side_mid:.2f} outside fee zone "
                f"[{min_zone:.2f},{max_zone:.2f}]")

        side_exec, _src = resolve_side_exec(market, side, side_mid)
        ok, why = mid_ask_gap_ok(side_mid, side_exec)
        if not ok:
            return _hold(f"fzm: book integrity — {why}")

        maker_bid = round(max(0.01, side_mid - half_spread), 2)
        maker_ask = round(min(0.99, side_mid + half_spread), 2)
        maker_mid = side_mid

        fee = taker_fee(side_exec)
        fee_bps = fee * 10000
        min_fee = float(p["min_fee_bps"])
        if fee_bps < min_fee:
            return _hold(
                f"fzm: fee {fee_bps:.0f}bps < {min_fee:.0f}bps at ask={side_exec:.2f}")

        signed_drift = drift if side == "yes" else -drift
        implied_p = 0.5 + 0.5 * signed_drift
        # Lag rule: crowd mid must still sit below implied (market lags drift).
        max_mid_vs = float(p.get("max_mid_vs_implied", 0.02))
        if side_mid > implied_p - max_mid_vs:
            return _hold(
                f"fzm: mid={side_mid:.2f} not lagging implied={implied_p:.2f} "
                f"(need mid ≤ {implied_p - max_mid_vs:.2f})")

        fzm_edge = implied_p - float(side_exec) - fee
        min_edge = float(p.get("min_edge", 0.025))
        if fzm_edge < min_edge:
            return _hold(
                f"fzm: edge {fzm_edge:+.3f} < {min_edge:.3f} "
                f"(implied={implied_p:.2f} ask={side_exec:.2f})")

        inventory = self._inventory_usd(market, side)
        max_inv = float(p.get("max_inventory_usd", 20.0))
        inv_headroom = max_inv - inventory
        if inv_headroom <= 0:
            return _hold(
                f"fzm: inventory cap — ${inventory:.2f} open on {side} ≥ ${max_inv:.2f}")

        prices = sv.prices
        lb = int(p["lookback_candles"])
        momentum = 0.0
        if len(prices) >= lb and prices[-lb] > 0:
            momentum = (prices[-1] - prices[-lb]) / prices[-lb]
        signed_mom = momentum if side == "yes" else -momentum
        if signed_mom < -float(p.get("mom_contradict", 0.0012)):
            return _hold(
                f"fzm: BTC momentum contradicts {side} (mom={momentum:+.5f})")

        conf = quality_confidence(
            edge=fzm_edge,
            abs_drift=abs(drift),
            side_mid=side_mid,
            side=side,
        )
        min_conf = float(p.get("min_confidence", 0.20))
        if conf < min_conf:
            return _hold(f"fzm: quality conf {conf:.3f} < {min_conf:.2f}")

        # Size: calibrated Kelly against a bankroll slice, capped by inventory.
        try:
            import db as _db
            bankroll = float(_db.get_paper_available())
        except Exception:
            bankroll = float(getattr(config, "PAPER_BANKROLL_DEFAULT", 200.0))
        try:
            from arena.portfolio import get_weight
            w = float(get_weight(self.name) or (1.0 / 8.0))
        except Exception:
            w = 0.125
        amount = maker_kelly_amount(
            fzm_edge, float(side_exec), bankroll * w,
            size_pct_cap=float(p["position_size_pct"]),
            inv_headroom=inv_headroom,
        )
        min_usd = float(getattr(config, "POLYMARKET_MIN_SHARES", 5)) * float(side_exec) * 0.5
        if amount < max(0.50, min_usd * 0.25):
            return _hold(f"fzm: size ${amount:.2f} too small")

        of_data = sv.orderflow
        features = learning.extract_features(
            market_price, momentum,
            volume=of_data.get("volume_24h"),
            time_rem=time_rem,
        )

        return strategy_decision(
            "buy", side,
            edge=fzm_edge,
            confidence=conf,
            reasoning=(
                f"fzm: {side} mid={side_mid:.2f} exec={float(side_exec):.2f} "
                f"fee={fee_bps:.0f}bps drift={drift:+.3f} implied={implied_p:.2f} "
                f"edge={fzm_edge:+.3f} conf={conf:.3f} "
                f"quote_bid={maker_bid:.2f} quote_ask={maker_ask:.2f} "
                f"inv=${inventory:.2f}"
            ),
            signals={
                "drift": drift, "signed_drift": signed_drift,
                "momentum": momentum, "fee_bps": fee_bps,
                "implied_p": implied_p, "inventory_usd": inventory,
            },
            suggested_amount=amount,
            entry_price=round(float(side_exec), 4),
            features=features,
            maker_bid=maker_bid,
            maker_ask=maker_ask,
            maker_mid=maker_mid,
            maker_side=side,
        )

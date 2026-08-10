"""LateWindowMaker — late-window drift lag sniper (taker fills today).

Thesis: near expiry, BTC **TWAP** direction is largely locked; buy the lagging
side when drift is strong and the book still underprices it.

Under TWAP resolution (2026-08-07+) the last 30s are an averaging window, not
a single print race — enter before/through that window on TWAP moneyness
(``btc_drift``), not last-tick snipes.

2026-08 redesign:
  * Mid/ask integrity gate (no more mid=0.90 ask=0.49 fantasy edges).
  * Edge always on ask; lag rule on mid vs implied_P.
  * Tighter price band [0.55, 0.80] — above ~0.80 BE gap is brutal as taker.
  * Size via calibrated Kelly; conf from structure.
  * Slightly shorter window (120s) so entries aren't early-window noise.
  * TWAP certainty boost inside the settlement window.
"""

from __future__ import annotations

import config
import learning
import polymarket_fills
from bots.base_bot import BaseBot, strategy_decision
from bots.edge_calibration import quality_confidence
from bots.maker_utils import maker_kelly_amount, mid_ask_gap_ok, resolve_side_exec
from signals.lab import SignalView

DEFAULT_PARAMS = {
    # Cover pre-TWAP lock-in + full 30s settlement averaging window.
    "entry_window_sec": 120,
    "min_drift": 0.28,
    "min_momentum": 0.0004,     # momentum must not contradict drift
    "min_price_yes": 0.55,
    "max_price_yes": 0.80,      # was 0.90 — expensive favorites lost $ despite WR
    "min_edge": 0.03,
    "max_mid_vs_implied": 0.02,
    "maker_offset_pct": 0.04,   # logged limit metric only
    "position_size_pct": 0.08,
    "lookback_candles": 3,
    "max_inventory_usd": 25.0,
    "min_confidence": 0.22,
}


class LateWindowMakerBot(BaseBot):
    """Drift-first late-window lag entries on the winning side's token."""

    strategy_type = "late_window_maker"

    def __init__(self, name="late-window-maker-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="late_window_maker",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def analyze(self, market: dict, signals: dict) -> dict:
        p = self.strategy_params
        time_rem = market.get("time_remaining_seconds")
        market_price = market.get("current_price") or 0.5

        def _hold(reason, signals_out=None):
            return strategy_decision(
                "hold", reasoning=reason, signals=signals_out or {},
                maker_bid=round(max(0.01, market_price - 0.02), 2),
                maker_ask=round(min(0.99, market_price + 0.02), 2),
                maker_mid=market_price,
                maker_side="both",
            )

        entry_window = int(p["entry_window_sec"])
        if time_rem is None or time_rem > entry_window:
            return _hold(f"lwm: waiting (rem={time_rem}s, window={entry_window}s)")

        sv = SignalView.of(signals)
        drift = float(sv.btc_drift or 0.0)
        min_drift = float(p.get("min_drift", 0.28))
        if abs(drift) < min_drift:
            return _hold(f"lwm: weak drift ({drift:+.3f} < {min_drift})")

        prices = sv.prices
        lb = int(p["lookback_candles"])
        momentum = 0.0
        if len(prices) >= lb and prices[-lb] > 0:
            momentum = (prices[-1] - prices[-lb]) / prices[-lb]
        signed_mom = momentum if drift > 0 else -momentum
        min_mom = float(p["min_momentum"])
        if signed_mom < -min_mom:
            return _hold(
                f"lwm: momentum contradicts drift "
                f"(drift={drift:+.3f} mom={momentum:+.5f})")

        no_price = market.get("no_price")
        if no_price is None:
            no_price = round(1.0 - market_price, 4)
        if drift > 0:
            side, side_mid = "yes", float(market_price)
        else:
            side, side_mid = "no", float(no_price)

        min_price = float(p["min_price_yes"])
        max_price = float(p["max_price_yes"])
        if side_mid < min_price:
            return _hold(
                f"lwm: {side} mid {side_mid:.2f} < {min_price} (no book confirmation)")
        if side_mid > max_price:
            return _hold(
                f"lwm: {side} mid {side_mid:.2f} > {max_price} (margin too thin)")

        side_exec, _ = resolve_side_exec(market, side, side_mid)
        ok, why = mid_ask_gap_ok(side_mid, side_exec)
        if not ok:
            return _hold(f"lwm: book integrity — {why}")

        implied_p = 0.5 + 0.5 * abs(drift)
        max_mid_vs = float(p.get("max_mid_vs_implied", 0.02))
        if side_mid > implied_p - max_mid_vs:
            return _hold(
                f"lwm: mid={side_mid:.2f} not lagging implied={implied_p:.2f}")

        fee = polymarket_fills.taker_fee(1.0, side_exec)
        min_edge = float(p.get("min_edge", 0.03))
        lwm_edge = implied_p - float(side_exec) - fee
        if lwm_edge < min_edge:
            return _hold(
                f"lwm: edge {lwm_edge:+.3f} < {min_edge:.3f} "
                f"(implied={implied_p:.2f} ask={side_exec:.2f})")

        inventory = self._inventory_usd(market, side)
        max_inv = float(p.get("max_inventory_usd", 25.0))
        inv_headroom = max_inv - inventory
        if inv_headroom <= 0:
            return _hold(
                f"lwm: inventory cap — ${inventory:.2f} open on {side} ≥ ${max_inv:.2f}")

        maker_ask = round(min(max_price, side_mid + float(p["maker_offset_pct"])), 2)
        maker_bid = round(max(0.01, side_mid - 0.02), 2)
        maker_mid = round((maker_bid + maker_ask) / 2, 3)
        edge_bps = float(p["maker_offset_pct"]) * 10000

        time_weight = 1.0 - (float(time_rem) / entry_window)
        conf = quality_confidence(
            edge=lwm_edge,
            abs_drift=abs(drift),
            side_mid=side_mid,
            side=side,
        )
        # Mild urgency boost to conf only (not size-from-conf).
        conf = min(0.92, conf + 0.08 * time_weight)
        # TWAP settlement policy: conf boost when averaging window is locking.
        pol = sv.settlement_policy or {}
        twap_cert = float(pol.get("certainty") or sv.twap_certainty or 0.0)
        if pol.get("policy_active") and float(pol.get("conf_boost") or 0) > 0:
            conf = min(0.95, conf + float(pol["conf_boost"]))
        elif sv.in_settlement_window and twap_cert > 0:
            conf = min(0.95, conf + 0.10 * twap_cert)
        # Low-certainty settlement: demand more edge (spot spikes are noise).
        if pol.get("policy_active"):
            e_mult = float(pol.get("edge_mult") or 1.0)
            if e_mult > 1.0:
                min_edge = float(p.get("min_edge", 0.03)) * e_mult
                if lwm_edge < min_edge:
                    return _hold(
                        f"lwm: TWAP settle edge {lwm_edge:+.3f} < {min_edge:.3f} "
                        f"(phase={pol.get('phase')} cert={twap_cert:.2f})"
                    )
        if conf < float(p.get("min_confidence", 0.22)):
            return _hold(f"lwm: conf {conf:.3f} too low")

        try:
            import db as _db
            bankroll = float(_db.get_paper_available())
        except Exception:
            bankroll = float(getattr(config, "PAPER_BANKROLL_DEFAULT", 200.0))
        try:
            from arena.portfolio import get_weight
            w = float(get_weight(self.name) or 0.125)
        except Exception:
            w = 0.125
        amount = maker_kelly_amount(
            lwm_edge, float(side_exec), bankroll * w,
            size_pct_cap=float(p["position_size_pct"]),
            inv_headroom=inv_headroom,
        )
        min_usd = float(getattr(config, "POLYMARKET_MIN_SHARES", 5)) * float(side_exec) * 0.5
        if amount < max(0.50, min_usd * 0.25):
            return _hold(f"lwm: size ${amount:.2f} too small")

        of_data = sv.orderflow
        features = learning.extract_features(
            market_price, momentum,
            volume=of_data.get("volume_24h"),
            time_rem=time_rem,
        )

        return strategy_decision(
            "buy", side,
            edge=lwm_edge,
            confidence=conf,
            reasoning=(
                f"lwm: time={time_rem:.0f}s mom={momentum:+.5f} "
                f"{side} mid={side_mid:.2f} ask={float(side_exec):.2f} "
                f"limit={maker_ask:.2f} edge={lwm_edge:+.3f} "
                f"bps={edge_bps:.0f} tw={time_weight:.2f} inv=${inventory:.2f}"
            ),
            signals={
                "drift": drift, "momentum": momentum,
                "implied_p": implied_p, "time_weight": time_weight,
                "inventory_usd": inventory,
                "twap_certainty": twap_cert,
                "in_settlement_window": bool(sv.in_settlement_window),
                "resolution_source": sv.resolution_source,
            },
            suggested_amount=amount,
            entry_price=round(float(side_exec), 4),
            features=features,
            maker_bid=maker_bid,
            maker_ask=maker_ask,
            maker_mid=maker_mid,
            maker_side=side,
        )

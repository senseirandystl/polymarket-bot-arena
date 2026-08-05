"""True maker — limit-first GTC-style quoting on the lagging side.

Posts (paper: simulate_limit_buy; live: place_limit_order GTC) at a
passive limit near mid, only when:
  * |drift| ≥ min_drift toward the chosen side
  * side mid still lags (≤ max_side_mid)
  * mid/ask integrity gap is acceptable

Unlike late-window / fee-zone makers that still fill as aggressive takers
when the book walks, this bot always prefers the limit path and sizes
conservatively. Menu only — not default slate until live fill rates proven.
"""

from __future__ import annotations

import config
import polymarket_fills
from bots.base_bot import BaseBot, strategy_decision
from bots.edge_calibration import quality_confidence
from bots.maker_utils import mid_ask_gap_ok, resolve_side_exec
from signals.lab import SignalView

DEFAULT_PARAMS = {
    "min_drift": 0.18,
    "min_edge": 0.015,
    "max_side_mid": 0.62,
    "min_side_mid": 0.38,
    "maker_offset": 0.01,       # tick inside the mid toward passive
    "position_size_pct": 0.05,
    "max_inventory_usd": 20.0,
    "min_confidence": 0.18,
    "entry_window_sec": 240,    # avoid last seconds chaos
}


class TrueMakerBot(BaseBot):
    """Limit-first passive maker (menu only)."""

    strategy_type = "true_maker"

    def __init__(self, name="true-maker-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="true_maker",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def analyze(self, market: dict, signals: dict) -> dict:
        p = self.strategy_params
        sv = SignalView.of(signals)
        time_rem = market.get("time_remaining_seconds")
        yes_mid = market.get("current_price") or 0.5
        no_mid = market.get("no_price")
        if no_mid is None:
            no_mid = round(1.0 - yes_mid, 4)

        def _hold(reason):
            bid = round(max(0.01, yes_mid - 0.02), 2)
            ask = round(min(0.99, yes_mid + 0.02), 2)
            return strategy_decision(
                "hold", reasoning=reason,
                maker_bid=bid, maker_ask=ask, maker_mid=yes_mid, maker_side="both",
            )

        win = int(p.get("entry_window_sec", 240))
        if time_rem is not None and time_rem > win:
            return _hold(f"true_maker: waiting rem={time_rem}s > {win}s")

        drift = float(sv.btc_drift or 0.0)
        min_drift = float(p.get("min_drift", 0.18))
        if abs(drift) < min_drift:
            return _hold(f"true_maker: weak drift {drift:+.3f}")

        side = "yes" if drift > 0 else "no"
        mid = float(yes_mid if side == "yes" else no_mid)
        ask, _src = resolve_side_exec(market, side, mid)
        max_mid = float(p.get("max_side_mid", 0.62))
        min_mid = float(p.get("min_side_mid", 0.38))
        if not (min_mid <= mid <= max_mid):
            return _hold(f"true_maker: mid {mid:.2f} outside band")

        ok, gap_why = mid_ask_gap_ok(mid, ask)
        if not ok:
            return _hold(f"true_maker: {gap_why}")

        implied = 0.5 + 0.5 * abs(drift)
        residual = implied - mid
        fee = polymarket_fills.fee_per_share(mid, is_maker=True)  # maker fee ≈ 0
        edge = residual - fee
        min_edge = float(p.get("min_edge", 0.015))
        if edge < min_edge:
            return _hold(f"true_maker: edge {edge:+.3f} < {min_edge:.3f}")

        offset = float(p.get("maker_offset", 0.01))
        # Passive buy: bid slightly below mid (or at mid − offset).
        limit = round(max(0.01, min(0.99, mid - offset)), 2)
        conf = quality_confidence(
            edge=edge, abs_drift=abs(drift), side_mid=mid, side=side,
            regime_label=self.regime_context(signals).get("label"),
        )
        try:
            from bots.base_bot import _sizing_bankroll
            bankroll = _sizing_bankroll(self.trading_mode)
        except Exception:
            bankroll = float(getattr(config, "PAPER_BANKROLL_DEFAULT", 200.0))
        max_inv = float(p.get("max_inventory_usd", 20.0))
        amount = min(bankroll * float(p.get("position_size_pct", 0.05)), max_inv)
        amount = max(amount, config.POLYMARKET_MIN_SHARES * limit * 1.15)

        return strategy_decision(
            "buy", side,
            edge=edge,
            confidence=conf,
            reasoning=(
                f"true_maker: drift={drift:+.3f} mid={mid:.2f} lim={limit:.2f} "
                f"edge={edge:+.3f} residual={residual:+.3f}"
            ),
            signals={"drift": drift, "lag": residual},
            suggested_amount=amount,
            entry_price=limit,
            limit_price=limit,
            maker_bid=limit if side == "yes" else round(1.0 - limit, 2),
            maker_ask=round(min(0.99, mid + offset), 2),
            maker_mid=mid,
            maker_side=side,
        )

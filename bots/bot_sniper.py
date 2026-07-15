"""Sniper bot — only trades when historical data shows 65%+ win rate.

v2 adjustments from sniper-v1 trade data (13 trades):
- cheap YES (40-48c): 100% WR, +$8.60 — KEEP, this is the money zone
- strong YES (58-65c): 20% WR, -$10.03 — REMOVED, widened skip zone to 64c
- strong YES (65-85c): 33% WR, -$6.49 — TIGHTENED, max YES now 78c
- strong NO (0-35c): only <25c won — TIGHTENED, max NO now 25c
- Momentum threshold tightened (0.0005 → 0.0003) to filter marginal trades

Trades less often but with much higher accuracy.

NO side (BUG_HISTORY #20): the old NO ban is removed. The sniper now applies the
SAME cheap/strong zone rules to the NO token's price (with DOWN-momentum
confirmation) that it applies to the YES token — a symmetric mirror of its own
strategy, not a banned side. The NO zones are unvalidated by live data yet
(mirror assumption); recheck once NO trades accumulate.
"""

import config
import learning
from bots.base_bot import BaseBot

DEFAULT_PARAMS = {
    "min_price_yes": 0.40,     # Min YES price for YES bets
    "max_price_yes": 0.78,     # Max YES price for YES bets (was 0.85 — 80c+ lost money)
    "max_price_no": 0.25,      # Max YES price for NO bets (was 0.35 — 30-35c lost, only <25c won)
    "skip_zone_low": 0.48,     # Start of coin-flip dead zone (cheap-YES 40-48¢ has 100% WR)
    "skip_zone_high": 0.64,    # End of coin-flip dead zone (was 0.58 — 58-65c was 20% WR)
    "require_momentum": True,  # Only trade when BTC momentum confirms
    "momentum_threshold": 0.0003,  # Tighter threshold (was 0.0005 hardcoded)
    "position_size_pct": 0.08, # Larger positions since we're more selective
    "min_confidence": 0.10,    # Only trade with real edge
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
        """Only emit a signal when conditions match high-WR patterns."""
        return {"action": "hold", "side": "yes", "confidence": 0, "reasoning": "sniper: no signal"}

    def _zone_signal(self, price):
        """Snipe-worthiness of a token priced ``price`` (side-agnostic).

        Mirrors the sniper's YES zones onto whichever token is being priced, so
        the SAME data-driven pattern (cheap favorite / strong signal) is applied
        to YES and NO alike. Returns ``(tradeable, confidence, label)``.
        """
        p = self.strategy_params
        skip_lo = p.get("skip_zone_low", 0.48)
        skip_hi = p.get("skip_zone_high", 0.64)
        max_price = p.get("max_price_yes", 0.78)
        min_price = p.get("min_price_yes", 0.40)
        if min_price <= price < skip_lo:
            # cheap zone: token priced just under 50c, historically wins
            return True, 0.20 + (0.50 - price) * 2.0, "cheap"
        if skip_hi < price <= max_price:
            # strong zone: clear market signal
            return True, 0.15 + (price - 0.50) * 1.5, "strong"
        return False, 0.0, "skip"

    def make_decision(self, market, signals):
        """Override full decision logic — pure data-driven rules.

        Ignores the base class signal hierarchy. Instead uses simple
        rules derived from historical trade data analysis.
        """
        market_price = market.get("current_price") or 0.5  # None if book down
        p = self.strategy_params

        # Zone thresholds are applied inside _zone_signal (per token price).
        require_mom = p.get("require_momentum", True)

        # Extract BTC momentum from signals
        prices = signals.get("prices", [])
        btc_momentum = 0.0
        if len(prices) >= 2 and prices[-1] > 0:
            btc_momentum = (prices[-1] - prices[-2]) / prices[-2]

        of_data = signals.get("orderflow", {})
        volume = of_data.get("volume_24h")
        time_rem = market.get("time_remaining_seconds")

        features = learning.extract_features(
            market_price, btc_momentum,
            volume=volume, time_rem=time_rem
        )

        # --- Determine side: snipe whichever token's price is in a good zone ---
        # Evaluate the SAME data-driven zones on both the YES token (yes price)
        # and the NO token (no price). BTC momentum must confirm the side's
        # direction: YES needs BTC not dropping, NO needs BTC not rising. Since
        # yes+no ~= 1, at most one side's price lands in a buy zone.
        no_price = market.get("no_price")
        if no_price is None:
            no_price = round(1.0 - market_price, 4)

        mom_thresh = p.get("momentum_threshold", 0.0003)
        yes_ok, yes_conf, yes_label = self._zone_signal(market_price)
        no_ok, no_conf, no_label = self._zone_signal(no_price)
        if require_mom:
            yes_ok = yes_ok and btc_momentum >= -mom_thresh
            no_ok = no_ok and btc_momentum <= mom_thresh

        side = None
        confidence = 0
        reasoning_parts = [f"yes={market_price:.2f} no={no_price:.2f}"]
        if yes_ok and yes_conf >= no_conf:
            side, confidence = "yes", yes_conf
            reasoning_parts.append(f"{yes_label}-YES zone ({market_price:.0%})")
        elif no_ok:
            side, confidence = "no", no_conf
            reasoning_parts.append(f"{no_label}-NO zone ({no_price:.0%})")
        else:
            return {
                "action": "skip", "side": "yes", "confidence": 0,
                "reasoning": (
                    f"sniper: no snipe zone (yes={market_price:.2f} "
                    f"no={no_price:.2f} mom={btc_momentum:+.4f})"
                ),
                "suggested_amount": 0, "features": features,
            }

        # --- Learned bias adjustment ---
        prior = 0.50
        learned_bias = learning.get_learned_bias(self.name, features, prior)
        # Slight adjustment from learning (don't let it override data rules)
        if side == "yes" and learned_bias < 0.35:
            confidence *= 0.7  # reduce confidence if learning says NO
        elif side == "no" and learned_bias > 0.65:
            confidence *= 0.7

        confidence = min(0.95, confidence)

        # --- Minimum confidence gate ---
        min_conf = p.get("min_confidence", 0.10)
        if confidence < min_conf:
            return {
                "action": "skip", "side": side, "confidence": confidence,
                "reasoning": f"sniper: conf {confidence:.2f} < {min_conf}",
                "suggested_amount": 0, "features": features,
            }

        # --- Early-window boost ---
        window_age = market.get("window_age_seconds")
        if window_age is not None and 0 <= window_age < 90:
            confidence *= 1.25
            confidence = min(0.95, confidence)
            reasoning_parts.append(f"early-window-boost(age={window_age:.0f}s)")

        # --- Late-window boost ---
        # Mirror of early-window: BTC direction increasingly certain in final 60s.
        time_rem = market.get("time_remaining_seconds")
        if time_rem is not None and 0 < time_rem < 60:
            confidence = min(0.95, confidence * 1.30)
            reasoning_parts.append(f"late-window-boost(rem={time_rem:.0f}s)")

        # --- Position sizing ---
        max_pos = config.get_max_position()
        size_pct = p.get("position_size_pct", 0.08)
        if window_age is not None and 0 <= window_age < 90:
            size_pct *= 1.2  # Larger positions in early window
        if time_rem is not None and 0 < time_rem < 60:
            size_pct *= 1.2  # Larger positions in late window too
        amount = max_pos * size_pct * (0.5 + confidence)
        amount = min(amount, max_pos)

        mom_str = f"mom={btc_momentum:+.4f}" if btc_momentum != 0 else "mom=flat"
        reasoning_parts.append(mom_str)
        reasoning_parts.append(f"=> {side} conf={confidence:.2f}")

        return {
            "action": "buy",
            "side": side,
            "confidence": confidence,
            "reasoning": "sniper: " + " ".join(reasoning_parts),
            "suggested_amount": amount,
            "features": features,
        }

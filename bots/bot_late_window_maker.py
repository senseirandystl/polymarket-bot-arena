"""LateWindowMaker — models the article's "T-10s maker" strategy.

The article:
  "At T-10 seconds before window close, BTC direction is ~85% determined.
   Post a maker order on the winning side at 90-95¢."

We use a 90-second entry window because Simmer polling runs every 15s and
the bot needs reaction time. At T-90s conviction is ~70%; the trade-off vs
LateWindowMaker's peer (FeeZoneMaker) is: fewer trades, higher WR target.

Paper mode: Simmer has no limit-order book, so we execute at market price
like all other bots. The "maker" logic controls WHEN and IF we enter.
Logs theoretical maker metrics (what limit we'd post, edge in bps) so we
can compare against real maker results if this ever goes live.

Competing hypothesis:
  High-conviction, time-gated, momentum-confirmed entries beat
  always-on fee-zone bets because the signal is strongest in the final seconds.
"""

import config
import learning
import polymarket_fills
from bots.base_bot import BaseBot

# Retune (was: window=90s, min_mom=0.0008, price [0.58,0.92]): the bot had ZERO
# trades all session. By T-90s a 5-min BTC market has almost always resolved to
# an extreme (price near 0 or 1), so the [0.58,0.92] band was rarely occupied
# that late and only ~4 evaluations fit in 90s at the ~23s maker cadence.
# Widening the window to 150s lets it catch the market earlier, while it is
# still in the profitable mid-high band; the max cap stays ≤0.90 to respect the
# break-even rule (buying YES as a taker above ~0.90 needs an implausible WR).
#
# Retune 2 (2026-07-16, harness net-edge data, ~300 resolved markets): side
# selection is now DRIFT-first — the validated fundamental (83% WR in the final
# minute) picks the side, momentum only confirms. Late-window band entries
# gated on |drift| ≥ 0.25 measured 85.8% WR / +7.7c per share net of price+fee;
# requiring momentum agreement on top raised WR to 88.4%. Momentum-only side
# selection is the weaker signal and is demoted to a non-contradiction check +
# confidence booster.
DEFAULT_PARAMS = {
    "entry_window_sec": 150,   # Activate in the last 150s (more shots, less extreme prices)
    "min_drift": 0.25,         # |btc_drift| needed for conviction (drift picks the side)
    "min_momentum": 0.0005,    # Momentum must not CONTRADICT the drift side by more than this
    "min_price_yes": 0.56,     # Chosen side's price must be ≥ 56¢ (direction confirmed by book)
    "max_price_yes": 0.90,     # Cap: above 90¢ taker margin is too thin to profit
    # Price must be JUSTIFIED by drift's calibrated probability (2026-07-17
    # overnight run: 69 trades, 71% WR but -$41.66 at avg entry 0.788 — WR ran
    # ~5-10pp BELOW the price paid at every level; the price already contained
    # the conviction). Drift's implied P = 0.5 + 0.5*|drift| is empirically
    # well-calibrated (see BUG #23 calibration table), so require
    # implied_P >= side_price + taker_fee + min_edge.
    "min_edge": 0.03,
    "maker_offset_pct": 0.06,  # Simulated limit = market_price + 6¢ (logged maker metric only)
    "position_size_pct": 0.10, # 10% of max — large because entries are highly selective
    "lookback_candles": 3,     # BTC candles used for momentum calculation
}


class LateWindowMakerBot(BaseBot):
    """Posts a directional maker quote in the final window when BTC momentum and
    price align — YES on up-momentum, NO on down-momentum (symmetric mirror)."""

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
        market_price = market.get("current_price") or 0.5  # None if book down

        # Maker quote fields always returned so run_maker_section() can log them
        def _hold(reason):
            return {
                "action": "hold",
                "side": "yes",
                "confidence": 0.0,
                "reasoning": reason,
                "maker_bid": round(max(0.01, market_price - 0.02), 2),
                "maker_ask": round(min(0.99, market_price + 0.02), 2),
                "maker_mid": market_price,
                "maker_side": "both",
            }

        # ── Time gate ────────────────────────────────────────────────────────
        entry_window = p["entry_window_sec"]
        if time_rem is None or time_rem > entry_window:
            return _hold(f"lwm: waiting (rem={time_rem}s, window={entry_window}s)")

        # ── Drift conviction gate (primary) ──────────────────────────────────
        # btc_drift is the validated "price to beat" fundamental (signals/
        # strike.py) — time-scaled, so late in the window a strong value means
        # the direction is close to locked in. It PICKS the side.
        drift = float(signals.get("btc_drift", 0.0) or 0.0)
        min_drift = p.get("min_drift", 0.25)
        if abs(drift) < min_drift:
            return _hold(f"lwm: weak drift ({drift:+.3f} < {min_drift})")

        # ── BTC momentum (confirmation only) ─────────────────────────────────
        prices = signals.get("prices", [])
        lb = p["lookback_candles"]
        momentum = 0.0
        if len(prices) >= lb and prices[-lb] > 0:
            momentum = (prices[-1] - prices[-lb]) / prices[-lb]

        # Momentum must not contradict the drift side (agreement measured
        # +2.6pp WR in the harness; contradiction is a warning sign).
        signed_mom = momentum if drift > 0 else -momentum
        min_mom = p["min_momentum"]
        if signed_mom < -min_mom:
            return _hold(
                f"lwm: momentum contradicts drift side "
                f"(drift={drift:+.3f} mom={momentum:+.5f})")

        # ── Side selection: quote the DRIFT side on its own token price ──────
        # The band check confirms the book agrees with the direction (yes+no
        # ~= 1, so only one side's price sits in the band). NO is a first-class
        # mirror of the YES entry, not a banned side.
        min_price = p["min_price_yes"]
        max_price = p["max_price_yes"]
        if drift > 0:
            side, side_price = "yes", market_price
        else:
            no_price = market.get("no_price")
            if no_price is None:
                no_price = round(1.0 - market_price, 4)
            side, side_price = "no", no_price

        if side_price < min_price:
            return _hold(
                f"lwm: {side} price {side_price:.2f} < {min_price} (no confirmation)")
        if side_price > max_price:
            return _hold(
                f"lwm: {side} price {side_price:.2f} > {max_price} (margin too thin)")

        # ── Edge gate: the price must be justified by drift's implied P ──────
        # implied_P = 0.5 + 0.5*|drift| is calibrated against resolved markets.
        # Buying above it is paying for conviction the fundamental doesn't
        # have — exactly how a 71%-WR bot lost money at 79c entries.
        implied_p = 0.5 + 0.5 * abs(drift)
        fee = polymarket_fills.taker_fee(1.0, side_price)
        min_edge = p.get("min_edge", 0.03)
        lwm_edge = implied_p - side_price - fee
        if lwm_edge < min_edge:
            return _hold(
                f"lwm: price {side_price:.2f} not justified by drift "
                f"(implied_P={implied_p:.2f}, edge={lwm_edge:+.3f} < {min_edge})")

        # ── Maker quote computation (on the chosen side's price) ──────────────
        # What we'd post as a limit order: slightly ahead of market to capture spread
        maker_ask = round(min(max_price, side_price + p["maker_offset_pct"]), 2)
        maker_bid = round(max(0.01, side_price - 0.02), 2)
        maker_mid = round((maker_bid + maker_ask) / 2, 3)
        edge_bps = p["maker_offset_pct"] * 10000  # spread captured if filled

        # ── Confidence: drift conviction × urgency × momentum agreement ──────
        time_weight = 1.0 - (time_rem / entry_window)  # 0 at window-open, 1 at close
        drift_strength = min(1.0, abs(drift))
        mom_strength = min(1.0, max(0.0, signed_mom) / (min_mom * 5))
        confidence = min(0.92, 0.35 + drift_strength * 0.30
                         + time_weight * 0.20 + mom_strength * 0.10)

        # ── Features ─────────────────────────────────────────────────────────
        of_data = signals.get("orderflow", {})
        features = learning.extract_features(
            market_price, momentum,
            volume=of_data.get("volume_24h"),
            time_rem=time_rem,
        )

        amount = config.get_max_position() * p["position_size_pct"]

        return {
            "action": "buy",
            "side": side,
            "confidence": confidence,
            "reasoning": (
                f"lwm: time={time_rem:.0f}s mom={momentum:+.5f} "
                f"{side} price={side_price:.2f} limit={maker_ask:.2f} "
                f"edge={edge_bps:.0f}bps tw={time_weight:.2f}"
            ),
            "suggested_amount": amount,
            # Expected taker price: the fill walks the real book from the best
            # ask, so expect ~the side's current price. Feeds the execute()
            # slippage guard (config.MAX_FILL_SLIPPAGE). Using maker_ask here
            # (side_price + 6c) silently widened the guard to ~9c over mid —
            # the maker_* fields below are logged metrics, not fill targets.
            "entry_price": round(side_price, 4),
            "features": features,
            "maker_bid": maker_bid,
            "maker_ask": maker_ask,
            "maker_mid": maker_mid,
            "maker_side": side,
        }

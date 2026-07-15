"""FeeZoneMaker — fee-aware directional maker for BTC 5-min markets.

Taker fee formula (official Polymarket docs — https://docs.polymarket.com/trading/fees):
    fee_usdc = feeRate × shares × p × (1 - p)     (makers pay ZERO)

BTC markets are the *crypto* tier, feeRate = 0.07 (peaks at $1.75 / 100 shares
at 50¢). Per-share taker fee across the quoting zone (feeRate 0.07):
    50¢ → 175 bps   60¢ → 168 bps   65¢ → 159 bps
    70¢ → 147 bps   75¢ → 131 bps   80¢ → 112 bps
    82¢ → 103 bps   85¢ →  89 bps   90¢ →  63 bps

The fee math lives in ONE place — :func:`polymarket_fills.taker_fee` (the same
function paper + live P&L use) — so this bot never re-derives it. (Earlier
versions carried a bogus quadratic ``0.25 × (p(1-p))²`` here; see BUG_HISTORY #17.)

Strategy:
  Post maker orders in the 60-82¢ YES zone where:
    1. Taker fee is ~103–168 bps (significant friction for takers crossing us)
    2. Market price gives a clear directional signal (>60¢ = YES favored)
    3. Maker pays ZERO fees — double advantage over takers

We run throughout the full market window (not time-gated), quoting
whenever the price is in the fee-advantage zone. Smaller positions,
higher frequency than LateWindowMaker.

Competing hypothesis:
  Always-on fee-zone quoting beats late-window time-gating because
  price signal alone (no momentum requirement) is sufficient in the
  60-82¢ range, and more trades → more learning data.
"""

import config
import learning
import polymarket_fills
from bots.base_bot import BaseBot


def taker_fee(price: float) -> float:
    """Canonical Polymarket taker fee for ONE share at ``price`` (USDC).

    Thin wrapper over :func:`polymarket_fills.taker_fee` (crypto tier,
    ``config.POLYMARKET_TAKER_FEE_RATE``) so the fee-zone gate uses the exact
    same formula as settled P&L. Do not re-derive the formula here.
    """
    return polymarket_fills.taker_fee(1.0, price)


DEFAULT_PARAMS = {
    # Zone widened 0.60-0.82 → 0.56-0.86 (still all ≥80 bps taker fee) because
    # 5-min BTC markets only sat in the old band ~9% of the time — the bot was
    # starved (1 trade/session). The wider band ~doubles quoting opportunities
    # while staying in the fee-friction range. Tune from live data.
    "min_price_zone": 0.56,    # Only quote at YES price ≥ 56¢
    "max_price_zone": 0.86,    # Only quote at YES price ≤ 86¢ (fee still ≥80 bps here)
    "min_fee_bps": 80,         # Require taker fee ≥ 80 bps at this price to justify quoting
    "spread_ticks": 2,         # Half-spread: 2 ticks (±2¢ around market price)
    "momentum_weight": 0.30,   # Weight of momentum signal in confidence (vs price signal)
    "position_size_pct": 0.06, # 6% of max — smaller per-trade, higher frequency
    "lookback_candles": 5,     # BTC candles for momentum context
    "min_confidence": 0.25,    # Skip if we can't reach this confidence
}


class FeeZoneMakerBot(BaseBot):
    """Quotes in the taker-fee-friction zone (56-86¢) throughout the window —
    whichever side (YES or NO) has its price in the zone (symmetric mirror)."""

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
        market_price = market.get("current_price") or 0.5  # None if book down
        time_rem = market.get("time_remaining_seconds")

        # Maker quote fields — always computed so run_maker_section() can log
        tick = 0.01
        half_spread = p["spread_ticks"] * tick
        maker_bid = round(max(0.01, market_price - half_spread), 2)
        maker_ask = round(min(0.99, market_price + half_spread), 2)
        maker_mid = market_price

        def _hold(reason):
            return {
                "action": "hold",
                "side": "yes",
                "confidence": 0.0,
                "reasoning": reason,
                "maker_bid": maker_bid,
                "maker_ask": maker_ask,
                "maker_mid": maker_mid,
                "maker_side": "both",
            }

        # ── Fee-zone gate: quote whichever SIDE's price is in the fee zone ────
        # The fee zone [56¢,86¢] never contains both yes and no (yes+no ~= 1),
        # so at most one side qualifies. NO is a first-class mirror: quote the NO
        # token when its price sits in the same fee-friction band. Fee is
        # symmetric (fee(p) == fee(1-p)), so the advantage is identical per side.
        min_zone = p["min_price_zone"]
        max_zone = p["max_price_zone"]
        no_price = market.get("no_price")
        if no_price is None:
            no_price = round(1.0 - market_price, 4)
        if min_zone <= market_price <= max_zone:
            side, side_price = "yes", market_price
        elif min_zone <= no_price <= max_zone:
            side, side_price = "no", no_price
        else:
            return _hold(
                f"fzm: neither side in fee zone [{min_zone},{max_zone}] "
                f"(yes={market_price:.2f} no={no_price:.2f})"
            )

        # Recompute the maker quote around the CHOSEN side's price.
        maker_bid = round(max(0.01, side_price - half_spread), 2)
        maker_ask = round(min(0.99, side_price + half_spread), 2)
        maker_mid = side_price

        # Verify taker fee is large enough to justify quoting
        fee = taker_fee(side_price)
        fee_bps = fee * 10000
        min_fee = p["min_fee_bps"]
        if fee_bps < min_fee:
            return _hold(f"fzm: fee {fee_bps:.0f}bps < {min_fee}bps at price={side_price:.2f}")

        # ── BTC momentum context ──────────────────────────────────────────────
        prices = signals.get("prices", [])
        lb = p["lookback_candles"]
        momentum = 0.0
        if len(prices) >= lb and prices[-lb] > 0:
            momentum = (prices[-1] - prices[-lb]) / prices[-lb]

        # Momentum must not contradict the side. YES price >56¢ says "up"; hard
        # BTC drop contradicts. For NO (betting "down"), a hard BTC rally
        # contradicts. Signed so the check mirrors per side.
        signed_mom = momentum if side == "yes" else -momentum
        if signed_mom < -0.0015:
            return _hold(f"fzm: BTC momentum contradicts {side} zone (mom={momentum:+.5f})")

        # ── Confidence ────────────────────────────────────────────────────────
        # Price signal: how far into the fee zone is the chosen side?
        price_signal = (side_price - min_zone) / (max_zone - min_zone)

        # Momentum signal: momentum confirming the side boosts confidence
        mw = p["momentum_weight"]
        mom_boost = min(0.30, max(0.0, signed_mom * 50))  # up to +0.30 from momentum
        confidence = min(0.88, 0.30 + price_signal * (1.0 - mw) * 0.50 + mom_boost * mw)

        min_conf = p["min_confidence"]
        if confidence < min_conf:
            return _hold(f"fzm: conf {confidence:.3f} < {min_conf}")

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
                f"fzm: {side} price={side_price:.2f} fee={fee_bps:.0f}bps "
                f"mom={momentum:+.5f} psig={price_signal:.2f} conf={confidence:.3f} "
                f"bid={maker_bid:.2f} ask={maker_ask:.2f}"
            ),
            "suggested_amount": amount,
            # Expected taker price = the ask we'd cross; feeds the execute()
            # slippage guard (config.MAX_FILL_SLIPPAGE) so an adverse book move
            # between decision and fill rejects rather than fills worse.
            "entry_price": maker_ask,
            "features": features,
            "maker_bid": maker_bid,
            "maker_ask": maker_ask,
            "maker_mid": maker_mid,
            "maker_side": side,
        }

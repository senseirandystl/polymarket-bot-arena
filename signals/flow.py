"""Improved CVD / aggressive-flow classification (pure, deterministic).

Refines the plain CVD lane (signals/orderflow_signals.cvd_from_trades) with
three reads over the same data-api trade tape. Every function takes the trade
list AND an explicit ``now`` timestamp — no clocks inside, so the offline
harness and live replay produce identical values from identical inputs.

Trade dicts carry ``side`` ('BUY'/'SELL'), ``outcome`` ('Up'/'Down'),
``size`` (shares) and ``timestamp`` (epoch seconds). Sign convention is
Up-space, same as the plain CVD: BUY Up / SELL Down = bullish (+),
SELL Up / BUY Down = bearish (−).

Directional outputs (YES/Up-frame, bounded [-1, 1]):
- ``flow_cvd_decay``: exponentially time-decayed signed volume — a burst of
  buying 20s ago outweighs the same volume 4 minutes ago (the plain CVD
  weighs the whole 100-trade tape flat). Volume-floored like the plain lane
  (BUG #27: unfloored thin tapes saturate to sign(tape)).
- ``flow_whale``: signed volume of LARGE prints only (>= WHALE_MIN_SHARES).
  Small orders are noise/retail; size is conviction. Floored separately.

Context outputs (non-directional):
- ``flow_rate``: 0..1 tape-activity score (decayed volume per minute vs a
  typical active tape) — a directional read off a dead tape means little.

CANDIDATES: the plain CVD lane is kill-switched pending re-validation; these
carry no live weight until validated (tools/validate_signals.py / live shadow
attribution — historical tape is not archived at fidelity, so live
attribution is the primary path).
"""

import math

from signals.curves import smooth_ramp

CVD_HALF_LIFE_SEC = 60.0    # decayed-CVD half-life
WHALE_MIN_SHARES = 100.0    # a print this size is "aggressive/informed" flow
VOLUME_FLOOR = 200.0        # shares; matches config.CVD_VOLUME_FLOOR intent
WHALE_VOLUME_FLOOR = 300.0  # whales are rarer — demand more volume to saturate
RATE_FULL_SHARES_PER_MIN = 500.0   # decayed volume/min that reads fully active


def _signed(trade: dict) -> float:
    """Signed size of one trade in Up-space (0.0 for malformed trades)."""
    try:
        size = float(trade.get("size", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    if size <= 0:
        return 0.0
    is_up = str(trade.get("outcome", "")).lower() == "up"
    is_buy = str(trade.get("side", "")).upper() == "BUY"
    return size if (is_up == is_buy) else -size


def _age_weight(trade: dict, now: float, half_life: float) -> float:
    """exp-decay weight by trade age; missing/future timestamps weigh 1.0."""
    try:
        ts = float(trade.get("timestamp", 0) or 0)
    except (TypeError, ValueError):
        return 1.0
    if ts <= 0 or ts > now:
        return 1.0
    return math.exp(-(now - ts) * math.log(2.0) / half_life)


def decayed_cvd(trades: list, now: float,
                half_life: float = CVD_HALF_LIFE_SEC,
                floor: float = VOLUME_FLOOR) -> float:
    """Time-decayed CVD in [-1, 1] (0.0 on an empty tape)."""
    net = total = 0.0
    for t in trades or []:
        signed = _signed(t)
        if signed == 0.0:
            continue
        w = _age_weight(t, now, half_life)
        net += signed * w
        total += abs(signed) * w
    if total <= 0:
        return 0.0
    return max(-1.0, min(1.0, net / max(total, floor)))


def whale_delta(trades: list, now: float,
                min_shares: float = WHALE_MIN_SHARES,
                half_life: float = CVD_HALF_LIFE_SEC,
                floor: float = WHALE_VOLUME_FLOOR) -> float:
    """Time-decayed CVD of large prints only, in [-1, 1]."""
    big = [t for t in (trades or []) if abs(_signed(t)) >= min_shares]
    return decayed_cvd(big, now, half_life=half_life, floor=floor)


def trade_rate(trades: list, now: float,
               half_life: float = CVD_HALF_LIFE_SEC,
               full_per_min: float = RATE_FULL_SHARES_PER_MIN) -> float:
    """0..1 activity score: decayed traded volume per minute, smooth-ramped."""
    total = 0.0
    for t in trades or []:
        signed = _signed(t)
        if signed == 0.0:
            continue
        total += abs(signed) * _age_weight(t, now, half_life)
    # Decayed total over one half-life ~= volume/min for a steady tape.
    per_min = total / (half_life / 60.0)
    return smooth_ramp(per_min, 0.0, full_per_min)


def compute(trades: list, now: float) -> dict:
    """All flow features from a trade tape at explicit time ``now``."""
    return {
        "flow_cvd_decay": decayed_cvd(trades, now),
        "flow_whale": whale_delta(trades, now),
        "flow_rate": trade_rate(trades, now),
    }

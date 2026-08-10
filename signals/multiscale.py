"""Multi-timeframe BTC momentum + volatility features (pure, deterministic).

All functions take an explicit list of closed 1m closes (oldest first) and
return plain floats — no clocks, no network, no module state — so the same
inputs always produce the same outputs and the offline harness validates
exactly what ships.

Directional outputs (YES/Up-frame, tanh-bounded to (-1, 1)):
- ``ms_mom_{1,3,5,15}m``: per-horizon momentum, each judged against its own
  typical magnitude (scale grows with sqrt(horizon), same convention as
  signals/technicals.multi_timeframe_score).

Context outputs (non-directional — used for regime conditioning / sizing,
never as a side-picker):
- ``ms_rvol_{5,15,30}m``: realized vol (stdev of 1m log-returns) per window.
- ``ms_atr_5m``: mean absolute 1m return over the last 5 candles.
- ``ms_vol_ratio``: short-vs-long vol expansion score in (-1, 1) — positive
  when 5m vol is running hot vs 30m (breakout conditions), negative when
  compressing.

ALL DIRECTIONAL OUTPUTS ARE CANDIDATES: they carry no live lane weight until
the offline harness (tools/validate_signals.py --rank) measures positive NET
edge (house rule: validate-before-weighting).
"""

import math

from signals.curves import soft_saturate

MOM_HORIZONS = (1, 3, 5, 15)
MOM_BASE_SCALE = 0.0015     # 0.15% 1m move reads ~0.76; sqrt-scaled per horizon
VOL_WINDOWS = (5, 15, 30)
VOL_RATIO_SCALE = 0.7       # |ln(vol_5m / vol_30m)| of 0.7 (2x ratio) reads ~0.76


def _base_mom_scale() -> float:
    """Adaptive 1m soft-sat scale when live estimator is warm; else prior."""
    try:
        from signals.drift_scale import get_drift_scale_estimator
        return float(get_drift_scale_estimator().mom_saturate_scale())
    except Exception:
        return float(MOM_BASE_SCALE)


def momentum_score(prices: list, horizon: int) -> float:
    """Tanh-bounded return over ``horizon`` closed candles (0.0 if too short)."""
    if len(prices) <= horizon or prices[-1 - horizon] <= 0:
        return 0.0
    move = (prices[-1] - prices[-1 - horizon]) / prices[-1 - horizon]
    # Scale grows with √horizon; base tracks live 1m vol (2026-08-07).
    return soft_saturate(move, _base_mom_scale() * (horizon ** 0.5))


def realized_vol(prices: list, window: int) -> float:
    """Stdev of 1m log-returns over the last ``window`` candles (fraction/min)."""
    w = prices[-(window + 1):]
    if len(w) < 3:
        return 0.0
    rets = [math.log(w[i] / w[i - 1]) for i in range(1, len(w)) if w[i - 1] > 0]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return math.sqrt(var)


def atr_pct(prices: list, window: int = 5) -> float:
    """Mean absolute 1m return over the last ``window`` candles."""
    w = prices[-(window + 1):]
    if len(w) < 2:
        return 0.0
    moves = [abs(w[i] - w[i - 1]) / w[i - 1] for i in range(1, len(w))
             if w[i - 1] > 0]
    return (sum(moves) / len(moves)) if moves else 0.0


def vol_ratio_score(prices: list) -> float:
    """Vol expansion: +ve when 5m vol runs hot vs 30m, -ve when compressing."""
    short = realized_vol(prices, 5)
    long = realized_vol(prices, 30)
    if short <= 0 or long <= 0:
        return 0.0
    return soft_saturate(math.log(short / long), VOL_RATIO_SCALE)


def compute(prices: list) -> dict:
    """All multiscale features from closed 1m closes (oldest first)."""
    clean = [p for p in (prices or []) if p and p > 0]
    out = {}
    for h in MOM_HORIZONS:
        out[f"ms_mom_{h}m"] = momentum_score(clean, h)
    for w in VOL_WINDOWS:
        out[f"ms_rvol_{w}m"] = realized_vol(clean, w)
    out["ms_atr_5m"] = atr_pct(clean, 5)
    out["ms_vol_ratio"] = vol_ratio_score(clean)
    return out

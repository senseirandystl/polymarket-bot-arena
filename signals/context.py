"""Composes existing per-window signals into one structured *context* vector.

Pure function — no network reads (all inputs are already computed on the warm
path) and no module state, so it is safe on the 1s warm path, the offline
harness, and in tests. This is Layer 1 of the regime-discovery design
(docs/superpowers/specs/2026-07-24-regime-discovery-context-attribution-design.md):
the vector is stamped on every trade at decision time; Layer 2 attributes
per-bot performance to `context_cell(...)` groupings.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

_SESSIONS = ("asia", "eu", "us", "overnight")


def _session_for_hour_et(hour_et: int) -> str:
    # Coarse crypto session buckets in ET.
    if 3 <= hour_et < 9:
        return "eu"
    if 9 <= hour_et < 16:
        return "us"
    if 16 <= hour_et < 21:
        return "overnight"
    return "asia"


def _btc_trend_slope(prices: Sequence[float]) -> float:
    """Signed, bounded macro-trend slope from first vs last of a long window."""
    clean = [p for p in (prices or []) if p and p > 0]
    if len(clean) < 10:
        return 0.0
    span = clean[-1] - clean[0]
    base = clean[0] or 1.0
    return math.tanh((span / base) / 0.01)  # 1% move over the window ~ 0.76


def build_context(
    prices: Sequence[float],
    signals: Optional[dict] = None,
    now_utc: Optional[datetime] = None,
) -> dict[str, Any]:
    """Return the structured context vector for the current window."""
    from signals import multiscale, volatility_regime
    from arena.session_filter import _to_et
    from signals.macro_calendar import macro_caution

    now_utc = now_utc or datetime.now(tz=timezone.utc)
    clean = [p for p in (prices or []) if p and p > 0]

    vr = volatility_regime.compute(clean)
    ms = multiscale.compute(clean)

    sv = signals or {}
    flow = 0.0
    try:
        flow = 0.5 * (abs(float(sv.get("cvd", 0.0))) + abs(float(sv.get("obi", 0.0))))
    except (TypeError, ValueError):
        flow = 0.0

    et = _to_et(now_utc)
    hour_et = et.hour
    caution = macro_caution(now_utc)
    macro_prox = 2 if caution >= 0.75 else (1 if caution >= 0.25 else 0)

    return {
        # continuous
        "vol": max(0.0, min(1.0, float(vr.get("vol_score") or 0.0))),
        "trend": max(0.0, min(1.0, float(vr.get("trend_score") or 0.0))),
        "flow": max(0.0, min(1.0, flow)),
        "realized_vol": float(vr.get("realized_vol") or 0.0),
        "btc_mom_1m": float(ms.get("ms_mom_1m") or 0.0),
        "btc_mom_5m": float(ms.get("ms_mom_5m") or 0.0),
        "btc_mom_15m": float(ms.get("ms_mom_15m") or 0.0),
        "btc_trend_slope": _btc_trend_slope(clean),
        # categorical
        "weekday": int(et.weekday()),
        "hour_block": int(hour_et // 3),
        "session": _session_for_hour_et(hour_et),
        "macro_prox": int(macro_prox),
        # derived
        "vol_trend_regime": str(vr.get("regime") or "unknown"),
    }


def context_cell(ctx: dict) -> tuple:
    """Discretized grouping key for attribution (hashable)."""
    slope = float(ctx.get("btc_trend_slope") or 0.0)
    trend_bucket = 1 if slope > 0.2 else (-1 if slope < -0.2 else 0)
    return (
        str(ctx.get("vol_trend_regime") or "unknown"),
        int(ctx.get("weekday") or 0),
        int(ctx.get("hour_block") or 0),
        str(ctx.get("session") or "asia"),
        int(ctx.get("macro_prox") or 0),
        trend_bucket,
    )

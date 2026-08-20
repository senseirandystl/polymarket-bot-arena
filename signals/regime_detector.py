"""Robust market-regime detection with continuous online updates.

Combines volatility, trend efficiency, short-horizon momentum, and
order-flow intensity into a small set of actionable regimes:

  * ``high_vol_trend`` — violent, directional tape
  * ``low_vol_range``  — quiet, mean-reverting / rangebound
  * ``high_vol_chop``  — violent but non-directional (whipsaw)
  * ``low_vol_trend``  — quiet grind / efficient drift
  * ``normal``         — middle band (no strong classification)
  * ``unknown``        — insufficient data

Classification is primarily **rule-based** on EMA-smoothed features, with
optional **lightweight online centroids** (running mean feature vectors per
regime) for soft confidence. A hysteresis layer stops one-tick flapping.

This module is *context*, not a directional signal. Consumers:

- Signal Lab (regime-specific lane damps / weight tilts)
- Every bot via ``BaseBot.regime_context`` / ``signals["market_regime"]``
- Hybrid meta-learner (bucket mapping for online multipliers)
- Evolution fitness (regime-conditioned multi-objective scoring)
- Dashboard / arena_state (transitions + per-regime performance)

Updates run on every ``build_combined_signals`` call (1s warm path) — **not**
only after market resolution. Performance impact is tracked both online
(when trades resolve) and at entry time (regime stamped into trade features).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from typing import Any, Optional, Sequence

import config

logger = logging.getLogger(__name__)

# Canonical regime ids (stable API for Lab / bots / GA / dashboard).
REGIME_IDS = (
    "high_vol_trend",
    "low_vol_range",
    "high_vol_chop",
    "low_vol_trend",
    "normal",
    "unknown",
)

# Map rich regime → legacy quiet/normal/trending/volatile labels used by
# older code paths (sniper quiet bump, older logs).
LEGACY_MAP = {
    "high_vol_trend": "trending",
    "low_vol_range": "quiet",
    "high_vol_chop": "volatile",
    "low_vol_trend": "trending",
    "normal": "normal",
    "unknown": "unknown",
}

# Map rich regime → hybrid meta-learner buckets.
META_BUCKET_MAP = {
    "high_vol_trend": "trending",
    "low_vol_trend": "trending",
    "low_vol_range": "ranging",
    "high_vol_chop": "chop",
    "normal": "mixed",
    "unknown": "mixed",
}

STATE_KEY = "regime_detector"
PERF_KEY = "regime_performance"
HISTORY_KEY = "regime_transitions"

# Feature vector keys used by the rule classifier + EMA centroids.
# ``vol`` is VOLATILITY (realized log-return stdev score), not volume.
# ``volume`` is a separate activity feature (dashboard / context only) —
# filled from Binance BTC 1m kline volume (price stays Chainlink). Classifier
# rules do not use volume as a grid axis (see compute_features).
FEATURE_KEYS = ("vol", "trend", "mom", "flow")
# Live-path EMA also holds the directionality composite and its inputs so
# classify_rules sees the same axes compute_features produced (chop +
# multi-scale align were previously dropped before scoring).
SMOOTH_KEYS = FEATURE_KEYS + ("volume", "chop", "direction", "ms_mom_align", "vol_rel")
PASS_KEYS = (
    "flow_align", "realized_vol", "trend_sign", "vol_abs", "calibration",
    "twap_blend", "sample_ok", "xasset_align",
    "pm_spread_score", "pm_book_sum", "pm_mid", "pm_lag", "pm_book_quality",
)

# Human labels for dashboard / ops (vol ≠ volume).
FEATURE_LABELS = {
    "vol": "volatility",
    "trend": "trend",
    "mom": "momentum",
    "flow": "flow",
    "volume": "volume",
    "flow_align": "flow_align",
    "realized_vol": "realized_vol",
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _feat(features: dict, key: str, default: float) -> float:
    """Read a float feature. ``0.0`` is a valid extreme — never use ``or``."""
    if not features or key not in features or features[key] is None:
        return float(default)
    try:
        return float(features[key])
    except (TypeError, ValueError):
        return float(default)


def _volume_score(volumes: Optional[Sequence[float]]) -> float:
    """Relative tape activity in ~[0, 1].

    Compares recent mean volume to the longer baseline. 0.5 ≈ typical;
    near 0 = dead tape; near 1 = expansion. Empty/missing series → 0
    (unknown — e.g. Chainlink BTC has no volume).
    """
    if not volumes:
        return 0.0
    clean = [float(v) for v in volumes if v is not None and float(v) >= 0]
    if len(clean) < 5:
        return 0.0
    recent_n = min(5, len(clean))
    recent = sum(clean[-recent_n:]) / recent_n
    baseline = sum(clean) / len(clean)
    if baseline <= 1e-12:
        return 0.0 if recent <= 1e-12 else 1.0
    ratio = recent / baseline
    # Soft map: half baseline → ~0.2, equal → ~0.5, 2× → ~0.85
    return _clip01(1.0 / (1.0 + math.exp(-2.2 * (ratio - 1.0))))


def _pm_sidecar(pm_state: Optional[dict]) -> dict[str, float]:
    """Polymarket book-quality sidecar — never a BTC-grid axis.

    Spread tightness + YES+NO book-sum consistency. Used to *damp
    confidence* when the tradeable book is gapped/wide, not to relabel
    the underlying BTC tape.
    """
    if not pm_state:
        return {}
    try:
        spread_score = _clip01(float(pm_state.get("spread_score", 0.5) or 0.5))
    except (TypeError, ValueError):
        spread_score = 0.5
    try:
        yes = float(pm_state.get("yes_price") or pm_state.get("mid") or 0.0)
    except (TypeError, ValueError):
        yes = 0.0
    try:
        no = float(pm_state.get("no_price") or 0.0)
    except (TypeError, ValueError):
        no = 0.0
    try:
        book_sum = float(pm_state.get("book_sum") or 0.0)
    except (TypeError, ValueError):
        book_sum = 0.0
    if book_sum <= 0 and yes > 0 and no > 0:
        book_sum = yes + no
    if book_sum > 0:
        consistency = _clip01(1.0 - abs(book_sum - 1.0) / 0.08)
    else:
        consistency = 0.5
    out: dict[str, float] = {
        "pm_spread_score": spread_score,
        "pm_book_sum": book_sum if book_sum > 0 else 0.0,
        "pm_book_quality": _clip01(0.6 * spread_score + 0.4 * consistency),
    }
    if yes > 0:
        out["pm_mid"] = yes
    lag = pm_state.get("lag_residual")
    if lag is not None:
        try:
            out["pm_lag"] = float(max(-1.0, min(1.0, float(lag))))
        except (TypeError, ValueError):
            pass
    return out


def _apply_xasset(feats: dict[str, float], xasset_score: float) -> dict[str, float]:
    """Sidecar ETH/SOL alignment. Never writes the BTC direction axis."""
    out = dict(feats)
    try:
        xa = float(xasset_score)
    except (TypeError, ValueError):
        return out
    ts = _feat(out, "trend_sign", 0.0)
    if abs(xa) > 0.2 and abs(ts) > 0.15:
        out["xasset_align"] = 1.0 if (xa > 0) == (ts > 0) else 0.0
    else:
        out["xasset_align"] = 0.5
    return out


def compute_features(
    prices: Sequence[float],
    *,
    cvd: float = 0.0,
    obi: float = 0.0,
    vol_score: Optional[float] = None,
    trend_score: Optional[float] = None,
    realized_vol: Optional[float] = None,
    volumes: Optional[Sequence[float]] = None,
    volume_score: Optional[float] = None,
    calibrate: bool = True,
) -> dict[str, float]:
    """Derive continuous feature vector in ~[0, 1] from market inputs.

    Pure function — no module state. Safe for offline harness / tests.

    Keys:
      * ``vol`` — **volatility** score (not volume)
      * ``volume`` — relative tape activity (separate; may be 0 for Chainlink)
      * ``trend``, ``mom``, ``flow``, ``flow_align``, ``realized_vol``
    """
    clean = [p for p in (prices or []) if p and p > 0]

    # --- Volatility / trend from provided scores or local recompute ---
    if vol_score is None or trend_score is None:
        from signals import volatility_regime
        vr = volatility_regime.compute(clean)
        vol_score = float(vr.get("vol_score") or 0.0)
        trend_score = float(vr.get("trend_score") or 0.0)
        realized_vol = float(vr.get("realized_vol") or 0.0)
    else:
        vol_score = float(vol_score)
        trend_score = float(trend_score)
        realized_vol = float(realized_vol or 0.0)

    # --- Momentum intensity: |1-candle return| soft-scaled at 0.2% ---
    # Also keep signed trend_sign in YES-frame (BTC up → positive) for
    # side attribution: which way the tape is grinding, not just how hard.
    mom = 0.0
    trend_sign = 0.0
    if len(clean) >= 2 and clean[-2] > 0:
        ret = (clean[-1] - clean[-2]) / clean[-2]
        # soft saturate: 0.002 (~p97) → ~0.76; map to 0..1 via tanh
        signed = math.tanh(ret / 0.002)
        mom = abs(signed)
        trend_sign = float(signed)

    # --- Order-flow intensity: mean absolute CVD/OBI (already ~[-1,1]) ---
    flow = _clip01(0.5 * (abs(float(cvd)) + abs(float(obi))))

    # --- Flow/momentum alignment (context only; not in classifier core) ---
    flow_sign = 0.0
    if abs(float(cvd)) + abs(float(obi)) > 1e-9:
        flow_sign = 1.0 if (float(cvd) + float(obi)) > 0 else -1.0
    mom_sign = 0.0
    if len(clean) >= 2 and clean[-2] > 0:
        d = clean[-1] - clean[-2]
        if abs(d) > 1e-12:
            mom_sign = 1.0 if d > 0 else -1.0
    flow_align = 0.5  # neutral
    if flow_sign != 0.0 and mom_sign != 0.0:
        flow_align = 1.0 if flow_sign == mom_sign else 0.0

    if volume_score is None:
        volume_score = _volume_score(volumes)
    else:
        volume_score = _clip01(float(volume_score))

    # Multi-horizon path structure (context + directionality axis)
    chop = 0.5
    ms_mom_align = 0.5
    try:
        from signals import regime as regime_mod
        from signals import multiscale
        reg_feats = regime_mod.compute(clean)
        chop = _feat(reg_feats, "regime_chop", 0.5)
        # Prefer multi-window trend mean when available
        if reg_feats.get("regime_trend") is not None:
            trend_score = float(reg_feats["regime_trend"])
        ms = multiscale.compute(clean)
        m1 = float(ms.get("ms_mom_1m") or 0.0)
        m5 = float(ms.get("ms_mom_5m") or 0.0)
        if abs(m1) > 0.05 and abs(m5) > 0.05:
            ms_mom_align = 1.0 if (m1 > 0) == (m5 > 0) else 0.0
        elif abs(m1) <= 0.05 and abs(m5) <= 0.05:
            ms_mom_align = 0.5
        else:
            ms_mom_align = 0.5
        # Blend short + medium signed mom for stabler trend-side stamp
        if abs(m5) > 1e-9 or abs(m1) > 1e-9:
            blend = 0.4 * m1 + 0.6 * m5
            # multiscale moms are already roughly in [-1,1]
            trend_sign = float(max(-1.0, min(1.0, blend)))
    except Exception:
        pass

    feats = {
        "vol": _clip01(vol_score),  # volatility (abs score; may be replaced by rel)
        "vol_abs": _clip01(vol_score),
        "trend": _clip01(trend_score),
        "mom": _clip01(mom),
        "flow": _clip01(flow),
        "volume": volume_score,  # activity — NOT the same as vol
        "flow_align": _clip01(flow_align),
        "realized_vol": float(realized_vol),
        "chop": _clip01(chop),
        "ms_mom_align": _clip01(ms_mom_align),
        # Signed YES-frame direction of the tape ∈ [-1, 1]
        "trend_sign": float(max(-1.0, min(1.0, trend_sign))),
        # 0 → classify_rules returns unknown (cold start ≠ quiet range).
        # Gate on series length only: the live path always passes
        # vol_score=0.0 / trend_score=0.0 (not None) for a cold feed, which
        # is not "enough data".
        "sample_ok": 1.0 if len(clean) >= 5 else 0.0,
    }
    feats["direction"] = directionality(feats)

    # Relative calibration (percentile of raw realized_vol / trend / chop)
    try:
        from arena.regime_settings import get_bool as _reg_bool
        _use_rel = bool(_reg_bool("use_relative"))
    except Exception:
        _use_rel = bool(getattr(config, "REGIME_USE_RELATIVE", True))
    if _use_rel:
        try:
            from signals.regime_calibration import get_calibrator
            cal = get_calibrator()
            if calibrate:
                # Fingerprint the *candle*, not the feature values — consecutive
                # quiet minutes can share the same rounded vol/chop/mom and
                # must still enter the reservoir.
                cal.update_if_changed(
                    (
                        len(clean),
                        round(float(clean[-1]), 2) if clean else 0.0,
                    ),
                    realized_vol=float(realized_vol),
                    trend_eff=float(trend_score),
                    chop=float(chop),
                    mom_abs=float(mom),
                )
            vol_rel = cal.percentile(
                "realized_vol", float(realized_vol),
                fallback=_clip01(vol_score),
            )
            trend_rel = cal.percentile(
                "trend_eff", float(trend_score),
                fallback=_clip01(trend_score),
            )
            chop_rel = cal.percentile(
                "chop", float(chop), fallback=_clip01(chop),
            )
            feats["vol_rel"] = vol_rel
            feats["trend_rel"] = trend_rel
            feats["chop_rel"] = chop_rel
            # Classifier uses relative vol as primary "vol" axis when relative on
            feats["vol"] = vol_rel
            # Direction uses relative trend in the composite when available
            feats["trend"] = 0.5 * float(trend_score) + 0.5 * trend_rel
            feats["direction"] = directionality(feats)
            feats["calibration"] = 1.0 if cal.ready("realized_vol") else 0.0
        except Exception:
            feats["vol_rel"] = feats["vol"]
            feats["calibration"] = 0.0
    else:
        feats["vol_rel"] = feats["vol"]
        feats["calibration"] = 0.0

    return feats


def directionality(features: dict[str, float]) -> float:
    """Composite directionality in [0, 1]: trend + anti-chop + multi-scale align.

    Used as the second axis of the regime grid (alongside vol). High =
    efficient directional tape; low = range / churn.
    """
    t = _feat(features, "trend", 0.5)
    chop = _feat(features, "chop", 0.5)
    align = _feat(features, "ms_mom_align", 0.5)
    return _clip01(0.45 * t + 0.35 * (1.0 - chop) + 0.20 * align)


def classify_rules(features: dict[str, float],
                   *,
                   vol_hi: float | None = None,
                   vol_lo: float | None = None,
                   trend_hi: float | None = None,
                   trend_lo: float | None = None) -> tuple[str, float]:
    """Rule-based regime id + confidence from a feature dict.

    When relative calibration is on, ``vol`` should already be a relative
    score and thresholds default to REGIME_CLASSIFY_VOL_* / DIR_*.
    Falls back to classic absolute thresholds when relative mode is off.

    Returns (regime_id, confidence in 0..1).
    """
    try:
        _sample_ok = float(features["sample_ok"]) if "sample_ok" in features else 1.0
    except (TypeError, ValueError):
        _sample_ok = 1.0
    if _sample_ok < 0.5:
        return "unknown", 0.0
    try:
        from arena.regime_settings import get_bool as _reg_bool
        use_rel = bool(_reg_bool("use_relative"))
    except Exception:
        use_rel = bool(getattr(config, "REGIME_USE_RELATIVE", True))
    if use_rel:
        vol_hi = float(vol_hi if vol_hi is not None
                       else getattr(config, "REGIME_CLASSIFY_VOL_HI", 0.70))
        vol_lo = float(vol_lo if vol_lo is not None
                       else getattr(config, "REGIME_CLASSIFY_VOL_LO", 0.30))
        trend_hi = float(trend_hi if trend_hi is not None
                         else getattr(config, "REGIME_CLASSIFY_DIR_HI", 0.55))
        trend_lo = float(trend_lo if trend_lo is not None
                         else getattr(config, "REGIME_CLASSIFY_DIR_LO", 0.40))
        # Prefer relative vol when present
        v = float(features.get("vol_rel", features.get("vol", 0.0)) or 0.0)
        t = float(features.get("direction", directionality(features)) or 0.0)
    else:
        vol_hi = float(vol_hi if vol_hi is not None else 0.55)
        vol_lo = float(vol_lo if vol_lo is not None else 0.35)
        trend_hi = float(trend_hi if trend_hi is not None else 0.50)
        trend_lo = float(trend_lo if trend_lo is not None else 0.35)
        v = float(features.get("vol", 0.0))
        t = float(features.get("trend", 0.0))

    m = float(features.get("mom", 0.0))
    f = float(features.get("flow", 0.0))

    high_vol = v >= vol_hi
    low_vol = v <= vol_lo
    high_trend = t >= trend_hi
    low_trend = t <= trend_lo

    if high_vol and high_trend:
        rid = "high_vol_trend"
        conf = 0.55 + 0.25 * _clip01((v - vol_hi) / max(1e-6, 1 - vol_hi)) \
            + 0.20 * _clip01((t - trend_hi) / max(1e-6, 1 - trend_hi))
    elif high_vol and low_trend:
        rid = "high_vol_chop"
        conf = 0.55 + 0.25 * _clip01((v - vol_hi) / max(1e-6, 1 - vol_hi)) \
            + 0.20 * _clip01((trend_lo - t) / max(1e-6, trend_lo))
    elif low_vol and high_trend:
        rid = "low_vol_trend"
        conf = 0.50 + 0.25 * _clip01((vol_lo - v) / max(1e-6, vol_lo)) \
            + 0.25 * _clip01((t - trend_hi) / max(1e-6, 1 - trend_hi))
    elif low_vol and low_trend:
        rid = "low_vol_range"
        conf = 0.55 + 0.25 * _clip01((vol_lo - v) / max(1e-6, vol_lo)) \
            + 0.20 * _clip01((trend_lo - t) / max(1e-6, trend_lo))
    else:
        rid = "normal"
        # Closer to a corner → lower confidence in "normal"
        edge = max(abs(v - 0.5), abs(t - 0.5))
        conf = 0.45 + 0.2 * (1.0 - edge)

    # Mild boost when momentum/flow/volume agree with the classification
    vol_act = float(features.get("volume", 0.0) or 0.0)
    if rid in ("high_vol_trend", "low_vol_trend") and m > 0.4:
        conf = min(1.0, conf + 0.05)
    if rid == "high_vol_chop" and f > 0.4 and m < 0.3:
        conf = min(1.0, conf + 0.05)
    if rid == "low_vol_range" and m < 0.25:
        conf = min(1.0, conf + 0.05)
    if rid in ("high_vol_trend", "high_vol_chop") and vol_act > 0.6:
        conf = min(1.0, conf + 0.04)
    if rid == "low_vol_range" and 0.0 < vol_act < 0.35:
        conf = min(1.0, conf + 0.04)

    return rid, _clip01(conf)


def legacy_label(regime_id: str) -> str:
    return LEGACY_MAP.get(regime_id, "unknown")


def meta_bucket(regime_id: str, trend_score: Optional[float] = None) -> str:
    """Hybrid meta-learner bucket for a regime id (fallback to trend_score)."""
    if regime_id in META_BUCKET_MAP and regime_id != "unknown":
        return META_BUCKET_MAP[regime_id]
    if trend_score is None:
        return "mixed"
    if trend_score >= 0.65:
        return "trending"
    if trend_score <= 0.35:
        return "ranging"
    return "mixed"


class RegimeDetector:
    """Stateful online regime detector (EMA features + hysteresis + perf).

    Thread-safe. One process-wide instance via :func:`get_detector`.
    """

    def __init__(
        self,
        *,
        ema_alpha: float | None = None,
        hold_ticks: int | None = None,
        switch_margin: float | None = None,
        use_centroids: bool | None = None,
    ):
        self.ema_alpha = float(
            ema_alpha if ema_alpha is not None
            else getattr(config, "REGIME_EMA_ALPHA", 0.25)
        )
        self.hold_ticks = int(
            hold_ticks if hold_ticks is not None
            else getattr(config, "REGIME_HOLD_TICKS", 20)
        )
        self.switch_margin = float(
            switch_margin if switch_margin is not None
            else getattr(config, "REGIME_SWITCH_MARGIN", 0.12)
        )
        self.use_centroids = bool(
            use_centroids if use_centroids is not None
            else getattr(config, "REGIME_USE_CENTROIDS", True)
        )
        self._lock = threading.Lock()
        self._ema: dict[str, float] = {}
        self._regime = "unknown"
        self._confidence = 0.0
        self._candidate: Optional[str] = None
        self._candidate_ticks = 0
        self._ticks = 0
        self._last_features: dict[str, float] = {}
        self._last_change_ts: Optional[float] = None
        self._last_change_from: Optional[str] = None
        self._transitions: list[dict] = []  # recent, capped
        # Online centroids: regime -> {n, mean: [vol,trend,mom,flow]}
        self._centroids: dict[str, dict] = {
            rid: {"n": 0, "mean": [0.5, 0.5, 0.3, 0.2]}
            for rid in REGIME_IDS if rid != "unknown"
        }
        # Per-regime performance (online, resolution-driven)
        self._perf: dict[str, dict] = {
            rid: {"n": 0, "wins": 0, "pnl": 0.0, "sum_pnl_sq": 0.0}
            for rid in REGIME_IDS
        }
        self._loaded = False
        self._last_persist = 0.0
        # True once this process drives the detector (update/record_outcome).
        # Read-only consumers (the dashboard process) leave it False and so
        # refresh from the DB on every read instead of trusting stale memory.
        self._live = False
        # Soft market-id stamp for rollover notes (no state reset — regime is
        # continuous BTC tape context, not per-window moneyness).
        self._last_market_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load_from_db()

    def _load_from_db(self) -> None:
        """Hydrate in-memory state from arena_state.

        Runs once at startup for the arena's live detector, and on every
        read for detached read-only consumers (the dashboard process, which
        never calls :meth:`update`). The detector is a per-process singleton;
        without this refresh a reader would freeze at whatever state existed
        the first time it was touched — e.g. the ``unknown`` default if the
        reader started before the arena committed its first regime.
        """
        try:
            import db
            raw = db.get_arena_state(STATE_KEY)
            if raw:
                data = json.loads(raw)
                self._regime = data.get("regime", self._regime)
                self._confidence = float(data.get("confidence") or 0.0)
                self._ema = dict(data.get("ema") or {})
                # Restore the smoothed feature vector so `known`/features are
                # populated for readers (drives the dashboard features table).
                self._last_features = dict(
                    data.get("last_features") or self._last_features
                )
                self._last_change_from = data.get(
                    "last_change_from", self._last_change_from
                )
                ts = data.get("last_change_ts")
                if ts is not None:
                    try:
                        self._last_change_ts = float(ts)
                    except (TypeError, ValueError):
                        ts = None
                # Old snapshots had no last_change_ts. A restored committed
                # label must already count as held or every gate stays off
                # until the next flip (which may be hours).
                if ts is None and self._regime not in ("unknown", "", None):
                    try:
                        updated = float(data.get("updated_at") or 0.0)
                    except (TypeError, ValueError):
                        updated = 0.0
                    hold = float(getattr(config, "REGIME_ACTION_MIN_HOLD_SEC", 20.0))
                    self._last_change_ts = (
                        updated if updated > 0 else time.time() - hold
                    )
                cents = data.get("centroids") or {}
                for rid, c in cents.items():
                    if rid in self._centroids and isinstance(c, dict):
                        self._centroids[rid] = {
                            "n": int(c.get("n") or 0),
                            "mean": list(c.get("mean") or self._centroids[rid]["mean"]),
                        }
            perf_raw = db.get_arena_state(PERF_KEY)
            if perf_raw:
                perf = json.loads(perf_raw)
                for rid, p in (perf or {}).items():
                    if rid in self._perf and isinstance(p, dict):
                        self._perf[rid].update({
                            "n": int(p.get("n") or 0),
                            "wins": int(p.get("wins") or 0),
                            "pnl": float(p.get("pnl") or 0.0),
                            "sum_pnl_sq": float(p.get("sum_pnl_sq") or 0.0),
                        })
        except Exception as e:
            logger.debug("regime_detector load failed: %s", e)

    def _persist(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_persist) < 5.0:
            return
        self._last_persist = now
        try:
            import db
            snap = {
                "regime": self._regime,
                "confidence": self._confidence,
                "ema": dict(self._ema),
                "last_features": dict(self._last_features),
                "last_change_from": self._last_change_from,
                "last_change_ts": self._last_change_ts,
                "centroids": {
                    rid: {"n": c["n"], "mean": list(c["mean"])}
                    for rid, c in self._centroids.items()
                },
                "updated_at": now,
                "ticks": self._ticks,
            }
            db.set_arena_state(STATE_KEY, json.dumps(snap))
            db.set_arena_state(PERF_KEY, json.dumps(self._perf))
        except Exception as e:
            logger.debug("regime_detector persist failed: %s", e)

    # ------------------------------------------------------------------
    # Market identity (soft note only — no EMA/centroid reset)
    # ------------------------------------------------------------------

    def note_market(self, market_id: Optional[str]) -> None:
        """Soft-annotate a Polymarket window rollover.

        Logs once when ``market_id`` changes so soaks can align the regime
        timeline to market boundaries. Does **not** reset EMA, hysteresis,
        centroids, or confidence — regime describes continuous BTC tape, not
        a single 5-min window's moneyness.
        """
        if not market_id:
            return
        with self._lock:
            prev = self._last_market_id
            if prev is None:
                self._last_market_id = str(market_id)
                return
            if str(market_id) == prev:
                return
            self._last_market_id = str(market_id)
            rid = self._regime
            conf = float(self._confidence)
            ticks = self._ticks
        logger.info(
            "REGIME MARKET ROLLOVER %s -> %s "
            "(soft note; state retained: regime=%s conf=%.2f ticks=%d)",
            prev, market_id, rid, conf, ticks,
        )

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(
        self,
        prices: Sequence[float],
        *,
        cvd: float = 0.0,
        obi: float = 0.0,
        vol_score: Optional[float] = None,
        trend_score: Optional[float] = None,
        realized_vol: Optional[float] = None,
        volumes: Optional[Sequence[float]] = None,
        volume_score: Optional[float] = None,
        market_id: Optional[str] = None,
        twap_prices: Optional[Sequence[float]] = None,
        pm_state: Optional[dict] = None,
        xasset_score: Optional[float] = None,
    ) -> dict[str, Any]:
        """Ingest one market tick; return current regime snapshot.

        Continuous / online — safe to call every second from the warm path.
        When ``market_id`` is provided, a soft rollover note is emitted if
        the live window changed (no detector state is cleared).

        ``twap_prices`` (optional): recent resolution-object TWAP series. When
        present, trend/mom features blend spot microstructure with TWAP path
        so "trend" is more resolution-relevant (2026-08-11).

        ``pm_state`` (optional): Polymarket book sidecar (spread / YES+NO
        sum / mid). Damps confidence on gapped books; never changes the
        BTC vol×direction label.

        ``xasset_score`` (optional): ETH/SOL confirmation in ~[-1, 1].
        Sidecar alignment only (confidence ±0.03) — not a classification axis.
        """
        self._ensure_loaded()
        self._live = True
        if market_id is not None:
            self.note_market(market_id)
        raw = compute_features(
            prices, cvd=cvd, obi=obi,
            vol_score=vol_score, trend_score=trend_score,
            realized_vol=realized_vol,
            volumes=volumes, volume_score=volume_score,
        )
        # Blend resolution-relevant TWAP trend/mom (spot still owns vol/flow).
        # Only blend a TWAP series that itself cleared the sample floor —
        # a 3-print cold TWAP has trend=0 and would drag spot toward range.
        if twap_prices is not None and len([p for p in twap_prices if p and p > 0]) >= 5:
            try:
                tw = compute_features(twap_prices, cvd=0.0, obi=0.0, calibrate=False)
                if _feat(tw, "sample_ok", 0.0) >= 0.5:
                    blend = float(getattr(config, "REGIME_TWAP_BLEND", 0.45) or 0.45)
                    blend = max(0.0, min(0.8, blend))
                    for k in ("trend", "mom"):
                        if k in raw and k in tw:
                            raw[k] = (1.0 - blend) * float(raw[k]) + blend * float(tw[k])
                    raw["twap_blend"] = blend
                    raw["direction"] = directionality(raw)
            except Exception:
                pass
        raw.update(_pm_sidecar(pm_state))
        # Thin tape: do not EMA zeros over a restored state or commit
        # low_vol_range from a 2–4 candle restart.
        if _feat(raw, "sample_ok", 1.0) < 0.5:
            with self._lock:
                if self._regime == "unknown":
                    self._last_features = dict(raw)
                    self._confidence = 0.0
                    return self._snapshot_unlocked()
                # Keep the last committed regime; always stamp sample_ok=0
                # so snapshot() does not default to 1.0 on an empty restore.
                prev = dict(self._last_features) if self._last_features else {}
                prev["sample_ok"] = 0.0
                self._last_features = prev
                return self._snapshot_unlocked()
        with self._lock:
            # EMA the classifier axes + directionality inputs
            for k in SMOOTH_KEYS:
                if k not in raw:
                    continue
                prev = self._ema.get(k)
                cur = float(raw[k])
                if prev is None:
                    self._ema[k] = cur
                else:
                    a = self.ema_alpha
                    self._ema[k] = a * cur + (1.0 - a) * prev
            smoothed = {k: self._ema.get(k, raw[k]) for k in SMOOTH_KEYS if k in raw or k in self._ema}
            for k in PASS_KEYS:
                if k in raw:
                    smoothed[k] = raw[k]
            # Recompute direction from *smoothed* chop/trend/align so the
            # classifier sees a coherent composite, not a mix of raw+EMA.
            if "chop" in smoothed or "ms_mom_align" in smoothed:
                smoothed["direction"] = directionality(smoothed)
            if xasset_score is not None:
                smoothed = _apply_xasset(smoothed, xasset_score)
            self._last_features = dict(smoothed)

            rule_id, rule_conf = classify_rules(smoothed)
            # Optional centroid soft vote
            final_id, conf = rule_id, rule_conf
            if self.use_centroids:
                cent_id, cent_conf = self._nearest_centroid(smoothed)
                if cent_id and cent_id == rule_id:
                    conf = _clip01(0.6 * rule_conf + 0.4 * cent_conf)
                elif cent_id and cent_conf > rule_conf + 0.15:
                    # Strong centroid disagreement → slight pull, keep rules primary
                    conf = _clip01(rule_conf - 0.05)
                # else keep rule winner
            # Book quality + xasset are SIDECARS: they scale confidence
            # after the vote so they cannot flip the BTC vol×direction label
            # and cannot be overwritten by the centroid mix.
            pq = smoothed.get("pm_book_quality")
            if pq is not None and float(pq) < 0.55:
                conf = _clip01(conf * (0.70 + 0.30 * (float(pq) / 0.55)))
            xa = smoothed.get("xasset_align")
            if xa is not None:
                if float(xa) >= 0.99:
                    conf = _clip01(conf + 0.03)
                elif float(xa) <= 0.01:
                    conf = _clip01(conf - 0.03)

            self._ticks += 1
            changed = self._apply_hysteresis(final_id, conf)

            # Update centroid of the *committed* regime (online clustering)
            if self._regime not in ("unknown",) and self.use_centroids:
                self._update_centroid(self._regime, smoothed)

            snap = self._snapshot_unlocked()
            if changed:
                self._log_transition_unlocked(snap)
            # Throttled persist
            if changed or (self._ticks % 30 == 0):
                self._persist(force=changed)
            return snap

    def _nearest_centroid(self, features: dict) -> tuple[Optional[str], float]:
        vec = [float(features.get(k, 0.0)) for k in FEATURE_KEYS]
        best_id, best_dist = None, float("inf")
        for rid, c in self._centroids.items():
            if c["n"] < 5:
                continue  # cold centroid
            mean = c["mean"]
            dist = math.sqrt(sum((vec[i] - mean[i]) ** 2 for i in range(4)))
            if dist < best_dist:
                best_dist, best_id = dist, rid
        if best_id is None:
            return None, 0.0
        # Distance 0 → conf 1; distance ~1.0 → conf ~0.2
        conf = _clip01(1.0 - best_dist)
        return best_id, conf

    def _update_centroid(self, rid: str, features: dict) -> None:
        if rid not in self._centroids:
            return
        c = self._centroids[rid]
        n = int(c["n"])
        mean = list(c["mean"])
        vec = [float(features.get(k, 0.0)) for k in FEATURE_KEYS]
        n2 = n + 1
        for i in range(4):
            mean[i] = mean[i] + (vec[i] - mean[i]) / n2
        c["n"] = n2
        c["mean"] = mean

    def _apply_hysteresis(self, candidate: str, conf: float) -> bool:
        """Return True if the committed regime changed."""
        if self._regime == "unknown" and candidate != "unknown":
            self._regime = candidate
            self._confidence = conf
            self._candidate = None
            self._candidate_ticks = 0
            self._last_change_ts = time.time()
            self._last_change_from = "unknown"
            return True

        if candidate == self._regime:
            self._confidence = 0.8 * self._confidence + 0.2 * conf
            self._candidate = None
            self._candidate_ticks = 0
            return False

        # Need margin over current confidence to even start counting
        if conf + 1e-9 < self._confidence - self.switch_margin:
            self._candidate = None
            self._candidate_ticks = 0
            return False

        if candidate == self._candidate:
            self._candidate_ticks += 1
        else:
            self._candidate = candidate
            self._candidate_ticks = 1

        if self._candidate_ticks >= max(1, self.hold_ticks):
            prev = self._regime
            self._last_change_from = prev
            self._regime = candidate
            self._confidence = conf
            self._candidate = None
            self._candidate_ticks = 0
            self._last_change_ts = time.time()
            return True

        # Absolute-confidence escape hatch: when a regime candidate is very
        # confident (>0.75) and the current regime is weak (<0.45), avoid the
        # slow-drift deadlock where gradually rising confidence never clears
        # switch_margin (EMA-smoothed features drift too slowly for the margin
        # check to fire on consecutive ticks).
        if (
            candidate != self._regime
            and conf > 0.75
            and self._confidence < 0.45
            and self._candidate_ticks >= max(1, self.hold_ticks // 2)
        ):
            prev = self._regime
            old_conf = float(self._confidence)
            self._last_change_from = prev
            self._regime = candidate
            # Blend in a fraction of the old confidence so a single fleeting
            # spike cannot fully determine the new confidence.
            blended = 0.5 * old_conf + 0.5 * conf
            self._confidence = blended
            self._candidate = None
            self._candidate_ticks = 0
            self._last_change_ts = time.time()
            logger.info(
                "REGIME FAST TRANSITION %s -> %s conf=%.2f "
                "(absolute-confidence escape: old_conf=%.2f margin=%.2f "
                "blended=%.2f)",
                prev, candidate, conf,
                old_conf, self.switch_margin,
                blended,
            )
            return True

        return False

    def _log_transition_unlocked(self, snap: dict) -> None:
        rec = {
            "ts": time.time(),
            "from": self._last_change_from,
            "to": snap["regime_id"],
            "confidence": snap["confidence"],
            "features": dict(snap.get("features") or {}),
            "perf_at_change": {
                rid: {"n": p["n"], "pnl": p["pnl"]}
                for rid, p in self._perf.items() if p["n"] > 0
            },
        }
        self._transitions.append(rec)
        self._transitions = self._transitions[-40:]
        logger.info(
            "REGIME CHANGE %s -> %s conf=%.2f vol=%.2f trend=%.2f mom=%.2f flow=%.2f",
            rec["from"], rec["to"], rec["confidence"],
            snap["features"].get("vol", 0), snap["features"].get("trend", 0),
            snap["features"].get("mom", 0), snap["features"].get("flow", 0),
        )
        try:
            import db
            hist_raw = db.get_arena_state(HISTORY_KEY)
            hist = json.loads(hist_raw) if hist_raw else []
            if not isinstance(hist, list):
                hist = []
            hist.append(rec)
            db.set_arena_state(HISTORY_KEY, json.dumps(hist[-50:]))
            # Structured event table (best-effort)
            try:
                db.log_regime_event(
                    rec["from"], rec["to"], rec["confidence"],
                    rec["features"], rec.get("perf_at_change"),
                )
            except Exception:
                pass
        except Exception as e:
            logger.debug("regime transition persist failed: %s", e)
        # Production alerts (debounced; never raise into detector path)
        try:
            from arena.alerts import alert_regime_shift
            alert_regime_shift(
                rec.get("from") or "?",
                rec.get("to") or "?",
                float(rec.get("confidence") or 0.0),
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Performance impact (resolution-time online update)
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        regime_id: str,
        pnl: float,
        *,
        won: bool | None = None,
    ) -> None:
        """Update per-regime P&L stats when a trade resolves (online)."""
        self._ensure_loaded()
        self._live = True
        rid = regime_id if regime_id in self._perf else "unknown"
        with self._lock:
            p = self._perf[rid]
            p["n"] += 1
            p["pnl"] += float(pnl)
            p["sum_pnl_sq"] += float(pnl) ** 2
            if won is None:
                won = float(pnl) > 0
            if won:
                p["wins"] += 1
            if p["n"] % 5 == 0:
                self._persist(force=True)

    def performance_snapshot(self) -> dict[str, dict]:
        self._ensure_loaded()
        with self._lock:
            if not self._live:
                # Read-only consumer (dashboard): reflect the arena's latest
                # persisted perf, not this process's stale first load.
                self._load_from_db()
            out = {}
            for rid, p in self._perf.items():
                n = int(p["n"])
                pnl = float(p["pnl"])
                mean = pnl / n if n else 0.0
                var = (p["sum_pnl_sq"] / n - mean ** 2) if n > 1 else 0.0
                out[rid] = {
                    "n": n,
                    "wins": int(p["wins"]),
                    "pnl": pnl,
                    "win_rate": (p["wins"] / n) if n else None,
                    "avg_pnl": mean,
                    "pnl_std": math.sqrt(max(0.0, var)) if n > 1 else 0.0,
                }
            return out

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def _snapshot_unlocked(self) -> dict[str, Any]:
        feats = dict(self._last_features)
        rid = self._regime
        try:
            tsign = float(feats.get("trend_sign") or 0.0)
        except (TypeError, ValueError):
            tsign = 0.0
        if tsign > 0.15:
            trend_side = "yes"
        elif tsign < -0.15:
            trend_side = "no"
        else:
            trend_side = "flat"
        sample_ok = _feat(feats, "sample_ok", 0.0)
        held_sec = 0.0
        if self._last_change_ts is not None:
            held_sec = max(0.0, time.time() - float(self._last_change_ts))
        min_conf = float(getattr(config, "REGIME_ACTION_MIN_CONF", 0.50))
        min_hold = float(getattr(config, "REGIME_ACTION_MIN_HOLD_SEC", 20.0))
        actionable = (
            rid not in ("unknown", "", None)
            and sample_ok >= 0.5
            and float(self._confidence) >= min_conf
            and held_sec >= min_hold
        )
        return {
            "regime_id": rid,
            "regime": legacy_label(rid),          # legacy quiet/normal/...
            "label": rid,                         # rich id (preferred)
            "legacy": legacy_label(rid),
            "confidence": float(self._confidence),
            "sample_ok": sample_ok,
            "held_sec": held_sec,
            "actionable": bool(actionable),
            "features": feats,
            "vol_score": float(feats.get("vol", 0.0)),
            "trend_score": float(feats.get("trend", 0.0)),
            "mom_score": float(feats.get("mom", 0.0)),
            "flow_score": float(feats.get("flow", 0.0)),
            "trend_sign": tsign,
            "trend_side": trend_side,             # yes | no | flat
            "meta_bucket": meta_bucket(rid, feats.get("trend")),
            "known": rid != "unknown" and bool(feats),
            "ticks": self._ticks,
            "last_change_ts": self._last_change_ts,
            "last_change_from": self._last_change_from,
            "candidate": self._candidate,
            "candidate_ticks": self._candidate_ticks,
            # Soft stamp only — does not imply per-window regime state.
            "market_id": self._last_market_id,
        }

    def snapshot(self) -> dict[str, Any]:
        self._ensure_loaded()
        with self._lock:
            if not self._live:
                # Read-only consumer (dashboard): reflect the arena's latest
                # persisted regime, not this process's stale first load.
                self._load_from_db()
            return self._snapshot_unlocked()

    def transitions(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return list(self._transitions[-limit:])

    def status(self) -> dict[str, Any]:
        """Full dashboard payload."""
        snap = self.snapshot()
        return {
            "current": snap,
            "performance": self.performance_snapshot(),
            "transitions": self.transitions(15),
            "regimes": list(REGIME_IDS),
        }


# Process-wide singleton
_detector: Optional[RegimeDetector] = None
_detector_lock = threading.Lock()


def is_actionable(snap: Optional[dict] = None) -> bool:
    """True when downstream may apply regime policy (adapt / tilt / boost)."""
    if snap is None:
        try:
            snap = get_detector().snapshot()
        except Exception:
            return False
    if not isinstance(snap, dict):
        return False
    if "actionable" in snap:
        return bool(snap["actionable"])
    rid = snap.get("regime_id") or snap.get("label") or "unknown"
    if rid in ("unknown", "", None):
        return False
    return True


def get_detector() -> RegimeDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = RegimeDetector()
        return _detector


def reset_detector() -> RegimeDetector:
    """Test helper — replace the singleton with a fresh instance."""
    global _detector
    with _detector_lock:
        _detector = RegimeDetector()
        return _detector


def detect_once(
    prices: Sequence[float],
    *,
    cvd: float = 0.0,
    obi: float = 0.0,
    vol_score: Optional[float] = None,
    trend_score: Optional[float] = None,
) -> dict[str, Any]:
    """Stateless one-shot classification (no hysteresis / no singleton).

    Useful for offline harness and unit tests.
    """
    feats = compute_features(
        prices, cvd=cvd, obi=obi,
        vol_score=vol_score, trend_score=trend_score,
    )
    rid, conf = classify_rules(feats)
    return {
        "regime_id": rid,
        "regime": legacy_label(rid),
        "label": rid,
        "legacy": legacy_label(rid),
        "confidence": conf,
        "features": feats,
        "vol_score": feats["vol"],
        "trend_score": feats["trend"],
        "mom_score": feats["mom"],
        "flow_score": feats["flow"],
        "meta_bucket": meta_bucket(rid, feats["trend"]),
        "known": rid != "unknown",
    }

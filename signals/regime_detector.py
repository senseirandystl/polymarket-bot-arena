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

# Feature vector keys (order fixed for centroid distance).
FEATURE_KEYS = ("vol", "trend", "mom", "flow")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_features(
    prices: Sequence[float],
    *,
    cvd: float = 0.0,
    obi: float = 0.0,
    vol_score: Optional[float] = None,
    trend_score: Optional[float] = None,
    realized_vol: Optional[float] = None,
) -> dict[str, float]:
    """Derive continuous feature vector in ~[0, 1] from market inputs.

    Pure function — no module state. Safe for offline harness / tests.
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
    mom = 0.0
    if len(clean) >= 2 and clean[-2] > 0:
        ret = (clean[-1] - clean[-2]) / clean[-2]
        # soft saturate: 0.002 (~p97) → ~0.76; map to 0..1 via tanh
        mom = abs(math.tanh(ret / 0.002))

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

    return {
        "vol": _clip01(vol_score),
        "trend": _clip01(trend_score),
        "mom": _clip01(mom),
        "flow": _clip01(flow),
        "flow_align": _clip01(flow_align),
        "realized_vol": float(realized_vol),
    }


def classify_rules(features: dict[str, float],
                   *,
                   vol_hi: float = 0.55,
                   vol_lo: float = 0.35,
                   trend_hi: float = 0.50,
                   trend_lo: float = 0.35) -> tuple[str, float]:
    """Rule-based regime id + confidence from a feature dict.

    Returns (regime_id, confidence in 0..1).
    """
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

    # Mild boost when momentum/flow agree with the classification
    if rid in ("high_vol_trend", "low_vol_trend") and m > 0.4:
        conf = min(1.0, conf + 0.05)
    if rid == "high_vol_chop" and f > 0.4 and m < 0.3:
        conf = min(1.0, conf + 0.05)
    if rid == "low_vol_range" and m < 0.25:
        conf = min(1.0, conf + 0.05)

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
            else getattr(config, "REGIME_HOLD_TICKS", 3)
        )
        self.switch_margin = float(
            switch_margin if switch_margin is not None
            else getattr(config, "REGIME_SWITCH_MARGIN", 0.08)
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import db
            raw = db.get_arena_state(STATE_KEY)
            if raw:
                data = json.loads(raw)
                self._regime = data.get("regime", self._regime)
                self._confidence = float(data.get("confidence") or 0.0)
                self._ema = dict(data.get("ema") or {})
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
    ) -> dict[str, Any]:
        """Ingest one market tick; return current regime snapshot.

        Continuous / online — safe to call every second from the warm path.
        """
        self._ensure_loaded()
        raw = compute_features(
            prices, cvd=cvd, obi=obi,
            vol_score=vol_score, trend_score=trend_score,
            realized_vol=realized_vol,
        )
        with self._lock:
            # EMA smooth the classifier features
            for k in FEATURE_KEYS:
                prev = self._ema.get(k)
                cur = float(raw[k])
                if prev is None:
                    self._ema[k] = cur
                else:
                    a = self.ema_alpha
                    self._ema[k] = a * cur + (1.0 - a) * prev
            smoothed = {k: self._ema.get(k, raw[k]) for k in FEATURE_KEYS}
            smoothed["flow_align"] = raw.get("flow_align", 0.5)
            smoothed["realized_vol"] = raw.get("realized_vol", 0.0)
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
        return {
            "regime_id": rid,
            "regime": legacy_label(rid),          # legacy quiet/normal/...
            "label": rid,                         # rich id (preferred)
            "legacy": legacy_label(rid),
            "confidence": float(self._confidence),
            "features": feats,
            "vol_score": float(feats.get("vol", 0.0)),
            "trend_score": float(feats.get("trend", 0.0)),
            "mom_score": float(feats.get("mom", 0.0)),
            "flow_score": float(feats.get("flow", 0.0)),
            "meta_bucket": meta_bucket(rid, feats.get("trend")),
            "known": rid != "unknown" and bool(feats),
            "ticks": self._ticks,
            "last_change_ts": self._last_change_ts,
            "last_change_from": self._last_change_from,
            "candidate": self._candidate,
            "candidate_ticks": self._candidate_ticks,
        }

    def snapshot(self) -> dict[str, Any]:
        self._ensure_loaded()
        with self._lock:
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

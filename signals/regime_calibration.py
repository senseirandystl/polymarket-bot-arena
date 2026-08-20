"""Relative regime feature calibration (percentile scores over rolling history).

Absolute vol scores (sigmoid around a fixed VOL_TYPICAL) drift in meaning as
BTC's base volatility changes. This module maps raw scalars to [0, 1] via an
empirical CDF over a capped reservoir of recent observations so "high vol"
means high *for recent tape*.

Thread-safe; warm-path cheap (binary search on a sorted copy when needed).
Persists to arena_state so restarts keep calibration warm.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

STATE_KEY = "regime_calibration"
_DEFAULT_KEYS = ("realized_vol", "trend_eff", "chop", "mom_abs")


class RelativeCalibrator:
    """Reservoir percentile calibrator for named raw feature streams."""

    def __init__(
        self,
        *,
        max_samples: int | None = None,
        min_samples: int | None = None,
        window_days: float | None = None,
        keys: tuple[str, ...] = _DEFAULT_KEYS,
    ):
        self.max_samples = int(
            max_samples
            if max_samples is not None
            else getattr(config, "REGIME_REL_RESERVOIR_MAX", 20_000)
        )
        self.min_samples = int(
            min_samples
            if min_samples is not None
            else getattr(config, "REGIME_REL_MIN_SAMPLES", 500)
        )
        self.keys = tuple(keys)
        days = float(
            window_days
            if window_days is not None
            else getattr(config, "REGIME_REL_WINDOW_DAYS", 14) or 14
        )
        self.window_sec = max(60.0, days * 86400.0)
        self._lock = threading.Lock()
        # Timestamped points: [{"t": unix, "v": float}, ...]
        self._data: dict[str, list[dict[str, float]]] = {k: [] for k in self.keys}
        self._n_updates = 0
        self._last_persist = 0.0
        self._loaded = False
        self._last_fingerprint: Any = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                import db
                raw = db.get_arena_state(STATE_KEY)
                if raw:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(data, dict):
                        now = time.time()
                        for k in self.keys:
                            vals = data.get(k) or []
                            if not isinstance(vals, list):
                                continue
                            self._data[k] = self._migrate_points(
                                vals, now
                            )[-self.max_samples:]
                        self._n_updates = int(data.get("n_updates") or 0)
            except Exception as e:
                logger.debug("regime_calibration load failed: %s", e)
            self._loaded = True

    def _migrate_points(self, vals: list, now: float) -> list[dict[str, float]]:
        """Accept timestamped dicts or bare floats (legacy reservoir)."""
        stamped: list[dict[str, float]] = []
        bares: list[float] = []
        for v in vals:
            if isinstance(v, dict) and v.get("v") is not None:
                try:
                    raw_t = v.get("t")
                    ts = float(raw_t) if raw_t is not None else now
                    stamped.append({"t": ts, "v": float(v["v"])})
                except (TypeError, ValueError):
                    continue
            else:
                try:
                    bares.append(float(v))
                except (TypeError, ValueError):
                    continue
        # Legacy bare floats: space them 1m apart ending at `now` so a deploy
        # does not flash-cold the CDF.
        n = len(bares)
        for i, fv in enumerate(bares):
            stamped.append({"t": now - (n - 1 - i) * 60.0, "v": fv})
        cutoff = now - self.window_sec
        return [p for p in stamped if p["t"] >= cutoff]

    def _evict_unlocked(self, now: float) -> None:
        cutoff = now - self.window_sec
        for k, buf in self._data.items():
            kept = [p for p in buf if p["t"] >= cutoff]
            if len(kept) > self.max_samples:
                kept = kept[-self.max_samples:]
            self._data[k] = kept

    def persist(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_persist) < 60.0:
            return
        self._last_persist = now
        try:
            import db
            with self._lock:
                self._evict_unlocked(now)
                payload = {
                    k: list(v[-self.max_samples:]) for k, v in self._data.items()
                }
                payload["n_updates"] = self._n_updates
                payload["updated_at"] = now
                payload["max_samples"] = self.max_samples
                payload["window_days"] = self.window_sec / 86400.0
            db.set_arena_state(STATE_KEY, json.dumps(payload))
        except Exception as e:
            logger.debug("regime_calibration persist failed: %s", e)

    def update_if_changed(self, fingerprint: Any, **raw: float) -> bool:
        """Ingest only when ``fingerprint`` differs from the last ingest.

        The detector ticks at 1 Hz on 1-minute candles; without this the
        reservoir is 60×-duplicated copies of the same realized_vol.
        Returns True when a new observation was stored.
        """
        if fingerprint is None:
            return False
        self._ensure_loaded()
        with self._lock:
            if fingerprint == self._last_fingerprint:
                return False
        self.update(**raw)
        with self._lock:
            self._last_fingerprint = fingerprint
        # Unique candles arrive ~1/min; persist() already has a 60s gate.
        self.persist()
        return True

    def update(self, now: float | None = None, **raw: float) -> None:
        """Ingest one observation dict of raw feature values."""
        self._ensure_loaded()
        ts = float(now if now is not None else time.time())
        with self._lock:
            for k, v in raw.items():
                if k == "now":
                    continue
                if k not in self._data:
                    self._data[k] = []
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(fv):
                    continue
                self._data[k].append({"t": ts, "v": fv})
            self._evict_unlocked(ts)
            self._n_updates += 1
        # Unique 1m candles; persist() has a 60s gate.
        self.persist()

    def n_samples(self, key: str = "realized_vol") -> int:
        self._ensure_loaded()
        with self._lock:
            return len(self._data.get(key) or [])

    def ready(self, key: str = "realized_vol") -> bool:
        return self.n_samples(key) >= self.min_samples

    def percentile(self, key: str, value: float,
                   fallback: float | None = None) -> float:
        """Empirical CDF rank of ``value`` in [0, 1].

        If the reservoir is cold, returns ``fallback`` if given, else 0.5.
        """
        self._ensure_loaded()
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return 0.5 if fallback is None else float(fallback)
        with self._lock:
            self._evict_unlocked(time.time())
            buf = self._data.get(key) or []
            vals = [p["v"] if isinstance(p, dict) else float(p) for p in buf]
            n = len(vals)
            if n < max(10, self.min_samples // 10):
                return 0.5 if fallback is None else max(0.0, min(1.0, float(fallback)))
            # Fraction of samples strictly less + half ties
            less = sum(1 for x in vals if x < fv)
            equal = sum(1 for x in vals if x == fv)
            rank = (less + 0.5 * equal) / n
            cold = n < self.min_samples
        if cold and fallback is not None:
            # Blend toward absolute fallback until fully warm
            w = n / float(self.min_samples)
            fb = max(0.0, min(1.0, float(fallback)))
            return max(0.0, min(1.0, w * rank + (1.0 - w) * fb))
        return max(0.0, min(1.0, rank))

    def status(self) -> dict[str, Any]:
        self._ensure_loaded()
        with self._lock:
            self._evict_unlocked(time.time())
            oldest = None
            for buf in self._data.values():
                if buf:
                    t0 = buf[0]["t"] if isinstance(buf[0], dict) else None
                    if t0 is not None and (oldest is None or t0 < oldest):
                        oldest = t0
            return {
                "n_updates": self._n_updates,
                "min_samples": self.min_samples,
                "max_samples": self.max_samples,
                "window_days": self.window_sec / 86400.0,
                "oldest_ts": oldest,
                "counts": {k: len(v) for k, v in self._data.items()},
                "ready": {
                    k: len(v) >= self.min_samples for k, v in self._data.items()
                },
            }


_calibrator: Optional[RelativeCalibrator] = None
_cal_lock = threading.Lock()


def get_calibrator() -> RelativeCalibrator:
    global _calibrator
    with _cal_lock:
        if _calibrator is None:
            _calibrator = RelativeCalibrator()
        return _calibrator


def reset_calibrator() -> None:
    global _calibrator
    with _cal_lock:
        _calibrator = RelativeCalibrator()
        _calibrator._loaded = True  # skip DB reload in tests

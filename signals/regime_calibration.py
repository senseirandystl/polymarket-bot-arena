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
import random
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
        self._lock = threading.Lock()
        self._data: dict[str, list[float]] = {k: [] for k in self.keys}
        self._n_updates = 0
        self._last_persist = 0.0
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            import db
            raw = db.get_arena_state(STATE_KEY)
            if not raw:
                return
            data = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(data, dict):
                return
            for k in self.keys:
                vals = data.get(k) or []
                if isinstance(vals, list):
                    clean = []
                    for v in vals[-self.max_samples:]:
                        try:
                            clean.append(float(v))
                        except (TypeError, ValueError):
                            continue
                    self._data[k] = clean
            self._n_updates = int(data.get("n_updates") or 0)
        except Exception as e:
            logger.debug("regime_calibration load failed: %s", e)

    def persist(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_persist) < 60.0:
            return
        self._last_persist = now
        try:
            import db
            with self._lock:
                payload = {
                    k: list(v[-self.max_samples:]) for k, v in self._data.items()
                }
                payload["n_updates"] = self._n_updates
                payload["updated_at"] = now
                payload["max_samples"] = self.max_samples
            db.set_arena_state(STATE_KEY, json.dumps(payload))
        except Exception as e:
            logger.debug("regime_calibration persist failed: %s", e)

    def update(self, **raw: float) -> None:
        """Ingest one observation dict of raw feature values."""
        self._ensure_loaded()
        with self._lock:
            for k, v in raw.items():
                if k not in self._data:
                    self._data[k] = []
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(fv):
                    continue
                buf = self._data[k]
                if len(buf) < self.max_samples:
                    buf.append(fv)
                else:
                    # Reservoir sample: replace random slot
                    i = random.randint(0, self.max_samples - 1)
                    buf[i] = fv
            self._n_updates += 1
            n = self._n_updates
        if n % 120 == 0:  # ~2 min at 1 Hz
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
            buf = self._data.get(key) or []
            if len(buf) < max(10, self.min_samples // 10):
                return 0.5 if fallback is None else max(0.0, min(1.0, float(fallback)))
            # Fraction of samples strictly less + half ties
            less = sum(1 for x in buf if x < fv)
            equal = sum(1 for x in buf if x == fv)
            rank = (less + 0.5 * equal) / len(buf)
            cold = len(buf) < self.min_samples
        if cold and fallback is not None:
            # Blend toward absolute fallback until fully warm
            w = len(buf) / float(self.min_samples)
            fb = max(0.0, min(1.0, float(fallback)))
            return max(0.0, min(1.0, w * rank + (1.0 - w) * fb))
        return max(0.0, min(1.0, rank))

    def status(self) -> dict[str, Any]:
        self._ensure_loaded()
        with self._lock:
            return {
                "n_updates": self._n_updates,
                "min_samples": self.min_samples,
                "max_samples": self.max_samples,
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

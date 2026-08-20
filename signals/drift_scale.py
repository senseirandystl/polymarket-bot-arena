"""Adaptive full-window vol scale for ``btc_drift``.

``drift_signal`` divides moneyness by a remaining-window sigma. Historically
that sigma used a fixed ``config.DRIFT_VOL_SCALE``. When BTC vol rises, the
same dollar move is less special; when vol falls, it is more special.

**2026-08-07 recal:** moneyness is Chainlink **TWAP**, so adaptive σ prefers
TWAP tick samples (same object). Spot 1m candles are fallback only. Under-
scaled σ + √time made $10–20 late wiggles print as strong drift and broke
directional follow-WR.

Design
------
* Prefer TWAP tick series → resample → log-return stdev → scale to 5m window.
* Fallback: 1m spot closes → σ_1m · √5 (legacy).
* EMA-smooth; cold-start stays on the config prior.
* Hard clamps around the prior.

Thread-safe. Hot path: update once per trader tick in ``build_combined_signals``.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Optional

import config

logger = logging.getLogger(__name__)

# 5 one-minute returns in a 5-minute window under Brownian independence.
_WINDOW_BARS = 5.0
_SQRT_WINDOW = math.sqrt(_WINDOW_BARS)


class DriftScaleEstimator:
    """EMA of full-window fractional vol for drift normalization."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ema: Optional[float] = None
        self._n = 0
        self._last_raw: Optional[float] = None
        self._source: str = "prior"  # prior | twap | spot

    def reset(self) -> None:
        with self._lock:
            self._ema = None
            self._n = 0
            self._last_raw = None
            self._source = "prior"

    def update_from_prices(self, prices: list) -> Optional[float]:
        """Ingest closed 1m closes (spot fallback); return EMA or None."""
        raw = estimate_window_vol_scale(prices)
        if raw is None:
            return None
        return self.update_raw(raw, source="spot")

    def update_from_twap_ticks(
        self, ticks: list, sample_sec: Optional[float] = None,
    ) -> Optional[float]:
        """Ingest (epoch, twap) ticks — preferred adaptive path."""
        raw = estimate_vol_scale_from_ticks(ticks, sample_sec=sample_sec)
        if raw is None:
            return None
        return self.update_raw(raw, source="twap")

    def update_raw(self, raw_scale: float, *, source: str = "spot") -> float:
        """Push one full-window scale observation into the EMA."""
        try:
            raw = float(raw_scale)
        except (TypeError, ValueError):
            return self.current()
        if not math.isfinite(raw) or raw <= 0:
            return self.current()

        lo, hi = _clamp_bounds()
        raw = max(lo, min(hi, raw))
        alpha = float(getattr(config, "DRIFT_ADAPT_EMA_ALPHA", 0.08) or 0.08)
        alpha = max(0.01, min(0.5, alpha))

        with self._lock:
            self._last_raw = raw
            self._source = source or "spot"
            if self._ema is None:
                self._ema = raw
            else:
                self._ema = (1.0 - alpha) * self._ema + alpha * raw
            self._ema = max(lo, min(hi, self._ema))
            self._n += 1
            return self._ema

    def current(self) -> float:
        """Scale to use in ``drift_signal`` (clamped, prior if cold)."""
        prior = float(getattr(config, "DRIFT_VOL_SCALE", 0.0022) or 0.0022)
        if not getattr(config, "DRIFT_ADAPTIVE_SCALE", True):
            return max(1e-8, prior)

        lo, hi = _clamp_bounds()
        min_n = int(getattr(config, "DRIFT_ADAPT_MIN_SAMPLES", 20) or 20)
        with self._lock:
            ema = self._ema
            n = self._n
        if ema is None or n <= 0:
            return max(lo, min(hi, prior))
        if n < min_n:
            w = n / float(min_n)
            blended = w * ema + (1.0 - w) * prior
        else:
            blended = ema
        return max(lo, min(hi, float(blended)))

    def last_source(self) -> str:
        with self._lock:
            return self._source

    def sigma_1m(self) -> float:
        """Implied 1m log-return σ from current full-window scale (≈ scale/√5)."""
        return float(self.current()) / _SQRT_WINDOW

    def mom_saturate_scale(self) -> float:
        """Soft-saturate scale for 1m momentum lanes (adaptive)."""
        prior = float(getattr(config, "MOM_SCALE_PRIOR", 0.002) or 0.002)
        if not getattr(config, "MOM_ADAPTIVE_SCALE", True):
            return prior
        lo = float(getattr(config, "MOM_SCALE_MIN", 0.0015) or 0.0015)
        hi = float(getattr(config, "MOM_SCALE_MAX", 0.005) or 0.005)
        mult = float(getattr(config, "MOM_SCALE_VOL_MULT", 1.35) or 1.35)
        raw = mult * self.sigma_1m()
        if not math.isfinite(raw) or raw <= 0:
            return prior
        # Blend toward prior when cold
        min_n = int(getattr(config, "DRIFT_ADAPT_MIN_SAMPLES", 20) or 20)
        with self._lock:
            n = self._n
        if n < min_n:
            w = n / float(max(1, min_n))
            raw = w * raw + (1.0 - w) * prior
        return max(lo, min(hi, float(raw)))

    def status(self) -> dict:
        prior = float(getattr(config, "DRIFT_VOL_SCALE", 0.0022) or 0.0022)
        lo, hi = _clamp_bounds()
        cur = self.current()
        with self._lock:
            return {
                "adaptive": bool(getattr(config, "DRIFT_ADAPTIVE_SCALE", True)),
                "ema": self._ema,
                "n": self._n,
                "last_raw": self._last_raw,
                "current": cur,
                "prior": prior,
                "min": lo,
                "max": hi,
                "source": self._source,
                "sigma_1m": cur / _SQRT_WINDOW,
                "mom_scale": self.mom_saturate_scale(),
            }


def _clamp_bounds() -> tuple[float, float]:
    prior = float(getattr(config, "DRIFT_VOL_SCALE", 0.0022) or 0.0022)
    lo = float(getattr(config, "DRIFT_VOL_SCALE_MIN", prior * 0.45) or prior * 0.45)
    hi = float(getattr(config, "DRIFT_VOL_SCALE_MAX", prior * 2.5) or prior * 2.5)
    if lo <= 0:
        lo = prior * 0.45
    if hi <= lo:
        hi = max(lo * 2.0, prior * 2.5)
    return lo, hi


def estimate_window_vol_scale(prices: list) -> Optional[float]:
    """Estimate full-window fractional vol from 1m closes (spot fallback).

    Uses stdev of 1m log-returns over the last 20 bars, scaled by √5 for a
    5-minute horizon.
    """
    if not prices or len(prices) < 5:
        return None
    window = [p for p in prices[-20:] if p and p > 0]
    if len(window) < 5:
        return None
    rets = [math.log(window[i] / window[i - 1]) for i in range(1, len(window))]
    if len(rets) < 4:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    rvol_1m = math.sqrt(max(var, 0.0))
    atr = sum(abs(r) for r in rets) / len(rets)
    atr_sigma = atr * math.sqrt(math.pi / 2.0) if atr > 0 else 0.0
    sigma_1m = max(rvol_1m, 0.5 * atr_sigma)
    if sigma_1m <= 0:
        return None
    return float(sigma_1m * _SQRT_WINDOW)


def resample_tick_prices(
    ticks: list,
    sample_sec: float = 5.0,
    max_points: int = 80,
) -> Optional[list[float]]:
    """Downsample ``(epoch, price)`` ticks to ~sample_sec spacing (oldest first)."""
    if not ticks:
        return None
    try:
        sample_sec = float(sample_sec)
    except (TypeError, ValueError):
        sample_sec = 5.0
    sample_sec = max(1.0, sample_sec)
    ordered = sorted(
        ((float(t), float(p)) for t, p in ticks if p and float(p) > 0),
        key=lambda x: x[0],
    )
    if len(ordered) < 5:
        return None
    out: list[float] = []
    last_t: Optional[float] = None
    for t, p in ordered:
        if last_t is None or (t - last_t) >= sample_sec * 0.85:
            out.append(p)
            last_t = t
    if len(out) > max_points:
        out = out[-max_points:]
    return out if len(out) >= 5 else None


def estimate_vol_scale_from_ticks(
    ticks: list,
    sample_sec: Optional[float] = None,
) -> Optional[float]:
    """Full-window fractional vol from TWAP (or any) tick series.

    Resamples to ``sample_sec``, takes log-return stdev of that sampling
    frequency, then scales to ``MARKET_WINDOW_SEC``:
    ``σ_window = σ_sample · √(window / sample_sec)``.
    """
    dt = float(
        sample_sec
        if sample_sec is not None
        else getattr(config, "DRIFT_ADAPT_TWAP_SAMPLE_SEC", 60.0) or 60.0
    )
    series = resample_tick_prices(ticks, sample_sec=dt)
    if not series or len(series) < 5:
        return None
    rets = [
        math.log(series[i] / series[i - 1])
        for i in range(1, len(series))
        if series[i - 1] > 0 and series[i] > 0
    ]
    if len(rets) < 4:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    sigma_s = math.sqrt(max(var, 0.0))
    atr = sum(abs(r) for r in rets) / len(rets)
    atr_sigma = atr * math.sqrt(math.pi / 2.0) if atr > 0 else 0.0
    sigma_s = max(sigma_s, 0.5 * atr_sigma)
    if sigma_s <= 0:
        return None
    window = float(getattr(config, "MARKET_WINDOW_SEC", 300) or 300)
    # Brownian scale from sample interval → full market window
    return float(sigma_s * math.sqrt(window / max(dt, 1.0)))


_estimator: Optional[DriftScaleEstimator] = None
_est_lock = threading.Lock()


def get_drift_scale_estimator() -> DriftScaleEstimator:
    global _estimator
    with _est_lock:
        if _estimator is None:
            _estimator = DriftScaleEstimator()
        return _estimator


def reset_drift_scale_estimator() -> None:
    """Test helper — clear singleton state."""
    global _estimator
    with _est_lock:
        _estimator = DriftScaleEstimator()


def resolve_vol_scale(explicit: Optional[float] = None) -> float:
    """Vol scale for one drift computation.

    Preference: explicit override → adaptive estimator → config prior.
    """
    if explicit is not None:
        try:
            v = float(explicit)
            if math.isfinite(v) and v > 0:
                lo, hi = _clamp_bounds()
                return max(lo, min(hi, v))
        except (TypeError, ValueError):
            pass
    return get_drift_scale_estimator().current()


def update_estimator_from_feeds(
    *,
    twap_ticks: Optional[list] = None,
    spot_prices: Optional[list] = None,
) -> float:
    """Hot-path helper: prefer TWAP ticks, else spot 1m closes. Returns current."""
    est = get_drift_scale_estimator()
    use_twap = bool(getattr(config, "DRIFT_ADAPT_USE_TWAP", True))
    if use_twap and twap_ticks:
        out = est.update_from_twap_ticks(twap_ticks)
        if out is not None:
            return float(est.current())
    if spot_prices:
        est.update_from_prices(spot_prices)
    return float(est.current())

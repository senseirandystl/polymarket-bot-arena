"""Relative regime calibrator unit tests."""

import time

import pytest

from signals.regime_calibration import RelativeCalibrator, reset_calibrator


def test_percentile_constant_series_near_half():
    reset_calibrator()
    cal = RelativeCalibrator(max_samples=500, min_samples=50)
    cal._loaded = True
    for _ in range(200):
        cal.update(realized_vol=0.0005)
    p = cal.percentile("realized_vol", 0.0005, fallback=0.5)
    assert 0.4 <= p <= 0.6


def test_percentile_monotonic():
    cal = RelativeCalibrator(max_samples=1000, min_samples=20)
    cal._loaded = True
    for i in range(100):
        cal.update(realized_vol=0.0001 * (i + 1))
    lo = cal.percentile("realized_vol", 0.001)
    hi = cal.percentile("realized_vol", 0.009)
    assert hi >= lo


def test_cold_fallback():
    cal = RelativeCalibrator(max_samples=100, min_samples=500)
    cal._loaded = True
    cal.update(realized_vol=0.001)
    p = cal.percentile("realized_vol", 0.001, fallback=0.42)
    assert 0.0 <= p <= 1.0


def test_status_keys():
    cal = RelativeCalibrator()
    cal._loaded = True
    st = cal.status()
    assert "counts" in st and "ready" in st


def test_update_if_changed_skips_duplicate_fingerprint():
    """1s ticks on the same 1m candle must not 60×-duplicate the reservoir."""
    cal = RelativeCalibrator(max_samples=100, min_samples=10)
    cal._loaded = True
    assert cal.update_if_changed("c1", realized_vol=0.0006) is True
    assert cal.update_if_changed("c1", realized_vol=0.0006) is False
    assert cal.update_if_changed("c2", realized_vol=0.0007) is True
    assert cal.n_samples("realized_vol") == 2


def test_time_window_evicts_old_samples():
    cal = RelativeCalibrator(max_samples=1000, min_samples=10, window_days=14)
    cal._loaded = True
    now = 1_700_000_000.0
    cal.update(now=now - 16 * 86400, realized_vol=0.0001)
    cal.update(now=now, realized_vol=0.0009)
    assert cal.n_samples("realized_vol") == 1
    # Remaining point is the recent high one
    p = cal.percentile("realized_vol", 0.0009, fallback=0.5)
    assert p >= 0.4


def test_migrate_bare_floats_keeps_reservoir():
    cal = RelativeCalibrator(max_samples=100, min_samples=10, window_days=14)
    cal._loaded = True
    now = time.time()
    pts = cal._migrate_points([0.001, 0.002, 0.003], now)
    assert len(pts) == 3
    assert all("t" in p and "v" in p for p in pts)
    assert pts[-1]["v"] == pytest.approx(0.003)

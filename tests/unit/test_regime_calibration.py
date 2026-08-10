"""Relative regime calibrator unit tests."""

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

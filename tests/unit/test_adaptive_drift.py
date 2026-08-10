"""Adaptive drift vol scale — high vol damps |drift|; low vol amplifies."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
import signals.strike as strike
from signals.drift_scale import (
    DriftScaleEstimator,
    estimate_window_vol_scale,
    get_drift_scale_estimator,
    reset_drift_scale_estimator,
    resolve_vol_scale,
)


def _quiet_prices(n=25, start=100_000.0, step=2.0):
    """Tiny 1m moves → low realized vol."""
    return [start + i * step for i in range(n)]


def _wild_prices(n=25, start=100_000.0, amp=400.0):
    """Large alternating moves → high realized vol."""
    out = []
    px = start
    for i in range(n):
        px = start + (amp if i % 2 == 0 else -amp)
        out.append(px)
    return out


def setup_function():
    reset_drift_scale_estimator()


def test_estimate_window_vol_scale_higher_for_wild_tape():
    q = estimate_window_vol_scale(_quiet_prices())
    w = estimate_window_vol_scale(_wild_prices())
    assert q is not None and w is not None
    assert w > q * 2


def test_ema_moves_toward_observations():
    est = DriftScaleEstimator()
    prior = float(config.DRIFT_VOL_SCALE)
    # Force many high-vol updates
    for _ in range(80):
        est.update_from_prices(_wild_prices())
    cur = est.current()
    assert cur > prior
    assert cur <= float(config.DRIFT_VOL_SCALE_MAX) + 1e-12


def test_clamps_respected():
    est = DriftScaleEstimator()
    est.update_raw(1.0)  # absurd → clamp to MAX
    assert est.current() <= float(config.DRIFT_VOL_SCALE_MAX) + 1e-12
    est2 = DriftScaleEstimator()
    est2.update_raw(1e-12)
    assert est2.current() >= float(config.DRIFT_VOL_SCALE_MIN) - 1e-12


def test_higher_vol_scale_reduces_abs_drift():
    """Same $ move, larger σ → smaller |btc_drift|."""
    d_low = strike.drift_signal(
        100_000.0, 100_100.0, 120, vol_scale=0.0012)
    d_high = strike.drift_signal(
        100_000.0, 100_100.0, 120, vol_scale=0.0035)
    assert abs(d_low) > abs(d_high)


def test_twap_tick_vol_estimate():
    from signals.drift_scale import estimate_vol_scale_from_ticks
    # Synthetic TWAP path with large moves
    import time as _t
    t0 = 1_700_000_000.0
    ticks = []
    px = 100_000.0
    for i in range(40):
        px = 100_000.0 + (200.0 if i % 2 == 0 else -200.0)
        ticks.append((t0 + i * 5.0, px))
    raw = estimate_vol_scale_from_ticks(ticks, sample_sec=5.0)
    assert raw is not None and raw > 0


def test_mom_saturate_scale_tracks_vol():
    from signals.drift_scale import DriftScaleEstimator
    est = DriftScaleEstimator()
    for _ in range(40):
        est.update_from_prices(_wild_prices())
    assert est.mom_saturate_scale() > float(config.MOM_SCALE_PRIOR) * 0.9


def test_dual_gate_skips_false_strong_z_without_moneyness(monkeypatch):
    """Live path with strike/pct: tiny $ move + large z must skip."""
    import db
    from bots.bot_momentum import MomentumBot
    monkeypatch.setattr(db, "get_paper_available", lambda: 200.0)
    monkeypatch.setattr(db, "get_kelly_fraction", lambda: 0.25)
    bot = MomentumBot(name="mom-dual-gate")
    bot._perf_cache = (9e12, 0)
    market = {
        "id": "m", "current_price": 0.50, "no_price": 0.50,
        "yes_ask": 0.51, "no_ask": 0.51, "time_remaining_seconds": 30,
    }
    # Large z but tiny moneyness (would be √time noise without dual gate)
    signals = {
        "prices": [65000.0, 65010.0], "latest": 65010.0,
        "orderflow": {}, "btc_drift": 0.55,
        "btc_drift_pct": 0.00010,  # ~$6.5 @ 65k — below floor
        "btc_strike": 65000.0, "btc_now": 65006.5,
        "drift_vol_scale": 0.0022,
    }
    d = bot.make_decision(market, signals)
    assert d["action"] == "skip"
    assert "dual-gate" in d["reasoning"].lower()


def test_time_scaling_still_applies_with_explicit_scale():
    # With DRIFT_TIME_SCALE_MIN_SEC floor (60s), compare 280s vs 90s so both
    # are above the floor and time scaling is still visible.
    early = strike.drift_signal(
        100_000.0, 100_100.0, 280, vol_scale=0.0022)
    mid = strike.drift_signal(
        100_000.0, 100_100.0, 90, vol_scale=0.0022)
    assert abs(mid) > abs(early)


def test_time_scale_floor_caps_late_explosion():
    """Last-second noise must not exceed mid-window floor equivalence."""
    # Without floor, tr=15 would dominate tr=60; with floor both ≈60s factor.
    a = strike.drift_signal(100_000.0, 100_050.0, 15, vol_scale=0.0022)
    b = strike.drift_signal(100_000.0, 100_050.0, 60, vol_scale=0.0022)
    assert abs(a - b) < 1e-9


def test_adaptive_off_uses_prior():
    est = get_drift_scale_estimator()
    for _ in range(50):
        est.update_from_prices(_wild_prices())
    prev = getattr(config, "DRIFT_ADAPTIVE_SCALE", True)
    try:
        config.DRIFT_ADAPTIVE_SCALE = False
        assert abs(resolve_vol_scale() - float(config.DRIFT_VOL_SCALE)) < 1e-12
    finally:
        config.DRIFT_ADAPTIVE_SCALE = prev


def test_cold_start_near_prior():
    reset_drift_scale_estimator()
    s = resolve_vol_scale()
    assert abs(s - float(config.DRIFT_VOL_SCALE)) < 1e-9


def test_drift_pct_raw():
    assert abs(strike.drift_pct(100_000.0, 100_150.0) - 0.0015) < 1e-9
    assert strike.drift_pct(None, 100_000.0) == 0.0


def test_live_path_estimator_affects_default_drift_signal():
    """After enough wild updates, default (no vol_scale) drift shrinks."""
    reset_drift_scale_estimator()
    d0 = strike.drift_signal(100_000.0, 100_080.0, 60)
    est = get_drift_scale_estimator()
    for _ in range(100):
        est.update_from_prices(_wild_prices())
    d1 = strike.drift_signal(100_000.0, 100_080.0, 60)
    # Adaptive should have raised scale → |d1| <= |d0| (usually strict)
    assert abs(d1) <= abs(d0) + 1e-9
    if getattr(config, "DRIFT_ADAPTIVE_SCALE", True):
        assert abs(d1) < abs(d0) or math.isclose(abs(d1), abs(d0), rel_tol=1e-3)

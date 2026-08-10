"""Continuous regime residual weights."""

import config
from arena import regime_continuous as rc


def test_disabled_returns_zero_residual(monkeypatch):
    monkeypatch.setattr(config, "REGIME_CONTINUOUS_BLEND", False)
    monkeypatch.setattr(
        "arena.regime_settings.get_bool", lambda name: False, raising=False,
    )
    rc.reset_for_tests()
    assert rc.residual("mom", "momentum", {"vol": 0.8}) == 0.0


def test_apply_respects_cap(monkeypatch):
    monkeypatch.setattr(config, "REGIME_CONTINUOUS_BLEND", True)
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: name == "continuous_blend",
        raising=False,
    )
    monkeypatch.setattr(config, "REGIME_CONTINUOUS_MIN_SAMPLES", 0)
    monkeypatch.setattr(config, "REGIME_CONTINUOUS_MAX_DELTA", 0.08)
    rc.reset_for_tests()
    # Force large coeffs
    with rc._lock:
        rc._B = {"mom": {"momentum": [10.0, 0.0, 0.0, 0.0]}}
        rc._n_obs = 500
    d = rc.residual("mom", "momentum", {"vol_rel": 1.0, "direction": 0.5,
                                        "chop": 0.5, "flow": 0.3})
    assert abs(d) <= 0.08 + 1e-9


def test_apply_residuals_preserves_unknown_lanes(monkeypatch):
    monkeypatch.setattr(config, "REGIME_CONTINUOUS_BLEND", True)
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: name == "continuous_blend",
        raising=False,
    )
    monkeypatch.setattr(config, "REGIME_CONTINUOUS_MIN_SAMPLES", 0)
    rc.reset_for_tests()
    with rc._lock:
        rc._n_obs = 500
    w = rc.apply_residuals({"drift": 0.5, "fut": 0.1}, "momentum", {})
    assert "fut" in w and w["fut"] == 0.1

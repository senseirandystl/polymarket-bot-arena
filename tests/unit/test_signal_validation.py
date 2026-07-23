"""Unit tests for the pure signal-validation logic (no network)."""

from tools.signal_validation import build_samples, predictiveness, time_buckets


def _traj_up():
    # BTC rises steadily from a 100000 strike over 5 one-min steps.
    return [(0, 100000.0), (60, 100050.0), (120, 100100.0),
            (180, 100150.0), (240, 100200.0)]


def test_build_samples_skips_open_and_sets_time_remaining():
    s = build_samples("m", 100000.0, _traj_up(), yes_won=True)
    assert len(s) == 4                      # open point (drift 0) skipped
    assert s[0].time_remaining == 240       # 300 - 60
    assert s[-1].time_remaining == 60
    assert s[0].signals["drift_raw"] > 0    # above strike -> Up-frame positive


def test_predictiveness_detects_a_perfect_signal():
    # A market that rose all window AND resolved Up -> drift_raw perfectly
    # predicts: following it wins 100%.
    s = build_samples("m", 100000.0, _traj_up(), yes_won=True)
    r = predictiveness(s, "drift_raw")
    assert r["n"] == 4
    assert r["follow_winrate"] == 1.0


def test_predictiveness_detects_an_inverted_signal():
    # BTC above strike all window (signal says Up) but market resolved DOWN ->
    # following the signal loses every time (the drift-blowup scenario).
    s = build_samples("m", 100000.0, _traj_up(), yes_won=False)
    r = predictiveness(s, "drift_raw")
    assert r["follow_winrate"] == 0.0       # INVERTED


def test_predictiveness_ignores_deadband():
    flat = [(0, 100000.0), (60, 100000.0), (120, 100000.0)]
    s = build_samples("m", 100000.0, flat, yes_won=True)
    r = predictiveness(s, "drift_raw")
    assert r["n"] == 0                       # no signal -> nothing counted


def test_time_buckets_partition_samples():
    s = build_samples("m", 100000.0, _traj_up(), yes_won=True)
    buckets = time_buckets(s, "drift_raw")
    total = sum(b["n"] for b in buckets)
    assert total == 4
    labels = [b["bucket"] for b in buckets]
    assert labels == ["0-60s", "60-120s", "120-180s", "180-300s"]


# --- pm_mom lane validation (BUG: live lane saturates at a 0.19c/step move) ---

def _pm_up():
    # PM YES mid rising 2c/min over the window.
    return [(0, 0.50), (60, 0.52), (120, 0.54), (180, 0.56), (240, 0.58)]


def test_build_samples_pm_mom_sign_follows_pm_trend():
    s = build_samples("m", 100000.0, _traj_up(), yes_won=True, pm_prices=_pm_up())
    # Every sample after the first PM point should carry a positive pm_mom
    vals = [x.signals.get("pm_mom") for x in s]
    assert all(v is not None for v in vals)
    assert all(v > 0 for v in vals if v is not None)


def test_build_samples_pm_mom_none_without_history():
    s = build_samples("m", 100000.0, _traj_up(), yes_won=True)
    assert all(x.signals.get("pm_mom") is None for x in s)


def test_build_samples_pm_mom_needs_two_points():
    # Only one PM point at/before decision time -> no momentum measurable.
    s = build_samples("m", 100000.0, _traj_up(), yes_won=True,
                      pm_prices=[(0, 0.50)])
    assert all(x.signals.get("pm_mom") is None for x in s)


def test_predictiveness_skips_none_signals():
    s = build_samples("m", 100000.0, _traj_up(), yes_won=True)
    r = predictiveness(s, "pm_mom")
    assert r["n"] == 0


def test_magnitude_distribution_percentiles():
    from tools.signal_validation import magnitude_distribution
    s = build_samples("m", 100000.0, _traj_up(), yes_won=True, pm_prices=_pm_up())
    dist = magnitude_distribution(s, "pm_mom", percentiles=(50, 97))
    assert dist["n"] == len(s)
    # steady 2c/min PM trend -> per-minute |delta| = 0.02 at every sample
    assert abs(dist["p50"] - 0.02) < 1e-9
    assert abs(dist["p97"] - 0.02) < 1e-9

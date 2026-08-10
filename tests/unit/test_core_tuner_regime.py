import inspect

from arena import core_lane_tuner


def test_compute_core_attribution_accepts_cell_filter():
    sig = inspect.signature(core_lane_tuner.compute_core_attribution)
    assert "cell_filter" in sig.parameters


def test_tune_respects_conditioning_toggle(monkeypatch):
    # When conditioning is OFF, tune() must call attribution with cell_filter=None
    # (global behavior, unchanged).
    captured = {}
    monkeypatch.setattr(core_lane_tuner.db, "get_regime_conditioning", lambda: False)

    def spy(conn, deadband, *, cell_filter=None, regime_id=None, lanes=None):
        captured["cell_filter"] = cell_filter
        captured["regime_id"] = regime_id
        captured["lanes"] = lanes
        return {}

    monkeypatch.setattr(core_lane_tuner, "compute_core_attribution", spy)
    monkeypatch.setattr(core_lane_tuner.config, "CORE_TUNE_ENABLED", True,
                        raising=False)
    # Disable live-regime path so we exercise cell_filter=None global path
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: False if name == "profile_adapt" else True,
    )
    monkeypatch.setattr(core_lane_tuner.config, "REGIME_PROFILE_ADAPT_ENABLED",
                        False, raising=False)
    core_lane_tuner.tune()
    assert captured.get("cell_filter") is None


def test_tune_passes_current_cell_when_conditioning_on(monkeypatch):
    # When conditioning is ON and a current_cell exists, tune() passes it through
    # as a tuple cell_filter. (May fall back to a second global call when the
    # cell has no samples — first call must still receive the cell.)
    calls = []
    monkeypatch.setattr(core_lane_tuner.db, "get_regime_conditioning", lambda: True)
    monkeypatch.setattr(core_lane_tuner.db, "get_regime_map",
                        lambda: {"current_cell": ["low_vol_range", 2, 3, "us", 0, 0]})
    # Force cell_filter path (not live detector regime_id path)
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: False if name == "profile_adapt" else True,
    )
    monkeypatch.setattr(core_lane_tuner.config, "REGIME_PROFILE_ADAPT_ENABLED",
                        False, raising=False)

    def spy(conn, deadband, *, cell_filter=None, regime_id=None, lanes=None):
        calls.append(cell_filter)
        # Enough samples so fallback is not required.
        if cell_filter is not None:
            return {"momentum": {"drift": {"n": 100, "accuracy": 0.60}}}
        return {}

    monkeypatch.setattr(core_lane_tuner, "compute_core_attribution", spy)
    monkeypatch.setattr(core_lane_tuner.config, "CORE_TUNE_ENABLED", True,
                        raising=False)
    core_lane_tuner.tune()
    assert calls[0] == ("low_vol_range", 2, 3, "us", 0, 0)

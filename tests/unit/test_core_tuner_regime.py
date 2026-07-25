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

    def spy(conn, deadband, *, cell_filter=None):
        captured["cell_filter"] = cell_filter
        return {}

    monkeypatch.setattr(core_lane_tuner, "compute_core_attribution", spy)
    monkeypatch.setattr(core_lane_tuner.config, "CORE_TUNE_ENABLED", True,
                        raising=False)
    core_lane_tuner.tune()
    assert captured.get("cell_filter") is None


def test_tune_passes_current_cell_when_conditioning_on(monkeypatch):
    # When conditioning is ON and a current_cell exists, tune() passes it through
    # as a tuple cell_filter.
    captured = {}
    monkeypatch.setattr(core_lane_tuner.db, "get_regime_conditioning", lambda: True)
    monkeypatch.setattr(core_lane_tuner.db, "get_regime_map",
                        lambda: {"current_cell": ["low_vol_range", 2, 3, "us", 0, 0]})

    def spy(conn, deadband, *, cell_filter=None):
        captured["cell_filter"] = cell_filter
        return {}

    monkeypatch.setattr(core_lane_tuner, "compute_core_attribution", spy)
    monkeypatch.setattr(core_lane_tuner.config, "CORE_TUNE_ENABLED", True,
                        raising=False)
    core_lane_tuner.tune()
    assert captured.get("cell_filter") == ("low_vol_range", 2, 3, "us", 0, 0)

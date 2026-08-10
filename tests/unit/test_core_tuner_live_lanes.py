"""Core tuner includes enabled live candidate overrides."""

import pytest

from arena import core_lane_tuner as ct


def test_live_tune_lanes_includes_enabled_candidates():
    ov = {
        "drift": {"enabled": True, "core": True, "profile": {"momentum": 0.7}},
        "xasset": {"enabled": True, "profile": {"momentum": 0.12}},
        "fut": {"enabled": False, "profile": {"momentum": 0.1}},
        "tech": {"profile": {"momentum": 0.1}},  # no enabled flag
    }
    lanes = ct.live_tune_lanes(ov)
    assert "drift" in lanes and "mom" in lanes and "strat" in lanes
    assert "xasset" in lanes
    assert "fut" not in lanes
    assert "tech" not in lanes


def test_parse_readings_core_and_cand():
    text = (
        "fair=0.55 => yes edge=+0.05 "
        "drift=+0.40 mom=-0.10 strat=+0.05 "
        "cand(fut=+0.20 tech=-0.30 xa=+0.55 lag=+0.1 ms=+0.0 fd=-0.2)"
    )
    r = ct._parse_readings(
        text, ("drift", "mom", "strat", "fut", "tech", "xasset", "lag"),
    )
    assert r["drift"] == pytest.approx(0.40)
    assert r["mom"] == pytest.approx(-0.10)
    assert r["xasset"] == pytest.approx(0.55)
    assert r["fut"] == pytest.approx(0.20)
    assert r["tech"] == pytest.approx(-0.30)


def test_tune_reports_candidate_lane(monkeypatch):
    monkeypatch.setattr(ct.config, "CORE_TUNE_ENABLED", True, raising=False)
    monkeypatch.setattr(ct.db, "get_auto_core_tune", lambda: False)
    monkeypatch.setattr(ct.db, "get_lane_overrides", lambda: {
        "xasset": {
            "enabled": True,
            "profile": {"momentum": 0.15, "hybrid": 0.10},
        },
    })
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: False if name == "profile_adapt" else True,
    )
    monkeypatch.setattr(ct.config, "REGIME_PROFILE_ADAPT_ENABLED", False,
                        raising=False)
    monkeypatch.setattr(ct.db, "get_regime_conditioning", lambda: False)

    attr = {
        "momentum": {
            "drift": {"n": 50, "accuracy": 0.60},
            "mom": {"n": 50, "accuracy": 0.55},
            "strat": {"n": 50, "accuracy": 0.52},
            "xasset": {"n": 40, "accuracy": 0.42},  # should DOWN
        },
    }
    monkeypatch.setattr(
        ct, "compute_core_attribution",
        lambda *a, **k: attr,
    )
    monkeypatch.setattr(ct.db, "set_arena_state", lambda *a, **k: None)
    monkeypatch.setattr(ct.config, "CORE_TUNE_MIN_TRADES", 30, raising=False)
    monkeypatch.setattr(ct.config, "CANDIDATE_TUNE_MIN_TRADES", 30, raising=False)

    report = ct.tune()
    assert "xasset" in report.get("tune_lanes", [])
    assert "xasset" in report["lanes"]
    xa = report["lanes"]["xasset"].get("momentum")
    assert xa is not None
    assert xa["kind"] == "candidate"
    # accuracy 0.42 <= LOW → down from 0.15
    assert xa["action"] in ("down", "collecting", "hold", "revert")
    if xa["action"] == "down":
        assert xa["suggested"] < xa["current"]

"""Core-lane tuner P&L gate: red strategy×regime $ must block accuracy UP."""

from unittest import mock

import config
from arena import core_lane_tuner


def test_pnl_gate_blocks_up_with_low_regime_n(monkeypatch):
    """With CORE_TUNE_PNL_MIN_TRADES_REGIME=5, n=5 red $ blocks UP."""
    monkeypatch.setattr(config, "CORE_TUNE_PNL_GATE", True, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_PNL_MIN_TRADES", 15, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_PNL_MIN_TRADES_REGIME", 5, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_HIGH_ACC", 0.56, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_LOW_ACC", 0.48, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_STEP", 0.05, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_MIN_TRADES", 40, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_MIN_TRADES_REGIME", 40, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_ENABLED", True, raising=False)

    # Attribution: high acc on drift for momentum with n≥40
    attribution = {
        "momentum": {
            "drift": {"n": 59, "accuracy": 0.593, "correct": 35, "wrong": 24},
        },
    }
    # Only 5 resolved trades in regime — previously below global min 15
    strat_pnl = {
        "momentum": {"n": 5, "wins": 1, "pnl": -8.0, "wr": 0.2},
    }

    monkeypatch.setattr(
        core_lane_tuner, "compute_core_attribution",
        lambda *a, **k: attribution,
    )
    monkeypatch.setattr(
        core_lane_tuner, "_strategy_regime_pnl",
        lambda *a, **k: strat_pnl,
    )
    monkeypatch.setattr(
        core_lane_tuner, "live_tune_lanes",
        lambda *a, **k: ["drift"],
    )

    import db
    monkeypatch.setattr(db, "get_auto_core_tune", lambda: True)
    monkeypatch.setattr(db, "get_lane_overrides", lambda: {
        "drift": {
            "enabled": True,
            "core": True,
            "profile": {"momentum": 0.75},
            "by_regime": {
                "low_vol_trend": {"momentum": 0.75},
            },
        },
    })
    monkeypatch.setattr(db, "get_regime_conditioning", lambda: False)

    class _Det:
        def status(self):
            return {"current": {"regime_id": "low_vol_trend"}}

    monkeypatch.setattr(
        "signals.regime_detector.get_detector", lambda: _Det(), raising=False,
    )
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: name == "profile_adapt",
        raising=False,
    )

    # Avoid DB writes
    monkeypatch.setattr(db, "set_arena_state", lambda *a, **k: None)
    monkeypatch.setattr(
        "bots.base_bot.BaseBot.STRATEGY_SIGNAL_PROFILE",
        {"momentum": {"drift": 0.55, "mom": 0.3, "strat": 0.15}},
        raising=False,
    )

    # Fake conn context for compute path
    class _Conn:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def execute(self, *a, **k):
            class R:
                def fetchall(self):
                    return []
                def fetchone(self):
                    return None
            return R()

    monkeypatch.setattr(db, "get_conn", lambda: _Conn())

    report = core_lane_tuner.tune()
    drift = (report.get("lanes") or {}).get("drift") or {}
    mom = drift.get("momentum") or {}
    # Accuracy would UP 0.75→0.80, but P&L gate must hold or revert
    assert mom.get("action") in ("hold_pnl_gate", "pnl_revert", "hold")
    assert mom.get("action") != "up"

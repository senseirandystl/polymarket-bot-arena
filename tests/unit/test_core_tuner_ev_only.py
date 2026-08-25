"""Core-lane tuner: every UP/DOWN is EV-triggered; accuracy is an UP veto only."""

from unittest import mock

import config
from arena import core_lane_tuner


def _conn():
    class R:
        def fetchall(self):
            return []

        def fetchone(self):
            return None

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            return R()

    return _Conn()


def _patch_common(monkeypatch, *, lane="mom", attribution=None,
                  scorecard=None, pnl=None, current=0.40, apply=False):
    monkeypatch.setattr(config, "CORE_TUNE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_EV_PRIMARY", True, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_PNL_GATE", True, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_MIN_TRADES", 40, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_MIN_TRADES_REGIME", 40, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_EV_MIN_TRADES", 20, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_EV_UP_MIN", 0.0, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_EV_DOWN_MAX", -0.05, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_UP_ACC_FLOOR", 0.50, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_HIGH_ACC", 0.56, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_LOW_ACC", 0.48, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_STEP", 0.05, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_RESET_SEED_ON_RED", True, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_SCORECARD_MIN", 20, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_SCORECARD_DOWN_MAX", 0.0, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_SCORECARD_FORCE_DOWN", -0.005, raising=False)
    monkeypatch.setattr(config, "CORE_TUNE_NEVER_CUT_DRIFT", True, raising=False)

    monkeypatch.setattr(
        core_lane_tuner, "compute_core_attribution",
        lambda *a, **k: attribution or {},
    )
    monkeypatch.setattr(core_lane_tuner, "_strategy_regime_pnl",
                        lambda *a, **k: pnl or {})
    monkeypatch.setattr(core_lane_tuner, "_strategy_global_pnl",
                        lambda *a, **k: pnl or {})
    sc = {} if scorecard is None else scorecard
    monkeypatch.setattr(
        core_lane_tuner, "_scorecard_net_by_strategy",
        lambda hours=None: sc,
    )
    monkeypatch.setattr(core_lane_tuner, "live_tune_lanes",
                        lambda *a, **k: [lane])

    import db
    monkeypatch.setattr(db, "get_auto_core_tune", lambda: apply)
    monkeypatch.setattr(db, "get_lane_overrides", lambda: {
        lane: {
            "enabled": True,
            "core": True,
            "profile": {"momentum": current},
        },
    })
    monkeypatch.setattr(db, "get_regime_conditioning", lambda: False)
    monkeypatch.setattr(db, "set_arena_state", lambda *a, **k: None)
    monkeypatch.setattr(db, "get_conn", lambda: _conn())

    class _Det:
        def status(self):
            return {"current": {"regime_id": "normal"}}

    monkeypatch.setattr(
        "signals.regime_detector.get_detector", lambda: _Det(), raising=False,
    )
    monkeypatch.setattr(
        "arena.regime_settings.get_bool",
        lambda name: False,
        raising=False,
    )
    monkeypatch.setattr(
        "bots.base_bot.BaseBot.STRATEGY_SIGNAL_PROFILE",
        {"momentum": {"drift": 0.55, "mom": 0.20, "strat": 0.15}},
        raising=False,
    )


def _row(report, lane, strat="momentum"):
    return ((report.get("lanes") or {}).get(lane) or {}).get(strat) or {}


def test_missing_ev_high_acc_does_not_up(monkeypatch):
    """Soak bug: acc 0.57 with mean_ev=None used to UP mom 0.20→0.40."""
    _patch_common(monkeypatch, lane="mom", current=0.40, attribution={
        "momentum": {
            "mom": {"n": 80, "accuracy": 0.57, "n_ev": 0, "mean_ev": None},
        },
    }, scorecard={
        "momentum": {"mom": {"n_priced": 5, "net_edge": None}},
    })
    row = _row(core_lane_tuner.tune(), "mom")
    assert row.get("action") != "up"
    assert row.get("suggested") <= row.get("current")


def test_red_ev_downs_mom_even_if_acc_high(monkeypatch):
    _patch_common(monkeypatch, lane="mom", current=0.40, attribution={
        "momentum": {
            "mom": {"n": 80, "accuracy": 0.57, "n_ev": 25, "mean_ev": -0.08},
        },
    }, scorecard={
        "momentum": {"mom": {"n_priced": 40, "net_edge": -0.014}},
    })
    row = _row(core_lane_tuner.tune(), "mom")
    assert row.get("action") in ("ev_down", "reset_seed")
    assert row.get("suggested") < row.get("current")
    assert row.get("suggested") <= 0.20 + 1e-9  # seed snap or step toward seed


def test_green_ev_and_acc_allows_up(monkeypatch):
    _patch_common(monkeypatch, lane="mom", current=0.20, attribution={
        "momentum": {
            "mom": {"n": 80, "accuracy": 0.57, "n_ev": 30, "mean_ev": 0.04},
        },
    }, scorecard={
        "momentum": {"mom": {"n_priced": 40, "net_edge": 0.01}},
    }, pnl={"momentum": {"n": 20, "pnl": 5.0, "wins": 12, "wr": 0.6}})
    row = _row(core_lane_tuner.tune(), "mom")
    assert row.get("action") == "up"
    assert row.get("suggested") > row.get("current")


def test_scorecard_outage_green_ev_holds_not_revert(monkeypatch):
    """Missing scorecard must block UP, not cut a green-EV elevated weight."""
    _patch_common(monkeypatch, lane="mom", current=0.40, attribution={
        "momentum": {
            "mom": {"n": 80, "accuracy": 0.57, "n_ev": 30, "mean_ev": 0.04},
        },
    }, pnl={"momentum": {"n": 20, "pnl": 5.0}})
    from arena import core_lane_tuner
    monkeypatch.setattr(
        core_lane_tuner, "_scorecard_net_by_strategy", lambda hours=None: None,
    )
    row = _row(core_lane_tuner.tune(), "mom")
    assert row.get("action") != "up"
    assert row.get("action") not in ("ev_revert", "ev_down", "reset_seed", "pnl_revert")
    assert row.get("suggested") == row.get("current")


def test_green_ev_low_acc_vetoes_up(monkeypatch):
    _patch_common(monkeypatch, lane="mom", current=0.20, attribution={
        "momentum": {
            "mom": {"n": 80, "accuracy": 0.45, "n_ev": 30, "mean_ev": 0.10},
        },
    }, scorecard={
        "momentum": {"mom": {"n_priced": 40, "net_edge": 0.02}},
    }, pnl={"momentum": {"n": 20, "pnl": 8.0}})
    row = _row(core_lane_tuner.tune(), "mom")
    assert row.get("action") != "up"
    assert row.get("suggested") == row.get("current")


def test_low_acc_does_not_down_without_red_ev(monkeypatch):
    """Accuracy is not a DOWN trigger — collecting/hold if EV is not red."""
    _patch_common(monkeypatch, lane="mom", current=0.20, attribution={
        "momentum": {
            "mom": {"n": 80, "accuracy": 0.40, "n_ev": 30, "mean_ev": 0.01},
        },
    }, scorecard={
        "momentum": {"mom": {"n_priced": 40, "net_edge": 0.002}},
    })
    row = _row(core_lane_tuner.tune(), "mom")
    assert row.get("action") not in ("down", "ev_down", "revert")
    assert row.get("suggested") == row.get("current")


def test_red_ev_drift_does_not_cut_below_floor(monkeypatch):
    _patch_common(monkeypatch, lane="drift", current=0.75, attribution={
        "momentum": {
            "drift": {"n": 80, "accuracy": 0.76, "n_ev": 30, "mean_ev": -0.08},
        },
    }, scorecard={
        "momentum": {"drift": {"n_priced": 40, "net_edge": -0.006}},
    })
    row = _row(core_lane_tuner.tune(), "drift")
    assert row.get("action") in ("hold_pnl_gate", "reset_seed", "hold")
    if row.get("action") == "reset_seed":
        assert row.get("suggested") >= 0.05
    else:
        assert row.get("suggested") == row.get("current")


def test_core_lane_attribution_includes_mean_ev_from_hyp_pnl(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "ev.db")
    db_module.init_db()
    with db_module.get_conn() as conn:
        conn.execute(
            """INSERT INTO decision_events
               (bot_name, strategy_type, market_id, action, side, drift, mom,
                market_up, hyp_pnl, entry_price)
               VALUES ('momentum-v1','momentum','m1','buy','yes', 0.4, 0.2, 1, 0.12, 0.45)"""
        )
        conn.execute(
            """INSERT INTO decision_events
               (bot_name, strategy_type, market_id, action, side, drift, mom,
                market_up, hyp_pnl, entry_price)
               VALUES ('momentum-v1','momentum','m2','buy','yes', 0.4, 0.2, 0, -0.20, 0.56)"""
        )
        conn.commit()
    from arena.decision_log import core_lane_attribution
    with db_module.get_conn() as conn:
        attr = core_lane_attribution(conn, 0.05)
    mom = (attr.get("momentum") or {}).get("mom") or {}
    assert mom.get("n", 0) >= 1
    assert "mean_ev" in mom
    assert mom.get("n_ev", 0) >= 1

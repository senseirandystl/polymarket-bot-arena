"""Gate tuner: only loosen a gate when unique-market hyp P&L is known and cheap."""

import config
import db
from arena import gate_tuner


def test_does_not_loosen_expensive_high_wr_gate(monkeypatch):
    """95% WR at 93c is sweeper territory — not a directional loosen."""
    report = {
        "gates": {
            "no_side_gate": {
                "markets": 80, "n_hyp": 80, "wr": 0.95, "avg_hyp_pnl": 0.021,
                "avg_entry": 0.93,
            }
        }
    }
    monkeypatch.setattr(gate_tuner, "_load_scorecard", lambda hours=None: report)
    out = gate_tuner.suggest(apply=False)
    no_side = out["suggestions"].get("NO_SIDE_MAX_MID")
    assert no_side is None or no_side["action"] == "hold"


def test_suggests_loosen_dead_zone_when_hyp_positive_and_cheap(monkeypatch):
    report = {
        "gates": {
            "dead_zone": {
                "markets": 40, "n_hyp": 40, "wr": 0.67, "avg_hyp_pnl": 0.16,
                "avg_entry": 0.51,
            }
        }
    }
    monkeypatch.setattr(gate_tuner, "_load_scorecard", lambda hours=None: report)
    out = gate_tuner.suggest(apply=False)
    dz = out["suggestions"]["DEAD_ZONE_DRIFT_MIN"]
    assert dz["action"] == "loosen"
    assert dz["suggested"] < float(config.DEAD_ZONE_DRIFT_MIN)


def test_suggests_loosen_dual_gate_when_hyp_positive_and_cheap(monkeypatch):
    """Priced dual-gate skips that would have been cheap + profitable move Z."""
    report = {
        "gates": {
            "drift_dual_gate": {
                "markets": 80, "n_hyp": 80, "wr": 0.70, "avg_hyp_pnl": 0.04,
                "avg_entry": 0.48,
            }
        }
    }
    monkeypatch.setattr(gate_tuner, "_load_scorecard", lambda hours=None: report)
    out = gate_tuner.suggest(apply=False)
    z = out["suggestions"]["DRIFT_MIN_ABS_Z"]
    assert z["action"] == "loosen"
    assert z["suggested"] < float(config.DRIFT_MIN_ABS_Z)


def test_apply_writes_dual_gate_z_override(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "gt-z.db")
    db.init_db()
    gate_tuner._OV_CACHE = (0.0, {})
    report = {
        "gates": {
            "drift_dual_gate": {
                "markets": 80, "n_hyp": 80, "wr": 0.70, "avg_hyp_pnl": 0.04,
                "avg_entry": 0.48,
            }
        }
    }
    monkeypatch.setattr(gate_tuner, "_load_scorecard", lambda hours=None: report)
    out = gate_tuner.suggest(apply=True)
    ov = gate_tuner.load_overrides()
    assert "DRIFT_MIN_ABS_Z" in ov
    assert ov["DRIFT_MIN_ABS_Z"] < float(config.DRIFT_MIN_ABS_Z)
    assert out["applied"] is True


def test_skips_dual_gate_without_hyp_pnl(monkeypatch):
    report = {
        "gates": {
            "drift_dual_gate": {
                "markets": 300, "wr": 0.81, "avg_hyp_pnl": None,
                "avg_entry": None,
            }
        }
    }
    monkeypatch.setattr(gate_tuner, "_load_scorecard", lambda hours=None: report)
    out = gate_tuner.suggest(apply=False)
    assert "DRIFT_MIN_ABS_Z" not in out["suggestions"] or (
        out["suggestions"]["DRIFT_MIN_ABS_Z"]["action"] == "hold"
    )


def test_apply_writes_override_within_band(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "gt.db")
    db.init_db()
    report = {
        "gates": {
            "dead_zone": {
                "markets": 40, "n_hyp": 40, "wr": 0.70, "avg_hyp_pnl": 0.12,
                "avg_entry": 0.50,
            }
        }
    }
    monkeypatch.setattr(gate_tuner, "_load_scorecard", lambda hours=None: report)
    out = gate_tuner.suggest(apply=True)
    ov = gate_tuner.load_overrides()
    assert "DEAD_ZONE_DRIFT_MIN" in ov
    assert ov["DEAD_ZONE_DRIFT_MIN"] < float(config.DEAD_ZONE_DRIFT_MIN)
    assert out["applied"] is True

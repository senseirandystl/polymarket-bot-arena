"""Paper gate profile (Pass A) — live constants stay conservative."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config


def test_paper_profile_loosens_z_and_lean_only_in_paper():
    assert config.TRADING_MODE == "paper"
    # Profit-mode default is off; activate data_gather_v1 for this check.
    assert not config.paper_gates_active()
    old = config.PAPER_GATE_PROFILE
    try:
        config.PAPER_GATE_PROFILE = "data_gather_v1"
        assert config.paper_gates_active()
        assert config.DRIFT_MIN_ABS_Z == 0.35  # live/base unchanged
        assert config.effective_float("DRIFT_MIN_ABS_Z") == 0.28
        assert config.effective_float("MODEL_LEAN_MIN") == 0.04
        assert config.effective_float("MIN_EDGE_DEFAULT") == 0.010
        # PCT intentionally not loosened in data_gather_v1
        assert "DRIFT_MIN_ABS_PCT" not in config.PAPER_GATE_OVERRIDES["data_gather_v1"]
        assert config.effective_float("DRIFT_MIN_ABS_PCT", config.DRIFT_MIN_ABS_PCT) == config.DRIFT_MIN_ABS_PCT
    finally:
        config.PAPER_GATE_PROFILE = old


def test_paper_profile_off_uses_base():
    old = config.PAPER_GATE_PROFILE
    try:
        config.PAPER_GATE_PROFILE = "off"
        assert not config.paper_gates_active()
        assert config.effective_float("DRIFT_MIN_ABS_Z", config.DRIFT_MIN_ABS_Z) == config.DRIFT_MIN_ABS_Z
    finally:
        config.PAPER_GATE_PROFILE = old


def test_live_mode_ignores_paper_profile():
    old_mode = config.TRADING_MODE
    old_prof = config.PAPER_GATE_PROFILE
    try:
        config.PAPER_GATE_PROFILE = "data_gather_v1"
        config.TRADING_MODE = "live"
        assert not config.paper_gates_active()
        assert config.effective_float("DRIFT_MIN_ABS_Z", 0.35) == 0.35
    finally:
        config.TRADING_MODE = old_mode
        config.PAPER_GATE_PROFILE = old_prof

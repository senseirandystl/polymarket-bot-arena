"""Config fail-fast validation + env-override helpers (slice F).

config.py validates its safety invariants at import; these exercise the model
and the env-override caster directly so a bad edit is caught in CI, not at
arena startup against (simulated) money.
"""

import pytest
from pydantic import ValidationError

import config


def _valid_kwargs(**over):
    base = dict(
        trading_mode="paper",
        taker_fee_rate=0.07,
        kelly_fraction=0.25,
        model_lean_min=0.05,
        model_conviction_scale=0.06,
        book_sum_tolerance=0.04,
        consensus_guard=0.35,
        high_price_guard=0.72,
        dead_zone_lo=0.42,
        dead_zone_hi=0.58,
        market_side_exposure_cap=0.10,
        paper_bankroll=200.0,
        live_max_position=10.0,
        evolution_window_hours=24.0,
        trade_loop_interval_sec=1.0,
        market_data_interval_sec=1.0,
        http_max_retries=2,
    )
    base.update(over)
    return base


def test_current_config_is_valid():
    # Import already ran _validate_config(); re-run to be explicit.
    config._validate_config()  # must not raise


def test_valid_kwargs_construct():
    config._ConfigInvariants(**_valid_kwargs())


def test_consensus_must_be_below_high_price_guard():
    with pytest.raises(ValidationError):
        config._ConfigInvariants(**_valid_kwargs(consensus_guard=0.80))


def test_dead_zone_lo_below_hi():
    with pytest.raises(ValidationError):
        config._ConfigInvariants(**_valid_kwargs(dead_zone_lo=0.60))


def test_kelly_fraction_bounded():
    with pytest.raises(ValidationError):
        config._ConfigInvariants(**_valid_kwargs(kelly_fraction=2.0))
    with pytest.raises(ValidationError):
        config._ConfigInvariants(**_valid_kwargs(kelly_fraction=0.0))


def test_fee_rate_bounded():
    with pytest.raises(ValidationError):
        config._ConfigInvariants(**_valid_kwargs(taker_fee_rate=1.5))


def test_trading_mode_enum():
    with pytest.raises(ValidationError):
        config._ConfigInvariants(**_valid_kwargs(trading_mode="turbo"))


def test_negative_retries_rejected():
    with pytest.raises(ValidationError):
        config._ConfigInvariants(**_valid_kwargs(http_max_retries=-1))


def test_env_num_passthrough_when_unset():
    assert config._env_num("ARENA_DEFINITELY_UNSET_XYZ", 3.5, float) == 3.5


def test_env_num_casts(monkeypatch):
    monkeypatch.setenv("ARENA_TEST_KNOB", "0.42")
    assert config._env_num("ARENA_TEST_KNOB", 1.0, float) == 0.42


def test_env_num_malformed_raises(monkeypatch):
    monkeypatch.setenv("ARENA_TEST_KNOB", "not-a-number")
    with pytest.raises(RuntimeError):
        config._env_num("ARENA_TEST_KNOB", 1.0, float)

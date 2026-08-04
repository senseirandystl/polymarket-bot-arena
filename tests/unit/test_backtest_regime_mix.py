"""GA backtest gate regime-mix check."""

from __future__ import annotations

import pytest

import config
from evolution.backtest_gate import GateResult, clear_cache, evaluate_offspring


class _Bot:
    def __init__(self, name="c"):
        self.name = name
        self.strategy_type = "momentum"


def test_regime_mix_rejects_worse_in_live_regime(monkeypatch):
    monkeypatch.setattr(config, "GA_BACKTEST_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_REGIME_MIX", True, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_BEAT_BASELINE", True, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_EPS", 0.5, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_REGIME_EPS", 0.5, raising=False)
    clear_cache()

    class Data:
        markets = [1, 2, 3]

    def load_fn(n):
        return Data()

    def run_fn(bot, data):
        if bot.name == "child":
            # Better overall, worse in chop
            return (10.0, {"high_vol_chop": -5.0, "high_vol_trend": 15.0})
        return (5.0, {"high_vol_chop": 2.0, "high_vol_trend": 3.0})

    child = _Bot("child")
    base = _Bot("base")
    res = evaluate_offspring(
        child, baseline_bot=base, load_fn=load_fn, run_fn=run_fn,
        live_regime="high_vol_chop",
    )
    assert res.passed is False
    assert res.reason == "worse_in_live_regime"


def test_regime_mix_passes_when_better_in_regime(monkeypatch):
    monkeypatch.setattr(config, "GA_BACKTEST_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_REGIME_MIX", True, raising=False)
    monkeypatch.setattr(config, "GA_BACKTEST_BEAT_BASELINE", True, raising=False)
    clear_cache()

    class Data:
        markets = [1]

    def run_fn(bot, data):
        if bot.name == "child":
            return (8.0, {"low_vol_range": 3.0})
        return (5.0, {"low_vol_range": 1.0})

    res = evaluate_offspring(
        _Bot("child"), baseline_bot=_Bot("base"),
        load_fn=lambda n: Data(), run_fn=run_fn,
        live_regime="low_vol_range",
    )
    assert res.passed is True

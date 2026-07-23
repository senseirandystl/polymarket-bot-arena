"""Centralized Risk Engine — limits, VaR, kill switch, pre_trade gates."""

from unittest import mock

import pytest

import config
from arena import risk_engine


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_max_drawdown_pct():
    # Run-up then 50% retrace of peak
    pnls = [10, 10, 10, -15]  # equity 30 peak, then 15 → dd 0.5
    assert risk_engine.max_drawdown_pct(pnls) == pytest.approx(0.5)
    assert risk_engine.max_drawdown_pct([]) == 0.0


def test_equity_stats():
    s = risk_engine.equity_stats([5, -2, 1])
    assert s["equity"] == pytest.approx(4.0)
    assert s["peak"] == pytest.approx(5.0)
    assert s["drawdown"] == pytest.approx(0.2)
    assert s["n"] == 3


def test_historical_var_needs_samples():
    assert risk_engine.historical_var([1.0] * 5, 0.95) is None
    # Many small wins + a few large losses → positive VaR
    pnls = [0.5] * 30 + [-5.0, -4.0, -3.0]
    with mock.patch.object(config, "RISK_VAR_MIN_TRADES", 20):
        var = risk_engine.historical_var(pnls, 0.95)
    assert var is not None
    assert var > 0


def test_dd_size_mult_taper():
    # No reduction below start
    assert risk_engine._dd_size_mult(0.10, 0.40, 0.50, 0.25) == 1.0
    # At max DD → min mult
    assert risk_engine._dd_size_mult(0.40, 0.40, 0.50, 0.25) == pytest.approx(0.25)
    # Midway between start (0.20) and max (0.40)
    mid = risk_engine._dd_size_mult(0.30, 0.40, 0.50, 0.25)
    assert 0.25 < mid < 1.0


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_arms_and_blocks(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(risk_engine.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(risk_engine.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(risk_engine.db, "log_risk_event",
                        lambda **kw: 1)
    kill_file = tmp_path / "KILL_SWITCH"
    monkeypatch.setattr(config, "RISK_KILL_SWITCH_FILE", str(kill_file))
    risk_engine.bust_cache()

    state = risk_engine.set_kill_switch(True, reason="test", source="unit")
    assert state["kill_switch"] is True
    assert kill_file.is_file()
    risk_engine.bust_cache()
    assert risk_engine.is_killed() is True

    d = risk_engine.pre_trade("bot-a", mode="paper", amount=10.0)
    assert d.allow is False
    assert d.action == "kill"
    assert d.reason == "kill_switch"

    risk_engine.set_kill_switch(False, reason="clear", source="unit")
    risk_engine.bust_cache()
    assert risk_engine.is_killed() is False
    assert not kill_file.is_file()


def test_file_kill_switch(monkeypatch, tmp_path):
    saved = {}
    monkeypatch.setattr(risk_engine.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(risk_engine.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(risk_engine.db, "log_risk_event", lambda **kw: 1)
    kill_file = tmp_path / "KILL_SWITCH"
    kill_file.write_text("kill\n")
    monkeypatch.setattr(config, "RISK_KILL_SWITCH_FILE", str(kill_file))
    risk_engine.bust_cache()
    assert risk_engine.is_killed() is True


# ---------------------------------------------------------------------------
# Evaluate + pre_trade
# ---------------------------------------------------------------------------

def test_evaluate_pauses_on_daily_loss(monkeypatch):
    saved = {}
    events = []

    monkeypatch.setattr(risk_engine.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(risk_engine.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(risk_engine.db, "log_risk_event",
                        lambda **kw: events.append(kw) or 1)
    monkeypatch.setattr(risk_engine.db, "get_active_bots",
                        lambda: [{"bot_name": "loser"}])
    monkeypatch.setattr(config, "RISK_PAPER_BOT_DAILY_LOSS", 20.0)
    monkeypatch.setattr(config, "RISK_BOT_DAILY_LOSS", 20.0)

    # Daily P&L = -30 → over limit
    monkeypatch.setattr(
        risk_engine, "_pnls_for_bots",
        lambda names, hours=None, today_only=False, mode=None: {
            "loser": [-10.0, -20.0] if today_only else [-10.0, -20.0],
        },
    )
    monkeypatch.setattr(risk_engine, "_portfolio_pnls",
                        lambda hours=None, today_only=False, mode=None: [-10.0, -20.0])
    monkeypatch.setattr(risk_engine, "_file_kill_armed", lambda: False)

    state = risk_engine.evaluate(bot_names=["loser"], mode="paper")
    bot = state["bots"]["loser"]
    assert bot["status"] == "paused"
    assert bot["size_mult"] == 0.0
    assert "bot_daily_loss" in (bot["reason"] or "")

    risk_engine.bust_cache()
    d = risk_engine.pre_trade("loser", mode="paper", amount=5.0)
    assert d.allow is False
    assert d.action == "pause"


def test_evaluate_size_reduction_on_drawdown(monkeypatch):
    saved = {}
    monkeypatch.setattr(risk_engine.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(risk_engine.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(risk_engine.db, "log_risk_event", lambda **kw: 1)
    monkeypatch.setattr(risk_engine.db, "get_active_bots",
                        lambda: [{"bot_name": "fader"}])
    # High max DD so we taper but don't pause
    monkeypatch.setattr(config, "RISK_BOT_MAX_DRAWDOWN", 0.50)
    monkeypatch.setattr(config, "RISK_SIZE_REDUCE_DD_FRAC", 0.40)
    monkeypatch.setattr(config, "RISK_BOT_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PAPER_BOT_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PAPER_PORTFOLIO_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PORTFOLIO_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PORTFOLIO_MAX_DRAWDOWN", 0.99)
    monkeypatch.setattr(config, "RISK_UNDERPERFORM_PAUSE_PNL", -9999.0)

    # Equity path: climb to 20 then drop to 12 → dd = 8/20 = 0.40
    # start taper at 0.4*0.5=0.20, so at 0.40 we're mid-taper
    series = [5, 5, 5, 5, -8]
    monkeypatch.setattr(
        risk_engine, "_pnls_for_bots",
        lambda names, hours=None, today_only=False, mode=None: {
            "fader": [] if today_only else series,
        },
    )
    monkeypatch.setattr(risk_engine, "_portfolio_pnls",
                        lambda hours=None, today_only=False, mode=None: (
                            [] if today_only else series))
    monkeypatch.setattr(risk_engine, "_file_kill_armed", lambda: False)

    state = risk_engine.evaluate(bot_names=["fader"], mode="paper")
    bot = state["bots"]["fader"]
    assert bot["status"] in ("reduced", "active")
    # At 40% DD with max 50% and start at 50% of max (25%), should be reduced
    assert bot["drawdown"] == pytest.approx(0.4, abs=0.05)
    if bot["status"] == "reduced":
        assert 0.0 < bot["size_mult"] < 1.0


def test_pre_trade_allow_when_healthy(monkeypatch):
    state = {
        "enabled": True,
        "kill_switch": False,
        "limits": risk_engine._default_limits("paper"),
        "bots": {
            "ok-bot": {
                "status": "active", "size_mult": 1.0,
                "daily_pnl": 5.0, "reason": None,
            },
        },
        "portfolio": {"status": "active", "size_mult": 1.0, "daily_pnl": 10.0},
    }
    monkeypatch.setattr(risk_engine, "load_state", lambda: state)
    monkeypatch.setattr(risk_engine, "_file_kill_armed", lambda: False)
    monkeypatch.setattr(risk_engine, "_pnls_for_bots",
                        lambda *a, **k: {"ok-bot": [1.0, 2.0]})
    monkeypatch.setattr(risk_engine, "_portfolio_pnls",
                        lambda *a, **k: [1.0, 2.0])
    risk_engine.bust_cache()
    # Force cache to use our load_state
    risk_engine._cache = (0.0, {}, False)

    d = risk_engine.pre_trade("ok-bot", mode="paper", amount=3.0)
    assert d.allow is True
    assert d.size_mult == pytest.approx(1.0)


def test_size_multiplier_combines_bot_and_portfolio(monkeypatch):
    state = {
        "enabled": True,
        "kill_switch": False,
        "bots": {"a": {"status": "reduced", "size_mult": 0.5}},
        "portfolio": {"status": "reduced", "size_mult": 0.8},
        "limits": {},
    }
    monkeypatch.setattr(risk_engine, "load_state", lambda: state)
    monkeypatch.setattr(risk_engine, "_file_kill_armed", lambda: False)
    risk_engine._cache = (0.0, {}, False)
    # Refresh cache via _cached_state path
    mult = risk_engine.size_multiplier("a")
    assert mult == pytest.approx(0.4)


def test_manual_pause_sticky(monkeypatch):
    saved = {}
    monkeypatch.setattr(risk_engine.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(risk_engine.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(risk_engine.db, "log_risk_event", lambda **kw: 1)
    monkeypatch.setattr(risk_engine, "_file_kill_armed", lambda: False)

    risk_engine.pause_bot("x", reason="ops")
    import json
    raw = json.loads(saved[risk_engine.STATE_KEY])
    assert raw["bots"]["x"]["manual_pause"] is True
    assert raw["bots"]["x"]["status"] == "paused"


def test_legacy_daily_check_when_disabled(monkeypatch):
    state = {
        "enabled": False,
        "kill_switch": False,
        "bots": {},
        "portfolio": {},
        "limits": {},
    }
    monkeypatch.setattr(risk_engine, "load_state", lambda: state)
    monkeypatch.setattr(risk_engine, "_file_kill_armed", lambda: False)
    monkeypatch.setattr(risk_engine.db, "get_bot_daily_loss",
                        lambda n, m: 100.0)
    monkeypatch.setattr(risk_engine.db, "get_total_daily_loss",
                        lambda m: 0.0)
    monkeypatch.setattr(config, "get_max_daily_loss_per_bot",
                        lambda: 50.0)
    monkeypatch.setattr(config, "get_max_daily_loss_total",
                        lambda: 200.0)
    risk_engine._cache = (0.0, {}, False)

    d = risk_engine.pre_trade("b", mode="paper")
    assert d.allow is False
    assert d.reason == "daily_loss_limit"

"""Centralized Risk Engine — limits, VaR, kill switch, pre_trade gates."""

from unittest import mock

import pytest

import config
from arena import risk_engine


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_max_drawdown_pct():
    # Run-up then 50% retrace of peak (zero-based, still valid when start=0)
    pnls = [10, 10, 10, -15]  # equity 30 peak, then 15 → dd 0.5
    assert risk_engine.max_drawdown_pct(pnls) == pytest.approx(0.5)
    assert risk_engine.max_drawdown_pct([]) == 0.0


def test_max_drawdown_pct_bankroll_anchored():
    # Pure losses on $1000 base → ~1.45% DD, not 100%
    pnls = [-5.0, -5.0, -4.53]
    dd = risk_engine.max_drawdown_pct(pnls, starting_equity=1000.0)
    assert dd == pytest.approx(14.53 / 1000.0, rel=1e-6)
    # Without capital base, pure losses no longer invent a false 100%
    assert risk_engine.max_drawdown_pct(pnls, starting_equity=0.0) == 0.0


def test_equity_stats():
    s = risk_engine.equity_stats([5, -2, 1])
    assert s["equity"] == pytest.approx(4.0)
    assert s["peak"] == pytest.approx(5.0)
    assert s["drawdown"] == pytest.approx(0.2)
    assert s["n"] == 3


def test_equity_stats_bankroll_anchored_pure_loss():
    """Regression: -$14.53 on $1000 bankroll must not report 100% DD."""
    pnls = [-2.0, -3.5, -4.0, -5.03]
    s = risk_engine.equity_stats(pnls, starting_equity=1000.0)
    assert s["starting_equity"] == pytest.approx(1000.0)
    assert s["peak"] == pytest.approx(1000.0)
    assert s["equity"] == pytest.approx(985.47)
    # equity_stats rounds drawdown to 4 dp
    assert s["drawdown"] == pytest.approx(round(14.53 / 1000.0, 4))
    assert s["drawdown"] < 0.40  # under portfolio max DD
    assert s["drawdown"] != pytest.approx(1.0)


def test_equity_stats_bankroll_anchored_runup_then_drawdown():
    # Start 1000, +50 peak 1050, then -80 → equity 970, dd = 80/1050
    s = risk_engine.equity_stats([50.0, -80.0], starting_equity=1000.0)
    assert s["peak"] == pytest.approx(1050.0)
    assert s["equity"] == pytest.approx(970.0)
    assert s["drawdown"] == pytest.approx(round(80.0 / 1050.0, 4))


def test_window_start_equity_reconstruction():
    capital_now = 985.47
    pnls = [-5.0, -5.0, -4.53]
    start = risk_engine._window_start_equity(pnls, capital_now)
    assert start == pytest.approx(1000.0)


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
    monkeypatch.setattr(risk_engine, "_capital_now", lambda: 1000.0)
    monkeypatch.setattr(risk_engine, "_bot_capital_weight", lambda _n: 1.0)

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

    # Bankroll-anchored: capital_now=60, series sum=-40 → window start=100.
    # Path 100→92→…→60, peak=100, dd=0.40. Taper starts at 0.5*0.50=0.25.
    series = [-8.0, -8.0, -8.0, -8.0, -8.0]
    monkeypatch.setattr(risk_engine, "_capital_now", lambda: 60.0)
    monkeypatch.setattr(risk_engine, "_bot_capital_weight", lambda _n: 1.0)
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


def test_evaluate_does_not_pause_on_small_pure_loss(monkeypatch):
    """User bug: -$14.53 on $1000 bankroll must not trip portfolio_max_drawdown."""
    saved = {}
    monkeypatch.setattr(risk_engine.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(risk_engine.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(risk_engine.db, "log_risk_event", lambda **kw: 1)
    monkeypatch.setattr(risk_engine.db, "get_active_bots",
                        lambda: [{"bot_name": "a"}, {"bot_name": "b"}])
    monkeypatch.setattr(config, "RISK_BOT_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PAPER_BOT_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PAPER_PORTFOLIO_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PORTFOLIO_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PORTFOLIO_MAX_DRAWDOWN", 0.40)
    monkeypatch.setattr(config, "RISK_BOT_MAX_DRAWDOWN", 0.35)
    monkeypatch.setattr(config, "RISK_UNDERPERFORM_PAUSE_PNL", -9999.0)

    # 12 small losses totaling -14.53 (n>=10 so portfolio DD gate is eligible)
    losses = [-1.21] * 11 + [-1.22]  # sum ≈ -14.53
    assert abs(sum(losses) + 14.53) < 0.01
    # capital_now after losses; window start reconstructs to ~1000
    monkeypatch.setattr(risk_engine, "_capital_now",
                        lambda: 1000.0 + sum(losses))
    monkeypatch.setattr(risk_engine, "_bot_capital_weight", lambda _n: 1.0)
    monkeypatch.setattr(
        risk_engine, "_pnls_for_bots",
        lambda names, hours=None, today_only=False, mode=None: {
            n: (list(losses) if not today_only else list(losses))
            for n in names
        },
    )
    # Portfolio series = chronological pool P&Ls (same losses once)
    monkeypatch.setattr(
        risk_engine, "_portfolio_pnls",
        lambda hours=None, today_only=False, mode=None: list(losses),
    )
    monkeypatch.setattr(risk_engine, "_file_kill_armed", lambda: False)

    state = risk_engine.evaluate(bot_names=["a", "b"], mode="paper")
    port = state["portfolio"]
    assert port["drawdown"] == pytest.approx(round(abs(sum(losses)) / 1000.0, 4),
                                             abs=1e-4)
    assert port["drawdown"] < 0.05  # ~1.5%, nowhere near 40%
    assert port["status"] == "active"
    assert port["size_mult"] == pytest.approx(1.0)
    assert port.get("reason") is None


def test_evaluate_pauses_portfolio_on_real_drawdown(monkeypatch):
    """Large loss vs bankroll still trips portfolio_max_drawdown."""
    saved = {}
    monkeypatch.setattr(risk_engine.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(risk_engine.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(risk_engine.db, "log_risk_event", lambda **kw: 1)
    monkeypatch.setattr(risk_engine.db, "get_active_bots",
                        lambda: [{"bot_name": "deep"}])
    monkeypatch.setattr(config, "RISK_BOT_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PAPER_BOT_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PAPER_PORTFOLIO_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PORTFOLIO_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PORTFOLIO_MAX_DRAWDOWN", 0.40)
    monkeypatch.setattr(config, "RISK_BOT_MAX_DRAWDOWN", 0.99)
    monkeypatch.setattr(config, "RISK_UNDERPERFORM_PAUSE_PNL", -9999.0)

    # 10 equal losses of $50 on $1000 → 50% DD ≥ 40%
    series = [-50.0] * 10
    monkeypatch.setattr(risk_engine, "_capital_now", lambda: 500.0)
    monkeypatch.setattr(risk_engine, "_bot_capital_weight", lambda _n: 1.0)
    monkeypatch.setattr(
        risk_engine, "_pnls_for_bots",
        lambda names, hours=None, today_only=False, mode=None: {
            "deep": [] if today_only else list(series),
        },
    )
    monkeypatch.setattr(
        risk_engine, "_portfolio_pnls",
        lambda hours=None, today_only=False, mode=None: (
            [] if today_only else list(series)),
    )
    monkeypatch.setattr(risk_engine, "_file_kill_armed", lambda: False)

    state = risk_engine.evaluate(bot_names=["deep"], mode="paper")
    port = state["portfolio"]
    assert port["drawdown"] == pytest.approx(0.5, abs=0.01)
    assert port["status"] == "paused"
    assert "portfolio_max_drawdown" in (port.get("reason") or "")


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


def test_resume_sticky_overrides_auto_pause(monkeypatch):
    """Resume must not be immediately undone by the same max-DD auto-pause."""
    saved = {}
    monkeypatch.setattr(risk_engine.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(risk_engine.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(risk_engine.db, "log_risk_event", lambda **kw: 1)
    monkeypatch.setattr(risk_engine.db, "get_active_bots",
                        lambda: [{"bot_name": "sniper-v1"}])
    monkeypatch.setattr(config, "RISK_BOT_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PAPER_BOT_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PAPER_PORTFOLIO_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PORTFOLIO_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_BOT_MAX_DRAWDOWN", 0.35)
    monkeypatch.setattr(config, "RISK_PORTFOLIO_MAX_DRAWDOWN", 0.99)
    monkeypatch.setattr(config, "RISK_UNDERPERFORM_PAUSE_PNL", -9999.0)
    monkeypatch.setattr(config, "RISK_SIZE_REDUCE_MIN_MULT", 0.25)

    # ~50% DD on $200 bankroll → auto-pause without resume override
    series = [-20.0] * 5  # start 200 → equity 100, dd=50%
    monkeypatch.setattr(risk_engine, "_capital_now", lambda: 100.0)
    monkeypatch.setattr(risk_engine, "_bot_capital_weight", lambda _n: 0.05)
    monkeypatch.setattr(
        risk_engine, "_pnls_for_bots",
        lambda names, hours=None, today_only=False, mode=None: {
            "sniper-v1": [] if today_only else list(series),
        },
    )
    monkeypatch.setattr(
        risk_engine, "_portfolio_pnls",
        lambda hours=None, today_only=False, mode=None: (
            [] if today_only else list(series)),
    )
    monkeypatch.setattr(risk_engine, "_file_kill_armed", lambda: False)

    state = risk_engine.evaluate(bot_names=["sniper-v1"], mode="paper")
    assert state["bots"]["sniper-v1"]["status"] == "paused"
    assert "bot_max_drawdown" in (state["bots"]["sniper-v1"]["reason"] or "")

    entry = risk_engine.resume_bot("sniper-v1")
    assert entry["status"] in ("active", "reduced")
    assert entry.get("manual_resume") is True
    assert entry["size_mult"] > 0.0
    assert "manual_resume" in (entry.get("reason") or "")


def test_bot_dd_uses_full_pool_not_portfolio_weight(monkeypatch):
    """Regression: 5% weight micro-book must not invent 35%+ DD on tiny $ swings."""
    saved = {}
    monkeypatch.setattr(risk_engine.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(risk_engine.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(risk_engine.db, "log_risk_event", lambda **kw: 1)
    monkeypatch.setattr(risk_engine.db, "get_active_bots",
                        lambda: [{"bot_name": "sniper-v1"}])
    monkeypatch.setattr(config, "RISK_BOT_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PAPER_BOT_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PAPER_PORTFOLIO_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_PORTFOLIO_DAILY_LOSS", 9999.0)
    monkeypatch.setattr(config, "RISK_BOT_MAX_DRAWDOWN", 0.35)
    monkeypatch.setattr(config, "RISK_PORTFOLIO_MAX_DRAWDOWN", 0.99)
    monkeypatch.setattr(config, "RISK_UNDERPERFORM_PAUSE_PNL", -9999.0)

    # Live shape of the sniper false-pause: weight 5%, ~$7 giveback from peak
    # after a run-up, window still net positive, daily ~−$0.50.
    series = [2.0, 3.0, 4.0, 2.0, -1.0, -2.0, -2.5, -1.0, 0.5, -0.5]
    assert len(series) >= 5
    window_sum = sum(series)
    capital_now = 200.0 + window_sum
    monkeypatch.setattr(risk_engine, "_capital_now", lambda: capital_now)
    monkeypatch.setattr(risk_engine, "_bot_capital_weight", lambda _n: 0.05)
    monkeypatch.setattr(
        risk_engine, "_pnls_for_bots",
        lambda names, hours=None, today_only=False, mode=None: {
            "sniper-v1": ([-0.53] if today_only else list(series)),
        },
    )
    monkeypatch.setattr(
        risk_engine, "_portfolio_pnls",
        lambda hours=None, today_only=False, mode=None: (
            [] if today_only else list(series)),
    )
    monkeypatch.setattr(risk_engine, "_file_kill_armed", lambda: False)

    state = risk_engine.evaluate(bot_names=["sniper-v1"], mode="paper")
    bot = state["bots"]["sniper-v1"]
    # Against full pool (~$200), this path is a few % DD — not a pause.
    assert bot["starting_equity"] == pytest.approx(200.0, abs=0.01)
    assert bot["drawdown"] < 0.10
    assert bot["status"] == "active"
    assert bot["size_mult"] == pytest.approx(1.0)


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


def test_paper_daily_loss_getters_use_risk_paper_floors(monkeypatch):
    """Engine-off fallback must never see uncapped 999999 paper limits."""
    monkeypatch.setattr(config, "TRADING_MODE", "paper")
    monkeypatch.setattr(config, "RISK_PAPER_BOT_DAILY_LOSS", 75.0)
    monkeypatch.setattr(config, "RISK_PAPER_PORTFOLIO_DAILY_LOSS", 150.0)
    assert config.get_max_daily_loss_per_bot() == 75.0
    assert config.get_max_daily_loss_total() == 150.0
    assert config.PAPER_MAX_DAILY_LOSS_PER_BOT < 1000
    assert config.PAPER_MAX_DAILY_LOSS_TOTAL < 1000


def test_warm_max_age_raised_for_dual_venue():
    assert float(config.WARM_MAX_AGE_SEC) >= 5.5


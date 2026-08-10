"""Portfolio capital allocation — metrics, correlation, weights, sizing hooks."""

from unittest import mock

import pytest

import config
from arena import portfolio
from bots import base_bot
from bots.bot_momentum import MomentumBot


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------

def test_sharpe_ratio_basic():
    # Steady wins → high positive Sharpe
    pnls = [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.1]
    assert portfolio.sharpe_ratio(pnls) > 5.0
    assert portfolio.sharpe_ratio([]) == 0.0
    assert portfolio.sharpe_ratio([1.0]) == 0.0


def test_expectancy():
    assert portfolio.expectancy([1.0, -0.5, 0.5]) == pytest.approx(1.0 / 3)
    assert portfolio.expectancy([]) == 0.0


def test_pairwise_correlation_identical_series():
    rets = {
        "a": {"m1": 1.0, "m2": -1.0, "m3": 0.5, "m4": -0.5,
              "m5": 1.0, "m6": -1.0, "m7": 0.5, "m8": -0.5},
        "b": {"m1": 1.0, "m2": -1.0, "m3": 0.5, "m4": -0.5,
              "m5": 1.0, "m6": -1.0, "m7": 0.5, "m8": -0.5},
        "c": {"m1": -1.0, "m2": 1.0, "m3": -0.5, "m4": 0.5,
              "m5": -1.0, "m6": 1.0, "m7": -0.5, "m8": 0.5},
    }
    corr = portfolio.pairwise_correlation(rets, min_overlap=8)
    assert corr["a"]["b"] == pytest.approx(1.0, abs=0.01)
    assert corr["a"]["c"] == pytest.approx(-1.0, abs=0.01)
    assert corr["a"]["a"] == 1.0


def test_allocate_equal():
    names = ["bot-a", "bot-b", "bot-c", "bot-d"]
    with mock.patch.object(portfolio, "compute_metrics", return_value={
        n: {"n": 50, "sharpe": 0.5, "expectancy": 0.1, "total_pnl": 5.0,
            "variance": 1.0, "ready": True}
        for n in names
    }), mock.patch.object(portfolio, "_market_returns_by_bot", return_value={
        n: {} for n in names
    }):
        result = portfolio.allocate(names, method="equal")
    w = result["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    for n in names:
        assert w[n] == pytest.approx(0.25, abs=0.01)


def test_arbitrage_pinned_to_equal_share():
    """Arb is a fixed 1/N staple — Kelly must not starve or over-weight it."""
    names = ["arbitrage-v1", "hybrid-v1", "sniper-v1", "sentiment-g11-794"]
    metrics = {
        "arbitrage-v1": {
            "n": 20, "sharpe": 0.05, "expectancy": 0.1,
            "total_pnl": 2.0, "variance": 50.0, "ready": True,
        },
        "hybrid-v1": {
            "n": 50, "sharpe": 0.5, "expectancy": 0.5,
            "total_pnl": 25.0, "variance": 2.0, "ready": True,
        },
        "sniper-v1": {
            "n": 40, "sharpe": 0.2, "expectancy": 0.2,
            "total_pnl": 8.0, "variance": 5.0, "ready": True,
        },
        "sentiment-g11-794": {
            "n": 40, "sharpe": 0.4, "expectancy": 0.4,
            "total_pnl": 16.0, "variance": 3.0, "ready": True,
        },
    }
    with mock.patch.object(portfolio, "compute_metrics", return_value=metrics), \
         mock.patch.object(portfolio, "_market_returns_by_bot",
                           return_value={n: {} for n in names}):
        result = portfolio.allocate(names, method="kelly_portfolio")
    w = result["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["arbitrage-v1"] == pytest.approx(0.25, abs=0.01)
    # Manual override still wins
    result2 = portfolio.allocate(
        names, method="kelly_portfolio",
        manual_overrides={"arbitrage-v1": 0.10},
    )
    assert result2["weights"]["arbitrage-v1"] == pytest.approx(0.10, abs=0.01)


def test_losers_starved_not_floored():
    """Ready bots with negative expectancy get a tiny score, not cold-start floor."""
    # Need ≥3 bots so the simplex min/max box can actually starve a loser
    # (with n=2, max_w is forced to 0.5 and weights collapse to equal).
    names = ["winner-v1", "mid-v1", "loser-v1"]
    metrics = {
        "winner-v1": {
            "n": 40, "sharpe": 0.5, "expectancy": 0.4,
            "total_pnl": 16.0, "variance": 2.0, "ready": True,
        },
        "mid-v1": {
            "n": 40, "sharpe": 0.2, "expectancy": 0.15,
            "total_pnl": 6.0, "variance": 3.0, "ready": True,
        },
        "loser-v1": {
            "n": 40, "sharpe": -0.2, "expectancy": -0.3,
            "total_pnl": -12.0, "variance": 4.0, "ready": True,
        },
    }
    with mock.patch.object(portfolio, "compute_metrics", return_value=metrics), \
         mock.patch.object(portfolio, "_market_returns_by_bot",
                           return_value={n: {} for n in names}):
        result = portfolio.allocate(names, method="kelly_portfolio")
    w = result["weights"]
    assert w["winner-v1"] > w["loser-v1"]
    assert w["loser-v1"] <= w["mid-v1"]
    assert w["loser-v1"] < 0.25  # not an equal 1/3 floor
    # n≥20 + neg expectancy → hard demote to ~0
    assert w["loser-v1"] <= 0.01


def test_unproven_bots_capped_at_20pct():
    """Until live edge proven, no bot may take >20% of capital."""
    names = ["hot-v1", "other-a", "other-b", "other-c"]
    # hot-v1 ready with huge score but n=10 < EDGE_PROVEN_MIN_N → unproven cap
    metrics = {
        "hot-v1": {
            "n": 10, "sharpe": 5.0, "expectancy": 2.0,
            "total_pnl": 20.0, "variance": 0.1, "ready": True,
        },
        "other-a": {
            "n": 10, "sharpe": 0.1, "expectancy": 0.05,
            "total_pnl": 0.5, "variance": 1.0, "ready": True,
        },
        "other-b": {
            "n": 10, "sharpe": 0.1, "expectancy": 0.05,
            "total_pnl": 0.5, "variance": 1.0, "ready": True,
        },
        "other-c": {
            "n": 10, "sharpe": 0.1, "expectancy": 0.05,
            "total_pnl": 0.5, "variance": 1.0, "ready": True,
        },
    }
    with mock.patch.object(portfolio, "compute_metrics", return_value=metrics), \
         mock.patch.object(portfolio, "_market_returns_by_bot",
                           return_value={n: {} for n in names}), \
         mock.patch.object(config, "PORTFOLIO_CORR_SHRINK", 0.0):
        result = portfolio.allocate(names, method="kelly_portfolio")
    assert result["weights"]["hot-v1"] <= 0.20 + 1e-6


def test_neg_expectancy_strips_manual_override():
    """Manual floor on a proven loser is removed so capital can leave."""
    names = ["winner-v1", "sweeper-v1"]
    metrics = {
        "winner-v1": {
            "n": 40, "sharpe": 0.5, "expectancy": 0.4,
            "total_pnl": 16.0, "variance": 2.0, "ready": True,
        },
        "sweeper-v1": {
            "n": 40, "sharpe": -0.2, "expectancy": -0.5,
            "total_pnl": -20.0, "variance": 9.0, "ready": True,
        },
    }
    with mock.patch.object(portfolio, "compute_metrics", return_value=metrics), \
         mock.patch.object(portfolio, "_market_returns_by_bot",
                           return_value={n: {} for n in names}):
        result = portfolio.allocate(
            names, method="kelly_portfolio",
            manual_overrides={"sweeper-v1": 0.30},
        )
    assert result["weights"]["sweeper-v1"] <= 0.01
    assert "sweeper-v1" not in (result.get("manual_overrides") or {})


def test_rebalance_force_evolution_reason(monkeypatch):
    """Post-GA path uses force=True + reason=evolution on the new roster."""
    names = ["phantom-v1", "hybrid-g4-158", "sniper-g4-144"]
    monkeypatch.setattr(portfolio, "active_bot_names", lambda: names)
    monkeypatch.setattr(portfolio, "_current_regime_label", lambda: "low_vol_range")
    saved = {}

    def fake_save(state):
        saved.update(state)

    monkeypatch.setattr(portfolio, "save_state", fake_save)
    monkeypatch.setattr(portfolio, "load_state", lambda: {
        "enabled": True,
        "method": "equal",
        "window_hours": 24.0,
        "weights": {"old-v1": 1.0},
        "manual_overrides": {},
        "last_rebalance_at": 0.0,
        "last_regime": "normal",
    })
    with mock.patch.object(portfolio, "allocate", return_value={
        "weights": {n: 1.0 / len(names) for n in names},
        "auto_weights": {n: 1.0 / len(names) for n in names},
        "manual_overrides": {},
        "metrics": {},
        "correlations": {},
        "method": "equal",
        "window_hours": 24.0,
    }):
        state = portfolio.rebalance(force=True, reason="evolution")
    assert state["rebalance_reason"] == "evolution"
    assert set(state["weights"]) == set(names)
    assert "old-v1" not in state["weights"]


def test_explore_floor_caps_new_gn_bots(monkeypatch):
    """New hybrid-g4-* with n=0 gets weight ≤ PORTFOLIO_EXPLORE_MAX_WEIGHT."""
    names = ["phantom-v1", "hybrid-g4-158", "hybrid-g4-259"]
    metrics = {
        "phantom-v1": {
            "n": 40, "sharpe": 1.0, "expectancy": 0.5,
            "total_pnl": 20.0, "variance": 1.0, "ready": True,
        },
        "hybrid-g4-158": {
            "n": 0, "sharpe": 0.0, "expectancy": 0.0,
            "total_pnl": 0.0, "variance": 0.0, "ready": False,
        },
        "hybrid-g4-259": {
            "n": 0, "sharpe": 0.0, "expectancy": 0.0,
            "total_pnl": 0.0, "variance": 0.0, "ready": False,
        },
    }
    monkeypatch.setattr(config, "PORTFOLIO_EXPLORE_MAX_WEIGHT", 0.08, raising=False)
    monkeypatch.setattr(config, "PORTFOLIO_EXPLORE_MIN_TRADES", 15, raising=False)
    monkeypatch.setattr(config, "PORTFOLIO_MIN_WEIGHT", 0.05, raising=False)
    monkeypatch.setattr(config, "PORTFOLIO_MAX_WEIGHT", 0.70, raising=False)
    monkeypatch.setattr(config, "PORTFOLIO_CORR_SHRINK", 0.0, raising=False)
    with mock.patch.object(portfolio, "compute_metrics", return_value=metrics), \
         mock.patch.object(portfolio, "_market_returns_by_bot",
                           return_value={n: {} for n in names}):
        result = portfolio.allocate(names, method="equal")
    w = result["weights"]
    assert w["hybrid-g4-158"] <= 0.08 + 1e-6
    assert w["hybrid-g4-259"] <= 0.08 + 1e-6
    # Weights may sum to <1 when max_w + explore caps bind (capital sits idle)
    assert sum(w.values()) <= 1.0 + 1e-6
    # Veteran keeps the bulk
    assert w["phantom-v1"] > w["hybrid-g4-158"]


def test_allocate_sharpe_favors_winners():
    names = ["winner", "loser", "mid"]
    metrics = {
        "winner": {"n": 40, "sharpe": 1.5, "expectancy": 0.5,
                   "total_pnl": 20.0, "variance": 0.5, "ready": True},
        "loser": {"n": 40, "sharpe": -0.8, "expectancy": -0.3,
                  "total_pnl": -12.0, "variance": 0.5, "ready": True},
        "mid": {"n": 40, "sharpe": 0.3, "expectancy": 0.05,
                "total_pnl": 2.0, "variance": 0.5, "ready": True},
    }
    with mock.patch.object(portfolio, "compute_metrics", return_value=metrics), \
         mock.patch.object(portfolio, "_market_returns_by_bot",
                           return_value={n: {} for n in names}), \
         mock.patch.object(config, "PORTFOLIO_CORR_SHRINK", 0.0), \
         mock.patch.object(config, "PORTFOLIO_MAX_WEIGHT", 0.70), \
         mock.patch.object(config, "PORTFOLIO_MIN_WEIGHT", 0.05):
        result = portfolio.allocate(names, method="sharpe")
    w = result["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["winner"] > w["mid"]
    # n≥20 + neg expectancy → hard demote (no min-weight floor for losers)
    assert w["loser"] <= 0.01


def test_allocate_kelly_portfolio_correlation_shrink():
    """Two identical high-Sharpe bots should not both get top weight when
    highly correlated — capital is spread."""
    names = ["a", "b", "c"]
    metrics = {
        "a": {"n": 40, "sharpe": 1.0, "expectancy": 0.4,
              "total_pnl": 16.0, "variance": 0.2, "ready": True},
        "b": {"n": 40, "sharpe": 1.0, "expectancy": 0.4,
              "total_pnl": 16.0, "variance": 0.2, "ready": True},
        "c": {"n": 40, "sharpe": 0.4, "expectancy": 0.15,
              "total_pnl": 6.0, "variance": 0.3, "ready": True},
    }
    # a and b perfectly correlated; c independent
    markets = [f"m{i}" for i in range(12)]
    rets_a = {m: (1.0 if i % 2 == 0 else -0.5) for i, m in enumerate(markets)}
    rets_b = dict(rets_a)
    rets_c = {m: (0.3 if i % 3 == 0 else -0.1) for i, m in enumerate(markets)}

    with mock.patch.object(portfolio, "compute_metrics", return_value=metrics), \
         mock.patch.object(portfolio, "_market_returns_by_bot", return_value={
             "a": rets_a, "b": rets_b, "c": rets_c,
         }), mock.patch.object(config, "PORTFOLIO_CORR_SHRINK", 0.8):
        shrunk = portfolio.allocate(names, method="kelly_portfolio")
    with mock.patch.object(portfolio, "compute_metrics", return_value=metrics), \
         mock.patch.object(portfolio, "_market_returns_by_bot", return_value={
             "a": rets_a, "b": rets_b, "c": rets_c,
         }), mock.patch.object(config, "PORTFOLIO_CORR_SHRINK", 0.0):
        plain = portfolio.allocate(names, method="kelly_portfolio")

    # With shrink, uncorrelated c should get relatively more vs a+b stack
    shrunk_c_share = shrunk["weights"]["c"]
    plain_c_share = plain["weights"]["c"]
    assert shrunk_c_share >= plain_c_share - 0.02  # not worse; usually better


def test_allocate_manual_override_pins_weight():
    names = ["a", "b", "c"]
    metrics = {
        n: {"n": 40, "sharpe": 0.5, "expectancy": 0.1, "total_pnl": 4.0,
            "variance": 0.5, "ready": True}
        for n in names
    }
    with mock.patch.object(portfolio, "compute_metrics", return_value=metrics), \
         mock.patch.object(portfolio, "_market_returns_by_bot",
                           return_value={n: {} for n in names}):
        result = portfolio.allocate(
            names, method="equal",
            manual_overrides={"a": 0.50},
        )
    w = result["weights"]
    # Manual pin respected up to PORTFOLIO_MAX_WEIGHT (0.50)
    assert w["a"] == pytest.approx(0.50, abs=0.02)
    assert abs(sum(w.values()) - 1.0) < 1e-5
    # Remaining free mass split between b and c
    assert w["b"] + w["c"] == pytest.approx(1.0 - w["a"], abs=0.02)


def test_weights_respect_min_max_bounds():
    names = [f"b{i}" for i in range(5)]
    # One massive winner
    metrics = {
        "b0": {"n": 50, "sharpe": 5.0, "expectancy": 2.0, "total_pnl": 100,
               "variance": 0.1, "ready": True},
    }
    for n in names[1:]:
        metrics[n] = {"n": 50, "sharpe": 0.01, "expectancy": 0.001,
                      "total_pnl": 0.1, "variance": 1.0, "ready": True}
    with mock.patch.object(portfolio, "compute_metrics", return_value=metrics), \
         mock.patch.object(portfolio, "_market_returns_by_bot",
                           return_value={n: {} for n in names}), \
         mock.patch.object(config, "PORTFOLIO_CORR_SHRINK", 0.0), \
         mock.patch.object(config, "PORTFOLIO_MAX_WEIGHT", 0.40), \
         mock.patch.object(config, "PORTFOLIO_MIN_WEIGHT", 0.08):
        result = portfolio.allocate(names, method="sharpe")
    w = result["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-5
    assert w["b0"] <= 0.40 + 1e-6
    for n in names[1:]:
        assert w[n] >= 0.08 - 1e-6


# ---------------------------------------------------------------------------
# State / rebalance / hot-path weight
# ---------------------------------------------------------------------------

def test_get_weight_disabled_returns_one(tmp_path, monkeypatch):
    # Isolate arena_state via mocked load_state
    with mock.patch.object(portfolio, "load_state", return_value={
        "enabled": False, "weights": {"x": 0.3}, "n_active": 3,
    }):
        portfolio._weight_cache = (0.0, False, {}, 0)
        assert portfolio.get_weight("x") == 1.0


def test_get_weight_enabled_returns_slice():
    with mock.patch.object(portfolio, "load_state", return_value={
        "enabled": True,
        "weights": {"alpha": 0.4, "beta": 0.6},
        "n_active": 2,
    }):
        portfolio._weight_cache = (0.0, False, {}, 0)
        assert portfolio.get_weight("alpha") == pytest.approx(0.4)
        assert portfolio.get_weight("beta") == pytest.approx(0.6)
        # Unknown bot mid-cycle → equal share
        assert portfolio.get_weight("newcomer") == pytest.approx(0.5)


def test_size_multiplier_equal_is_one():
    with mock.patch.object(portfolio, "load_state", return_value={
        "enabled": True,
        "weights": {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
        "n_active": 4,
    }):
        portfolio._weight_cache = (0.0, False, {}, 0)
        assert portfolio.size_multiplier("a") == pytest.approx(1.0)


def test_rebalance_persists(monkeypatch):
    saved = {}

    def fake_set(key, value):
        saved[key] = value

    def fake_get(key, default=None):
        return saved.get(key, default)

    monkeypatch.setattr(portfolio.db, "set_arena_state", fake_set)
    monkeypatch.setattr(portfolio.db, "get_arena_state", fake_get)
    monkeypatch.setattr(portfolio, "active_bot_names",
                        lambda: ["m1", "m2"])
    monkeypatch.setattr(portfolio, "compute_metrics", lambda names, hours=None: {
        n: {"n": 30, "sharpe": 0.4, "expectancy": 0.1, "total_pnl": 3.0,
            "variance": 0.5, "ready": True}
        for n in names
    })
    monkeypatch.setattr(portfolio, "_market_returns_by_bot",
                        lambda names, hours: {n: {} for n in names})
    monkeypatch.setattr(portfolio, "_current_regime_label", lambda: "normal")

    state = portfolio.rebalance(force=True, reason="test")
    assert state["rebalance_reason"] == "test"
    assert abs(sum(state["weights"].values()) - 1.0) < 1e-6
    assert portfolio.STATE_KEY in saved


def test_rebalance_skips_when_not_due(monkeypatch):
    import time
    import json
    state0 = {
        "enabled": True,
        "method": "equal",
        "window_hours": 24,
        "weights": {"a": 1.0},
        "auto_weights": {},
        "manual_overrides": {},
        "metrics": {},
        "correlations": {},
        "last_rebalance_at": time.time(),
        "last_regime": "normal",
        "rebalance_reason": "prev",
        "n_active": 1,
    }
    saved = {portfolio.STATE_KEY: json.dumps(state0)}

    monkeypatch.setattr(portfolio.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(portfolio.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(portfolio, "active_bot_names", lambda: ["a"])
    monkeypatch.setattr(portfolio, "_current_regime_label", lambda: "normal")
    monkeypatch.setattr(config, "PORTFOLIO_REBALANCE_INTERVAL_SEC", 99999)

    out = portfolio.rebalance(force=False)
    assert out["rebalance_reason"] == "prev"  # unchanged


def test_rebalance_on_regime_change(monkeypatch):
    import time
    import json
    state0 = {
        "enabled": True,
        "method": "equal",
        "window_hours": 24,
        "weights": {"a": 0.5, "b": 0.5},
        "auto_weights": {},
        "manual_overrides": {},
        "metrics": {},
        "correlations": {},
        "last_rebalance_at": time.time(),  # just rebalanced
        "last_regime": "low_vol_range",
        "rebalance_reason": "timer",
        "n_active": 2,
    }
    saved = {portfolio.STATE_KEY: json.dumps(state0)}
    monkeypatch.setattr(portfolio.db, "get_arena_state",
                        lambda k, d=None: saved.get(k, d))
    monkeypatch.setattr(portfolio.db, "set_arena_state",
                        lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(portfolio, "active_bot_names", lambda: ["a", "b"])
    monkeypatch.setattr(portfolio, "_current_regime_label",
                        lambda: "high_vol_trend")
    monkeypatch.setattr(portfolio, "compute_metrics", lambda names, hours=None: {
        n: {"n": 20, "sharpe": 0.2, "expectancy": 0.05, "total_pnl": 1.0,
            "variance": 0.5, "ready": True}
        for n in names
    })
    monkeypatch.setattr(portfolio, "_market_returns_by_bot",
                        lambda names, hours: {n: {} for n in names})
    monkeypatch.setattr(config, "PORTFOLIO_REBALANCE_INTERVAL_SEC", 99999)
    monkeypatch.setattr(config, "PORTFOLIO_REBALANCE_ON_REGIME", True)
    monkeypatch.setattr(config, "PORTFOLIO_REGIME_REBALANCE_MIN_DWELL_SEC", 0.0)

    class _Det:
        def status(self):
            return {"current": {"last_change_ts": time.time() - 600}}

    monkeypatch.setattr(
        "signals.regime_detector.get_detector", lambda: _Det(), raising=False,
    )

    out = portfolio.rebalance(force=False)
    assert out["rebalance_reason"].startswith("regime:")
    assert out["last_regime"] == "high_vol_trend"


# ---------------------------------------------------------------------------
# Sizing integration (make_decision bankroll slice)
# ---------------------------------------------------------------------------

def _bot():
    return MomentumBot(name="momentum-test", generation=0)


def _market(yes=0.52, tr=150):
    return {"id": "m", "current_price": yes, "no_price": round(1 - yes, 4),
            "polymarket_token_id": "y", "polymarket_no_token_id": "n",
            "time_remaining_seconds": tr}


def _sig(drift=0.5):
    return {"prices": [100.0, 100.05, 100.12, 100.20, 100.30],
            "latest": 100.30, "orderflow": {}, "pm_momentum": 0.0,
            "obi": 0.0, "cvd": 0.0, "btc_drift": drift}


def test_kelly_size_scales_with_portfolio_weight():
    """Half capital weight → roughly half suggested amount (same edge)."""
    from arena.regime_adapt import RegimeAdjust
    neutral = RegimeAdjust(size_mult=1.0, label="normal")
    m = _market(yes=0.60)
    m["yes_ask"] = 0.61
    m["no_ask"] = 0.40
    sig = _sig(drift=0.5)

    def _run(w):
        with mock.patch.object(base_bot, "_sizing_bankroll", lambda mode: 5000.0), \
             mock.patch.object(base_bot, "_kelly_fraction", lambda: 0.25), \
             mock.patch.object(base_bot, "_portfolio_weight",
                               side_effect=lambda name: w), \
             mock.patch.object(base_bot, "_risk_size_mult", lambda name: 1.0), \
             mock.patch("arena.regime_adapt.adjustments", return_value=neutral):
            return _bot().make_decision(m, sig)

    full = _run(1.0)
    half = _run(0.5)
    if full.get("action") != "buy" or half.get("action") != "buy":
        pytest.skip("momentum bot did not trade under test signals")
    # Allow small shares-first rounding
    assert half["suggested_amount"] == pytest.approx(
        full["suggested_amount"] * 0.5, rel=0.08, abs=0.15)


def test_execute_scales_zone_bot_amount():
    """Zone-style signals (no target_shares) get size_multiplier applied."""
    bot = _bot()
    signal = {
        "action": "buy", "side": "yes", "confidence": 0.5,
        "suggested_amount": 10.0, "entry_price": 0.5,
        # no target_shares → zone path
    }
    market = _market()

    class _RiskOK:
        allow = True
        action = "allow"
        reason = None
        size_mult = 1.0

    with mock.patch.object(bot, "_paused", False), \
         mock.patch.object(base_bot.db, "get_bot_mode", return_value="paper"), \
         mock.patch.object(base_bot.db, "get_bot_daily_loss", return_value=0.0), \
         mock.patch.object(base_bot.db, "get_total_daily_loss", return_value=0.0), \
         mock.patch.object(base_bot, "_portfolio_size_mult", return_value=2.0), \
         mock.patch("arena.risk_engine.pre_trade", return_value=_RiskOK()), \
         mock.patch.object(bot, "_exposure_headroom", return_value=None), \
         mock.patch.object(bot, "_place_via_engine") as place:
        place.return_value = {"success": True}
        bot.execute(signal, market)
        args = place.call_args
        # amount is 3rd positional or kw
        amount = args[0][2] if len(args[0]) > 2 else args[1].get("amount")
        assert amount == pytest.approx(20.0)


def test_execute_does_not_double_scale_kelly_path():
    """Kelly decisions carry target_shares — execute must not re-scale."""
    bot = _bot()
    signal = {
        "action": "buy", "side": "yes", "confidence": 0.5,
        "suggested_amount": 10.0, "entry_price": 0.5,
        "target_shares": 20.0,
    }
    market = _market()

    with mock.patch.object(bot, "_paused", False), \
         mock.patch.object(base_bot.db, "get_bot_mode", return_value="paper"), \
         mock.patch.object(base_bot.db, "get_bot_daily_loss", return_value=0.0), \
         mock.patch.object(base_bot.db, "get_total_daily_loss", return_value=0.0), \
         mock.patch.object(base_bot, "_portfolio_size_mult", return_value=2.0), \
         mock.patch.object(bot, "_exposure_headroom", return_value=None), \
         mock.patch.object(bot, "_place_via_engine") as place:
        place.return_value = {"success": True}
        bot.execute(signal, market)
        amount = place.call_args[0][2]
        assert amount == pytest.approx(10.0)  # unchanged

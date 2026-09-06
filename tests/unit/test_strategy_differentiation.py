import pytest
"""Phase 2: the directional strategies must be genuinely distinct.

Each strategy expresses a distinct, frequently-firing, data-backed thesis:
  momentum      -> follow BTC trend
  mean_reversion-> fade BTC z-score extremes (opposite of momentum)
  hybrid        -> ensemble of the subs (momentum / meanrev / phantom)
  lag_residual  -> pure market-lags-drift (menu specialist)

Sentiment bot removed (2026-08 audit).
"""

from bots.bot_momentum import MomentumBot
from bots.bot_mean_rev import MeanRevBot
from bots.bot_hybrid import HybridBot
from bots.bot_lag_residual import LagResidualBot

RISING = [100.0 + i * 0.05 for i in range(12)]   # steady BTC uptrend


def _mkt(**over):
    # 90s remaining: momentum still trades (late skip is last 80s) and
    # meanrev has 3 closed 1m bars for a window-local pullback.
    base = {"current_price": 0.5, "no_price": 0.5, "time_remaining_seconds": 90}
    base.update(over)
    return base


def _sig(prices=RISING, **over):
    # Default strike above the RISING path so a down-drift NO pullback
    # (TWAP still below PTB) can fire when tests inject down-drift.
    base = {"prices": prices, "latest": prices[-1], "volumes": [], "orderflow": {},
            "pm_momentum": 0.0, "obi": 0.0, "cvd": 0.0, "sentiment": {},
            "btc_strike": 100.70, "btc_now": prices[-1]}
    base.update(over)
    return base


def test_momentum_fires_on_trend():
    d = MomentumBot(name="m").analyze(_mkt(), _sig())
    assert d["action"] == "buy" and d["side"] == "yes"


def test_mean_rev_fires_and_opposes_momentum():
    # Overextended up-move WITH a down-drift backing the fade -> NO
    # (opposite of momentum's YES). BUG #28: the ungated fade went 0/11 live
    # (-$55) — the fade now requires signed drift agreement, so the identity
    # is "fade the move the fundamentals don't back", not "fade everything".
    # P0: strike above mean so reversion target still supports DOWN.
    d = MeanRevBot(name="mr").analyze(_mkt(), _sig(btc_drift=-0.3, btc_strike=100.70))
    assert d["action"] == "buy" and d["side"] == "no"
    assert "strike=" in d["reasoning"] and "pullback=" in d["reasoning"]
    assert "btc_now=" in d["reasoning"]


def test_mean_rev_holds_without_drift_backing():
    # Same overextension, drift flat -> no thesis (the 0/11 death class).
    d = MeanRevBot(name="mr").analyze(_mkt(), _sig())
    assert d["action"] == "hold"


def test_mean_rev_holds_when_twap_above_ptb_for_no():
    # TWAP above PTB is an UP window — do not bet DOWN just because drift was injected.
    d = MeanRevBot(name="mr").analyze(
        _mkt(), _sig(btc_drift=-0.3, btc_strike=99.0))
    assert d["action"] == "hold"


def test_lag_residual_skips_without_drift():
    d = LagResidualBot(name="lr").make_decision(_mkt(), _sig(btc_drift=0.0))
    assert d["action"] == "skip"


def test_momentum_and_meanrev_take_opposite_sides():
    # A BTC pop inside a DOWN window (drift negative): momentum follows the
    # pop (YES), meanrev fades it back toward the fundamentals (NO).
    m, s = _mkt(), _sig(btc_drift=-0.3, btc_strike=100.70)
    sides = {MomentumBot(name="m").analyze(m, s)["side"],
             MeanRevBot(name="mr").analyze(m, s)["side"]}
    assert sides == {"yes", "no"}   # genuinely distinct, not clones


def test_hybrid_fires_when_substrategies_lean():
    # BTC trend up -> hybrid should reach a buy, not hold.
    d = HybridBot(name="h").analyze(
        _mkt(),
        _sig(
            btc_drift=0.25,
            market_regime={"label": "high_vol_trend", "known": True,
                           "trend_score": 0.8},
        ),
    )
    # Hybrid's own analyze() requires ≥2-sub agreement before a BUY.
    # A single momentum lean must still show up in the ensemble vote/score.
    assert d["signals"]["votes"].get("momentum", 0) > 0
    assert d["signals"]["weighted_score"] > 0


def test_lag_residual_uses_btc_implied_yes_from_signals():
    """Phase 1: lag residual must use Phi(z)/btc_implied_yes, not 0.5+0.5*drift."""
    # tanh lie would map drift=0.80 -> implied 0.90; Phi path uses 0.62 from signals.
    m = _mkt(current_price=0.50, no_price=0.50, yes_ask=0.51, no_ask=0.51)
    s = _sig(
        btc_drift=0.80,
        btc_implied_yes=0.62,
        btc_drift_z=0.30,
    )
    d = LagResidualBot(name="lr").make_decision(m, s)
    # With implied=0.62 and mid=0.50, residual_yes=0.12 clears min_residual.
    assert d["action"] == "buy"
    assert d["side"] == "yes"
    assert d["signals"]["implied"] == pytest.approx(0.62)
    # Prove we did NOT use 0.5+0.5*0.80 = 0.90
    assert d["signals"]["implied"] < 0.80


def test_lag_residual_prefers_phi_over_tanh_lie():
    """Absent btc_implied_yes, still use Phi(z) via drift_z / artanh — not 0.5+0.5*tanh."""
    import math
    from signals.prob import _phi_from_z
    m = _mkt(current_price=0.50, no_price=0.50, yes_ask=0.51, no_ask=0.51)
    z = 0.25
    s = _sig(btc_drift=0.80, btc_drift_z=z)  # no btc_implied_yes
    d = LagResidualBot(name="lr").make_decision(m, s)
    expected = _phi_from_z(z)
    tanh_lie = 0.5 + 0.5 * 0.80
    assert d["signals"]["implied"] == pytest.approx(expected, abs=1e-6)
    assert abs(d["signals"]["implied"] - tanh_lie) > 0.1

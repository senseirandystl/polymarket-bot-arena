"""Per-strategy analyze() behavior with mocked signal dicts.

Each major strategy's analyze() is exercised on crafted signals: direction
follows its thesis, the returned dict always has the contract keys, holds
carry confidence 0 semantics, and insufficient data never trades.
"""

import pytest

from tests.conftest import make_market, make_signals

from bots.bot_momentum import MomentumBot
from bots.bot_mean_rev import MeanRevBot
from bots.bot_phantom import PhantomBot
from bots.bot_hybrid import HybridBot
from bots.bot_lag_residual import LagResidualBot
from bots.bot_regime_specialist import RegimeSpecialistBot

CONTRACT_KEYS = {"action", "side", "confidence", "reasoning"}


def _check_contract(sig):
    assert CONTRACT_KEYS <= set(sig)
    assert sig["action"] in ("buy", "hold")
    assert sig["side"] in ("yes", "no")
    assert 0.0 <= sig["confidence"] <= 1.0
    if sig["action"] == "buy":
        assert sig.get("suggested_amount", 0) > 0


ALL_BOTS = [
    (MomentumBot, "momentum"),
    (MeanRevBot, "meanrev"),
    (PhantomBot, "phantom"),
    (HybridBot, "hybrid"),
    (LagResidualBot, "lag-residual"),
    (RegimeSpecialistBot, "regime-specialist"),
]


@pytest.mark.parametrize("cls,name", ALL_BOTS, ids=[n for _, n in ALL_BOTS])
def test_insufficient_data_holds(cls, name):
    bot = cls(name=f"{name}-t", generation=0)
    sig = bot.analyze(make_market(), make_signals(prices=[100.0], volumes=[]))
    _check_contract(sig)
    assert sig["action"] == "hold"


@pytest.mark.parametrize("cls,name", ALL_BOTS, ids=[n for _, n in ALL_BOTS])
def test_contract_on_neutral_tape(cls, name):
    bot = cls(name=f"{name}-t", generation=0)
    _check_contract(bot.analyze(make_market(), make_signals()))


def test_momentum_follows_uptrend():
    bot = MomentumBot(name="momo-t", generation=0)
    prices = [100_000.0 * (1.001 ** i) for i in range(30)]  # steady uptrend
    sig = bot.analyze(make_market(), make_signals(prices=prices, latest=prices[-1]))
    _check_contract(sig)
    assert sig["action"] == "buy" and sig["side"] == "yes"


def test_momentum_follows_downtrend():
    bot = MomentumBot(name="momo-t", generation=0)
    prices = [100_000.0 * (0.999 ** i) for i in range(30)]
    sig = bot.analyze(make_market(), make_signals(prices=prices, latest=prices[-1]))
    assert sig["action"] == "buy" and sig["side"] == "no"


def test_momentum_holds_below_threshold():
    bot = MomentumBot(name="momo-t", generation=0)
    sig = bot.analyze(make_market(), make_signals())  # perfectly flat tape
    assert sig["action"] == "hold"


def test_meanrev_fade_requires_drift_backing():
    """BUG #28: the fade only fires toward the side signed drift favors."""
    bot = MeanRevBot(name="rev-t", generation=0)
    lookback = bot.strategy_params["lookback_candles"]
    # A sharp up-spike at the end of a flat tape → overextended UP → fade = NO.
    prices = [100_000.0] * (lookback + 20) + [101_500.0]
    # Without a DOWN drift, the NO fade must be vetoed.
    late = make_market(time_remaining=60)
    vetoed = bot.analyze(late,
                         make_signals(prices=prices, latest=prices[-1], btc_drift=0.0,
                                      btc_strike=102_000.0))
    assert vetoed["action"] == "hold"
    # With a down-drift ≥ min_drift AND mean ≤ strike the same fade is allowed.
    backed = bot.analyze(late,
                         make_signals(prices=prices, latest=prices[-1], btc_drift=-0.3,
                                      btc_strike=102_000.0))
    _check_contract(backed)
    if backed["action"] == "buy":
        assert backed["side"] == "no"


def test_meanrev_fade_requires_mean_on_ptb_side():
    """P0: fade NO only when reversion mean ≤ Price-to-Beat."""
    bot = MeanRevBot(name="rev-t", generation=0)
    lookback = bot.strategy_params["lookback_candles"]
    prices = [100_000.0] * (lookback + 20) + [101_500.0]
    # Drift would allow NO, but mean (~100k) is ABOVE strike → hold.
    late = make_market(time_remaining=60)
    blocked = bot.analyze(
        late,
        make_signals(prices=prices, latest=prices[-1], btc_drift=-0.3,
                     btc_strike=99_000.0))
    assert blocked["action"] == "hold"
    assert "PTB" in blocked["reasoning"] or "strike" in blocked["reasoning"]
    # Strike above mean → NO allowed.
    ok = bot.analyze(
        late,
        make_signals(prices=prices, latest=prices[-1], btc_drift=-0.3,
                     btc_strike=102_000.0))
    assert ok["action"] == "buy" and ok["side"] == "no"
    assert "strike=" in ok["reasoning"] and "mean=" in ok["reasoning"]


def test_meanrev_make_decision_skips_when_analyze_holds():
    """Identity: a held fade must not become a drift-only trend clone."""
    bot = MeanRevBot(name="rev-clone", generation=0)
    market = make_market()
    market["current_price"] = 0.50
    market["yes_price"] = 0.50
    market["no_price"] = 0.50
    market["yes_ask"] = 0.51
    market["no_ask"] = 0.51
    market["time_remaining_seconds"] = 180
    sigs = make_signals(
        prices=[100_000.0] * 30,
        latest=100_000.0,
        btc_drift=0.75,
        btc_strike=100_000.0,
        btc_now=100_080.0,
        btc_implied_yes=0.62,
    )
    d = bot.make_decision(market, sigs)
    assert d["action"] == "skip"
    why = (d.get("reasoning") or "").lower()
    assert d.get("skip_reason") == "no_thesis" or "fade thesis" in why


def test_meanrev_window_lookback_preferred_late_window():
    """P1: late in the window, z-score uses window-local closed 1m bars."""
    from bots.bot_mean_rev import resolve_lookback

    # 60s remaining of a 300s window → age 240s → 4 closed 1m bars.
    lb, src = resolve_lookback(
        {"time_remaining_seconds": 60}, n_prices=20,
        max_lookback=10, min_window_candles=3)
    assert src == "window" and lb == 4

    # Early window (age 90s → 1 closed bar) does not use prior windows.
    lb2, src2 = resolve_lookback(
        {"time_remaining_seconds": 210}, n_prices=20,
        max_lookback=10, min_window_candles=3)
    assert src2 == "none" and lb2 == 0


def test_regime_specialist_holds_outside_allow_list():
    bot = RegimeSpecialistBot(name="rs-t", generation=0)
    sig = bot.analyze(
        make_market(),
        make_signals(
            btc_drift=0.3,
            vol_regime={"regime": "high_vol_chop", "trend_score": 0.2},
            market_regime={"regime_id": "high_vol_chop", "confidence": 0.9},
        ),
    )
    assert sig["action"] == "hold"


def test_phantom_vol_gate_holds_on_dead_tape():
    """Zero-ATR tape is outside the min_atr_pct bound → hold, never trade."""
    bot = PhantomBot(name="ph-t", generation=0)
    n = bot.strategy_params["ema_slow"] + bot.strategy_params["breakout_lookback"] + 5
    sig = bot.analyze(make_market(),
                      make_signals(prices=[100_000.0] * n, latest=100_000.0))
    assert sig["action"] == "hold"


def test_hybrid_holds_when_all_subs_hold():
    bot = HybridBot(name="hy-t", generation=0)
    sig = bot.analyze(make_market(), make_signals())
    _check_contract(sig)
    assert sig["action"] == "hold"

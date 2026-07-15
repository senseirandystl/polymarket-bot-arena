"""BTC 'price to beat' (strike) registry + drift signal.

The single strongest fundamental for BTC 5-min Up/Down markets — where BTC sits
vs the window's open price — was entirely missing from the decision logic. The
drift signal is regime-agnostic: it favors YES when BTC is above the strike and
NO when below, and grows more decisive as expiry nears.
"""

import signals.strike as strike


def _fresh_registry():
    return strike.StrikeRegistry()


def test_strike_captured_on_first_live_observation():
    reg = _fresh_registry()
    reg.observe("mkt1", btc_price=100000.0, time_remaining=298)
    assert reg.strike("mkt1") == 100000.0


def test_strike_not_overwritten_by_later_observations():
    reg = _fresh_registry()
    reg.observe("mkt1", 100000.0, 298)
    reg.observe("mkt1", 100500.0, 120)   # later in window — must NOT move the strike
    assert reg.strike("mkt1") == 100000.0


def test_prewindow_observation_ignored():
    # A market seen 20 min early (tr > window) must not set the strike early.
    reg = _fresh_registry()
    reg.observe("mkt1", 99000.0, time_remaining=1200)
    assert reg.strike("mkt1") is None


def test_drift_positive_when_btc_above_strike():
    s = strike.drift_signal(strike_price=100000.0, btc_now=100200.0, time_remaining=60)
    assert s > 0


def test_drift_negative_when_btc_below_strike():
    s = strike.drift_signal(strike_price=100000.0, btc_now=99800.0, time_remaining=60)
    assert s < 0


def test_drift_zero_at_the_strike():
    assert strike.drift_signal(100000.0, 100000.0, 60) == 0.0


def test_drift_bounded():
    hi = strike.drift_signal(100000.0, 110000.0, 15)   # BTC 10% above, near expiry
    lo = strike.drift_signal(100000.0, 90000.0, 15)
    assert -1.0 <= lo < 0 < hi <= 1.0


def test_drift_more_decisive_near_expiry():
    # Same drift is a stronger signal with less time left to revert.
    early = strike.drift_signal(100000.0, 100100.0, time_remaining=280)
    late = strike.drift_signal(100000.0, 100100.0, time_remaining=30)
    assert abs(late) > abs(early)


def test_drift_zero_without_strike():
    assert strike.drift_signal(None, 100000.0, 60) == 0.0

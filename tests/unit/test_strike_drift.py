"""BTC 'price to beat' (strike) registry + drift signal.

The strike is fetched ACCURATELY as the Binance open at the market's
eventStartTime (validated ~76% predictive offline), not a mid-window snapshot.
The drift signal is regime-agnostic and grows more decisive near expiry.
"""

import signals.strike as strike


def test_get_strike_fetches_and_caches():
    calls = []

    def fake(est):
        calls.append(est)
        return 100000.0

    reg = strike.StrikeRegistry(fetcher=fake)
    assert reg.get_strike("m1", "2026-07-16T12:00:00Z") == 100000.0
    # second call is served from cache — fetcher not hit again
    assert reg.get_strike("m1", "2026-07-16T12:00:00Z") == 100000.0
    assert len(calls) == 1


def test_get_strike_none_without_event_start():
    reg = strike.StrikeRegistry(fetcher=lambda est: 100000.0)
    assert reg.get_strike("m1", None) is None


def test_get_strike_none_on_fetch_failure():
    reg = strike.StrikeRegistry(fetcher=lambda est: None)
    assert reg.get_strike("m1", "2026-07-16T12:00:00Z") is None


def test_drift_positive_when_btc_above_strike():
    assert strike.drift_signal(100000.0, 100200.0, 60) > 0


def test_drift_negative_when_btc_below_strike():
    assert strike.drift_signal(100000.0, 99800.0, 60) < 0


def test_drift_zero_at_the_strike():
    assert strike.drift_signal(100000.0, 100000.0, 60) == 0.0


def test_drift_bounded():
    hi = strike.drift_signal(100000.0, 110000.0, 15)
    lo = strike.drift_signal(100000.0, 90000.0, 15)
    assert -1.0 <= lo < 0 < hi <= 1.0


def test_drift_more_decisive_near_expiry():
    early = strike.drift_signal(100000.0, 100100.0, time_remaining=280)
    late = strike.drift_signal(100000.0, 100100.0, time_remaining=30)
    assert abs(late) > abs(early)


def test_drift_zero_without_strike():
    assert strike.drift_signal(None, 100000.0, 60) == 0.0

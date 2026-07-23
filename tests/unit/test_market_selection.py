"""Regression tests for 5-minute-window market selection.

Guards against the July-2026 bug where the arena selected
"Bitcoin Up or Down - July 13, 8:15AM-8:30AM ET" (a *next-day*, *15-minute*
window) as the current market and let bots trade it. Two root causes:

  1. No 5-minute filter -- 15-min BTC up/down markets passed classification.
  2. Date-blind window matching -- selection keyed off ET *time-of-day*, so a
     future-dated market whose clock window straddled "now" was chosen.
"""

from datetime import datetime, timezone

from arena.market_utils import (
    compute_time_remaining_seconds,
    is_5min_market,
    select_current_market,
)


def _decorate(markets, now_utc):
    for m in markets:
        m["time_remaining_seconds"] = compute_time_remaining_seconds(m, now_utc)
    return markets


# 8:29 AM ET (EDT, UTC-4) == 12:29 UTC -- the exact moment the bad trade fired.
NOW = datetime(2026, 7, 12, 12, 29, 0, tzinfo=timezone.utc)


def test_is_5min_market_rejects_15min_window():
    assert is_5min_market("Bitcoin Up or Down - July 12, 8:25AM-8:30AM ET")
    assert not is_5min_market("Bitcoin Up or Down - July 13, 8:15AM-8:30AM ET")


def test_selects_todays_5min_window_not_tomorrows_15min():
    live_5min = {
        "id": "today-5m",
        "question": "Bitcoin Up or Down - July 12, 8:25AM-8:30AM ET",
        "resolves_at": "2026-07-12T12:30:00Z",  # 60s out -> current window
    }
    tomorrow_15min = {
        "id": "tomorrow-15m",
        "question": "Bitcoin Up or Down - July 13, 8:15AM-8:30AM ET",
        "resolves_at": "2026-07-13T12:30:00Z",  # ~24h out, but same clock window
    }
    markets = _decorate([tomorrow_15min, live_5min], NOW)
    current = select_current_market(markets, NOW)
    assert current is not None
    assert current["id"] == "today-5m"


def test_never_selects_future_dated_market_even_when_only_option():
    # Only the next-day 15-min market is present. Nothing is live right now,
    # so no current market must be returned (date-blind logic used to return it).
    tomorrow_15min = {
        "id": "tomorrow-15m",
        "question": "Bitcoin Up or Down - July 13, 8:15AM-8:30AM ET",
        "resolves_at": "2026-07-13T12:30:00Z",
    }
    markets = _decorate([tomorrow_15min], NOW)
    assert select_current_market(markets, NOW) is None


def test_no_live_window_returns_none():
    upcoming_5min = {
        "id": "soon-5m",
        "question": "Bitcoin Up or Down - July 12, 8:35AM-8:40AM ET",
        "resolves_at": "2026-07-12T12:40:00Z",  # 11 min out -> not yet current
    }
    markets = _decorate([upcoming_5min], NOW)
    assert select_current_market(markets, NOW) is None

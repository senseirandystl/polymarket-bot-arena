# tests/unit/test_context.py
from datetime import datetime, timezone
from signals.context import build_context, context_cell


def _rising_prices(n=40):
    return [100.0 + i * 0.05 for i in range(n)]


def test_build_context_has_all_keys():
    now = datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc)  # Wed
    ctx = build_context(_rising_prices(), signals=None, now_utc=now)
    for k in ("vol", "trend", "flow", "realized_vol",
              "btc_mom_1m", "btc_mom_5m", "btc_mom_15m", "btc_trend_slope",
              "weekday", "hour_block", "session", "macro_prox", "vol_trend_regime"):
        assert k in ctx, f"missing {k}"
    assert 0.0 <= ctx["vol"] <= 1.0
    assert ctx["weekday"] in range(7)
    assert ctx["hour_block"] in range(8)
    assert ctx["session"] in ("asia", "eu", "us", "overnight")
    assert ctx["macro_prox"] in (0, 1, 2)


def test_build_context_is_pure_and_deterministic():
    now = datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc)
    a = build_context(_rising_prices(), signals=None, now_utc=now)
    b = build_context(_rising_prices(), signals=None, now_utc=now)
    assert a == b


def test_build_context_empty_prices_safe():
    now = datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc)
    ctx = build_context([], signals=None, now_utc=now)
    assert ctx["vol_trend_regime"]  # still a string, no crash


def test_context_cell_is_hashable_tuple():
    now = datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc)
    ctx = build_context(_rising_prices(), signals=None, now_utc=now)
    cell = context_cell(ctx)
    assert isinstance(cell, tuple)
    hash(cell)  # must be hashable for dict grouping

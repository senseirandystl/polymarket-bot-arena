"""Unit tests for Phase 2 audit fixes: discovery warm-only, toggle cache,
position-monitor warm preference, and window-fraction age gates.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Age-gate fraction helpers
# ---------------------------------------------------------------------------

def test_momentum_late_skip_pm_5m_matches_soak():
    from arena.market_utils import momentum_late_skip_sec

    sec = momentum_late_skip_sec(300, exchange="polymarket")
    assert abs(sec - 80.0) < 0.05


def test_sniper_min_age_pm_5m_matches_soak():
    from arena.market_utils import sniper_min_age_sec

    sec = sniper_min_age_sec(300, exchange="polymarket")
    assert abs(sec - 60.0) < 0.05


def test_age_gates_scale_with_15m_window():
    from arena.market_utils import momentum_late_skip_sec, sniper_min_age_sec

    late = momentum_late_skip_sec(900, exchange="kalshi")
    age = sniper_min_age_sec(900, exchange="kalshi")
    # Prefer fraction so 15m stays ~same % as 5m (not the old absolute 120/90).
    assert abs(late - 240.0) < 0.5  # 80/300 * 900
    assert abs(age - 180.0) < 0.5   # 60/300 * 900


def test_age_gates_cap_at_half_window():
    import config
    from arena.market_utils import momentum_late_skip_sec, sniper_min_age_sec

    old_late = getattr(config, "MOMENTUM_LATE_SKIP_FRAC", None)
    old_age = getattr(config, "SNIPER_MIN_AGE_FRAC", None)
    try:
        config.MOMENTUM_LATE_SKIP_FRAC = 0.9
        config.SNIPER_MIN_AGE_FRAC = 0.9
        assert momentum_late_skip_sec(300) == pytest.approx(150.0)
        assert sniper_min_age_sec(300) == pytest.approx(150.0)
    finally:
        if old_late is not None:
            config.MOMENTUM_LATE_SKIP_FRAC = old_late
        if old_age is not None:
            config.SNIPER_MIN_AGE_FRAC = old_age


def test_age_gates_fallback_absolute_when_window_missing():
    from arena.market_utils import momentum_late_skip_sec, sniper_min_age_sec

    assert momentum_late_skip_sec(None, exchange="polymarket") == pytest.approx(80.0)
    assert sniper_min_age_sec(None, exchange="polymarket") == pytest.approx(60.0)
    assert momentum_late_skip_sec(None, exchange="kalshi") == pytest.approx(120.0)
    assert sniper_min_age_sec(None, exchange="kalshi") == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Discovery: no CLOB refresh_price / get_order_book
# ---------------------------------------------------------------------------

def test_discovery_refresh_uses_warm_not_http(monkeypatch):
    import arena.discovery as disc_mod
    from arena.discovery import MarketDiscovery

    calls = {"refresh": 0, "book": 0}

    def _boom_refresh(*_a, **_k):
        calls["refresh"] += 1
        raise AssertionError("discovery must not call refresh_price")

    def _boom_book(*_a, **_k):
        calls["book"] += 1
        raise AssertionError("discovery must not call get_order_book")

    monkeypatch.setattr("polymarket_markets.refresh_price", _boom_refresh)
    monkeypatch.setattr("polymarket_markets.get_order_book", _boom_book)

    warm = {
        "yes_price": 0.55,
        "no_price": 0.45,
        "obi": 0.12,
        "yes_book": {"valid": True, "best_ask": 0.56, "asks": [[0.56, 10]]},
        "no_book": {"valid": True, "best_ask": 0.46, "asks": [[0.46, 10]]},
        "ts": time.time(),
    }

    class _Store:
        def get(self, mid):
            return warm if mid == "m1" else None

    monkeypatch.setattr("arena.market_data.store", lambda: _Store())
    monkeypatch.setattr("arena.market_data.is_warm_fresh", lambda w, **k: bool(w))
    monkeypatch.setattr(
        "arena.market_data.lay_warm_onto_market",
        lambda m, w: m.__setitem__("current_price", w["yes_price"])
        or m.__setitem__("orderflow", {"obi": w["obi"]}),
    )

    d = MarketDiscovery()
    m = {"id": "m1", "polymarket_token_id": "tok"}
    d._refresh_market_data(m)
    assert m.get("current_price") == 0.55
    assert calls["refresh"] == 0
    assert calls["book"] == 0


def test_discovery_refresh_skips_when_warm_stale(monkeypatch):
    from arena.discovery import MarketDiscovery

    monkeypatch.setattr(
        "polymarket_markets.refresh_price",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("no http")),
    )
    monkeypatch.setattr("arena.market_data.store", lambda: MagicMock(get=lambda *_: None))
    monkeypatch.setattr("arena.market_data.is_warm_fresh", lambda *_a, **_k: False)

    d = MarketDiscovery()
    m = {"id": "m1", "current_price": 0.4}
    d._refresh_market_data(m)
    # Untouched — skip rather than HTTP.
    assert m["current_price"] == 0.4


# ---------------------------------------------------------------------------
# Trader toggle TTL cache
# ---------------------------------------------------------------------------

def test_trader_toggle_cache_hits_db_once_within_ttl(monkeypatch):
    import arena.trader as trader_mod
    import config

    counts = {"one": 0, "hybrid": 0, "lock": 0}

    def _one():
        counts["one"] += 1
        return True

    def _hyb():
        counts["hybrid"] += 1
        return False

    def _lock():
        counts["lock"] += 1
        return True

    monkeypatch.setattr(trader_mod.db, "get_one_trade_per_tick", _one)
    monkeypatch.setattr(trader_mod.db, "get_hybrid_yield", _hyb)
    monkeypatch.setattr(trader_mod.db, "get_directional_window_lock", _lock)

    with trader_mod._TOGGLE_CACHE_LOCK:
        trader_mod._TOGGLE_CACHE.update(
            {"ts": 0.0, "one": None, "hybrid": None, "lock": None}
        )

    a = trader_mod._cached_trade_toggles(config)
    b = trader_mod._cached_trade_toggles(config)
    assert a == (True, False, True)
    assert b == a
    assert counts == {"one": 1, "hybrid": 1, "lock": 1}

    # Expire TTL and confirm refresh.
    with trader_mod._TOGGLE_CACHE_LOCK:
        trader_mod._TOGGLE_CACHE["ts"] = time.time() - 10.0
    trader_mod._cached_trade_toggles(config)
    assert counts["one"] == 2


# ---------------------------------------------------------------------------
# Position monitor: prefer warm store
# ---------------------------------------------------------------------------

def test_position_monitor_prefers_warm_over_http(monkeypatch, arena_db):
    import polymarket_markets
    from arena.position_monitor import PositionMonitorThread, FAST_POLL_INTERVAL
    import arena.market_data as md

    assert FAST_POLL_INTERVAL >= 0.99

    arena_db.log_trade(
        bot_name="mom-p", market_id="pm-warm-1", side="yes",
        amount=5.0, venue="polymarket", mode="paper", shares_bought=10.0,
        fill_source="paper_sim", entry_price=0.50, fee=0.1,
    )

    def _boom(*_a, **_k):
        raise AssertionError("HTTP must not run when warm is fresh")

    monkeypatch.setattr(polymarket_markets, "current_up_price", _boom)

    warm = {"yes_price": 0.63, "no_price": 0.37, "ts": time.time()}
    md.store().put("pm-warm-1", warm)
    monkeypatch.setattr(md, "is_warm_fresh", lambda w, **k: bool(w and w.get("yes_price") is not None))

    prices = PositionMonitorThread()._fetch_market_prices()
    assert prices.get("pm-warm-1") == 0.63


def test_position_monitor_http_fallback_when_warm_missing(monkeypatch, arena_db):
    import polymarket_markets
    from arena.position_monitor import PositionMonitorThread
    import arena.market_data as md

    arena_db.log_trade(
        bot_name="mom-p", market_id="pm-cold-1", side="yes",
        amount=5.0, venue="polymarket", mode="paper", shares_bought=10.0,
        fill_source="paper_sim", entry_price=0.50, fee=0.1,
    )
    monkeypatch.setattr(polymarket_markets, "current_up_price", lambda *_a, **_k: 0.71)
    monkeypatch.setattr(md, "is_warm_fresh", lambda *_a, **_k: False)

    prices = PositionMonitorThread()._fetch_market_prices()
    assert prices.get("pm-cold-1") == 0.71

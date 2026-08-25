"""Kalshi book normalize, market stamp, fee ceil, enable toggle."""

import json

import config
from exchanges import (
    KALSHI, POLYMARKET, exchange_enabled, exchange_of, load_toggles,
    namespace_market_id, native_market_id, save_toggles, stamp_exchange,
)
from kalshi_markets import (
    kalshi_taker_fee, normalize_kalshi_book, normalize_market,
    select_current, select_next, discover_live,
)


def test_namespace_and_native_roundtrip():
    assert namespace_market_id("kalshi", "KXBTC15M-1") == "kalshi:KXBTC15M-1"
    assert namespace_market_id("kalshi", "kalshi:KXBTC15M-1") == "kalshi:KXBTC15M-1"
    assert native_market_id("kalshi:KXBTC15M-1") == "KXBTC15M-1"
    assert native_market_id("0xabc") == "0xabc"


def test_stamp_exchange_sets_identity():
    m = stamp_exchange(
        {"id": "0xdead", "condition_id": "0xdead", "question": "BTC 5m"},
        POLYMARKET, window_sec=300, settlement="chainlink_twap",
    )
    assert m["exchange"] == "polymarket"
    assert m["id"] == "0xdead"
    assert m["window_sec"] == 300
    assert m["settlement"] == "chainlink_twap"
    assert exchange_of(m) == POLYMARKET


def test_normalize_kalshi_book_yes_ask_is_one_minus_no_bid():
    ob = {
        "yes_dollars": [["0.5400", "100.00"], ["0.5300", "50.00"]],
        "no_dollars": [["0.4700", "80.00"], ["0.4600", "20.00"]],
    }
    book = normalize_kalshi_book(ob)
    assert book["valid"] is True
    assert book["yes_bid"] == 0.54
    assert book["no_bid"] == 0.47
    assert book["yes_ask"] == 0.53  # 1 - 0.47
    assert book["no_ask"] == 0.46   # 1 - 0.54
    assert book["yes"]["asks"][0][0] == 0.53
    assert book["no"]["asks"][0][0] == 0.46


def test_normalize_kalshi_book_empty_invalid():
    assert normalize_kalshi_book({})["valid"] is False
    assert normalize_kalshi_book(None)["valid"] is False


def test_normalize_market_namespaces_and_strike():
    raw = {
        "ticker": "KXBTC15M-26AUG231430-00",
        "event_ticker": "KXBTC15M-26AUG231430",
        "series_ticker": "KXBTC15M",
        "title": "BTC Up or Down 15m",
        "status": "active",
        "open_time": "2026-08-23T18:30:00Z",
        "close_time": "2026-08-23T18:45:00Z",
        "floor_strike": 77440.45,
        "yes_bid_dollars": "0.54",
        "no_bid_dollars": "0.47",
    }
    m = normalize_market(raw)
    assert m["exchange"] == KALSHI
    assert m["id"] == "kalshi:KXBTC15M-26AUG231430-00"
    assert m["window_sec"] == 900
    assert m["settlement"] == "brti_last60"
    assert m["floor_strike"] == 77440.45
    assert m["yes_ask"] == 0.53
    assert m["no_ask"] == 0.46
    assert m["current_price"] is not None


def test_kalshi_taker_fee_ceils_to_cent():
    # 100 contracts @ 50¢ → $1.75 exactly (published table).
    assert kalshi_taker_fee(100.0, 0.50) == 1.75
    # 1 contract @ 50¢ → 1.75¢ raw → ceil to $0.02.
    assert kalshi_taker_fee(1.0, 0.50) == 0.02
    assert kalshi_taker_fee(0.0, 0.50) == 0.0
    assert kalshi_taker_fee(10.0, 0.0) == 0.0


def test_select_current_picks_shortest_remaining():
    markets = [
        normalize_market({
            "ticker": "far", "close_time": "2099-01-01T00:00:00Z",
            "status": "active",
        }),
    ]
    # Inject remaining directly (avoid clock dependence).
    a = {"id": "kalshi:a", "exchange": "kalshi", "time_remaining_seconds": 400}
    b = {"id": "kalshi:b", "exchange": "kalshi", "time_remaining_seconds": 80}
    c = {"id": "kalshi:c", "exchange": "kalshi", "time_remaining_seconds": 1200}
    assert select_current([a, b, c])["id"] == "kalshi:b"


def test_select_next_is_later_window():
    cur = {"id": "kalshi:A", "time_remaining_seconds": 400}
    nxt = {"id": "kalshi:B", "time_remaining_seconds": 1300}
    later = {"id": "kalshi:C", "time_remaining_seconds": 2200}
    assert select_next([cur, nxt, later], cur)["id"] == "kalshi:B"
    assert select_next([cur], cur) is None


def test_discover_live_respects_toggle(monkeypatch):
    monkeypatch.setattr("exchanges.exchange_enabled", lambda name: False)
    called = []

    def _client():
        called.append(True)
        return {"markets": []}

    assert discover_live(client=_client) == []
    assert called == []


def test_discover_live_normalizes_when_enabled(monkeypatch):
    monkeypatch.setattr("exchanges.exchange_enabled", lambda name: True)

    def _client():
        return {"markets": [{
            "ticker": "KXBTC15M-X-00",
            "status": "active",
            "close_time": "2099-01-01T00:00:00Z",
            "floor_strike": 100000.0,
            "yes_bid_dollars": "0.50",
            "no_bid_dollars": "0.50",
        }]}

    rows = discover_live(client=_client)
    assert len(rows) == 1
    assert rows[0]["id"] == "kalshi:KXBTC15M-X-00"
    assert rows[0]["exchange"] == "kalshi"


def test_toggles_default_both_on():
    assert config.EXCHANGE_POLYMARKET_ENABLED is True
    assert config.EXCHANGE_KALSHI_ENABLED is True


def test_save_toggles_roundtrip(tmp_path, monkeypatch):
    import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "ex.db")
    db.init_db()
    from exchanges import _TOGGLE_CACHE
    import exchanges as exmod
    exmod._TOGGLE_CACHE = (0.0, {})
    save_toggles({"kalshi": False, "polymarket": True})
    exmod._TOGGLE_CACHE = (0.0, {})
    t = load_toggles()
    assert t["kalshi"] is False
    assert t["polymarket"] is True
    assert exchange_enabled("kalshi") is False
    assert exchange_enabled("polymarket") is True

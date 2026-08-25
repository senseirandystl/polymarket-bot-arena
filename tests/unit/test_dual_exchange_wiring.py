"""Dual-exchange trader / paper / signals wiring (Polymarket + Kalshi)."""

from unittest.mock import MagicMock, patch

from exchanges import KALSHI, exchange_of, stamp_exchange
from kalshi_markets import normalize_kalshi_book
from tests.conftest import make_market, make_book
from venues.paper import PaperEngine


def test_paper_kalshi_fill_writes_venue(arena_db, monkeypatch):
    books = normalize_kalshi_book({
        "yes_dollars": [["0.54", "200"]],
        "no_dollars": [["0.47", "200"]],
    })
    yes = dict(books["yes"])
    yes["exchange"] = "kalshi"
    market = {
        "id": "kalshi:KXBTC15M-X-00",
        "ticker": "KXBTC15M-X-00",
        "exchange": "kalshi",
        "question": "BTC 15m",
        "yes_book": yes,
        "window_sec": 900,
    }
    res = PaperEngine.instance().place(
        bot_name="mom-k", side="yes", amount=10.0, market=market, mode="paper",
        book=yes,
    )
    assert res.success, res.reason
    with arena_db.get_conn() as conn:
        row = conn.execute("SELECT * FROM trades WHERE bot_name='mom-k'").fetchone()
    assert row["venue"] == "kalshi"
    assert row["fee"] > 0


def test_trader_evaluates_each_enabled_exchange(monkeypatch):
    from arena.trader import Trader
    from arena.state import SharedArenaState

    pm = make_market(market_id="pm-1", time_remaining=120)
    pm["exchange"] = "polymarket"
    ks = stamp_exchange(
        {"id": "KXBTC15M-Y", "ticker": "KXBTC15M-Y", "question": "k",
         "time_remaining_seconds": 400, "yes_ask": 0.51, "no_ask": 0.51,
         "current_price": 0.50, "no_price": 0.50},
        KALSHI, window_sec=900, settlement="brti_last60",
    )
    disc = MagicMock()
    disc.current_markets_snapshot.return_value = {
        "polymarket": pm, "kalshi": ks,
    }
    seen = []

    class _Bot:
        name = "t-bot"
        strategy_type = "momentum"
        trading_mode = "paper"

        def make_decision(self, market, signals):
            seen.append(market.get("id"))
            return {"action": "skip", "reasoning": "test", "skip_reason": "test"}

    tr = Trader(disc, SharedArenaState(), None, None, None)
    tr.set_bots([_Bot()])
    warm = {
        "yes_price": 0.50, "no_price": 0.50, "ts": 9e12,
        "yes_book": make_book(), "no_book": make_book(),
    }
    monkeypatch.setattr("arena.market_data.store", lambda: MagicMock(get=lambda _id: warm))
    monkeypatch.setattr("arena.market_data.is_warm_fresh", lambda w: True)
    monkeypatch.setattr("arena.market_data.lay_warm_onto_market", lambda m, w: None)
    monkeypatch.setattr("arena.trader.build_combined_signals", lambda *a, **k: {
        "btc_drift": 0.0, "prices": [100.0] * 10,
    })
    monkeypatch.setattr("arena.trader.session_skip", lambda _n: None)
    monkeypatch.setattr("arena.risk_engine.is_killed", lambda: False)
    tr._tick()
    assert "pm-1" in seen
    assert any(str(x).startswith("kalshi:") for x in seen)


def test_candle_prices_kalshi_uses_brti_not_twap():
    from signals import brti
    from signals.tape import candle_prices
    brti.record_tick(1_700_000.0, 77000.0, source="test")
    brti.record_tick(1_700_060.0, 77100.0, source="test")
    brti.record_tick(1_700_120.0, 77200.0, source="test")
    brti.record_tick(1_700_180.0, 77300.0, source="test")
    brti.record_tick(1_700_240.0, 77400.0, source="test")
    ticks = [
        (1_700_000.0, 77000.0), (1_700_060.0, 77100.0),
        (1_700_120.0, 77200.0), (1_700_180.0, 77300.0),
        (1_700_240.0, 77400.0),
    ]
    market = {"id": "kalshi:KXBTC15M-TAPE", "exchange": "kalshi"}
    px = candle_prices(market, {
        "btc_brti_ticks": ticks,
        "btc_twap_ticks": [(1.0, 99999.0)] * 10,
        "prices": [99999.0] * 10,
    }, sample_sec=60.0)
    assert px, "BRTI tape should resample"
    assert max(px) < 90000
    assert min(px) >= 77000


def test_candle_prices_polymarket_ignores_brti():
    from signals.tape import candle_prices
    market = {"id": "0xabc", "exchange": "polymarket"}
    px = candle_prices(market, {
        "btc_brti_ticks": [(1.0, 11111.0)] * 10,
        "prices": [100_000.0 + i for i in range(10)],
    }, sample_sec=60.0)
    assert px[0] >= 100_000


def test_kalshi_signals_tape_source_is_brti(monkeypatch):
    from arena.signals import build_combined_signals
    from signals import brti
    for i in range(8):
        brti.record_tick(1_800_000.0 + i * 60, 77000.0 + i, source="test")
    market = {
        "id": "kalshi:KXBTC15M-TAPE2",
        "exchange": "kalshi",
        "floor_strike": 76900.0,
        "resolves_at": "2099-01-01T00:00:00Z",
        "time_remaining_seconds": 400,
        "window_sec": 900,
    }
    sigs = build_combined_signals(None, None, None, market, warm={
        "strike": 76900.0, "obi": 0.0, "cvd": 0.0,
        "pm_momentum": 0.0, "pm_prices": [],
    })
    assert sigs.get("tape_source") == "brti"
    assert sigs.get("btc_twap_ticks") == []
    assert sigs.get("btc_brti_ticks")


def test_kalshi_signals_do_not_use_polymarket_twap(monkeypatch):
    from arena.signals import build_combined_signals
    from signals import brti
    brti.record_tick(1_000_000.0, 77000.0)
    market = {
        "id": "kalshi:KXBTC15M-Z",
        "exchange": "kalshi",
        "floor_strike": 76900.0,
        "resolves_at": "2099-01-01T00:00:00Z",
        "time_remaining_seconds": 200,
        "window_sec": 900,
    }
    sigs = build_combined_signals(None, None, None, market, warm={
        "strike": 76900.0, "obi": 0.0, "cvd": 0.0,
        "pm_momentum": 0.0, "pm_prices": [],
    })
    assert sigs.get("btc_strike") == 76900.0
    src = (sigs.get("resolution_source") or sigs.get("btc_now_source")
           or "")
    # Must not claim Chainlink TWAP for a Kalshi window.
    assert "twap" not in str(src).lower() or "brti" in str(sigs).lower()
    assert sigs.get("btc_strike_source") in ("kalshi_floor", "brti_open")


def test_kalshi_signals_never_call_pm_strike_registry(monkeypatch):
    from arena import signals as sigmod
    from signals import brti

    def _boom(*_a, **_k):
        raise AssertionError("PM strike registry must not run for Kalshi")

    monkeypatch.setattr(sigmod, "get_strike_registry", _boom)
    brti.record_tick(1_700_000.0, 77100.0)
    sigs = sigmod.build_combined_signals(None, None, None, {
        "id": "kalshi:KXBTC15M-REG",
        "exchange": "kalshi",
        "floor_strike": 77000.0,
        "resolves_at": "2099-01-01T00:00:00Z",
        "time_remaining_seconds": 400,
        "window_sec": 900,
    })
    assert sigs.get("btc_strike") == 77000.0
    assert sigs.get("btc_strike_source") == "kalshi_floor"


def test_get_engine_routes_live_kalshi():
    from venues import get_engine
    from venues.kalshi_live import KalshiLiveEngine
    from venues.live import LiveEngine
    from venues.paper import PaperEngine
    assert get_engine("live", exchange="kalshi") is KalshiLiveEngine.instance()
    assert get_engine("live", exchange="polymarket") is LiveEngine.instance()
    assert get_engine("paper", exchange="kalshi") is PaperEngine.instance()


def test_arb_kalshi_does_not_require_pm_tokens():
    from bots.bot_arbitrage import ArbitrageBot
    d = ArbitrageBot(name="arb-k").make_decision(
        {"id": "kalshi:X", "exchange": "kalshi", "ticker": "X"}, {},
    )
    assert d["action"] == "skip"
    assert "token" not in (d.get("reasoning") or "").lower()


def test_kalshi_live_toggle_fail_closed(monkeypatch, arena_db):
    import exchanges
    from venues.kalshi_live import KalshiLiveEngine

    monkeypatch.setattr(exchanges, "exchange_enabled",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("db")))
    res = KalshiLiveEngine().place(
        bot_name="x", side="yes", amount=1.0,
        market={"ticker": "KXBTC15M-T", "id": "kalshi:KXBTC15M-T"},
        mode="live",
    )
    assert res.success is False
    assert res.reason == "kalshi_toggle_error"


def test_kalshi_live_disabled_toggle(monkeypatch):
    import exchanges
    from venues.kalshi_live import KalshiLiveEngine

    monkeypatch.setattr(exchanges, "exchange_enabled", lambda *_a, **_k: False)
    res = KalshiLiveEngine().place(
        bot_name="x", side="yes", amount=1.0,
        market={"ticker": "KXBTC15M-T"}, mode="live",
    )
    assert res.success is False
    assert res.reason == "kalshi_disabled"


def test_kalshi_live_post_retries_zero(monkeypatch, arena_db):
    import exchanges
    import kalshi_client
    from venues.kalshi_live import KalshiLiveEngine

    captured = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"order": {
                "order_id": "oid-1", "count": 1, "fill_count": 1,
                "status": "executed", "yes_price_dollars": "0.55",
            }}

    def _req(method, path, **kw):
        captured.update(kw)
        captured["method"] = method
        captured["path"] = path
        return _Resp()

    monkeypatch.setattr(exchanges, "exchange_enabled", lambda *_a, **_k: True)
    monkeypatch.setattr(kalshi_client, "has_auth", lambda: True)
    monkeypatch.setattr(kalshi_client, "request", _req)
    book = {
        "valid": True, "asks": [(0.55, 50)], "bids": [(0.54, 50)],
        "exchange": "kalshi", "min_order_size": 1,
    }
    res = KalshiLiveEngine().place(
        bot_name="mom-k", side="yes", amount=5.0,
        market={"ticker": "KXBTC15M-T", "id": "kalshi:KXBTC15M-T"},
        mode="live", expected_price=0.55, book=book, target_shares=1,
    )
    assert res.success, res.reason
    assert captured.get("retries") == 0
    assert captured.get("method") == "POST"
    body = captured.get("json_body") or {}
    assert body.get("type") == "limit"
    assert body.get("time_in_force") == "fill_or_kill"
    assert body.get("yes_price") == 58  # 0.55 + MAX_FILL_SLIPPAGE 0.03


def test_kalshi_live_slippage_probe_blocks_post(monkeypatch):
    import exchanges
    import kalshi_client
    from venues.kalshi_live import KalshiLiveEngine

    called = {"n": 0}

    def _req(*_a, **_k):
        called["n"] += 1
        raise AssertionError("must not POST when slippage probe fails")

    monkeypatch.setattr(exchanges, "exchange_enabled", lambda *_a, **_k: True)
    monkeypatch.setattr(kalshi_client, "has_auth", lambda: True)
    monkeypatch.setattr(kalshi_client, "request", _req)
    book = {
        "valid": True, "asks": [(0.80, 50)], "bids": [(0.79, 50)],
        "exchange": "kalshi", "min_order_size": 1,
    }
    res = KalshiLiveEngine().place(
        bot_name="mom-k", side="yes", amount=5.0,
        market={"ticker": "KXBTC15M-T", "id": "kalshi:KXBTC15M-T"},
        mode="live", expected_price=0.50, book=book, target_shares=1,
    )
    assert res.success is False
    assert res.reason == "slippage_exceeded"
    assert called["n"] == 0


def test_kalshi_live_unfilled_status_does_not_log(monkeypatch, arena_db):
    import exchanges
    import kalshi_client
    from venues.kalshi_live import KalshiLiveEngine

    class _Resp:
        status_code = 200

        def json(self):
            return {"order": {
                "order_id": "oid-rest", "count": 1, "status": "resting",
                "remaining_count": 1, "yes_price_dollars": "0.55",
            }}

    monkeypatch.setattr(exchanges, "exchange_enabled", lambda *_a, **_k: True)
    monkeypatch.setattr(kalshi_client, "has_auth", lambda: True)
    monkeypatch.setattr(kalshi_client, "request", lambda *_a, **_k: _Resp())
    book = {
        "valid": True, "asks": [(0.55, 50)], "bids": [(0.54, 50)],
        "exchange": "kalshi", "min_order_size": 1,
    }
    res = KalshiLiveEngine().place(
        bot_name="mom-k", side="yes", amount=5.0,
        market={"ticker": "KXBTC15M-T", "id": "kalshi:KXBTC15M-REST"},
        mode="live", expected_price=0.55, book=book, target_shares=1,
    )
    assert res.success is False
    assert res.reason == "kalshi_unfilled"
    with arena_db.get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE market_id='kalshi:KXBTC15M-REST'"
        ).fetchone()[0]
    assert n == 0


def test_kalshi_live_slippage_no_book_blocks_post(monkeypatch):
    import exchanges
    import kalshi_client
    import kalshi_markets
    from venues.kalshi_live import KalshiLiveEngine

    called = {"n": 0}

    def _req(*_a, **_k):
        called["n"] += 1
        raise AssertionError("must not POST without a probe book")

    monkeypatch.setattr(exchanges, "exchange_enabled", lambda *_a, **_k: True)
    monkeypatch.setattr(kalshi_client, "has_auth", lambda: True)
    monkeypatch.setattr(kalshi_client, "request", _req)
    monkeypatch.setattr(
        kalshi_markets, "get_order_book", lambda *_a, **_k: {"valid": False},
    )
    res = KalshiLiveEngine().place(
        bot_name="mom-k", side="yes", amount=5.0,
        market={"ticker": "KXBTC15M-T", "id": "kalshi:KXBTC15M-T"},
        mode="live", expected_price=0.55, book=None, target_shares=1,
    )
    assert res.success is False
    assert res.reason == "slippage_no_book"
    assert called["n"] == 0


def test_position_monitor_does_not_hit_pm_clob_for_kalshi(monkeypatch, arena_db):
    import kalshi_markets
    import polymarket_markets
    from arena.position_monitor import PositionMonitorThread

    def _pm_boom(*_a, **_k):
        raise AssertionError("PM CLOB must not price Kalshi ids")

    monkeypatch.setattr(polymarket_markets, "current_up_price", _pm_boom)
    monkeypatch.setattr(kalshi_markets, "current_up_price", lambda *_a, **_k: 0.61)
    arena_db.log_trade(
        bot_name="mom-k", market_id="kalshi:KXBTC15M-PMON", side="yes",
        amount=5.0, venue="kalshi", mode="paper", shares_bought=10.0,
        fill_source="paper_sim", entry_price=0.50, fee=0.1,
    )
    prices = PositionMonitorThread()._fetch_market_prices()
    assert prices.get("kalshi:KXBTC15M-PMON") == 0.61

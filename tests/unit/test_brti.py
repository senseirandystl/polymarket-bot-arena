"""Kalshi BRTI last-60s nowcast and strike latch (not Chainlink TWAP)."""

from signals.brti import brti_certainty, last60_average, latch_strike


def test_last60_average_means_settlement_prints():
    expiry = 1_000.0
    ticks = [(expiry - 60 + i, 100.0 + i) for i in range(61)]  # 100..160
    out = last60_average(ticks, now=expiry, expiry=expiry)
    assert out["n"] == 61
    assert out["coverage"] == 1.0
    assert out["in_settlement"] is True
    assert abs(out["brti_now"] - sum(range(100, 161)) / 61) < 1e-9


def test_last60_ignores_prints_before_settlement_window():
    expiry = 1_000.0
    ticks = [(expiry - 400, 50.0), (expiry - 10, 110.0), (expiry - 5, 112.0)]
    out = last60_average(ticks, now=expiry, expiry=expiry)
    assert out["n"] == 2
    assert out["last"] == 112.0
    # Fill-forward uses last print for remaining settlement seconds.
    assert 110.0 < out["brti_now"] <= 112.0


def test_last60_empty_is_none():
    out = last60_average([], now=100.0, expiry=200.0)
    assert out["brti_now"] is None
    assert out["coverage"] == 0.0


def test_latch_strike_prefers_floor_not_mid_window():
    src, kind = latch_strike(77440.45, brti_at_open=77000.0)
    assert (src, kind) == (77440.45, "kalshi_floor")
    src2, kind2 = latch_strike(None, brti_at_open=77000.0)
    assert (src2, kind2) == (77000.0, "brti_open")
    src3, kind3 = latch_strike(None, None)
    assert kind3 == "none"


def test_certainty_zero_without_coverage():
    assert brti_certainty(coverage=0.0, elapsed_frac=1.0, abs_drift=1.0) == 0.0
    assert brti_certainty(coverage=1.0, elapsed_frac=1.0, abs_drift=0.20) == 1.0


def test_poll_px_ignores_price_to_beat():
    from signals.brti import _btc_scale_px
    assert _btc_scale_px({"price_to_beat": 77440.0, "floor_strike": 77440.0}) is None
    assert _btc_scale_px({"last_price": 77501.25}) == 77501.25
    assert _btc_scale_px({"yes_bid_dollars": "0.54"}) is None


def test_ingest_kalshi_ws_parses_nested_data_and_avg60():
    from signals.brti import ingest_kalshi_ws_msg
    msg = {
        "type": "cfbenchmarks_value",
        "msg": {
            "index_id": "BRTI",
            "received_at": 1710000000123,
            "data": '{"type":"value","id":"BRTI","time":1710000000123,"value":"68000.12"}',
            "avg_60s_data": {"value": "68001.50", "window_size": 3},
        },
    }
    out = ingest_kalshi_ws_msg(msg)
    assert out is not None
    assert abs(out["last"] - 68000.12) < 1e-6
    assert abs(out["avg60"] - 68001.50) < 1e-6
    assert out["ts"] == 1710000000.123


def test_parse_cfb_index_summary_value():
    from signals.brti import parse_cfb_index_summary
    html = (
        '<html><script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"pageProps":{"indexSummary":{"value":"77207.88",'
        '"valueChange24":"0.64"}}}}'
        "</script></html>"
    )
    assert parse_cfb_index_summary(html) == 77207.88
    assert parse_cfb_index_summary("<html></html>") is None


def test_snapshot_computes_local_avg60_without_ws(monkeypatch):
    import time as _t
    from signals import brti
    with brti._lock():
        brti._TICKS.clear()
        brti._LAST_AVG60 = None
        brti._LAST_SETTLE60 = None
        brti._LAST_SOURCE = "none"
    now = _t.time()
    brti.record_tick(now - 40.0, 100.0, source="cfb_page")
    brti.record_tick(now - 10.0, 200.0, source="cfb_page")
    snap = brti.snapshot()
    assert snap["avg60"] is not None
    assert 100.0 < snap["avg60"] < 200.0 or abs(snap["avg60"] - 150.0) < 1.0
    assert "local60" in str(snap.get("source") or "") or snap.get("source") == "cfb_page+local60"


def test_published_status_roundtrip(arena_db):
    from signals import brti
    with brti._lock():
        brti._TICKS.clear()
        brti._LAST_AVG60 = None
        brti._LAST_SETTLE60 = None
        brti._LAST_SOURCE = "none"
    brti.record_tick(1_700_000.0, 77111.25, source="test")
    snap = brti.publish_status()
    assert snap["last"] == 77111.25
    loaded = brti.load_published()
    assert loaded.get("last") == 77111.25
    assert str(loaded.get("source") or "").startswith("test")

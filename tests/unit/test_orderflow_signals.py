"""Tests for order-flow signals: OBI (order-book imbalance) and CVD."""

from signals import orderflow_signals as ofs


# --- OBI --------------------------------------------------------------------

def test_obi_bid_heavy_is_positive():
    book = {"valid": True,
            "bids": [(0.49, 100.0), (0.48, 100.0)],
            "asks": [(0.51, 10.0), (0.52, 10.0)]}
    obi = ofs.order_book_imbalance(book)
    assert obi > 0.5  # heavy bids, thin asks → strong upward pressure


def test_obi_ask_heavy_is_negative():
    book = {"valid": True,
            "bids": [(0.49, 10.0)],
            "asks": [(0.51, 100.0)]}
    assert ofs.order_book_imbalance(book) < 0


def test_obi_balanced_is_zero():
    book = {"valid": True,
            "bids": [(0.49, 50.0)],
            "asks": [(0.51, 50.0)]}
    assert ofs.order_book_imbalance(book) == 0.0


def test_obi_invalid_or_empty_book_is_zero():
    assert ofs.order_book_imbalance({"valid": False}) == 0.0
    assert ofs.order_book_imbalance({}) == 0.0
    assert ofs.order_book_imbalance({"valid": True, "bids": [], "asks": []}) == 0.0


def test_obi_respects_levels_cap():
    # Deep bid levels beyond `levels` must not count.
    book = {"valid": True,
            "bids": [(0.49, 10.0), (0.48, 10.0), (0.47, 1000.0)],
            "asks": [(0.51, 10.0), (0.52, 10.0), (0.53, 10.0)]}
    obi = ofs.order_book_imbalance(book, levels=2)
    assert obi == 0.0  # top-2 are balanced; the deep 1000-bid is ignored


# --- CVD --------------------------------------------------------------------

# Sizes below sit ABOVE config.CVD_VOLUME_FLOOR (200 shares) so the sign
# semantics read at full magnitude; thin tapes are damped by design (BUG #27
# — see test_cvd_calibration.py).

def test_cvd_buy_up_is_bullish():
    trades = [{"side": "BUY", "outcome": "Up", "size": 400}]
    assert ofs.cvd_from_trades(trades) == 1.0


def test_cvd_sell_down_is_bullish():
    trades = [{"side": "SELL", "outcome": "Down", "size": 400}]
    assert ofs.cvd_from_trades(trades) == 1.0


def test_cvd_buy_down_and_sell_up_are_bearish():
    assert ofs.cvd_from_trades([{"side": "BUY", "outcome": "Down", "size": 500}]) == -1.0
    assert ofs.cvd_from_trades([{"side": "SELL", "outcome": "Up", "size": 500}]) == -1.0


def test_cvd_nets_out():
    trades = [
        {"side": "BUY", "outcome": "Up", "size": 300},    # +300
        {"side": "BUY", "outcome": "Down", "size": 100},  # -100
    ]
    assert ofs.cvd_from_trades(trades) == (300 - 100) / (300 + 100)


def test_cvd_empty_and_bad_rows():
    assert ofs.cvd_from_trades([]) == 0.0
    assert ofs.cvd_from_trades([{"side": "BUY", "outcome": "Up", "size": 0}]) == 0.0
    assert ofs.cvd_from_trades([{"side": "BUY", "outcome": "Up", "size": "x"}]) == 0.0


# --- CvdFeed caching --------------------------------------------------------

def test_cvd_feed_caches_and_falls_back(monkeypatch):
    feed = ofs.CvdFeed()
    calls = {"n": 0}

    def fake_fetch(cond):
        calls["n"] += 1
        return [{"side": "BUY", "outcome": "Up", "size": 400}]

    monkeypatch.setattr(feed, "_fetch", fake_fetch)
    assert feed.get_cvd("cond1") == 1.0
    assert feed.get_cvd("cond1") == 1.0
    assert calls["n"] == 1  # second read served from cache

    # A transient empty fetch keeps the last good value, doesn't reset to 0.
    feed.clear("cond1")
    monkeypatch.setattr(feed, "_fetch", lambda c: [{"side": "BUY", "outcome": "Up", "size": 400}])
    feed.get_cvd("cond1")
    monkeypatch.setattr(feed, "_fetch", lambda c: [])
    feed.clear()  # force a fetch attempt but keep no cache → 0.0 on genuine empty
    assert feed.get_cvd("cond1") == 0.0

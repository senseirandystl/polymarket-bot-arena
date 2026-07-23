"""Unit tests for batch CLOB pricing (POST /midpoints)."""

import polymarket_markets as pm


class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def test_midpoints_batch_parses_map(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        return _FakeResp(200, {"tokA": "0.045", "tokB": "0.975", "bad": "x"})

    monkeypatch.setattr(pm.requests, "post", fake_post)
    out = pm.midpoints_batch(["tokA", "tokB", "bad", None, ""])

    assert captured["url"].endswith("/midpoints")
    # None / "" tokens are dropped before the request.
    assert captured["body"] == [{"token_id": "tokA"}, {"token_id": "tokB"},
                                {"token_id": "bad"}]
    assert out == {"tokA": 0.045, "tokB": 0.975}  # unparseable "bad" skipped


def test_midpoints_batch_empty_and_failure(monkeypatch):
    assert pm.midpoints_batch([]) == {}

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(pm.requests, "post", boom)
    assert pm.midpoints_batch(["tokA"]) == {}  # failure -> {} (caller falls back)


def test_price_markets_sets_yes_and_no_in_one_call(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, json=None, timeout=None):
        calls["n"] += 1
        return _FakeResp(200, {"up1": "0.30", "down1": "0.68", "up2": "0.90"})

    monkeypatch.setattr(pm.requests, "post", fake_post)
    markets = [
        {"polymarket_token_id": "up1", "polymarket_no_token_id": "down1"},
        {"polymarket_token_id": "up2", "polymarket_no_token_id": "down2"},
        None,  # tolerated
    ]
    pm.price_markets(markets)

    assert calls["n"] == 1  # ONE batch call for all markets
    assert markets[0]["current_price"] == 0.30
    assert markets[0]["no_price"] == 0.68            # real Down mid
    assert markets[1]["current_price"] == 0.90
    assert markets[1]["no_price"] == round(1.0 - 0.90, 4)  # derived (down2 absent)

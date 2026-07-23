"""Retry/backoff behavior of the shared HTTP client (slice C)."""

import requests

import http_client


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def _patch(monkeypatch, outcomes):
    """Feed `outcomes` (status ints or Exception instances) one per call.

    Returns a list that records how many times requests.request was invoked.
    time.sleep is stubbed so retries are instant.
    """
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url))
        item = outcomes[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return _Resp(item)

    monkeypatch.setattr(http_client.requests, "request", fake_request)
    monkeypatch.setattr(http_client.time, "sleep", lambda *_a, **_k: None)
    return calls


def test_success_first_try_no_retry(monkeypatch):
    calls = _patch(monkeypatch, [200])
    resp = http_client.get("http://x", retries=2)
    assert resp.status_code == 200
    assert len(calls) == 1  # no retries when the first call succeeds


def test_retries_transient_status_then_succeeds(monkeypatch):
    calls = _patch(monkeypatch, [503, 200])
    resp = http_client.get("http://x", retries=2)
    assert resp.status_code == 200
    assert len(calls) == 2  # one retry after the 503


def test_exhausts_retries_returns_last_response(monkeypatch):
    # A persistently-failing transient status returns the final response (the
    # caller inspects status_code exactly as before) — it does NOT raise.
    calls = _patch(monkeypatch, [503, 503, 503])
    resp = http_client.get("http://x", retries=2)
    assert resp.status_code == 503
    assert len(calls) == 3  # first + 2 retries


def test_no_retry_on_non_transient_status(monkeypatch):
    calls = _patch(monkeypatch, [404, 200])
    resp = http_client.get("http://x", retries=2)
    assert resp.status_code == 404
    assert len(calls) == 1  # 404 is not a retryable status


def test_retries_request_exception_then_succeeds(monkeypatch):
    calls = _patch(monkeypatch, [requests.ConnectionError("reset"), 200])
    resp = http_client.get("http://x", retries=2)
    assert resp.status_code == 200
    assert len(calls) == 2


def test_persistent_exception_raises_after_exhaustion(monkeypatch):
    calls = _patch(monkeypatch, [requests.Timeout("t")] * 3)
    try:
        http_client.get("http://x", retries=2)
        assert False, "expected the underlying exception to propagate"
    except requests.Timeout:
        pass
    assert len(calls) == 3  # first + 2 retries, then raise


def test_custom_retry_statuses(monkeypatch):
    # 418 is not in the default set, but callers can opt in.
    calls = _patch(monkeypatch, [418, 200])
    resp = http_client.get("http://x", retries=1, retry_statuses=(418,))
    assert resp.status_code == 200
    assert len(calls) == 2

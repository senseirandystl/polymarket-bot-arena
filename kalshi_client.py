"""Kalshi Trade API client (REST). Auth is RSA-PSS when keys are present.

Public GETs (markets list) work without keys. Orderbook / private / WS BRTI
need ``kalshi_api_key_id`` + ``kalshi_private_key_pem`` in the credential store.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Optional
from urllib.parse import urlparse

import config
import http_client

logger = logging.getLogger("kalshi.client")


def _base() -> str:
    return str(getattr(config, "KALSHI_API_BASE",
                       "https://external-api.kalshi.com/trade-api/v2")).rstrip("/")


def has_auth() -> bool:
    try:
        kid = config.get_credential("kalshi_api_key_id")
        pem = config.get_credential("kalshi_private_key_pem")
        return bool(kid and pem)
    except Exception:
        return False


def _sign_headers(method: str, path: str) -> dict:
    """Kalshi RSA-PSS headers. ``path`` is the URL path including query."""
    kid = config.get_credential("kalshi_api_key_id")
    pem = config.get_credential("kalshi_private_key_pem")
    if not kid or not pem:
        return {}
    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + path).encode("utf-8")
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
        sig = key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": kid,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("ascii"),
        }
    except Exception as e:
        logger.warning("Kalshi sign failed: %s", e)
        return {}


def request(method: str, path: str, *, params=None, json_body=None,
            timeout: float = 15.0, auth: bool = False,
            retries: int | None = None):
    rel = path if path.startswith("/") else "/" + path
    if params:
        from urllib.parse import urlencode
        items = []
        for k, v in params.items():
            if v is None:
                continue
            items.append((str(k), str(v)))
        qs = urlencode(sorted(items))
        if qs:
            rel = rel + ("&" if "?" in rel else "?") + qs
        params = None
    url = _base() + rel
    headers = {}
    if auth or has_auth():
        parsed = urlparse(url)
        prefix = urlparse(_base()).path or "/trade-api/v2"
        sign_path = parsed.path if parsed.path.startswith("/trade-api") else (
            prefix.rstrip("/") + parsed.path
        )
        if parsed.query:
            sign_path += "?" + parsed.query
        headers.update(_sign_headers(method, sign_path))
    kwargs = {"timeout": timeout}
    if retries is not None:
        kwargs["retries"] = retries
    if json_body is not None:
        kwargs["json"] = json_body
    if headers:
        kwargs["headers"] = headers
    return http_client.request_with_retry(method.upper(), url, **kwargs)


def get_json(path: str, *, params=None, timeout: float = 15.0,
             auth: bool = False) -> Optional[dict]:
    try:
        resp = request("GET", path, params=params, timeout=timeout, auth=auth)
        if resp is None or resp.status_code >= 400:
            if resp is not None and resp.status_code in (401, 403) and not auth:
                return get_json(path, params=params, timeout=timeout, auth=True)
            logger.debug("Kalshi GET %s -> %s", path,
                         getattr(resp, "status_code", None))
            return None
        return resp.json()
    except Exception as e:
        logger.debug("Kalshi GET %s failed: %s", path, e)
        return None

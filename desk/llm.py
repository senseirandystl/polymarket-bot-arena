"""Optional research LLM. Never used for order placement.

Providers:
  none   — caller should skip this module
  ollama — local OpenAI-compatible /api/chat (Umbrel / Jetson)
  grok   — xAI REST https://api.x.ai/v1/chat/completions
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("desk.llm")


def configured_provider() -> str:
    import config

    raw = (
        os.environ.get("DESK_LLM_PROVIDER")
        or getattr(config, "DESK_LLM_PROVIDER", "none")
        or "none"
    )
    return str(raw).strip().lower()


def complete_json(system: str, user: str, *, timeout: float = 45.0) -> dict[str, Any] | None:
    """Ask the configured provider for a JSON object. None on any failure."""
    provider = configured_provider()
    if provider in ("", "none", "off", "heuristic"):
        return None
    try:
        if provider == "ollama":
            text = _ollama(system, user, timeout=timeout)
        elif provider in ("grok", "xai"):
            text = _grok(system, user, timeout=timeout)
        else:
            logger.warning("unknown DESK_LLM_PROVIDER=%s", provider)
            return None
    except Exception:
        logger.warning("desk LLM call failed", exc_info=True)
        return None
    if not text:
        return None
    return _parse_json_object(text)


def _ollama(system: str, user: str, *, timeout: float) -> str:
    import requests
    import config

    host = os.environ.get("OLLAMA_HOST") or getattr(
        config, "OLLAMA_HOST", "http://127.0.0.1:11434"
    )
    model = os.environ.get("OLLAMA_MODEL") or getattr(
        config, "OLLAMA_MODEL", "llama3.1"
    )
    url = str(host).rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return str((data.get("message") or {}).get("content") or "")


def _grok(system: str, user: str, *, timeout: float) -> str:
    import requests
    import config

    key = os.environ.get("XAI_API_KEY") or getattr(config, "XAI_API_KEY", "") or ""
    if not key:
        raise RuntimeError("XAI_API_KEY missing")
    model = os.environ.get("XAI_MODEL") or getattr(config, "XAI_MODEL", "grok-4")
    url = "https://api.x.ai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
    }
    resp = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "")


def _parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None

"""Optional LLM assist for Strategy Lab (research / params / autopsy narrative).

Providers: none | ollama | grok (xAI). Honors STRATEGY_LAB_LLM_PROVIDER via
control overlay, then config / env. Never required — silent fallback to
deterministic path. Never executes LLM Python source; only JSON specs within
the compiler allowlist.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("strategy_pipeline.llm")

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

SYSTEM_RESEARCH = (
    "You design parameterized trading-bot specs for binary crypto Up/Down "
    "markets. Reply JSON only: a list of objects with keys primitive, name, "
    "thesis, params. Do not invent primitives. No arbitrage. "
    "Do not emit source code."
)

SYSTEM_NARRATE = (
    "You write a short postmortem narrative (2-4 sentences) for a failed "
    "trading strategy autopsy. Reply plain text only. No code. No secrets."
)

SYSTEM_PARAMS = (
    "You suggest parameter adjustments for one trading primitive under "
    "avoid constraints. Reply JSON object of numeric params only. No code."
)


def provider_name() -> str:
    try:
        from signals.strategy_pipeline.control import cfg

        p = str(cfg("STRATEGY_LAB_LLM_PROVIDER", "none") or "none").strip().lower()
    except Exception:
        try:
            import config
            p = str(getattr(config, "STRATEGY_LAB_LLM_PROVIDER", "none") or "none").lower()
        except Exception:
            p = "none"
    if p in ("off", "heuristic", ""):
        return "none"
    if p in ("xai", "x-ai", "x_ai"):
        return "grok"
    if p not in ("none", "ollama", "grok"):
        return "none"
    return p


def _chat(messages: list[dict[str, str]], *, temperature: float = 0.2) -> str | None:
    """Send chat messages; return assistant text or None on any failure."""
    prov = provider_name()
    if prov == "none":
        return None
    try:
        if prov == "ollama":
            return _ollama_chat(messages, temperature=temperature)
        if prov == "grok":
            return _xai_chat(messages, temperature=temperature)
    except Exception as e:
        logger.debug("llm chat failed (%s): %s", prov, e)
    return None


def _ollama_chat(messages: list[dict[str, str]], *, temperature: float) -> str | None:
    import config

    host = str(getattr(config, "OLLAMA_HOST", "http://127.0.0.1:11434") or "").rstrip("/")
    model = str(getattr(config, "OLLAMA_MODEL", "llama3.1") or "llama3.1")
    if not host:
        return None
    url = f"{host}/api/chat"
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    msg = (payload or {}).get("message") or {}
    content = msg.get("content")
    return str(content) if content else None


def _xai_chat(messages: list[dict[str, str]], *, temperature: float) -> str | None:
    import config
    import os

    key = (
        str(getattr(config, "XAI_API_KEY", "") or "").strip()
        or str(os.environ.get("XAI_API_KEY") or "").strip()
    )
    if not key:
        logger.debug("XAI_API_KEY missing; skipping grok assist")
        return None
    model = str(getattr(config, "XAI_MODEL", "grok-4") or "grok-4")
    url = "https://api.x.ai/v1/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    choices = (payload or {}).get("choices") or []
    if not choices:
        return None
    content = ((choices[0] or {}).get("message") or {}).get("content")
    return str(content) if content else None


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty")
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    # Find first JSON array/object.
    for starter, ender in (("[", "]"), ("{", "}")):
        i = text.find(starter)
        if i < 0:
            continue
        j = text.rfind(ender)
        if j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def research_assist(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Optional LLM research proposals. Validated later by compiler sanitize.

    Returns [] when provider is none or the call fails.
    """
    if provider_name() == "none":
        return []
    compact = {
        "universe": context.get("universe") or [],
        "avoid_fingerprints": (context.get("avoid_fingerprints") or [])[:6],
        "avoid_param_bands": (context.get("avoid_param_bands") or [])[:8],
        "prefer_factor_cells": [
            c for c in (context.get("prefer_factor_cells") or [])
            if c.get("kind") == "prefer"
        ][:6],
        "autopsy_lessons": [
            (a.get("autopsy") or {}).get("lesson")
            or (a.get("autopsy") or {}).get("verdict")
            or (a.get("autopsy") or {}).get("reason")
            for a in (context.get("autopsies") or [])[:5]
        ],
    }
    user = (
        "Propose up to 3 distinct strategy specs as a JSON list. "
        "Respect avoid fingerprints and param bands. Prefer positive factor cells. "
        f"Context:\n{json.dumps(compact, default=str)[:6000]}"
    )
    raw = _chat([
        {"role": "system", "content": SYSTEM_RESEARCH},
        {"role": "user", "content": user},
    ])
    if not raw:
        return []
    try:
        data = _extract_json(raw)
    except Exception as e:
        logger.debug("research_assist JSON parse failed: %s", e)
        return []
    if isinstance(data, dict):
        data = data.get("specs") or data.get("candidates") or [data]
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for item in data[:5]:
        if not isinstance(item, dict):
            continue
        # Refuse any source-code fields.
        if any(k in item for k in ("source", "code", "python", "script", "body")):
            continue
        item = dict(item)
        item["origin"] = "llm"
        out.append(item)
    return out


def narrate_autopsy(autopsy: dict[str, Any]) -> str:
    """Optional narrative string for autopsy; empty when LLM unavailable."""
    if provider_name() == "none":
        return ""
    compact = {
        "verdict": autopsy.get("verdict") or autopsy.get("reason"),
        "stage": autopsy.get("died_at_stage"),
        "fingerprint": autopsy.get("fingerprint"),
        "skip_codes": autopsy.get("skip_codes"),
        "regime_mix": autopsy.get("regime_mix"),
        "lean_drift_stats": autopsy.get("lean_drift_stats"),
        "lesson": autopsy.get("lesson"),
        "primitive": (autopsy.get("evidence") or {}).get("primitive")
        if isinstance(autopsy.get("evidence"), dict)
        else None,
    }
    raw = _chat([
        {"role": "system", "content": SYSTEM_NARRATE},
        {"role": "user", "content": json.dumps(compact, default=str)[:4000]},
    ], temperature=0.3)
    if not raw:
        return ""
    # Strip fences / refuse code-looking blocks.
    text = raw.strip()
    if "```" in text or "def " in text or "import " in text:
        text = _JSON_FENCE.sub("", text).strip()
        if "def " in text or "import " in text:
            return ""
    return text[:2000]


def suggest_params(primitive: str, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
    """Optional param suggestion dict; {} when LLM unavailable / invalid."""
    if provider_name() == "none":
        return {}
    constraints = constraints or {}
    compact = {
        "primitive": primitive,
        "avoid_param_bands": [
            b for b in (constraints.get("avoid_param_bands") or [])
            if not b.get("primitive") or b.get("primitive") == primitive
        ][:10],
        "prefer_factor_cells": (constraints.get("prefer_factor_cells") or [])[:6],
    }
    raw = _chat([
        {"role": "system", "content": SYSTEM_PARAMS},
        {"role": "user", "content": json.dumps(compact, default=str)[:4000]},
    ])
    if not raw:
        return {}
    try:
        data = _extract_json(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    if any(k in data for k in ("source", "code", "python", "script")):
        return {}
    # Keep only scalar numeric / bool params.
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, bool) or isinstance(v, (int, float)):
            out[str(k)] = v
    return out

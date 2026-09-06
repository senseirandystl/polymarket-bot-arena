"""Clone / dead-fingerprint detection for Lab strategy specs."""

from __future__ import annotations

import copy
import json
from typing import Any

from signals.strategy_pipeline.compiler import normalize_primitive


def _round_val(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, str):
        return v
    return str(v)


def params_fingerprint(params: dict[str, Any] | None) -> tuple:
    if not params:
        return ()
    items = []
    for k in sorted(params):
        v = params[k]
        if isinstance(v, (dict, list, tuple)):
            continue
        items.append((k, _round_val(v)))
    return tuple(items)


def effective_params(primitive: str, params: dict | None) -> dict[str, Any]:
    from evolution.ga import _default_params_for

    prim = normalize_primitive(primitive)
    merged = copy.deepcopy(_default_params_for(prim))
    for k, v in (params or {}).items():
        if k in merged:
            merged[k] = v
    return merged


def spec_fingerprint(spec: dict[str, Any]) -> tuple:
    prim = str(spec.get("primitive") or spec.get("strategy_type") or "")
    params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
    return (normalize_primitive(prim), params_fingerprint(effective_params(prim, params)))


def bot_fingerprint(bot_or_cfg: Any) -> tuple | None:
    if bot_or_cfg is None:
        return None
    if not isinstance(bot_or_cfg, dict):
        st = getattr(bot_or_cfg, "strategy_type", None)
        params = getattr(bot_or_cfg, "strategy_params", None) or {}
    else:
        st = bot_or_cfg.get("strategy_type")
        params = bot_or_cfg.get("params") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                params = {}
    if not st:
        return None
    try:
        return spec_fingerprint({
            "primitive": st,
            "params": params if isinstance(params, dict) else {},
        })
    except ValueError:
        return None


def _peer_meta(peer: Any) -> tuple[str | None, str | None, dict[str, Any]]:
    if not isinstance(peer, dict):
        name = getattr(peer, "name", None)
        st = getattr(peer, "strategy_type", None)
        params = getattr(peer, "strategy_params", None) or {}
        return name, st, params if isinstance(params, dict) else {}
    name = peer.get("bot_name") or peer.get("name")
    st = peer.get("strategy_type") or peer.get("primitive")
    params = peer.get("params") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            params = {}
    return name, st, params if isinstance(params, dict) else {}


def clone_min_distance() -> float:
    import config

    lab = getattr(config, "STRATEGY_LAB_CLONE_MIN_DISTANCE", None)
    if lab is not None:
        try:
            return float(lab)
        except (TypeError, ValueError):
            pass
    return float(getattr(config, "GA_DIVERSITY_MIN_DISTANCE", 0.08) or 0.0)


def clone_match(
    spec: dict[str, Any],
    peers: list[Any],
    *,
    min_distance: float | None = None,
) -> dict[str, Any] | None:
    """If spec matches a peer genome (exact or near), return match info."""
    try:
        fp = spec_fingerprint(spec)
        prim = fp[0]
        spec_params = effective_params(
            prim,
            spec.get("params") if isinstance(spec.get("params"), dict) else {},
        )
    except ValueError:
        return None
    self_id = str(spec.get("spec_id") or "")
    min_d = clone_min_distance() if min_distance is None else float(min_distance)
    from evolution.diversity import param_distance

    for peer in peers or []:
        if isinstance(peer, dict):
            peer_id = str(peer.get("spec_id") or "")
            if self_id and peer_id and peer_id == self_id:
                continue
        pfp = bot_fingerprint(peer)
        if pfp is None:
            continue
        if pfp[0] != prim:
            continue
        name, st, peer_raw = _peer_meta(peer)
        if pfp == fp:
            return {
                "bot_name": name,
                "strategy_type": st or prim,
                "reason": "exact",
                "distance": 0.0,
            }
        if min_d <= 0:
            continue
        try:
            peer_params = effective_params(prim, peer_raw)
        except ValueError:
            continue
        dist = param_distance(spec_params, peer_params, strategy_type=prim)
        if dist < min_d:
            return {
                "bot_name": name,
                "strategy_type": st or prim,
                "reason": "near",
                "distance": dist,
            }
    return None


def active_peers() -> list[Any]:
    import db

    try:
        return list(db.get_active_bots() or [])
    except Exception:
        return []


def open_spec_peers(store) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        rows = store.open_by_stage(
            "idea", "researched", "coded", "backtested", "paper", "ready", "live"
        )
    except Exception:
        return out
    for h in rows or []:
        inner = h.get("spec") if isinstance(h.get("spec"), dict) else h
        if isinstance(inner, dict):
            out.append(inner)
    return out


def autopsy_peers(store) -> list[dict[str, Any]]:
    """Closed specs whose fingerprints should not be resampled."""
    out: list[dict[str, Any]] = []
    try:
        rows = store.recent_autopsies(limit=24)
    except Exception:
        return out
    for row in rows or []:
        inner = row.get("spec") if isinstance(row.get("spec"), dict) else {}
        autopsy = row.get("autopsy") if isinstance(row.get("autopsy"), dict) else {}
        evidence = autopsy.get("evidence") if isinstance(autopsy.get("evidence"), dict) else {}
        prim = (
            (inner or {}).get("primitive")
            or row.get("primitive")
            or evidence.get("primitive")
        )
        params = (inner or {}).get("params")
        if not isinstance(params, dict):
            params = evidence.get("params") if isinstance(evidence.get("params"), dict) else {}
        if not prim:
            continue
        out.append({
            "spec_id": row.get("spec_id"),
            "bot_name": row.get("bot_name") or row.get("name"),
            "primitive": prim,
            "strategy_type": prim,
            "params": params or {},
        })
    return out


def is_clone(spec: dict[str, Any], store=None, *, extra_peers: list | None = None) -> dict | None:
    peers: list[Any] = list(extra_peers or [])
    peers.extend(active_peers())
    if store is not None:
        peers.extend(open_spec_peers(store))
    return clone_match(spec, peers)


def is_dead_clone(spec: dict[str, Any], store) -> dict | None:
    """True when this genome (or a near-clone) already has an autopsy."""
    if store is None:
        return None
    return clone_match(spec, autopsy_peers(store))


def is_clone_lab(spec: dict[str, Any], store=None, *, extra_peers: list | None = None):
    return is_clone(spec, store, extra_peers=extra_peers)


def is_dead_clone_lab(spec: dict[str, Any], store):
    return is_dead_clone(spec, store)

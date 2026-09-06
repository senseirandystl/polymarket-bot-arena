"""Unified learning spine for Strategy Lab.

Self-sufficient analyse → test → learn → improve loop. LLM is never required.
Durable constraints live in arena_state ``lab_learning_spine`` and feed
``research.propose()`` via :func:`get_constraints`.

Shared cell vocabulary with ``arena.learned_rules`` (regime|price|drift|side[|st])
is documented in ``_refs/PHASE4_NOTES.md``.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("strategy_pipeline.learning_spine")

STATE_KEY = "lab_learning_spine"
MAX_AVOID_FPS = 64
MAX_AVOID_BANDS = 48
MAX_PREFER_CELLS = 40

# Autopsy JSON keys required by Phase 4 contract.
AUTOPSY_KEYS = (
    "params",
    "fingerprint",
    "verdict",
    "factor_histograms",
    "skip_codes",
    "regime_mix",
    "venue_mix",
    "window_age_bins",
    "lean_drift_stats",
    "avoid_constraints",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _empty_spine() -> dict[str, Any]:
    return {
        "updated_at": None,
        "avoid_fingerprints": [],
        "avoid_param_bands": [],
        "prefer_factor_cells": [],
        "stats": {
            "autopsies_ingested": 0,
            "learned_rules_folded": 0,
            "last_mine_ts": None,
        },
    }


def load_spine() -> dict[str, Any]:
    try:
        import db

        raw = db.get_arena_state(STATE_KEY)
        if not raw:
            return _empty_spine()
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return _empty_spine()
        out = _empty_spine()
        out.update({k: data[k] for k in out if k in data})
        out.setdefault("avoid_fingerprints", [])
        out.setdefault("avoid_param_bands", [])
        out.setdefault("prefer_factor_cells", [])
        out.setdefault("stats", _empty_spine()["stats"])
        return out
    except Exception as e:
        logger.debug("learning_spine load failed: %s", e)
        return _empty_spine()


def save_spine(state: dict[str, Any]) -> dict[str, Any]:
    state = dict(state or _empty_spine())
    state["updated_at"] = _utc_now_iso()
    try:
        import db

        db.set_arena_state(STATE_KEY, json.dumps(state))
    except Exception as e:
        logger.warning("learning_spine save failed: %s", e)
    return state


def fingerprint_str(primitive: str | None, params: dict | None) -> str:
    """Stable string form of a genome fingerprint."""
    try:
        from signals.strategy_pipeline.fingerprint import (
            effective_params,
            params_fingerprint,
        )
        from signals.strategy_pipeline.compiler import normalize_primitive

        prim = normalize_primitive(str(primitive or "momentum"))
        fp = params_fingerprint(effective_params(prim, params or {}))
        return f"{prim}|{json.dumps(list(fp), separators=(',', ':'), sort_keys=False)}"
    except Exception:
        return f"{primitive or '?'}|{json.dumps(params or {}, sort_keys=True)}"


def _fp_tuple(primitive: str | None, params: dict | None):
    try:
        from signals.strategy_pipeline.fingerprint import spec_fingerprint

        return spec_fingerprint({"primitive": primitive, "params": params or {}})
    except Exception:
        return None


def build_structured_autopsy(
    *,
    params: dict[str, Any] | None,
    primitive: str | None,
    verdict: str,
    stage: str = "",
    evidence: dict[str, Any] | None = None,
    narrative: str | None = None,
) -> dict[str, Any]:
    """Build Phase-4 autopsy JSON (plus legacy postmortem fields)."""
    evidence = dict(evidence or {})
    params = dict(params or evidence.get("params") or {})
    primitive = primitive or evidence.get("primitive")
    fp = fingerprint_str(primitive, params)

    factor_histograms = dict(evidence.get("factor_histograms") or {})
    skip_codes = dict(evidence.get("skip_codes") or {})
    regime_mix = dict(evidence.get("regime_mix") or {})
    venue_mix = dict(evidence.get("venue_mix") or {})
    window_age_bins = dict(evidence.get("window_age_bins") or {})
    lean_drift_stats = dict(
        evidence.get("lean_drift_stats")
        or evidence.get("lean/drift stats")
        or {}
    )

    # Enrich from common backtest / paper evidence shapes when sparse.
    if not regime_mix and isinstance(evidence.get("regime"), str):
        regime_mix = {str(evidence["regime"]): 1}
    if not skip_codes:
        for k in ("skip_reason", "reason_code", "gate_reason"):
            if evidence.get(k):
                skip_codes[str(evidence[k])] = skip_codes.get(str(evidence[k]), 0) + 1
    if not lean_drift_stats:
        for k in ("drift", "lean", "avg_drift", "mean_lean"):
            if k in evidence and evidence[k] is not None:
                lean_drift_stats[k] = evidence[k]

    avoid_constraints: list[dict[str, Any]] = [
        {
            "kind": "fingerprint",
            "primitive": primitive,
            "params": params,
            "fingerprint": fp,
            "reason": verdict,
        }
    ]
    # Param-band hints: mark each numeric gene as a soft avoid band around it.
    for key, val in params.items():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        v = float(val)
        span = max(abs(v) * 0.08, 1e-6)
        avoid_constraints.append({
            "kind": "param_band",
            "primitive": primitive,
            "param": key,
            "lo": v - span,
            "hi": v + span,
            "reason": verdict,
        })

    autopsy: dict[str, Any] = {
        "params": params,
        "fingerprint": fp,
        "verdict": verdict,
        "factor_histograms": factor_histograms,
        "skip_codes": skip_codes,
        "regime_mix": regime_mix,
        "venue_mix": venue_mix,
        "window_age_bins": window_age_bins,
        "lean_drift_stats": lean_drift_stats,
        "avoid_constraints": avoid_constraints,
        # Legacy postmortem fields (research / UI still read these).
        "reason": verdict,
        "died_at_stage": stage or evidence.get("died_at_stage") or "",
        "evidence": evidence,
        "lesson": evidence.get("lesson") or _default_lesson(verdict, primitive),
    }
    if narrative:
        autopsy["narrative"] = str(narrative)
    if primitive:
        autopsy.setdefault("evidence", {})["primitive"] = primitive
    return autopsy


def _default_lesson(verdict: str, primitive: str | None) -> str:
    v = (verdict or "").lower()
    if "clone" in v:
        return "Avoid resampling an active or dead genome fingerprint."
    if "edge" in v or "pnl" in v:
        return "Do not promote on follow-WR; require fee-aware net edge."
    if "sample" in v or "trades" in v:
        return "Hold the candidate until n clears the promotion / backtest bar."
    if prim := primitive:
        return (
            f"Archive failure of this {prim} genome; "
            "the primitive may still be proposed with different genes."
        )
    return "Record the failure regime and do not resample the same fingerprint."


def ingest_autopsy(autopsy: dict[str, Any]) -> dict[str, Any]:
    """Fold one autopsy into durable spine constraints and persist."""
    state = load_spine()
    constraints = list(autopsy.get("avoid_constraints") or [])
    prim = None
    params = autopsy.get("params") if isinstance(autopsy.get("params"), dict) else {}
    evidence = autopsy.get("evidence") if isinstance(autopsy.get("evidence"), dict) else {}
    prim = evidence.get("primitive") or (constraints[0].get("primitive") if constraints else None)

    fp = autopsy.get("fingerprint") or fingerprint_str(prim, params)
    avoid_fps: list[dict] = list(state.get("avoid_fingerprints") or [])
    found = False
    for row in avoid_fps:
        if row.get("fingerprint") == fp:
            row["count"] = int(row.get("count") or 0) + 1
            row["last_reason"] = autopsy.get("verdict") or autopsy.get("reason")
            row["last_at"] = _utc_now_iso()
            found = True
            break
    if not found:
        avoid_fps.append({
            "fingerprint": fp,
            "primitive": prim,
            "params": params,
            "count": 1,
            "last_reason": autopsy.get("verdict") or autopsy.get("reason"),
            "last_at": _utc_now_iso(),
        })
    state["avoid_fingerprints"] = avoid_fps[-MAX_AVOID_FPS:]

    bands: list[dict] = list(state.get("avoid_param_bands") or [])
    for c in constraints:
        if c.get("kind") != "param_band":
            continue
        key = (c.get("primitive"), c.get("param"))
        merged = False
        for b in bands:
            if (b.get("primitive"), b.get("param")) == key:
                # Expand band envelope.
                b["lo"] = min(float(b.get("lo", c["lo"])), float(c["lo"]))
                b["hi"] = max(float(b.get("hi", c["hi"])), float(c["hi"]))
                b["count"] = int(b.get("count") or 0) + 1
                merged = True
                break
        if not merged:
            bands.append({
                "primitive": c.get("primitive"),
                "param": c.get("param"),
                "lo": float(c["lo"]),
                "hi": float(c["hi"]),
                "count": 1,
            })
    state["avoid_param_bands"] = bands[-MAX_AVOID_BANDS:]

    stats = dict(state.get("stats") or {})
    stats["autopsies_ingested"] = int(stats.get("autopsies_ingested") or 0) + 1
    state["stats"] = stats
    return save_spine(state)


def fold_learned_rules(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Adapter: fold arena.learned_rules skip/go cells into prefer/avoid cells.

    Does NOT break the existing mine path — read-only consume of their state.
    """
    state = dict(state or load_spine())
    prefer: list[dict] = [
        c for c in (state.get("prefer_factor_cells") or [])
        if c.get("source") != "learned_rules"
    ]
    folded = 0
    try:
        from arena.learned_rules import load_state, parse_cell

        lr = load_state()
        for rule in lr.get("rules") or []:
            if not isinstance(rule, dict):
                continue
            cell = rule.get("cell")
            rtype = rule.get("type") or rule.get("rule_type")
            if not cell or rtype not in ("skip", "go"):
                continue
            parsed = parse_cell(str(cell))
            prefer.append({
                "cell": cell,
                "kind": "avoid" if rtype == "skip" else "prefer",
                "rule_type": rtype,
                "source": "learned_rules",
                "effect": rule.get("effect") or {},
                "n": rule.get("n") or rule.get("buy_n"),
                "wr": rule.get("wr") or rule.get("buy_wr"),
                **{k: parsed.get(k) for k in (
                    "regime", "price_band", "drift_band", "side", "strategy_type"
                )},
            })
            folded += 1
        # Also surface continuous cells with strong WR as soft prefer.
        for cell_key, cell in (lr.get("cells") or {}).items():
            if not isinstance(cell, dict):
                continue
            bn = int(cell.get("buy_n") or 0)
            wr = cell.get("buy_wr")
            if bn < 25 or wr is None:
                continue
            if float(wr) >= 0.58:
                prefer.append({
                    "cell": cell_key,
                    "kind": "prefer",
                    "rule_type": "cell_go",
                    "source": "learned_rules",
                    "n": bn,
                    "wr": wr,
                })
                folded += 1
            elif float(wr) <= 0.47:
                prefer.append({
                    "cell": cell_key,
                    "kind": "avoid",
                    "rule_type": "cell_skip",
                    "source": "learned_rules",
                    "n": bn,
                    "wr": wr,
                })
                folded += 1
    except Exception as e:
        logger.debug("fold_learned_rules skipped: %s", e)

    # Dedup by cell+kind, keep last.
    seen: dict[str, dict] = {}
    for row in prefer:
        key = f"{row.get('cell')}|{row.get('kind')}|{row.get('source')}"
        seen[key] = row
    state["prefer_factor_cells"] = list(seen.values())[-MAX_PREFER_CELLS:]
    stats = dict(state.get("stats") or {})
    stats["learned_rules_folded"] = folded
    state["stats"] = stats
    return state


def mine_trends(store=None) -> dict[str, Any]:
    """Mine autopsies + decision/skip/scorecard signals into durable spine."""
    state = load_spine()

    # 1) Lab hypothesis autopsies
    try:
        if store is None:
            from signals.strategy_pipeline.store import HypothesisStore
            store = HypothesisStore()
        for row in store.recent_autopsies(limit=40) or []:
            autopsy = row.get("autopsy") if isinstance(row.get("autopsy"), dict) else {}
            if not autopsy:
                continue
            # Ensure structured keys; re-ingest fingerprint if present.
            if autopsy.get("fingerprint") or autopsy.get("avoid_constraints"):
                ingest_autopsy(autopsy)
            else:
                inner = row.get("spec") if isinstance(row.get("spec"), dict) else {}
                evidence = autopsy.get("evidence") if isinstance(autopsy.get("evidence"), dict) else {}
                structured = build_structured_autopsy(
                    params=inner.get("params") if isinstance(inner.get("params"), dict) else evidence.get("params"),
                    primitive=inner.get("primitive") or row.get("primitive") or evidence.get("primitive"),
                    verdict=str(autopsy.get("reason") or "unknown"),
                    stage=str(autopsy.get("died_at_stage") or row.get("stage") or ""),
                    evidence=evidence,
                )
                ingest_autopsy(structured)
        state = load_spine()
    except Exception as e:
        logger.debug("mine autopsies failed: %s", e)

    # 2) Fold learned trade rules (shared vocabulary)
    state = fold_learned_rules(state)

    # 3) skip_counts from arena_state
    try:
        import db

        raw = db.get_arena_state("skip_counts")
        if raw:
            skips = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(skips, dict) and skips:
                stats = dict(state.get("stats") or {})
                stats["skip_counts_top"] = dict(
                    sorted(skips.items(), key=lambda kv: -int(kv[1] or 0))[:12]
                )
                state["stats"] = stats
    except Exception as e:
        logger.debug("mine skip_counts failed: %s", e)

    # 4) live_scorecard factor hint (best-effort)
    try:
        import db

        raw = db.get_arena_state("live_scorecard")
        if raw:
            card = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(card, dict):
                stats = dict(state.get("stats") or {})
                stats["scorecard_ts"] = card.get("ts") or card.get("updated_at")
                # Prefer lanes / types with positive net if present.
                lanes = card.get("lanes") or card.get("by_lane") or {}
                if isinstance(lanes, dict):
                    good = []
                    for name, info in lanes.items():
                        if not isinstance(info, dict):
                            continue
                        pnl = info.get("pnl") or info.get("net_pnl")
                        if pnl is not None and float(pnl) > 0:
                            good.append({"lane": name, "pnl": float(pnl)})
                    if good:
                        stats["prefer_lanes"] = sorted(
                            good, key=lambda r: -r["pnl"]
                        )[:8]
                state["stats"] = stats
    except Exception as e:
        logger.debug("mine live_scorecard failed: %s", e)

    stats = dict(state.get("stats") or {})
    stats["last_mine_ts"] = time.time()
    state["stats"] = stats
    return save_spine(state)


def get_constraints() -> dict[str, Any]:
    """Constraints bundle for research.propose() — always safe / deterministic."""
    state = load_spine()
    # Refresh learned-rules fold cheaply (no full mine) so Lab Signals stay aligned.
    try:
        state = fold_learned_rules(state)
        # Persist only if fold changed something material.
        save_spine(state)
    except Exception:
        pass
    return {
        "updated_at": state.get("updated_at"),
        "avoid_fingerprints": list(state.get("avoid_fingerprints") or []),
        "avoid_param_bands": list(state.get("avoid_param_bands") or []),
        "prefer_factor_cells": list(state.get("prefer_factor_cells") or []),
        "stats": dict(state.get("stats") or {}),
    }


def fingerprint_blocked(primitive: str, params: dict | None, constraints: dict | None = None) -> bool:
    """True when this genome matches a durable avoid fingerprint."""
    cons = constraints if constraints is not None else get_constraints()
    fp = fingerprint_str(primitive, params)
    for row in cons.get("avoid_fingerprints") or []:
        if row.get("fingerprint") == fp:
            return True
        # Also compare via tuple when params present on the row.
        if row.get("primitive") == primitive and isinstance(row.get("params"), dict):
            try:
                if _fp_tuple(primitive, params) == _fp_tuple(primitive, row["params"]):
                    return True
            except Exception:
                pass
    return False


def bias_params_away_from_bands(
    primitive: str,
    params: dict[str, Any],
    constraints: dict | None = None,
    *,
    rng=None,
) -> dict[str, Any]:
    """Nudge numeric genes out of avoid bands (deterministic primary path)."""
    import copy
    import random as _random

    rng = rng or _random
    cons = constraints if constraints is not None else get_constraints()
    out = copy.deepcopy(params)
    bands = [
        b for b in (cons.get("avoid_param_bands") or [])
        if (not b.get("primitive") or b.get("primitive") == primitive)
    ]
    if not bands:
        return out
    for key, val in list(out.items()):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        v = float(val)
        for b in bands:
            if b.get("param") != key:
                continue
            lo, hi = float(b["lo"]), float(b["hi"])
            if lo <= v <= hi:
                # Push just outside the band toward the nearer edge + jitter.
                span = max(hi - lo, 1e-6)
                if abs(v - lo) <= abs(v - hi):
                    v = lo - span * (0.15 + 0.35 * rng.random())
                else:
                    v = hi + span * (0.15 + 0.35 * rng.random())
                try:
                    from evolution.bounds import clamp
                    out[key] = clamp(key, v, reference=val)
                except Exception:
                    out[key] = type(val)(v) if not isinstance(val, float) else v
                break
    return out


def prefer_primitives(constraints: dict | None = None) -> list[str]:
    """Primitives hinted by prefer_factor_cells (strategy_type tagged)."""
    cons = constraints if constraints is not None else get_constraints()
    scores: dict[str, float] = {}
    for cell in cons.get("prefer_factor_cells") or []:
        st = cell.get("strategy_type")
        if not st:
            continue
        kind = cell.get("kind")
        wr = float(cell.get("wr") or 0.5)
        if kind == "prefer":
            scores[st] = scores.get(st, 0.0) + max(0.1, wr)
        elif kind == "avoid":
            scores[st] = scores.get(st, 0.0) - max(0.1, 1.0 - wr)
    return [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1]) if scores[k] > 0]


def write_autopsy_from_bot(
    bot_or_cfg: Any,
    *,
    verdict: str,
    stage: str = "retired",
    evidence: dict[str, Any] | None = None,
    store=None,
    narrate: bool = True,
) -> dict[str, Any]:
    """GA / paper / promote death path: build + persist autopsy from a bot."""
    evidence = dict(evidence or {})
    name = None
    primitive = None
    params: dict[str, Any] = {}

    if isinstance(bot_or_cfg, dict):
        name = bot_or_cfg.get("bot_name") or bot_or_cfg.get("name")
        primitive = bot_or_cfg.get("strategy_type") or bot_or_cfg.get("primitive")
        raw_p = bot_or_cfg.get("params") or bot_or_cfg.get("strategy_params") or {}
        if isinstance(raw_p, str):
            try:
                raw_p = json.loads(raw_p)
            except (json.JSONDecodeError, TypeError):
                raw_p = {}
        params = dict(raw_p) if isinstance(raw_p, dict) else {}
    else:
        name = getattr(bot_or_cfg, "name", None)
        primitive = getattr(bot_or_cfg, "strategy_type", None)
        params = dict(getattr(bot_or_cfg, "strategy_params", None) or {})

    if name and "bot_name" not in evidence:
        evidence["bot_name"] = name
    if primitive:
        evidence["primitive"] = primitive
    evidence.setdefault("params", params)

    autopsy = build_structured_autopsy(
        params=params,
        primitive=primitive,
        verdict=verdict,
        stage=stage,
        evidence=evidence,
    )

    if narrate:
        try:
            from signals.strategy_pipeline import llm as llm_mod

            narrative = llm_mod.narrate_autopsy(autopsy)
            if narrative:
                autopsy["narrative"] = narrative
        except Exception:
            pass

    ingest_autopsy(autopsy)

    # If we have a lab hyp for this bot, close it too.
    if store is None:
        try:
            from signals.strategy_pipeline.store import HypothesisStore
            store = HypothesisStore()
        except Exception:
            store = None
    if store is not None and name:
        try:
            for hyp in store.list(limit=80):
                if hyp.get("bot_name") == name and hyp.get("status") == "open":
                    store.advance(
                        hyp["spec_id"],
                        "retired",
                        status="closed",
                        autopsy=autopsy,
                    )
                    store.log(hyp["spec_id"], "reviewer", "postmortem", "autopsy", verdict)
                    break
        except Exception as e:
            logger.debug("write_autopsy_from_bot store close failed: %s", e)

    return autopsy


def write_autopsy_for_spec(
    store,
    spec_id: str,
    *,
    stage: str,
    reason: str,
    evidence: dict[str, Any] | None = None,
    narrate: bool = True,
) -> dict[str, Any]:
    """Pipeline reject path: structured autopsy + spine ingest + store close."""
    evidence = dict(evidence or {})
    hyp = None
    try:
        hyp = store.get(spec_id) if hasattr(store, "get") else None
    except Exception:
        hyp = None
    inner = hyp.get("spec") if isinstance((hyp or {}).get("spec"), dict) else {}
    primitive = (
        inner.get("primitive")
        or (hyp or {}).get("primitive")
        or evidence.get("primitive")
    )
    params = inner.get("params") if isinstance(inner.get("params"), dict) else {}
    if not params and isinstance(evidence.get("params"), dict):
        params = dict(evidence["params"])
    if primitive and "primitive" not in evidence:
        evidence["primitive"] = primitive
    if params and "params" not in evidence:
        evidence["params"] = dict(params)

    autopsy = build_structured_autopsy(
        params=params,
        primitive=primitive,
        verdict=reason,
        stage=stage,
        evidence=evidence,
    )
    if narrate:
        try:
            from signals.strategy_pipeline import llm as llm_mod

            narrative = llm_mod.narrate_autopsy(autopsy)
            if narrative:
                autopsy["narrative"] = narrative
        except Exception:
            pass

    ingest_autopsy(autopsy)
    closed_stage = (
        "rejected" if stage in ("idea", "researched", "coded", "backtested") else "retired"
    )
    store.advance(
        spec_id,
        closed_stage,
        status="closed",
        autopsy=autopsy,
    )
    store.log(spec_id, "reviewer", "postmortem", "autopsy", reason)
    return autopsy

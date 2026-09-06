"""Deterministic proposer: catalog + gene-bank mutate; optional LLM assist.

Primary path is always heuristic / gene-bank. STRATEGY_LAB_LLM_PROVIDER=none
(or any LLM failure) leaves behavior unchanged aside from spine constraints.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any

from signals.strategy_pipeline.compiler import (
    PRIMITIVE_TYPES,
    LIVE_COMPAT_PRIMITIVES,
    new_spec_id,
    sanitize_spec,
)
from signals.strategy_pipeline.fingerprint import is_clone, is_dead_clone

logger = logging.getLogger("strategy_pipeline.research")


def _allowed_primitives() -> frozenset[str]:
    """Primitives invent may propose.

    While STRATEGY_LAB_PAPER_SLOTS==0 (or always for v1 profit-mode), restrict
    to live-compatible genomes so invent still learns without proposing culled
    momentum/meanrev/hybrid/phantom types.
    """
    try:
        import config
        slots = int(getattr(config, "STRATEGY_LAB_PAPER_SLOTS", 0) or 0)
    except Exception:
        slots = 0
    if slots <= 0:
        return LIVE_COMPAT_PRIMITIVES
    return frozenset(PRIMITIVE_TYPES.values())


SYSTEM = (
    "You design parameterized trading-bot specs for binary crypto Up/Down "
    "markets. Reply JSON only. Do not invent primitives. No arbitrage. "
    "Do not emit source code."
)


def propose(store, *, max_new: int = 3) -> list[dict[str, Any]]:
    """Return new sanitized specs inserted into the store."""
    context = _context(store)
    constraints = context.get("constraints") or {}
    raw_candidates = _heuristic_candidates(context)
    raw_candidates.extend(_gene_bank_mutations(context))
    llm_overlay = _llm_candidates(context)
    if llm_overlay:
        raw_candidates = llm_overlay + raw_candidates

    # Prefer primitives with positive factor evidence (soft ordering).
    try:
        from signals.strategy_pipeline.learning_spine import prefer_primitives

        preferred = set(prefer_primitives(constraints))
        if preferred:
            raw_candidates.sort(
                key=lambda c: (0 if c.get("primitive") in preferred else 1)
            )
    except Exception:
        pass

    existing = {h.get("name") for h in store.list(limit=200)}
    out: list[dict[str, Any]] = []
    for raw in raw_candidates:
        if len(out) >= max_new:
            break
        try:
            spec = sanitize_spec(raw)
        except ValueError:
            continue
        params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
        if not params:
            logger.info(
                "lab research skip empty-param lean clone primitive=%s name=%s",
                spec.get("primitive"), spec.get("name"),
            )
            continue

        # Spine: skip dead fingerprints / bias away from avoid bands.
        try:
            from signals.strategy_pipeline.learning_spine import (
                bias_params_away_from_bands,
                fingerprint_blocked,
            )

            if fingerprint_blocked(spec["primitive"], params, constraints):
                logger.info(
                    "lab research skip spine-avoid fingerprint primitive=%s",
                    spec.get("primitive"),
                )
                continue
            biased = bias_params_away_from_bands(
                spec["primitive"], params, constraints
            )
            if biased != params:
                spec = sanitize_spec({**spec, "params": biased})
                params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
                if fingerprint_blocked(spec["primitive"], params, constraints):
                    continue
        except Exception as e:
            logger.debug("spine constraint apply failed: %s", e)

        clone = is_clone(spec, store)
        if clone:
            logger.info(
                "lab research skip clone primitive=%s of=%s",
                spec.get("primitive"), clone.get("bot_name"),
            )
            continue
        dead = is_dead_clone(spec, store)
        if dead:
            logger.info(
                "lab research skip dead fingerprint primitive=%s of=%s",
                spec.get("primitive"), dead.get("bot_name"),
            )
            continue
        if spec["name"] in existing:
            spec["name"] = f"{spec['name']}-{spec['spec_id'][-4:]}"
        spec["stage"] = "researched"
        spec["origin"] = raw.get("origin") or spec.get("origin") or "heuristic"
        store.insert(spec)
        existing.add(spec["name"])
        out.append(spec)
    return out


def _context(store) -> dict[str, Any]:
    autopsies = store.recent_autopsies(limit=12)
    try:
        from signals.strategy_pipeline.universe import tradable_slots
        slots = [s.as_dict() for s in tradable_slots()]
    except Exception:
        slots = [{"slot_id": "polymarket:btc_5m"}]

    constraints: dict[str, Any] = {}
    try:
        from signals.strategy_pipeline.learning_spine import get_constraints, mine_trends

        # Periodic mine keeps prefer/avoid cells fresh; cheap when DB quiet.
        try:
            mine_trends(store)
        except Exception:
            pass
        constraints = get_constraints()
    except Exception as e:
        logger.debug("spine constraints unavailable: %s", e)
        constraints = {
            "avoid_fingerprints": [],
            "avoid_param_bands": [],
            "prefer_factor_cells": [],
            "stats": {},
        }

    avoid_fingerprints = list(constraints.get("avoid_fingerprints") or [])
    # Legacy autopsy-derived avoids (kept for LLM context + back-compat).
    for a in autopsies:
        inner = a.get("spec") if isinstance(a.get("spec"), dict) else {}
        autopsy = a.get("autopsy") if isinstance(a.get("autopsy"), dict) else {}
        evidence = autopsy.get("evidence") if isinstance(autopsy.get("evidence"), dict) else {}
        prim = inner.get("primitive") or a.get("primitive") or evidence.get("primitive")
        params = inner.get("params") if isinstance(inner.get("params"), dict) else (
            evidence.get("params") if isinstance(evidence.get("params"), dict) else {}
        )
        if prim and prim not in ("arbitrage",):
            avoid_fingerprints.append({
                "primitive": prim,
                "params": params or {},
                "fingerprint": autopsy.get("fingerprint"),
            })

    return {
        "universe": slots,
        "autopsies": autopsies,
        "avoid_fingerprints": avoid_fingerprints[-16:],
        "avoid_param_bands": list(constraints.get("avoid_param_bands") or [])[:16],
        "prefer_factor_cells": list(constraints.get("prefer_factor_cells") or [])[:16],
        "constraints": constraints,
    }


def _heuristic_candidates(context: dict[str, Any]) -> list[dict[str, Any]]:
    universe = [u["slot_id"] for u in context.get("universe") or []] or [
        "polymarket:btc_5m"
    ]
    catalog = [
        {
            "primitive": "momentum",
            "name": "mom-wide",
            "thesis": "Slower lookback + higher threshold genome.",
            "params": {
                "lookback_candles": 25,
                "momentum_threshold": 0.003,
                "min_confidence": 0.72,
            },
        },
        {
            "primitive": "momentum",
            "name": "mom-tight",
            "thesis": "Faster lookback + lower threshold genome.",
            "params": {
                "lookback_candles": 12,
                "momentum_threshold": 0.0015,
                "min_confidence": 0.55,
            },
        },
        {
            "primitive": "mean_reversion",
            "name": "fade-slow",
            "thesis": "Mean-rev with slower tape + stricter drift gate.",
            "params": {
                "lookback_candles": 24,
                "min_drift": 0.35,
                "rsi_period": 8,
                "min_confidence": 0.70,
            },
        },
        {
            "primitive": "sniper",
            "name": "sniper-strict",
            "thesis": "Sniper with harder drift/confidence than live default.",
            "params": {
                "min_drift": 0.28,
                "min_confidence": 0.25,
                "quiet_drift_bump": 0.12,
            },
        },
        {
            "primitive": "hybrid",
            "name": "hybrid-picky",
            "thesis": "Hybrid with higher min_confidence.",
            "params": {"min_confidence": 0.72},
        },
        {
            "primitive": "sweeper",
            "name": "sweep-lab",
            "thesis": "Fee-curve extreme sweep with non-default threshold.",
            "params": {"min_edge": 0.02},
        },
        {
            "primitive": "phantom",
            "name": "phantom-lab",
            "thesis": "EMA breakout only when drift leans.",
            "params": {
                "ema_fast": 8,
                "ema_slow": 21,
                "breakout_lookback": 12,
                "min_confidence": 0.40,
            },
        },
        {
            "primitive": "lag_residual",
            "name": "lag-lab",
            "thesis": "Lag residual with stricter confidence.",
            "params": {"min_confidence": 0.65},
        },
    ]
    allowed = _allowed_primitives()
    out = []
    for c in catalog:
        if c.get("primitive") not in allowed:
            continue
        c = dict(c)
        c["spec_id"] = new_spec_id(c["primitive"])
        c["universe"] = universe
        c["origin"] = "heuristic"
        c["parent_spec_ids"] = [
            a.get("spec_id") for a in (context.get("autopsies") or [])[:2]
            if a.get("spec_id")
        ]
        out.append(c)
    return out


def _gene_bank_mutations(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Mutate gene-bank elites into fresh candidates (avoid static-only stall)."""
    try:
        from evolution.gene_bank import load_bank
        from evolution.operators import mutate
        from evolution.ga import _default_params_for
    except Exception as e:
        logger.debug("gene bank mutate unavailable: %s", e)
        return []

    bank = load_bank() or []
    if not bank:
        return []
    universe = [u["slot_id"] for u in context.get("universe") or []] or [
        "polymarket:btc_5m"
    ]
    rng = random.Random()
    constraints = context.get("constraints") or {}
    out: list[dict[str, Any]] = []
    for entry in bank[-6:]:
        st = str(entry.get("strategy_type") or "").strip()
        if not st or st == "arbitrage" or st not in PRIMITIVE_TYPES.values() or st not in _allowed_primitives():
            continue
        base = copy.deepcopy(_default_params_for(st))
        params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
        for k, v in params.items():
            if k in base:
                base[k] = v
        try:
            mutated = mutate(base, rate=0.35, sigma=0.20, rng=rng)
        except Exception:
            continue
        try:
            from signals.strategy_pipeline.learning_spine import bias_params_away_from_bands

            mutated = bias_params_away_from_bands(st, mutated, constraints, rng=rng)
        except Exception:
            pass
        # Ensure at least one numeric gene differs from defaults.
        if mutated == _default_params_for(st):
            continue
        out.append({
            "primitive": st,
            "name": f"{st[:6]}-gb",
            "thesis": f"Gene-bank mutate of {entry.get('name') or st}.",
            "params": mutated,
            "spec_id": new_spec_id(st),
            "universe": universe,
            "origin": "gene_bank",
            "parent_spec_ids": [str(entry.get("name") or st)],
        })
    return out


def _llm_candidates(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Optional LLM hook. Default provider=none -> empty (deterministic path)."""
    try:
        from signals.strategy_pipeline import llm as llm_mod

        if llm_mod.provider_name() == "none":
            return []
        return list(llm_mod.research_assist(context) or [])
    except Exception as e:
        logger.debug("LLM research assist unavailable: %s", e)
        return []

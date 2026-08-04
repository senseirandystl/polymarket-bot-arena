"""Shadow gene bank — elites saved each cycle as future parents (not trading).

Each evolution cycle the top elite(s) are appended to a capped bank stored in
``arena_state['ga_gene_bank']``. Bank entries are never removed from the live
roster; they only expand the parent pool for tournament / type allocation so
good genomes survive beyond a single bad judgment window.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import config
import db

logger = logging.getLogger("arena")

STATE_KEY = "ga_gene_bank"


def _max_size() -> int:
    return max(1, int(getattr(config, "GA_GENE_BANK_SIZE", 20)))


def load_bank() -> list[dict]:
    """Return gene-bank entries (newest last)."""
    try:
        raw = db.get_arena_state(STATE_KEY)
        if not raw:
            return []
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            data = data.get("entries") or []
        if not isinstance(data, list):
            return []
        return [e for e in data if isinstance(e, dict) and e.get("params")]
    except Exception as e:
        logger.debug("gene_bank load failed: %s", e)
        return []


def _max_per_type() -> int:
    return max(1, int(getattr(config, "GA_GENE_BANK_MAX_PER_TYPE", 3)))


def apply_type_quotas(entries: list[dict]) -> list[dict]:
    """Keep at most N highest-fitness entries per strategy_type, then global cap.

    Prevents a single elite type (e.g. phantom) from filling the entire bank
    and dominating every future tournament parent pool.
    """
    per = _max_per_type()
    by_type: dict[str, list[dict]] = {}
    for e in entries:
        st = e.get("strategy_type") or "unknown"
        by_type.setdefault(st, []).append(e)
    kept: list[dict] = []
    for st, group in by_type.items():
        group_sorted = sorted(
            group,
            key=lambda x: (float(x.get("fitness") or 0.0), int(x.get("cycle") or 0)),
            reverse=True,
        )
        kept.extend(group_sorted[:per])
    # Global cap: prefer higher fitness, then newer cycle
    kept.sort(
        key=lambda x: (float(x.get("fitness") or 0.0), int(x.get("cycle") or 0)),
        reverse=True,
    )
    return kept[: _max_size()]


def save_bank(entries: list[dict]) -> None:
    """Persist bank (type quotas + global cap)."""
    trimmed = apply_type_quotas(entries)
    try:
        db.set_arena_state(STATE_KEY, json.dumps({
            "entries": trimmed,
            "max_size": _max_size(),
            "max_per_type": _max_per_type(),
        }))
    except Exception as e:
        logger.warning("gene_bank save failed: %s", e)


def record_elites(individuals: list[dict], cycle: int) -> list[dict]:
    """Append this cycle's elites into the bank; return the updated bank.

    Dedupes by (strategy_type, rounded params fingerprint) so identical elites
    don't flood the bank every 2h. Applies per-type quotas before persist.
    """
    bank = load_bank()
    existing_fps = {_fingerprint(e) for e in bank}
    for ind in individuals:
        if not ind.get("elite"):
            continue
        entry = {
            "name": ind.get("name"),
            "strategy_type": ind.get("strategy_type"),
            "generation": ind.get("generation"),
            "cycle": cycle,
            "fitness": float(ind.get("fitness") or 0.0),
            "pnl": ind.get("pnl"),
            "win_rate": ind.get("win_rate"),
            "trades": ind.get("trades"),
            "params": copy.deepcopy(ind.get("params") or {}),
            "lineage": ind.get("lineage"),
            "source": "elite",
        }
        fp = _fingerprint(entry)
        if fp in existing_fps:
            # Refresh fitness on matching entry (keep newest params)
            for i, old in enumerate(bank):
                if _fingerprint(old) == fp:
                    bank[i] = entry
                    break
            continue
        bank.append(entry)
        existing_fps.add(fp)
    save_bank(bank)
    return load_bank()


def as_parent_records(bank: list[dict] | None = None) -> list[dict]:
    """Shape bank rows like GA individuals for tournament_select.

    Fitness is taken from the stored elite fitness (rank-normalized score at
    deposit time). Missing fitness → 0.
    """
    bank = bank if bank is not None else load_bank()
    out = []
    for e in bank:
        out.append({
            "name": e.get("name") or "bank",
            "strategy_type": e.get("strategy_type"),
            "generation": e.get("generation") or 0,
            "params": copy.deepcopy(e.get("params") or {}),
            "fitness": float(e.get("fitness") or 0.0),
            "trades": int(e.get("trades") or 0),
            "pnl": float(e.get("pnl") or 0.0),
            "win_rate": float(e.get("win_rate") or 0.0),
            "be_gap": e.get("be_gap"),
            "elite": True,
            "status": "gene_bank",
            "lineage": e.get("lineage"),
            "from_gene_bank": True,
        })
    return out


def _fingerprint(entry: dict) -> str:
    st = entry.get("strategy_type") or ""
    params = entry.get("params") or {}
    try:
        keys = sorted(params.keys())
        parts = [f"{k}={params[k]!r}" for k in keys]
        return f"{st}|{'|'.join(parts)}"
    except Exception:
        return f"{st}|{id(params)}"

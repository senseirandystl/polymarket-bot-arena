"""Layer 2 of the regime-discovery design: per-bot performance attribution over
market context, empirical-Bayes shrunk toward coarser priors.

Part 1 is pure attribution math — `shrink` and `attribute`. Part 2 (this
addition) discovers named regimes from live resolved trades, validates each
candidate cell out-of-sample, and persists the resulting map via
`db.set_regime_map` / `db.get_regime_map`.
"""
from __future__ import annotations

import logging
import time
from typing import Sequence

import config
import db

logger = logging.getLogger("arena.regime_map")

STATE_KEY = "regime_map"


def shrink(cell_mean: float, cell_n: int, prior_mean: float, k: float) -> float:
    """Empirical-Bayes shrinkage of a cell mean toward a prior mean.

    A thin cell (small `cell_n` relative to `k`) is pulled mostly toward
    `prior_mean`; a rich cell (large `cell_n`) trusts its own `cell_mean`.
    """
    denom = cell_n + k
    if denom == 0:
        return prior_mean
    return (cell_n * cell_mean + k * prior_mean) / denom


def attribute(trades: Sequence[dict], k: float = 40.0) -> dict[tuple, dict]:
    """Group resolved trades by context cell + bot, shrinking each bot's mean
    PnL toward its cell's global mean.

    Returns:
        {cell: {"n": int, "global_pnl": float,
                "bots": {bot: {"n": int, "pnl": float, "shrunk_pnl": float}}}}
    """
    by_cell: dict[tuple, dict] = {}
    for trade in trades:
        cell = trade.get("cell")
        if cell is None:
            continue
        pnl = float(trade.get("pnl") or 0.0)
        bot_name = trade.get("bot_name")

        cell_bucket = by_cell.setdefault(cell, {"pnls": [], "bots": {}})
        cell_bucket["pnls"].append(pnl)
        bot_pnls = cell_bucket["bots"].setdefault(bot_name, [])
        bot_pnls.append(pnl)

    result: dict[tuple, dict] = {}
    for cell, cell_bucket in by_cell.items():
        cell_pnls = cell_bucket["pnls"]
        cell_n = len(cell_pnls)
        cell_mean = sum(cell_pnls) / cell_n if cell_n else 0.0

        bots: dict = {}
        for bot_name, bot_pnls in cell_bucket["bots"].items():
            n = len(bot_pnls)
            mean = sum(bot_pnls) / n if n else 0.0
            bots[bot_name] = {
                "n": n,
                "pnl": mean,
                "shrunk_pnl": shrink(mean, n, cell_mean, k),
            }

        result[cell] = {
            "n": cell_n,
            "global_pnl": cell_mean,
            "bots": bots,
        }

    return result


def validate_cell(train_trades: Sequence[dict], val_trades: Sequence[dict],
                   k: float) -> bool:
    """OOS check: the train-best bot must not lose on the validation half."""
    if not train_trades or not val_trades:
        return False
    train = attribute(train_trades, k)
    if not train:
        return False
    cell = next(iter(train))
    bots = train[cell]["bots"]
    best = max(bots, key=lambda b: bots[b]["shrunk_pnl"])
    val_pnls = [float(t["pnl"] or 0.0) for t in val_trades if t["bot_name"] == best]
    if not val_pnls:
        return False
    return (sum(val_pnls) / len(val_pnls)) >= 0.0


def rebuild() -> dict:
    """Recompute the regime map from live resolved trades and persist it."""
    k = float(getattr(config, "REGIME_SHRINKAGE_K", 40))
    min_n = int(getattr(config, "REGIME_MIN_SAMPLES", 60))
    trades = db.get_resolved_trades_with_context()
    by_cell = attribute(trades, k)

    # Group raw trades per cell for the OOS split (chronological).
    raw: dict[tuple, list] = {}
    for t in trades:
        if t.get("cell") is not None:
            raw.setdefault(t["cell"], []).append(t)

    regimes = []
    for cell, agg in by_cell.items():
        cell_trades = sorted(raw.get(cell, []), key=lambda t: t.get("created_at") or "")
        validated = False
        if agg["n"] >= min_n:
            # Per-bot chronological split: within EACH bot's own trade
            # stream, earlier half -> train, later half -> val. A single
            # interleaved (odd/even) split is a RANDOM split, not an
            # out-of-TIME holdout — it defeats the point of the validation,
            # since a bot whose edge decayed over the window would still
            # have both halves contain a 50/50 mix of its good and bad
            # periods. Splitting per-bot preserves true time-order within
            # each bot's stream (a real recency-aware holdout) while still
            # guaranteeing every bot with >=2 trades appears in both halves,
            # regardless of which time blocks different bots traded in.
            by_bot: dict = {}
            for t in cell_trades:
                by_bot.setdefault(t.get("bot_name"), []).append(t)
            train_trades: list = []
            val_trades: list = []
            for bot_trades in by_bot.values():
                bot_trades = sorted(bot_trades, key=lambda t: t.get("created_at") or "")
                mid = len(bot_trades) // 2
                if mid == 0:
                    # Single-trade bot: no split possible, contributes only
                    # to train (can't be evidenced OOS from one trade).
                    train_trades.extend(bot_trades)
                else:
                    train_trades.extend(bot_trades[:mid])
                    val_trades.extend(bot_trades[mid:])
            validated = validate_cell(train_trades, val_trades, k)
        regimes.append({
            "cell": cell,                # tuple; json.dumps encodes as an array,
                                          # matched back via tuple() after reload
            "n": agg["n"],
            "validated": bool(validated),
            "bot_edges": agg["bots"],
        })

    regimes.sort(key=lambda r: r["n"], reverse=True)
    payload = {"regimes": regimes, "updated_at": time.time()}
    db.set_regime_map(payload)
    return payload


def edges_for_cell(cell: tuple) -> dict | None:
    """Validated per-bot shrunk edges for a cell, or None if not validated."""
    payload = db.get_regime_map()
    for r in payload.get("regimes", []):
        if tuple(r.get("cell") or []) == tuple(cell) and r.get("validated"):
            return r.get("bot_edges") or {}
    return None

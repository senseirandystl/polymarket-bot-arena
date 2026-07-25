"""Layer 2 of the regime-discovery design: per-bot performance attribution over
market context, empirical-Bayes shrunk toward coarser priors.

This pass (part 1) contains ONLY pure attribution math — `shrink` and
`attribute`. Discovery of named regimes, out-of-sample validation, and
arena_state persistence are a LATER task; do not add them here.
"""
from __future__ import annotations

from typing import Sequence


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

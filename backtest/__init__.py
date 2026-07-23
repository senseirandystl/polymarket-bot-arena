"""Offline backtesting framework — replay resolved BTC 5-min markets.

Runs any combination of the arena's DIRECTIONAL bots (momentum, meanrev,
meanrev-tp, sniper, phantom, sentiment, hybrid) against historical resolved
markets through the bots' REAL ``make_decision`` path, with depth-walked fills
(:mod:`polymarket_fills`), taker fees and a slippage band — the same math the
paper/live venues use. Nothing here writes to the live trade tables; the only
DB write is the opt-in run record (``backtest_runs``, mirroring
``lane_validation_runs`` from the Signal Lab harness).

Honesty caveat (same as tools/validate_signals.py): historical order-book
DEPTH is not archived by Polymarket, so fills walk a synthetic ask ladder
anchored on the recorded PM mid (config.BACKTEST_* spread/depth). Results are
an optimistic upper bound — use them for ordering/sign and regime analysis,
and the live DB for ground truth.

Entry points:
  * CLI:  ``.venv/bin/python3 -m backtest --days 3 --bots momentum,hybrid``
  * Code: ``from backtest import run_backtest`` (arena/tools callable)
"""

from backtest.engine import run_backtest, BacktestResult
from backtest.walkforward import walk_forward

__all__ = ["run_backtest", "walk_forward", "BacktestResult"]

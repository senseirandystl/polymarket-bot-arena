"""Pre-promotion backtest gate for GA offspring.

A candidate bot must clear a short offline replay of recent resolved markets
before it is written to the live roster. This avoids swapping in a mutant that
fails on recent history — preferred over live shadow (which burns wall-clock
and capital while the bad bot trades).

Network/cache failures are non-fatal when ``GA_BACKTEST_REQUIRED`` is False:
the gate returns ``passed=True`` with ``reason='data_unavailable'`` so evolution
still completes; when True, the spawn is rejected and the slot keeps the
parent/defaults path at the call site.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import config

logger = logging.getLogger("arena")

# Module-level cache so we don't re-fetch Gamma/Binance every 2h cycle.
_cache: dict[str, Any] = {"data": None, "ts": 0.0, "n": 0}


@dataclass
class GateResult:
    passed: bool
    reason: str
    child_pnl: float | None = None
    baseline_pnl: float | None = None
    markets: int = 0
    elapsed_sec: float = 0.0
    detail: str = ""


def _cache_ttl() -> float:
    return float(getattr(config, "GA_BACKTEST_CACHE_SEC", 3600.0))


def _load_history(n_markets: int):
    """Load/cached HistoricalData for the most recent N resolved markets."""
    now = time.time()
    if (
        _cache["data"] is not None
        and _cache["n"] >= n_markets
        and (now - float(_cache["ts"])) < _cache_ttl()
    ):
        data = _cache["data"]
        # Trim to last n_markets if we cached more
        if len(data.markets) > n_markets:
            from backtest.data import HistoricalData
            return HistoricalData(
                markets=data.markets[-n_markets:],
                btc_opens=data.btc_opens,
                btc_closes=data.btc_closes,
                pm_prices=data.pm_prices,
            )
        return data

    from backtest.data import fetch_resolved_markets, load_historical_data
    markets = fetch_resolved_markets(limit=n_markets)
    data = load_historical_data(markets, use_cache=True)
    _cache["data"] = data
    _cache["ts"] = now
    _cache["n"] = len(data.markets) if data else 0
    return data


def _run_one(bot, data) -> float:
    """Replay one bot; return total_pnl (0.0 if no trades)."""
    from backtest.engine import run_backtest
    from backtest.metrics import trade_stats

    result = run_backtest([bot], data, compound=False)
    stats = trade_stats(result.trades)
    return float(stats.get("total_pnl") or 0.0)


def evaluate_offspring(
    child_bot,
    *,
    baseline_bot=None,
    strategy_type: str | None = None,
    load_fn: Callable[[int], Any] | None = None,
    run_fn: Callable[[Any, Any], float] | None = None,
) -> GateResult:
    """Backtest ``child_bot``; optionally require improvement vs baseline.

    ``load_fn`` / ``run_fn`` are injectable for unit tests (no network).
    """
    t0 = time.time()
    if not getattr(config, "GA_BACKTEST_GATE_ENABLED", True):
        return GateResult(True, "disabled", elapsed_sec=0.0)

    n = max(5, int(getattr(config, "GA_BACKTEST_MARKETS", 40)))
    required = bool(getattr(config, "GA_BACKTEST_REQUIRED", False))
    min_pnl = getattr(config, "GA_BACKTEST_MIN_PNL", None)
    beat_baseline = bool(getattr(config, "GA_BACKTEST_BEAT_BASELINE", True))

    try:
        data = (load_fn or _load_history)(n)
    except Exception as e:
        logger.warning("GA backtest gate: history load failed: %s", e)
        if required:
            return GateResult(
                False, "data_unavailable", elapsed_sec=time.time() - t0,
                detail=str(e),
            )
        return GateResult(
            True, "data_unavailable", elapsed_sec=time.time() - t0,
            detail=str(e),
        )

    if data is None or not getattr(data, "markets", None):
        if required:
            return GateResult(False, "no_markets", elapsed_sec=time.time() - t0)
        return GateResult(True, "no_markets", elapsed_sec=time.time() - t0)

    runner = run_fn or _run_one
    try:
        child_pnl = float(runner(child_bot, data))
    except Exception as e:
        logger.warning("GA backtest gate: child run failed: %s", e)
        if required:
            return GateResult(
                False, "run_failed", elapsed_sec=time.time() - t0, detail=str(e),
            )
        return GateResult(
            True, "run_failed_soft", elapsed_sec=time.time() - t0, detail=str(e),
        )

    baseline_pnl = None
    if baseline_bot is not None and beat_baseline:
        try:
            baseline_pnl = float(runner(baseline_bot, data))
        except Exception as e:
            logger.debug("GA backtest gate: baseline run failed: %s", e)
            baseline_pnl = None

    n_mkt = len(data.markets)
    elapsed = time.time() - t0

    # Floor: optional absolute min P&L
    if min_pnl is not None and child_pnl < float(min_pnl):
        return GateResult(
            False, "below_min_pnl", child_pnl=child_pnl,
            baseline_pnl=baseline_pnl, markets=n_mkt, elapsed_sec=elapsed,
            detail=f"child_pnl={child_pnl:.2f} < min={min_pnl}",
        )

    # Prefer not worse than the bot we're replacing (when baseline available)
    if baseline_pnl is not None and beat_baseline:
        # Allow small noise band so ties pass
        eps = float(getattr(config, "GA_BACKTEST_EPS", 0.50))
        if child_pnl + eps < baseline_pnl:
            return GateResult(
                False, "worse_than_baseline", child_pnl=child_pnl,
                baseline_pnl=baseline_pnl, markets=n_mkt, elapsed_sec=elapsed,
                detail=f"child={child_pnl:.2f} baseline={baseline_pnl:.2f}",
            )

    return GateResult(
        True, "passed", child_pnl=child_pnl, baseline_pnl=baseline_pnl,
        markets=n_mkt, elapsed_sec=elapsed,
    )


def clear_cache() -> None:
    """Test helper."""
    _cache["data"] = None
    _cache["ts"] = 0.0
    _cache["n"] = 0

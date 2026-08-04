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


def _run_detail(bot, data) -> tuple[float, dict[str, float], list]:
    """Replay one bot; return (total_pnl, pnl_by_regime, trades)."""
    from backtest.engine import run_backtest
    from backtest.metrics import trade_stats

    result = run_backtest([bot], data, compound=False)
    stats = trade_stats(result.trades)
    total = float(stats.get("total_pnl") or 0.0)
    by_reg: dict[str, float] = {}
    for t in result.trades or []:
        ctx = getattr(t, "context", None) or {}
        reg = str(ctx.get("regime") or "unknown")
        pnl = float(getattr(t, "pnl", 0.0) or 0.0)
        by_reg[reg] = by_reg.get(reg, 0.0) + pnl
    return total, by_reg, list(result.trades or [])


def _live_regime_label() -> str | None:
    """Best-effort current live regime for mix-aware gating."""
    try:
        from signals.regime_detector import get_detector
        snap = get_detector().snapshot() or {}
        lab = snap.get("label") or snap.get("regime_id") or snap.get("regime")
        if lab and lab not in ("unknown",):
            return str(lab)
    except Exception:
        pass
    try:
        raw = __import__("db").get_arena_state("market_regime")
        if raw:
            import json
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, dict):
                lab = data.get("label") or data.get("regime_id")
                if lab and lab not in ("unknown",):
                    return str(lab)
    except Exception:
        pass
    return None


def evaluate_offspring(
    child_bot,
    *,
    baseline_bot=None,
    strategy_type: str | None = None,
    load_fn: Callable[[int], Any] | None = None,
    run_fn: Callable[[Any, Any], float] | None = None,
    live_regime: str | None = None,
) -> GateResult:
    """Backtest ``child_bot``; optionally require improvement vs baseline.

    When ``GA_BACKTEST_REGIME_MIX`` is on, also require the child not be
    materially worse than baseline *in the current live regime subset* of
    trades (context.regime stamp from the backtest engine). That stops a
    mutant that only prints in trend hours from replacing a bot during chop.

    ``load_fn`` / ``run_fn`` are injectable for unit tests (no network).
    ``run_fn`` may return a bare float (legacy) or
    ``(total_pnl, by_regime_dict)``.
    """
    t0 = time.time()
    if not getattr(config, "GA_BACKTEST_GATE_ENABLED", True):
        return GateResult(True, "disabled", elapsed_sec=0.0)

    n = max(5, int(getattr(config, "GA_BACKTEST_MARKETS", 40)))
    required = bool(getattr(config, "GA_BACKTEST_REQUIRED", False))
    min_pnl = getattr(config, "GA_BACKTEST_MIN_PNL", None)
    beat_baseline = bool(getattr(config, "GA_BACKTEST_BEAT_BASELINE", True))
    regime_mix = bool(getattr(config, "GA_BACKTEST_REGIME_MIX", True))

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

    def _unpack(raw):
        if isinstance(raw, tuple) and len(raw) >= 2:
            return float(raw[0]), (raw[1] if isinstance(raw[1], dict) else {})
        return float(raw), {}

    child_by_reg: dict[str, float] = {}
    baseline_by_reg: dict[str, float] = {}

    try:
        if run_fn is not None:
            child_pnl, child_by_reg = _unpack(run_fn(child_bot, data))
        else:
            child_pnl, child_by_reg, _ = _run_detail(child_bot, data)
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
            if run_fn is not None:
                baseline_pnl, baseline_by_reg = _unpack(run_fn(baseline_bot, data))
            else:
                baseline_pnl, baseline_by_reg, _ = _run_detail(baseline_bot, data)
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
    eps = float(getattr(config, "GA_BACKTEST_EPS", 0.50))
    if baseline_pnl is not None and beat_baseline:
        if child_pnl + eps < baseline_pnl:
            return GateResult(
                False, "worse_than_baseline", child_pnl=child_pnl,
                baseline_pnl=baseline_pnl, markets=n_mkt, elapsed_sec=elapsed,
                detail=f"child={child_pnl:.2f} baseline={baseline_pnl:.2f}",
            )

    # Regime-mix: must not underperform baseline in the live regime subset
    if (
        regime_mix
        and baseline_pnl is not None
        and beat_baseline
        and (child_by_reg or baseline_by_reg)
    ):
        reg = live_regime if live_regime is not None else _live_regime_label()
        if reg:
            # Map robust ids ↔ legacy labels loosely
            c_reg = child_by_reg.get(reg, 0.0)
            b_reg = baseline_by_reg.get(reg, 0.0)
            # Also try partial key match (high_vol_chop vs chop)
            if reg not in child_by_reg and reg not in baseline_by_reg:
                for k in set(child_by_reg) | set(baseline_by_reg):
                    if reg in k or k in reg:
                        c_reg = child_by_reg.get(k, c_reg)
                        b_reg = baseline_by_reg.get(k, b_reg)
            reg_eps = float(getattr(config, "GA_BACKTEST_REGIME_EPS", eps))
            min_reg_n = int(getattr(config, "GA_BACKTEST_REGIME_MIN_TRADES", 3))
            # Only enforce when baseline actually traded that regime enough
            # (approx via whether baseline has non-zero stamp count — we use
            # non-zero pnl mass or presence as proxy)
            baseline_has = reg in baseline_by_reg or any(
                reg in k or k in reg for k in baseline_by_reg
            )
            if baseline_has and c_reg + reg_eps < b_reg:
                return GateResult(
                    False, "worse_in_live_regime",
                    child_pnl=child_pnl, baseline_pnl=baseline_pnl,
                    markets=n_mkt, elapsed_sec=elapsed,
                    detail=(
                        f"regime={reg} child={c_reg:.2f} baseline={b_reg:.2f} "
                        f"(overall child={child_pnl:.2f} base={baseline_pnl:.2f})"
                    ),
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

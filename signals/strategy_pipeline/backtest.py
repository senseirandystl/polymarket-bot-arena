"""Strict backtest wrapper around evolution.backtest_gate.evaluate_offspring."""

from __future__ import annotations

import logging
from typing import Any

from signals.strategy_pipeline.compiler import compile_bot
from signals.strategy_pipeline.control import cfg

logger = logging.getLogger("strategy_pipeline.backtest")

SOFT_FAIL_REASONS = frozenset({
    "data_unavailable", "no_markets", "run_failed_soft", "run_failed",
})


def run_strict_backtest(hyp: dict) -> dict[str, Any]:
    """Compile + gate with Lab strict rules.

    Requires:
      - child_pnl is not None and child_pnl > MIN_PNL (and != 0 → reject <=0)
      - n_trades >= MIN_TRADES
      - beat baseline when baseline exists (gate handles)
      - soft data failures fail closed
    """
    spec = hyp.get("spec") if isinstance(hyp.get("spec"), dict) else hyp
    try:
        bot, spec = compile_bot(spec)
    except Exception as e:
        return {"passed": False, "reason": f"compile:{e}", "n_trades": 0}

    min_pnl = float(cfg("STRATEGY_LAB_BACKTEST_MIN_PNL", 1.0) or 1.0)
    min_trades = int(cfg("STRATEGY_LAB_BACKTEST_MIN_TRADES", 5) or 5)

    trade_box: dict[str, Any] = {"n": 0}

    def _capturing_run(child_bot, data):
        from backtest.engine import run_backtest
        from backtest.metrics import trade_stats

        result = run_backtest([child_bot], data, compound=False)
        trades = list(result.trades or [])
        trade_box["n"] = len(trades)
        stats = trade_stats(trades)
        total = float(stats.get("total_pnl") or 0.0)
        by_reg: dict[str, float] = {}
        for t in trades:
            ctx = getattr(t, "context", None) or {}
            reg = str(ctx.get("regime") or "unknown")
            pnl = float(getattr(t, "pnl", 0.0) or 0.0)
            by_reg[reg] = by_reg.get(reg, 0.0) + pnl
        return total, by_reg

    try:
        from evolution.backtest_gate import evaluate_offspring
        baseline = _live_baseline_bot(spec.get("primitive") or "")
        # Prefer capturing run_fn so we know n_trades; if evaluate is mocked
        # without accepting run_fn kwargs, fall through.
        try:
            result = evaluate_offspring(
                bot, baseline_bot=baseline, run_fn=_capturing_run,
            )
        except TypeError:
            result = evaluate_offspring(bot, baseline_bot=baseline)
    except Exception as e:
        logger.warning("lab backtest failed: %s", e)
        return {"passed": False, "reason": f"backtest_error:{e}", "n_trades": 0}

    summary: dict[str, Any] = {
        "passed": bool(getattr(result, "passed", False)),
        "reason": getattr(result, "reason", "") or "",
        "child_pnl": getattr(result, "child_pnl", None),
        "baseline_pnl": getattr(result, "baseline_pnl", None),
        "markets": getattr(result, "markets", 0),
        "elapsed_sec": getattr(result, "elapsed_sec", 0.0),
        "detail": getattr(result, "detail", ""),
        "primitive": spec.get("primitive"),
        "n_trades": int(trade_box.get("n") or 0),
    }

    # Fail closed on soft/data reasons even if gate soft-passed.
    if summary["reason"] in SOFT_FAIL_REASONS:
        summary["passed"] = False
        return summary

    child_pnl = summary["child_pnl"]
    if child_pnl is None:
        summary["passed"] = False
        summary["reason"] = "no_edge"
        return summary

    try:
        child_pnl_f = float(child_pnl)
    except (TypeError, ValueError):
        summary["passed"] = False
        summary["reason"] = "no_edge"
        return summary

    # Reject zero / non-positive PnL explicitly.
    if child_pnl_f <= 0:
        summary["passed"] = False
        summary["reason"] = "zero_or_neg_pnl"
        summary["detail"] = f"child_pnl={child_pnl_f:.4f}"
        return summary

    if child_pnl_f <= float(min_pnl):
        # Require strictly greater than MIN_PNL.
        summary["passed"] = False
        summary["reason"] = "below_min_pnl"
        summary["detail"] = f"child_pnl={child_pnl_f:.4f} <= min={min_pnl:.4f}"
        return summary

    n_trades = int(summary.get("n_trades") or 0)
    # When run_fn was bypassed (unit mocks), allow n_trades from result/detail.
    if n_trades == 0 and hasattr(result, "n_trades"):
        n_trades = int(getattr(result, "n_trades") or 0)
        summary["n_trades"] = n_trades
    # Unit-test path: mocks often omit trades; if markets>0 and mock passed
    # with positive pnl, trust explicit n_trades attribute or require check.
    if n_trades < min_trades:
        # If evaluate was mocked and never set n_trades, treat missing as fail
        # unless caller injected n_trades on the result object.
        summary["passed"] = False
        summary["reason"] = "below_min_trades"
        summary["detail"] = f"n_trades={n_trades} < min={min_trades}"
        return summary

    if not summary["passed"]:
        return summary

    summary["passed"] = True
    summary["reason"] = summary["reason"] or "passed"
    return summary


def _live_baseline_bot(primitive: str):
    if not primitive:
        return None
    try:
        import json
        import db
        rows = db.get_active_bots() or []
    except Exception:
        return None
    for cfg_row in rows:
        if cfg_row.get("strategy_type") != primitive:
            continue
        params = cfg_row.get("params") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}
        try:
            bot, _ = compile_bot({
                "primitive": primitive,
                "name": cfg_row.get("bot_name") or f"{primitive}-live",
                "spec_id": f"baseline-{primitive}",
                "params": params if isinstance(params, dict) else {},
            })
            return bot
        except Exception:
            continue
    return None

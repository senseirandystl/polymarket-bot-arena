"""Performance metrics for backtest results (pure functions, no I/O)."""

from __future__ import annotations

import math


def _safe_div(a: float, b: float):
    return a / b if b else None


def trade_stats(trades: list) -> dict:
    """Core stats over resolved trades: expectancy, WR, profit factor, …"""
    n = len(trades)
    if n == 0:
        return {"n": 0, "wins": 0, "losses": 0, "win_rate": None,
                "total_pnl": 0.0, "expectancy": None, "profit_factor": None,
                "sharpe": None, "avg_entry_price": None, "breakeven_gap": None,
                "avg_win": None, "avg_loss": None, "total_fees": 0.0}
    wins = [t for t in trades if t.outcome == "win"]
    losses = [t for t in trades if t.outcome == "loss"]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    pnls = [t.pnl for t in trades]
    total = sum(pnls)
    mean = total / n
    var = sum((p - mean) ** 2 for p in pnls) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    wr = len(wins) / n
    avg_entry = sum(t.entry_price for t in trades) / n
    return {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "total_pnl": total,
        "expectancy": mean,                       # $/trade
        "profit_factor": _safe_div(gross_win, gross_loss),
        # Per-trade Sharpe (mean/sd of trade P&L) — unannualized on purpose:
        # 5-min windows make annualization numbers meaninglessly large.
        "sharpe": _safe_div(mean, sd),
        "avg_entry_price": avg_entry,
        # The core PBA metric: WR must beat avg entry by >=5c to break even.
        "breakeven_gap": wr - avg_entry,
        "avg_win": _safe_div(gross_win, len(wins)),
        "avg_loss": _safe_div(-gross_loss, len(losses)),
        "total_fees": sum(t.fee for t in trades),
    }


def max_drawdown(equity_curve: list) -> dict:
    """Max peak-to-trough drawdown over [(ts, equity)] points."""
    peak = float("-inf")
    dd = 0.0
    dd_pct = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
        if peak > 0:
            dd_pct = max(dd_pct, (peak - eq) / peak)
    return {"max_drawdown": dd if peak != float("-inf") else 0.0,
            "max_drawdown_pct": dd_pct}


def _grouped(trades: list, key) -> dict:
    groups: dict = {}
    for t in trades:
        groups.setdefault(key(t), []).append(t)
    return {k: trade_stats(v) for k, v in sorted(groups.items(),
                                                 key=lambda kv: str(kv[0]))}


def per_bot(trades: list) -> dict:
    return _grouped(trades, lambda t: t.bot_name)


def per_side(trades: list) -> dict:
    return _grouped(trades, lambda t: t.side)


def _drift_band(v: float) -> str:
    a = abs(v)
    if a < 0.10:
        return "|drift|<0.10"
    if a < 0.30:
        return "0.10<=|drift|<0.30"
    return "|drift|>=0.30"


def _price_bucket(p: float) -> str:
    for lo, hi in ((0.0, 0.35), (0.35, 0.45), (0.45, 0.55),
                   (0.55, 0.65), (0.65, 0.72), (0.72, 1.0)):
        if lo <= p < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return "other"


def _time_bucket(tr: float) -> str:
    for lo, hi in ((0, 60), (60, 120), (120, 180), (180, 300)):
        if lo < tr <= hi:
            return f"{lo}-{hi}s"
    return ">300s"


def per_regime(trades: list) -> dict:
    """Regime splits: volatility regime, drift band, entry bucket, time left."""
    return {
        "vol_regime": _grouped(trades, lambda t: t.context.get("regime") or "unknown"),
        "drift_band": _grouped(trades, lambda t: _drift_band(t.context.get("drift", 0.0))),
        "entry_price_bucket": _grouped(trades, lambda t: _price_bucket(t.entry_price)),
        "time_remaining": _grouped(trades, lambda t: _time_bucket(t.time_remaining)),
    }


def signal_contribution(samples: list, trades: list,
                        deadband: float = 0.02) -> dict:
    """Per-lane predictiveness over the replay + realized P&L attribution.

    ``follow_wr`` — across ALL decision ticks (not just trades), how often the
    lane's sign called the actual resolution (the Signal Lab predictiveness
    metric). ``traded_*`` — over executed trades, split by whether the lane
    agreed with the side actually bought, so a lane that argued against the
    book's losers shows up.
    """
    out = {}
    for lane in ("drift", "mom", "pm_mom"):
        n = wins = 0
        for s in samples:
            v = s.get(lane)
            if v is None or abs(v) <= deadband:
                continue
            n += 1
            wins += 1 if (v > 0) == s["yes_won"] else 0
        agree = [t for t in trades
                 if abs(t.context.get(lane, 0.0)) > deadband
                 and ((t.context[lane] > 0) == (t.side == "yes"))]
        contra = [t for t in trades
                  if abs(t.context.get(lane, 0.0)) > deadband
                  and ((t.context[lane] > 0) != (t.side == "yes"))]
        out[lane] = {
            "sample_n": n,
            "follow_wr": _safe_div(wins, n),
            "traded_agree": trade_stats(agree),
            "traded_contra": trade_stats(contra),
        }
    return out


def summarize(result) -> dict:
    """Full metrics bundle for a BacktestResult (JSON-serializable)."""
    return {
        "markets_replayed": result.markets_replayed,
        "decisions": result.decisions,
        "initial_bankroll": result.initial_bankroll,
        "final_bankroll": result.final_bankroll,
        "overall": trade_stats(result.trades),
        **max_drawdown(result.equity_curve),
        "per_bot": per_bot(result.trades),
        "per_side": per_side(result.trades),
        "per_regime": per_regime(result.trades),
        "signal_contribution": signal_contribution(result.samples, result.trades),
        "skips": dict(sorted(result.skips.items(), key=lambda kv: -kv[1])),
        "rejects": dict(result.rejects),
        "config": result.config_snapshot,
    }

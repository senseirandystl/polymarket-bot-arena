"""Walk-forward evaluation — train on window A, test on window B.

The arena's bots carry no fitted parameters, so "training" here is what the
evolution loop does live: SELECTION. Each fold ranks the bots on the train
window (by expectancy over a minimum trade count) and then measures how the
selected top-k slate performs on the strictly-later test window, next to the
all-bots baseline — the honest check that a slate chosen on the past keeps
its edge out of sample.
"""

from __future__ import annotations

import logging

from backtest.data import HistoricalData
from backtest.engine import run_backtest
from backtest.metrics import per_bot, summarize, trade_stats

logger = logging.getLogger("backtest.walkforward")


def _slice_data(data: HistoricalData, markets: list) -> HistoricalData:
    ids = {m.id for m in markets}
    return HistoricalData(
        markets=list(markets),
        btc_opens=data.btc_opens,
        btc_closes=data.btc_closes,
        pm_prices={k: v for k, v in data.pm_prices.items() if k in ids})


def walk_forward(bot_factory, data: HistoricalData, folds: int = 3,
                 train_frac: float = 0.6, top_k: int = 3,
                 min_train_trades: int = 5, **run_kwargs) -> dict:
    """Rolling-origin walk-forward over the chronological market list.

    ``bot_factory`` is a zero-arg callable returning a FRESH bot list (bot
    instances hold per-run caches, so folds must not share them). The market
    span is cut into ``folds`` contiguous segments; within each, the first
    ``train_frac`` of markets is the train window and the rest the test.
    """
    markets = data.markets
    if len(markets) < folds * 4:
        raise ValueError(f"Not enough markets ({len(markets)}) for {folds} folds")
    fold_size = len(markets) // folds
    out = {"folds": [], "params": {"folds": folds, "train_frac": train_frac,
                                   "top_k": top_k}}
    for i in range(folds):
        segment = markets[i * fold_size:
                          (i + 1) * fold_size if i < folds - 1 else len(markets)]
        cut = max(1, int(len(segment) * train_frac))
        train_mkts, test_mkts = segment[:cut], segment[cut:]
        if not test_mkts:
            continue

        train_res = run_backtest(bot_factory(), _slice_data(data, train_mkts),
                                 **run_kwargs)
        ranked = sorted(
            ((name, s) for name, s in per_bot(train_res.trades).items()
             if s["n"] >= min_train_trades),
            key=lambda kv: kv[1]["expectancy"], reverse=True)
        selected = [name for name, _ in ranked[:top_k]]

        test_all = run_backtest(bot_factory(), _slice_data(data, test_mkts),
                                **run_kwargs)
        selected_trades = [t for t in test_all.trades
                           if t.bot_name in selected]
        out["folds"].append({
            "fold": i + 1,
            "train_markets": len(train_mkts),
            "test_markets": len(test_mkts),
            "train_span": [train_mkts[0].close_ts, train_mkts[-1].close_ts],
            "test_span": [test_mkts[0].close_ts, test_mkts[-1].close_ts],
            "train_ranking": [
                {"bot": name, "expectancy": s["expectancy"], "n": s["n"],
                 "win_rate": s["win_rate"]} for name, s in ranked],
            "selected": selected,
            "test_all_bots": trade_stats(test_all.trades),
            "test_selected": trade_stats(selected_trades),
            "test_summary": summarize(test_all),
        })
    # Aggregate: does selection beat the baseline out of sample?
    sel = [f["test_selected"] for f in out["folds"] if f["test_selected"]["n"]]
    base = [f["test_all_bots"] for f in out["folds"] if f["test_all_bots"]["n"]]
    out["aggregate"] = {
        "selected_total_pnl": sum(s["total_pnl"] for s in sel),
        "baseline_total_pnl": sum(s["total_pnl"] for s in base),
        "selected_trades": sum(s["n"] for s in sel),
        "baseline_trades": sum(s["n"] for s in base),
    }
    return out


def format_walkforward(wf: dict) -> str:
    lines = ["=== Walk-forward report ===",
             f"params: {wf['params']}"]
    for f in wf["folds"]:
        lines.append(f"\n-- Fold {f['fold']}: train {f['train_markets']} mkts "
                     f"-> test {f['test_markets']} mkts --")
        lines.append(f"  train ranking: " + ", ".join(
            f"{r['bot']} (${r['expectancy']:+.3f}/tr, n={r['n']})"
            for r in f["train_ranking"]) if f["train_ranking"]
            else "  train ranking: (no bot cleared min trades)")
        lines.append(f"  selected: {f['selected'] or '(none)'}")
        a, s = f["test_all_bots"], f["test_selected"]
        lines.append(f"  test ALL:      n={a['n']:3d}  "
                     + (f"WR={a['win_rate']*100:.1f}%  P&L=${a['total_pnl']:+.2f}"
                        if a["n"] else ""))
        lines.append(f"  test SELECTED: n={s['n']:3d}  "
                     + (f"WR={s['win_rate']*100:.1f}%  P&L=${s['total_pnl']:+.2f}"
                        if s["n"] else ""))
    agg = wf["aggregate"]
    lines.append(f"\naggregate out-of-sample: selected "
                 f"${agg['selected_total_pnl']:+.2f} "
                 f"({agg['selected_trades']} trades) vs baseline "
                 f"${agg['baseline_total_pnl']:+.2f} "
                 f"({agg['baseline_trades']} trades)")
    return "\n".join(lines)

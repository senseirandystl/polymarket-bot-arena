"""Report rendering + persistence for backtest runs.

Reports go to JSON files under ``backtest/reports/`` (gitignored) and,
optionally, a summary row into bot_arena.db's ``backtest_runs`` table (the
same pattern as the Signal Lab's ``lane_validation_runs``) so the dashboard
can list historical runs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("backtest.report")

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def _fmt_stats(s: dict, label: str = "") -> str:
    if not s or not s.get("n"):
        return f"    {label:<24} n=0"
    pf = s["profit_factor"]
    parts = [
        f"    {label:<24} n={s['n']:4d}",
        f"WR={s['win_rate']*100:5.1f}%",
        f"P&L=${s['total_pnl']:+8.2f}",
        f"exp=${s['expectancy']:+.3f}/tr",
        f"PF={pf:.2f}" if pf is not None else "PF=inf",
    ]
    if s["sharpe"] is not None:
        parts.append(f"Sharpe={s['sharpe']:+.2f}")
    if s["breakeven_gap"] is not None:
        parts.append(f"BEgap={s['breakeven_gap']*100:+.1f}c")
    return "  ".join(parts)


def format_report(summary: dict, title: str = "Backtest report") -> str:
    """Human-readable console report from metrics.summarize() output."""
    L = [f"=== {title} ===",
         "NOTE: fills walk a synthetic book on recorded (stale) PM mids — "
         "treat results as an optimistic upper bound; use for ordering/sign, "
         "regime splits and relative bot ranking, not absolute P&L.",
         f"markets={summary['markets_replayed']}  "
         f"decision-calls={summary['decisions']}  "
         f"bankroll ${summary['initial_bankroll']:.2f} -> "
         f"${summary['final_bankroll']:.2f}  "
         f"maxDD ${summary['max_drawdown']:.2f} "
         f"({summary['max_drawdown_pct']*100:.1f}%)",
         "", "-- Overall --", _fmt_stats(summary["overall"], "all trades")]

    L += ["", "-- Per bot --"]
    for name, s in summary["per_bot"].items():
        L.append(_fmt_stats(s, name))
    L += ["", "-- Per side --"]
    for name, s in summary["per_side"].items():
        L.append(_fmt_stats(s, name))

    for dim, groups in summary["per_regime"].items():
        L += ["", f"-- By {dim} --"]
        for name, s in groups.items():
            L.append(_fmt_stats(s, str(name)))

    L += ["", "-- Signal contribution --"]
    for lane, m in summary["signal_contribution"].items():
        fw = m["follow_wr"]
        fw_s = f"{fw*100:5.1f}%" if fw is not None else "  n/a"
        L.append(f"    [{lane:<7}] ticks n={m['sample_n']:5d} follow-WR={fw_s}")
        L.append(_fmt_stats(m["traded_agree"], "  trades agreeing"))
        L.append(_fmt_stats(m["traded_contra"], "  trades contra"))

    if summary.get("skips"):
        L += ["", "-- Top skip reasons --"]
        for reason, count in list(summary["skips"].items())[:8]:
            L.append(f"    {count:6d}  {reason}")
    if summary.get("rejects"):
        L += ["", "-- Fill rejects --"]
        for reason, count in summary["rejects"].items():
            L.append(f"    {count:6d}  {reason}")
    L += ["", f"config: {summary.get('config')}"]
    return "\n".join(L)


def write_json(summary: dict, label: str, path: str | None = None) -> Path:
    """Write the summary JSON. Default path: backtest/reports/<ts>-<label>.json."""
    if path:
        out = Path(path)
    else:
        REPORTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:40]
        out = REPORTS_DIR / f"{stamp}-{safe or 'run'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Report written: {out}")
    return out


def record_to_db(summary: dict, label: str, report_path: Path | None) -> int:
    """Store a summary row in bot_arena.db (backtest_runs). Returns row id."""
    import db
    db.init_db()
    return db.record_backtest_run(
        label=label,
        markets=summary["markets_replayed"],
        trades=summary["overall"]["n"],
        summary=summary,
        report_path=str(report_path) if report_path else None)

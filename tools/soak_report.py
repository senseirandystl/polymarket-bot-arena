"""Paper/live soak report — bot/signal/regime/lane health snapshot.

Callable from the dashboard Settings button (sends via alerts) or CLI::

    .venv/bin/python3 -m tools.soak_report
    .venv/bin/python3 -m tools.soak_report --notify
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Repo root on path for `import db` when run as module or script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import db  # noqa: E402


def _be(wr, avg_entry):
    if wr is None or avg_entry is None:
        return None
    return round(float(wr) - float(avg_entry), 3)


def build_report() -> dict:
    """Return a structured soak report from the live DB + arena_state."""
    db.init_db()
    with db.get_conn() as conn:
        overall = conn.execute("""
            SELECT COUNT(*) n,
                SUM(CASE WHEN outcome IN ('win','exit_tp') THEN 1 ELSE 0 END) wins,
                ROUND(SUM(pnl), 2) pnl,
                ROUND(AVG(entry_price), 4) avg_entry,
                ROUND(1.0*SUM(CASE WHEN outcome IN ('win','exit_tp')
                    THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0), 4) wr,
                MIN(created_at) first_ts,
                MAX(created_at) last_ts,
                ROUND(SUM(amount), 2) volume,
                ROUND(SUM(COALESCE(fee, 0)), 2) fees
            FROM trades
            WHERE outcome IN ('win','loss','exit_tp','exit_sl')
        """).fetchone()
        overall = dict(overall) if overall else {}
        overall["be_gap"] = _be(overall.get("wr"), overall.get("avg_entry"))

        by_bot = []
        for r in conn.execute("""
            SELECT bot_name, COUNT(*) n,
                ROUND(100.0*SUM(CASE WHEN outcome IN ('win','exit_tp')
                    THEN 1 ELSE 0 END)/COUNT(*), 1) wr_pct,
                ROUND(SUM(pnl), 2) pnl,
                ROUND(AVG(entry_price), 3) avg_entry,
                ROUND(1.0*SUM(CASE WHEN outcome IN ('win','exit_tp')
                    THEN 1 ELSE 0 END)/COUNT(*) - AVG(entry_price), 3) be_gap
            FROM trades
            WHERE outcome IN ('win','loss','exit_tp','exit_sl')
            GROUP BY bot_name ORDER BY pnl DESC
        """).fetchall():
            by_bot.append(dict(r))

        by_side = []
        for r in conn.execute("""
            SELECT side, COUNT(*) n,
                ROUND(100.0*SUM(CASE WHEN outcome IN ('win','exit_tp')
                    THEN 1 ELSE 0 END)/COUNT(*), 1) wr_pct,
                ROUND(SUM(pnl), 2) pnl,
                ROUND(AVG(entry_price), 3) avg_entry
            FROM trades
            WHERE outcome IN ('win','loss','exit_tp','exit_sl')
            GROUP BY side
        """).fetchall():
            by_side.append(dict(r))

        windows = {}
        for label, hours in (("24h", 24), ("48h", 48)):
            r = conn.execute(f"""
                SELECT COUNT(*) n,
                    ROUND(1.0*SUM(CASE WHEN outcome IN ('win','exit_tp')
                        THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0), 4) wr,
                    ROUND(SUM(pnl), 2) pnl
                FROM trades
                WHERE outcome IN ('win','loss','exit_tp','exit_sl')
                  AND datetime(created_at) >= datetime('now', '-{hours} hours')
            """).fetchone()
            windows[label] = dict(r)

        def _state(key):
            raw = conn.execute(
                "SELECT value FROM arena_state WHERE key=?", (key,)
            ).fetchone()
            if not raw:
                return None
            v = raw["value"] if hasattr(raw, "keys") else raw[0]
            try:
                return json.loads(v)
            except Exception:
                return v

        lane_monitor = _state("lane_monitor") or {}
        lane_overrides = _state("lane_overrides") or {}
        core_tuner = _state("core_lane_tuner") or {}
        regime_perf = _state("regime_performance") or {}
        risk = _state("risk_engine") or {}
        skip_counts = _state("skip_counts") or {}
        health = _state("health_last_status")
        decision_rollup = _state("decision_rollup") or {}

    try:
        paper_avail = round(db.get_paper_available(), 2)
    except Exception:
        paper_avail = None
    try:
        from arena.regime_adapt import snapshot as regime_adapt_snap
        regime_adapt = regime_adapt_snap()
    except Exception as e:
        regime_adapt = {"error": str(e)}

    return {
        "overall": overall,
        "windows": windows,
        "by_bot": by_bot,
        "by_side": by_side,
        "paper_available": paper_avail,
        "lane_monitor": lane_monitor,
        "lane_overrides": {
            k: {"enabled": v.get("enabled"), "core": v.get("core", False)}
            for k, v in (lane_overrides or {}).items()
        },
        "core_lane_tuner": {
            "applied": (core_tuner or {}).get("applied"),
            "cell_filter": (core_tuner or {}).get("cell_filter"),
            "lanes_with_data": {
                lane: list((rep or {}).keys())
                for lane, rep in ((core_tuner or {}).get("lanes") or {}).items()
            },
        },
        "regime_performance": regime_perf,
        "regime_adapt": regime_adapt,
        "decision_rollup": {
            "n_total": (decision_rollup or {}).get("n_total"),
            "n_resolved": (decision_rollup or {}).get("n_resolved"),
            "by_action": (decision_rollup or {}).get("by_action"),
            "skip_counterfactual": (decision_rollup or {}).get(
                "skip_counterfactual"),
            "candidate_lanes": (decision_rollup or {}).get("candidate_lanes"),
        },
        "risk_kill": (risk or {}).get("kill_switch"),
        "skip_top": dict(sorted(
            (skip_counts or {}).items(), key=lambda x: -int(x[1] or 0)
        )[:8]),
        "health": health,
    }


def format_text(report: dict) -> str:
    """Compact Telegram-friendly multi-line summary."""
    o = report.get("overall") or {}
    lines = [
        f"n={o.get('n')} WR={100*(o.get('wr') or 0):.1f}% "
        f"PnL=${o.get('pnl')} BE={o.get('be_gap')} "
        f"fees=${o.get('fees')} pool=${report.get('paper_available')}",
        f"span {o.get('first_ts')} → {o.get('last_ts')}",
    ]
    w24 = (report.get("windows") or {}).get("24h") or {}
    lines.append(
        f"24h: n={w24.get('n')} WR={100*(w24.get('wr') or 0):.1f}% "
        f"PnL=${w24.get('pnl')}"
    )
    lines.append("Bots:")
    for b in (report.get("by_bot") or [])[:10]:
        lines.append(
            f"  {b.get('bot_name')}: n={b.get('n')} "
            f"WR={b.get('wr_pct')}% PnL=${b.get('pnl')} BE={b.get('be_gap')}"
        )
    lines.append("Sides:")
    for s in report.get("by_side") or []:
        lines.append(
            f"  {s.get('side')}: n={s.get('n')} WR={s.get('wr_pct')}% "
            f"PnL=${s.get('pnl')}"
        )
    lo = report.get("lane_overrides") or {}
    if lo:
        lines.append("Lanes: " + ", ".join(
            f"{k}={'ON' if v.get('enabled') else 'off'}"
            + ("(core)" if v.get("core") else "")
            for k, v in lo.items()
        ))
    lm = report.get("lane_monitor") or {}
    if lm:
        bits = []
        for k, v in lm.items():
            acc = v.get("accuracy")
            acc_s = f"{100*acc:.0f}%" if acc is not None else "n/a"
            bits.append(f"{k}:{acc_s}/{v.get('n')}[{v.get('verdict')}]")
        lines.append("Monitor: " + " ".join(bits))
    ct = report.get("core_lane_tuner") or {}
    lines.append(
        f"CoreTuner applied={ct.get('applied')} "
        f"cell={ct.get('cell_filter')} data={ct.get('lanes_with_data')}"
    )
    ra = report.get("regime_adapt") or {}
    if ra.get("regimes"):
        lines.append("Regime adapt:")
        for r in ra["regimes"]:
            wr = r.get("wr")
            wr_s = f"{100*wr:.0f}%" if wr is not None else "n/a"
            lines.append(
                f"  {r.get('regime')}: n={r.get('n')} WR={wr_s} "
                f"PnL=${r.get('pnl')} mult={r.get('size_mult')}"
            )
    sk = report.get("skip_top") or {}
    if sk:
        lines.append("Skips: " + ", ".join(f"{k}={v}" for k, v in sk.items()))
    dr = report.get("decision_rollup") or {}
    if dr.get("n_total"):
        lines.append(
            f"Decisions: total={dr.get('n_total')} resolved={dr.get('n_resolved')} "
            f"mix={dr.get('by_action')}"
        )
        scf = dr.get("skip_counterfactual") or {}
        if scf:
            bits = []
            for k, v in list(scf.items())[:6]:
                wr = v.get("wr")
                wr_s = f"{100*wr:.0f}%" if wr is not None else "n/a"
                bits.append(f"{k}:{wr_s}/{v.get('n')}")
            lines.append("Skip CF: " + " ".join(bits))
        cl = dr.get("candidate_lanes") or {}
        if cl:
            bits = []
            for k, v in cl.items():
                acc = v.get("accuracy")
                acc_s = f"{100*acc:.0f}%" if acc is not None else "n/a"
                bits.append(f"{k}:{acc_s}/n={v.get('n')}")
            lines.append("Cand shadow: " + " ".join(bits))
    lines.append(
        f"health={report.get('health')} kill={report.get('risk_kill')}"
    )
    return "\n".join(lines)


def notify(report: dict | None = None) -> dict:
    """Send the soak report on configured alert channels."""
    report = report or build_report()
    body = format_text(report)
    from arena.alerts import notify as alert_notify
    return alert_notify(
        "soak_report",
        "Arena soak report",
        body,
        level="info",
        key="soak_report",
        detail={"n": (report.get("overall") or {}).get("n"),
                "pnl": (report.get("overall") or {}).get("pnl")},
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Arena soak report")
    p.add_argument("--notify", action="store_true",
                   help="Send via alerts (Telegram/etc.)")
    p.add_argument("--json", action="store_true", help="Print raw JSON")
    args = p.parse_args(argv)
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_text(report))
    if args.notify:
        r = notify(report)
        print("notify:", r.get("sent"), r.get("channels") or r.get("reason"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Hypothesis graph + desk event log (SQLite via db.get_conn)."""

from __future__ import annotations

import json
from typing import Any

import db


SCHEMA = """
CREATE TABLE IF NOT EXISTS desk_hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    primitive TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    spec TEXT NOT NULL,
    thesis TEXT,
    parent_spec_ids TEXT,
    bot_name TEXT,
    backtest_summary TEXT,
    paper_summary TEXT,
    live_summary TEXT,
    autopsy TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_desk_hyp_stage ON desk_hypotheses(stage, status);

CREATE TABLE IF NOT EXISTS desk_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_id TEXT,
    role TEXT,
    stage TEXT,
    action TEXT NOT NULL,
    detail TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_desk_events_spec ON desk_events(spec_id, id);
"""


STAGES = (
    "idea",
    "researched",
    "coded",
    "backtested",
    "paper",
    "live",
    "retired",
    "rejected",
)


def init_desk_tables() -> None:
    with db.get_conn() as conn:
        conn.executescript(SCHEMA)


class HypothesisStore:
    def __init__(self) -> None:
        init_desk_tables()

    def insert(self, spec: dict[str, Any]) -> dict[str, Any]:
        spec_id = str(spec["spec_id"])
        row = {
            "spec_id": spec_id,
            "name": spec.get("name") or spec_id,
            "primitive": spec.get("primitive") or "momentum",
            "stage": spec.get("stage") or "idea",
            "status": spec.get("status") or "open",
            "spec": json.dumps(spec),
            "thesis": spec.get("thesis") or "",
            "parent_spec_ids": json.dumps(spec.get("parent_spec_ids") or []),
            "bot_name": spec.get("bot_name"),
        }
        with db.get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO desk_hypotheses
                   (spec_id, name, primitive, stage, status, spec, thesis,
                    parent_spec_ids, bot_name, updated_at)
                   VALUES (:spec_id, :name, :primitive, :stage, :status, :spec,
                           :thesis, :parent_spec_ids, :bot_name, datetime('now'))""",
                row,
            )
        self.log(spec_id, "researcher", "idea", "created", spec.get("thesis") or "")
        return self.get(spec_id) or row

    def get(self, spec_id: str) -> dict[str, Any] | None:
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM desk_hypotheses WHERE spec_id=?", (spec_id,)
            ).fetchone()
        return _row(row) if row else None

    def list(self, *, limit: int = 50, stage: str | None = None) -> list[dict]:
        sql = "SELECT * FROM desk_hypotheses"
        args: list[Any] = []
        if stage:
            sql += " WHERE stage=?"
            args.append(stage)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with db.get_conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [_row(r) for r in rows]

    def open_by_stage(self, *stages: str) -> list[dict]:
        if not stages:
            return []
        ph = ",".join("?" * len(stages))
        with db.get_conn() as conn:
            rows = conn.execute(
                f"""SELECT * FROM desk_hypotheses
                    WHERE status='open' AND stage IN ({ph})
                    ORDER BY id ASC""",
                stages,
            ).fetchall()
        return [_row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT stage, COUNT(*) AS n FROM desk_hypotheses GROUP BY stage"
            ).fetchall()
        out = {s: 0 for s in STAGES}
        for r in rows:
            out[str(r["stage"])] = int(r["n"])
        return out

    def advance(
        self,
        spec_id: str,
        stage: str,
        *,
        status: str = "open",
        bot_name: str | None = None,
        backtest_summary: dict | None = None,
        paper_summary: dict | None = None,
        live_summary: dict | None = None,
        autopsy: dict | None = None,
        spec_update: dict | None = None,
    ) -> None:
        fields = ["stage=?", "status=?", "updated_at=datetime('now')"]
        args: list[Any] = [stage, status]
        if bot_name is not None:
            fields.append("bot_name=?")
            args.append(bot_name)
        if backtest_summary is not None:
            fields.append("backtest_summary=?")
            args.append(json.dumps(backtest_summary))
        if paper_summary is not None:
            fields.append("paper_summary=?")
            args.append(json.dumps(paper_summary))
        if live_summary is not None:
            fields.append("live_summary=?")
            args.append(json.dumps(live_summary))
        if autopsy is not None:
            fields.append("autopsy=?")
            args.append(json.dumps(autopsy))
        if spec_update is not None:
            fields.append("spec=?")
            args.append(json.dumps(spec_update))
        args.append(spec_id)
        with db.get_conn() as conn:
            conn.execute(
                f"UPDATE desk_hypotheses SET {', '.join(fields)} WHERE spec_id=?",
                args,
            )

    def recent_autopsies(self, limit: int = 20) -> list[dict]:
        with db.get_conn() as conn:
            rows = conn.execute(
                """SELECT spec_id, name, primitive, stage, status, thesis, autopsy
                   FROM desk_hypotheses
                   WHERE autopsy IS NOT NULL AND autopsy != ''
                   ORDER BY id DESC LIMIT ?""",
                (int(limit),),
            ).fetchall()
        return [_row(r) for r in rows]

    def log(
        self,
        spec_id: str | None,
        role: str,
        stage: str,
        action: str,
        detail: str = "",
    ) -> None:
        with db.get_conn() as conn:
            conn.execute(
                """INSERT INTO desk_events (spec_id, role, stage, action, detail)
                   VALUES (?, ?, ?, ?, ?)""",
                (spec_id, role, stage, action, detail[:2000] if detail else ""),
            )

    def events(self, *, limit: int = 40) -> list[dict]:
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM desk_events ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [_row(r) for r in rows]


def _row(row) -> dict[str, Any]:
    d = dict(row)
    for key in (
        "spec", "parent_spec_ids", "backtest_summary",
        "paper_summary", "live_summary", "autopsy",
    ):
        raw = d.get(key)
        if isinstance(raw, str) and raw:
            try:
                d[key] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    return d

"""SQLite database for all trades, bot performance, evolution history."""

import sqlite3
import json
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import config

DB_PATH = config.DB_PATH

# Short-TTL cache for per-bot trading mode. get_bot_mode is read on every trade
# (base_bot.execute + the arb bot's execute); this keeps live/paper toggles from
# the dashboard applying within BOT_MODE_CACHE_TTL_SEC while removing a per-trade
# SQLite round-trip. Invalidated on set_bot_mode / retire.
_bot_mode_cache: dict = {}   # bot_name -> (ts, mode)
_bot_mode_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Eastern-Time helpers
#
# Timestamps are STORED as UTC (SQLite ``datetime('now')``). All day-boundary
# and "today/this week" reporting, however, is anchored to America/New_York so
# the dashboard rolls over at 00:00 ET — matching the BTC 5-min markets, which
# trade on ET. These helpers return UTC strings (in the same
# ``YYYY-MM-DD HH:MM:SS`` shape SQLite stores) representing ET day boundaries,
# so they can be compared directly against ``created_at``.
# ---------------------------------------------------------------------------
_ET_ZONE = "America/New_York"


def _et_now() -> datetime:
    """Current time as an aware ET datetime (DST-correct)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(timezone.utc).astimezone(ZoneInfo(_ET_ZONE))
    except Exception:
        # zoneinfo/tzdata unavailable — approximate US DST rules (2nd Sun Mar
        # .. 1st Sun Nov). Good enough for a day-boundary; storage stays UTC.
        now = datetime.now(timezone.utc)
        year = now.year
        mar1 = datetime(year, 3, 1, tzinfo=timezone.utc)
        dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7) + timedelta(weeks=1, hours=7)
        nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
        dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7) + timedelta(hours=6)
        offset = -4 if dst_start <= now < dst_end else -5
        return now + timedelta(hours=offset)


def et_day_start_utc(days_ago: int = 0) -> str:
    """UTC string for 00:00 ET of the day ``days_ago`` days before today (ET)."""
    et_midnight = (_et_now() - timedelta(days=days_ago)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    try:
        # et_midnight is aware (zoneinfo path) — convert to UTC.
        return et_midnight.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OverflowError):
        return et_midnight.strftime("%Y-%m-%d %H:%M:%S")


def utc_to_et_date(utc_str: str) -> str:
    """Convert a stored UTC ``created_at`` string to its ET calendar date."""
    if not utc_str:
        return ""
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "").replace("T", " ").strip())
        dt = dt.replace(tzinfo=timezone.utc)
        from zoneinfo import ZoneInfo
        return dt.astimezone(ZoneInfo(_ET_ZONE)).strftime("%Y-%m-%d")
    except Exception:
        return (utc_str or "")[:10]


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_name TEXT NOT NULL,
                market_id TEXT NOT NULL,
                market_question TEXT,
                side TEXT NOT NULL,
                amount REAL NOT NULL,
                confidence REAL,
                reasoning TEXT,
                trade_features TEXT,
                venue TEXT NOT NULL,
                mode TEXT NOT NULL,
                trade_id TEXT,
                shares_bought REAL,
                outcome TEXT,
                pnl REAL,
                resolved_at TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS bot_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_name TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                generation INTEGER DEFAULT 0,
                lineage TEXT,
                params TEXT NOT NULL,
                active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now')),
                retired_at TEXT
            );

            CREATE TABLE IF NOT EXISTS evolution_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_number INTEGER NOT NULL,
                survivors TEXT NOT NULL,
                replaced TEXT NOT NULL,
                new_bots TEXT NOT NULL,
                rankings TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            -- GA generation detail (fitness, elitism, lineage, operators).
            -- Complements evolution_events; one row per GA cycle.
            CREATE TABLE IF NOT EXISTS ga_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_number INTEGER NOT NULL,
                report TEXT NOT NULL,            -- JSON: full GA report
                best_fitness REAL,
                mean_fitness REAL,
                n_elites INTEGER DEFAULT 0,
                n_replaced INTEGER DEFAULT 0,
                n_spawned INTEGER DEFAULT 0,
                skipped INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            -- Regime detector transition log (online, continuous).
            CREATE TABLE IF NOT EXISTS regime_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_regime TEXT,
                to_regime TEXT NOT NULL,
                confidence REAL,
                features TEXT,                   -- JSON feature snapshot
                perf_snapshot TEXT,              -- JSON per-regime perf at change
                created_at TEXT DEFAULT (datetime('now'))
            );

            -- Risk engine decision log (pause / size-reduce / kill / block).
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'info',
                bot_name TEXT,
                reason TEXT,
                detail TEXT,                     -- JSON
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_name TEXT NOT NULL,
                date TEXT NOT NULL,
                trades_count INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                mode TEXT NOT NULL,
                UNIQUE(bot_name, date, mode)
            );

            CREATE TABLE IF NOT EXISTS bot_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_name TEXT NOT NULL,
                feature_key TEXT NOT NULL,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(bot_name, feature_key)
            );

            CREATE TABLE IF NOT EXISTS arena_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS copytrading_wallets (
                address TEXT PRIMARY KEY,
                label TEXT,
                tracked_since TEXT DEFAULT (datetime('now')),
                total_trades INTEGER DEFAULT 0,
                win_rate REAL,
                total_pnl REAL DEFAULT 0,
                active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS copytrading_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                market_id TEXT,
                side TEXT,
                amount REAL,
                our_trade_id TEXT,
                outcome TEXT,
                pnl REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS lane_validation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                markets INTEGER NOT NULL,
                samples INTEGER NOT NULL,
                results TEXT NOT NULL,           -- JSON: lane -> metrics
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                markets INTEGER NOT NULL,
                trades INTEGER NOT NULL,
                summary TEXT NOT NULL,           -- JSON: metrics.summarize()
                report_path TEXT,                -- full JSON report on disk
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS lane_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lane TEXT NOT NULL,              -- 'fut' | 'tech' | 'xasset'
                status TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|denied
                run_id INTEGER,                  -- lane_validation_runs.id
                metrics TEXT NOT NULL,           -- JSON: evidence behind it
                proposal TEXT NOT NULL,          -- JSON: {profile: {strategy: w}}
                created_at TEXT DEFAULT (datetime('now')),
                decided_at TEXT
            );
        """)

        # Migrations
        for migration in [
            "ALTER TABLE bot_configs ADD COLUMN trading_mode TEXT DEFAULT 'paper'",
            "ALTER TABLE copytrading_trades ADD COLUMN source_tx_hash TEXT",
            "ALTER TABLE copytrading_wallets ADD COLUMN trading_mode TEXT DEFAULT 'paper'",
            # How a trade was filled: 'local_sim' (priced locally, unlimited —
            # the primary paper path now that Simmer caps at 50 buys/day),
            # 'simmer' (confirmed on Simmer with a real trade_id), or
            # 'polymarket' (live CLOB fill). NULL on legacy rows.
            "ALTER TABLE trades ADD COLUMN fill_source TEXT",
            # Price per share at fill time (avg fill price after walking the
            # order book for depth/slippage).
            "ALTER TABLE trades ADD COLUMN entry_price REAL",
            # Polymarket taker fee (USDC) charged on the fill — applied to both
            # simulated (paper) and live trades; factored into resolved P&L.
            "ALTER TABLE trades ADD COLUMN fee REAL DEFAULT 0",
        ]:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Data migration (idempotent): the meanrev slate bot dropped its
        # stop-loss long ago (spec R3) and is now plain mean_reversion —
        # rename the historical rows so slate continuity ("Continue" at
        # startup) and per-bot stats carry over under the new name.
        conn.execute(
            "UPDATE bot_configs SET bot_name='meanrev-v1', "
            "strategy_type='mean_reversion' WHERE bot_name='meanrev-sl25-v1'")
        conn.execute(
            "UPDATE trades SET bot_name='meanrev-v1' "
            "WHERE bot_name='meanrev-sl25-v1'")


@contextmanager
def get_conn():
    # timeout covers brief multi-writer contention (arena + dashboard share
    # one SQLite file under docker-compose / dual launchd services).
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL lets the dashboard read while the arena writes without "database is
    # locked" storms; safe on a single host / shared volume.
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
    except sqlite3.Error:
        pass
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_trade(bot_name, market_id, side, amount, venue, mode, confidence=None,
              reasoning=None, market_question=None, trade_id=None, shares_bought=None,
              trade_features=None, fill_source=None, entry_price=None, fee=0.0):
    """Insert a filled trade and return its internal row id.

    ``amount`` is the USDC cost actually spent on shares (after order-book
    walk); ``fee`` is the Polymarket taker fee; ``entry_price`` is the avg fill
    price; ``fill_source`` records HOW it filled ('paper_sim' | 'polymarket').
    """
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO trades (bot_name, market_id, market_question, side, amount,
               confidence, reasoning, trade_features, venue, mode, trade_id,
               shares_bought, fill_source, entry_price, fee)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (bot_name, market_id, market_question, side, amount,
             confidence, reasoning,
             json.dumps(trade_features) if trade_features else None,
             venue, mode, trade_id, shares_bought, fill_source, entry_price, fee)
        )
        return cur.lastrowid


def resolve_trade(internal_id, outcome, pnl):
    with get_conn() as conn:
        conn.execute(
            "UPDATE trades SET outcome=?, pnl=?, resolved_at=datetime('now') WHERE id=?",
            (outcome, pnl, internal_id)
        )


def get_bot_trades(bot_name, hours=None, limit=50):
    with get_conn() as conn:
        if hours:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            rows = conn.execute(
                "SELECT * FROM trades WHERE bot_name=? AND created_at>=? ORDER BY created_at DESC LIMIT ?",
                (bot_name, cutoff, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE bot_name=? ORDER BY created_at DESC LIMIT ?",
                (bot_name, limit)
            ).fetchall()
        return [dict(r) for r in rows]


def get_bot_performance(bot_name, hours=12, mode=None):
    """Get bot performance stats. hours=None means all-time. mode filters by trading mode."""
    with get_conn() as conn:
        conditions = ["bot_name=?", "outcome IN ('win', 'loss', 'exit_tp', 'exit_sl')"]
        params = [bot_name]
        if hours is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            conditions.append("created_at>=?")
            params.append(cutoff)
        if mode is not None:
            conditions.append("mode=?")
            params.append(mode)
        where = " AND ".join(conditions)
        row = conn.execute(f"""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome IN ('win', 'exit_tp') THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome IN ('loss', 'exit_sl') THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(AVG(pnl), 0) as avg_pnl,
                AVG(entry_price) as avg_entry
            FROM trades WHERE {where}
        """, params).fetchone()
        result = dict(row)
        result["wins"] = result["wins"] or 0
        result["losses"] = result["losses"] or 0
        total = result["wins"] + result["losses"]
        result["win_rate"] = result["wins"] / total if total > 0 else 0
        # Break-even gap: WR must beat the avg entry price to profit (the
        # core PBA profitability lens). None without resolved entry data.
        avg_entry = result.get("avg_entry")
        result["breakeven_gap"] = (
            result["win_rate"] - avg_entry
            if total > 0 and avg_entry is not None else None
        )
        return result


def get_entry_price_buckets(mode=None, hours=None):
    """ROI by entry-price bucket for resolved trades — the core profitability lens.

    A high win rate bought at high prices still loses money: WR must exceed the
    average entry price by ≥5¢ to break even, ≥10¢ to profit (0xSurferX/0x_Punisher).
    This groups resolved trades into entry-price buckets and, per bucket, returns
    count / wins / win_rate / avg_entry / pnl / roi plus ``breakeven_gap`` =
    win_rate − avg_entry (the cents of edge over the break-even line: <0 losing,
    ≥0.05 healthy). ``roi`` is total pnl / total staked.

    Buckets (YES-price cents): 0-20, 20-40, 40-55, 55-65, 65-70, 70-75, 75-85,
    85-95, 95+. Only rows with a non-null ``entry_price`` and a win/loss outcome
    are counted.
    """
    edges = [0.0, 0.20, 0.40, 0.55, 0.65, 0.70, 0.75, 0.85, 0.95, 1.0001]
    labels = ["0-20", "20-40", "40-55", "55-65", "65-70",
              "70-75", "75-85", "85-95", "95+"]
    with get_conn() as conn:
        conditions = ["entry_price IS NOT NULL",
                      "outcome IN ('win', 'loss', 'exit_tp', 'exit_sl')"]
        params = []
        if mode is not None:
            conditions.append("mode=?")
            params.append(mode)
        if hours is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            conditions.append("created_at>=?")
            params.append(cutoff)
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT entry_price, amount, pnl, outcome FROM trades WHERE {where}",
            params,
        ).fetchall()

    buckets = [{"bucket": lbl, "count": 0, "wins": 0, "staked": 0.0,
                "pnl": 0.0, "entry_sum": 0.0} for lbl in labels]
    for r in rows:
        price = r["entry_price"]
        if price is None:
            continue
        idx = next((i for i in range(len(labels)) if edges[i] <= price < edges[i + 1]), None)
        if idx is None:
            continue
        b = buckets[idx]
        b["count"] += 1
        b["wins"] += 1 if r["outcome"] in ("win", "exit_tp") else 0
        b["staked"] += r["amount"] or 0.0
        b["pnl"] += r["pnl"] or 0.0
        b["entry_sum"] += price

    out = []
    for b in buckets:
        n = b["count"]
        if n == 0:
            continue
        wr = b["wins"] / n
        avg_entry = b["entry_sum"] / n
        out.append({
            "bucket": b["bucket"],
            "count": n,
            "wins": b["wins"],
            "win_rate": round(wr, 4),
            "avg_entry": round(avg_entry, 4),
            "pnl": round(b["pnl"], 2),
            "roi": round(b["pnl"] / b["staked"], 4) if b["staked"] else 0.0,
            "breakeven_gap": round(wr - avg_entry, 4),
        })
    return out


def get_all_bots_performance(hours=12):
    with get_conn() as conn:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        rows = conn.execute("""
            SELECT
                bot_name,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome IN ('win', 'exit_tp') THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome IN ('loss', 'exit_sl') THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl
            FROM trades
            WHERE created_at>=? AND outcome IN ('win', 'loss', 'exit_tp', 'exit_sl')
            GROUP BY bot_name
        """, (cutoff,)).fetchall()
        results = {}
        for r in rows:
            d = dict(r)
            d["wins"] = d["wins"] or 0
            d["losses"] = d["losses"] or 0
            total = d["wins"] + d["losses"]
            d["win_rate"] = d["wins"] / total if total > 0 else 0
            results[d["bot_name"]] = d
        return results


def save_bot_config(bot_name, strategy_type, generation, params, lineage=None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO bot_configs (bot_name, strategy_type, generation, lineage, params)
               VALUES (?, ?, ?, ?, ?)""",
            (bot_name, strategy_type, generation, lineage, json.dumps(params))
        )


def retire_bot(bot_name):
    with get_conn() as conn:
        conn.execute(
            "UPDATE bot_configs SET active=0, retired_at=datetime('now') WHERE bot_name=? AND active=1",
            (bot_name,)
        )
    with _bot_mode_lock:
        _bot_mode_cache.pop(bot_name, None)


def wipe_all():
    """Delete every row from all arena tables (schema is preserved).

    Backs the 'start fresh' startup option — the caller wipes the DB so a new
    bot slate isn't polluted by a previous run's trades, learning, evolution
    history or bankroll. Uses per-table DELETE (not file unlink) so it is safe
    while another process (the dashboard) holds an open connection.
    """
    with get_conn() as conn:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for t in tables:
            conn.execute(f"DELETE FROM {t}")
    with _bot_mode_lock:
        _bot_mode_cache.clear()
    return len(tables)


def add_copy_wallet(address: str, label: str | None = None, mode: str = "paper"):
    """Add or reactivate a wallet to copy-trade. mode: 'paper' or 'live'."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO copytrading_wallets (address, label, trading_mode, active)
               VALUES (?, ?, ?, 1)
               ON CONFLICT(address) DO UPDATE SET
                 label=excluded.label,
                 trading_mode=excluded.trading_mode,
                 active=1""",
            (address.lower(), label or address[:16], mode),
        )


def remove_copy_wallet(address: str):
    """Stop copying a wallet (soft delete)."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE copytrading_wallets SET active=0 WHERE address=?",
            (address.lower(),),
        )


def list_copy_wallets():
    """Return all active copy-trade wallets."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT address, label, trading_mode, tracked_since FROM copytrading_wallets WHERE active=1"
        ).fetchall()
        return [dict(r) for r in rows]


def get_active_bots():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bot_configs WHERE active=1 ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]


def log_evolution(cycle_number, survivors, replaced, new_bots, rankings):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO evolution_events (cycle_number, survivors, replaced, new_bots, rankings)
               VALUES (?, ?, ?, ?, ?)""",
            (cycle_number, json.dumps(survivors), json.dumps(replaced),
             json.dumps(new_bots), json.dumps(rankings))
        )


def get_evolution_history(limit=20):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM evolution_events ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def log_ga_generation(cycle_number: int, report: dict) -> None:
    """Persist a full GA cycle report for dashboard / offline analysis."""
    individuals = report.get("individuals") or []
    best = max((i.get("fitness", 0.0) for i in individuals), default=0.0)
    mean = (
        sum(i.get("fitness", 0.0) for i in individuals) / len(individuals)
        if individuals else 0.0
    )
    # Strip non-JSON-safe live bot refs if any leaked in
    safe_report = {
        k: v for k, v in report.items()
        if k not in ("bots",)
    }
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO ga_generations
               (cycle_number, report, best_fitness, mean_fitness,
                n_elites, n_replaced, n_spawned, skipped)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cycle_number,
                json.dumps(safe_report, default=str),
                best,
                mean,
                len(report.get("elites") or []),
                len(report.get("replaced") or []),
                len(report.get("spawned") or []),
                1 if report.get("skipped") else 0,
            ),
        )


def get_ga_history(limit: int = 20) -> list:
    """Recent GA generation rows with parsed report JSON."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM ga_generations
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            if isinstance(d.get("report"), str):
                try:
                    d["report"] = json.loads(d["report"])
                except Exception:
                    pass
            out.append(d)
        return out


def get_ga_status() -> dict:
    """Compact GA status for the dashboard (last cycle + fitness curve)."""
    last_raw = get_arena_state("ga_last_cycle")
    hist_raw = get_arena_state("ga_fitness_history")
    last = None
    hist = []
    if last_raw:
        try:
            last = json.loads(last_raw)
        except Exception:
            last = None
    if hist_raw:
        try:
            hist = json.loads(hist_raw)
        except Exception:
            hist = []
    return {"last_cycle": last, "fitness_history": hist or []}


def log_regime_event(from_regime, to_regime, confidence, features, perf_snapshot=None):
    """Persist a regime transition for dashboard / offline analysis."""
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO regime_events
               (from_regime, to_regime, confidence, features, perf_snapshot)
               VALUES (?, ?, ?, ?, ?)""",
            (
                from_regime,
                to_regime,
                float(confidence) if confidence is not None else None,
                json.dumps(features or {}, default=str),
                json.dumps(perf_snapshot or {}, default=str),
            ),
        )


def get_regime_events(limit: int = 30) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM regime_events
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("features", "perf_snapshot"):
                if isinstance(d.get(k), str):
                    try:
                        d[k] = json.loads(d[k])
                    except Exception:
                        pass
            out.append(d)
        return out


def log_risk_event(action: str, level: str = "info", reason: str = "",
                   bot_name: str = None, detail: dict = None) -> int:
    """Append one risk-engine decision for dashboard / audit trail."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO risk_events (action, level, bot_name, reason, detail)
               VALUES (?, ?, ?, ?, ?)""",
            (
                str(action),
                str(level or "info"),
                bot_name,
                (reason or "")[:1000],
                json.dumps(detail or {}, default=str),
            ),
        )
        # Cap table growth (keep newest N)
        max_keep = int(getattr(config, "RISK_EVENT_LOG_MAX", 500))
        conn.execute(
            """DELETE FROM risk_events WHERE id NOT IN (
                   SELECT id FROM risk_events ORDER BY id DESC LIMIT ?
               )""",
            (max_keep,),
        )
        return cur.lastrowid


def get_risk_events(limit: int = 40, bot_name: str = None) -> list:
    """Most recent risk decisions (newest first)."""
    with get_conn() as conn:
        if bot_name:
            rows = conn.execute(
                """SELECT * FROM risk_events WHERE bot_name=?
                   ORDER BY id DESC LIMIT ?""",
                (bot_name, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM risk_events ORDER BY id DESC LIMIT ?""",
                (int(limit),),
            ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("detail"), str):
            try:
                d["detail"] = json.loads(d["detail"])
            except Exception:
                pass
        out.append(d)
    return out


def get_total_daily_loss(mode="paper"):
    with get_conn() as conn:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = conn.execute("""
            SELECT COALESCE(SUM(pnl), 0) as total_loss
            FROM trades
            WHERE mode=? AND date(created_at)=? AND pnl < 0 AND outcome IS NOT NULL
        """, (mode, today)).fetchone()
        return abs(dict(row)["total_loss"])


def get_bot_daily_loss(bot_name, mode="paper"):
    with get_conn() as conn:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = conn.execute("""
            SELECT COALESCE(SUM(pnl), 0) as total_loss
            FROM trades
            WHERE bot_name=? AND mode=? AND date(created_at)=? AND pnl < 0 AND outcome IS NOT NULL
        """, (bot_name, mode, today)).fetchone()
        return abs(dict(row)["total_loss"])


def get_dashboard_stats():
    with get_conn() as conn:
        # ET-anchored day boundaries: "Today" rolls over at 00:00 ET (not
        # 00:00 UTC), and "This Week" spans the last 7 ET days. created_at is
        # stored UTC, so we compare it against UTC strings that represent the
        # ET midnight boundaries. "Current Session" spans since the arena last
        # booted (session_start written to arena_state on startup) — which may
        # be shorter or longer than a calendar day; it is omitted when no
        # session has been recorded (e.g. arena never started this DB).
        today_start = et_day_start_utc(0)
        week_start = et_day_start_utc(6)
        session_start = get_arena_state("session_start")

        # `trades` counts only RESOLVED trades (win/loss/expired); `pending`
        # (outcome IS NULL) is reported separately so the dashboard can render
        # e.g. "229 +2". 1h-stale-expired trades (outcome='expired', pnl=0)
        # count as resolved — they are real paper trades Simmer could not
        # settle in time, contributing 0 to P&L and to neither win nor loss.
        def _period(since):
            clause = "WHERE created_at>=?" if since else ""
            params = (since,) if since else ()
            row = conn.execute(f"""
                SELECT
                    SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) as trades,
                    SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) as pending,
                    COALESCE(SUM(pnl), 0) as pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 AND outcome IS NOT NULL THEN 1 ELSE 0 END) as losses
                FROM trades {clause}
            """, params).fetchone()
            d = dict(row)
            for k in ("trades", "pending", "wins", "losses"):
                d[k] = d[k] or 0
            return d

        return {
            "session": _period(session_start) if session_start else None,
            "today": _period(today_start),
            "week": _period(week_start),
            "all_time": _period(None),
        }


def get_arena_state(key, default=None):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM arena_state WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default


def set_arena_state(key, value):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO arena_state (key, value) VALUES (?, ?)
               ON CONFLICT(key) DO UPDATE SET value=?, updated_at=datetime('now')""",
            (key, str(value), str(value))
        )


def get_paper_bankroll():
    """The shared virtual USDC bankroll for paper mode (set in dashboard Settings)."""
    v = get_arena_state("paper_bankroll")
    try:
        return float(v) if v is not None else config.PAPER_BANKROLL_DEFAULT
    except (TypeError, ValueError):
        return config.PAPER_BANKROLL_DEFAULT


def set_paper_bankroll(amount):
    """Set the shared paper bankroll (USDC). Raises ValueError on bad input."""
    amount = float(amount)
    if amount < 0:
        raise ValueError("Bankroll must be non-negative")
    set_arena_state("paper_bankroll", amount)


def get_kelly_fraction():
    """The Kelly fraction used for bet sizing (editable in dashboard Settings).

    Falls back to ``config.KELLY_FRACTION`` until a value is saved.
    """
    v = get_arena_state("kelly_fraction")
    try:
        f = float(v) if v is not None else config.KELLY_FRACTION
    except (TypeError, ValueError):
        return config.KELLY_FRACTION
    return f if 0.0 < f <= 1.0 else config.KELLY_FRACTION


def set_kelly_fraction(fraction):
    """Set the Kelly fraction (0 < f <= 1). Raises ValueError on bad input."""
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("Kelly fraction must be in (0, 1]")
    set_arena_state("kelly_fraction", fraction)


# ---------------------------------------------------------------------------
# Candidate-lane validation runs, proposals + approved overrides
#
# The offline harness (tools/validate_signals.py --candidates --propose)
# records each validation run and, when a kill-switched lane clears the
# promotion thresholds, files a PENDING proposal. The dashboard lists pending
# proposals with approve/deny; APPROVING writes the lane into the
# 'lane_overrides' arena_state JSON, which bots/base_bot.py consults (cached)
# to weight the lane live — no config edit, no restart. Denying just closes
# the proposal (the harness may re-file after a later run with fresh data).
# ---------------------------------------------------------------------------

def record_backtest_run(label: str, markets: int, trades: int,
                        summary: dict, report_path=None) -> int:
    """Store one offline backtest run's summary (backtest/ package).

    Same pattern as lane_validation_runs: run records only, never trade
    tables — the dashboard can list runs via get_backtest_runs.
    """
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO backtest_runs (label, markets, trades, summary, "
            "report_path) VALUES (?, ?, ?, ?, ?)",
            (str(label), int(markets), int(trades), json.dumps(summary),
             report_path))
        return cur.lastrowid


def get_backtest_runs(limit: int = 20) -> list:
    """Most recent backtest runs, summaries parsed."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM backtest_runs ORDER BY id DESC LIMIT ?",
            (int(limit),)).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        try:
            d["summary"] = json.loads(d["summary"])
        except (TypeError, ValueError):
            d["summary"] = {}
        out.append(d)
    return out


def record_lane_validation_run(markets, samples, results: dict) -> int:
    """Store one harness run's per-lane metrics. Returns the run id."""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO lane_validation_runs (markets, samples, results) "
            "VALUES (?, ?, ?)",
            (int(markets), int(samples), json.dumps(results)))
        return cur.lastrowid


def get_latest_lane_run():
    """Most recent validation run (results parsed), or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM lane_validation_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["results"] = json.loads(d["results"])
    except (TypeError, ValueError):
        d["results"] = {}
    return d


def create_lane_proposal(lane, metrics: dict, proposal: dict, run_id=None):
    """File a pending proposal for ``lane`` unless one is already open or the
    lane is already approved (override active). Returns the proposal id, or
    None when skipped."""
    overrides = get_lane_overrides()
    if overrides.get(lane, {}).get("enabled"):
        return None
    with get_conn() as conn:
        exists = conn.execute(
            "SELECT id FROM lane_proposals WHERE lane=? AND status='pending'",
            (lane,)).fetchone()
        if exists:
            # Refresh the evidence on the open proposal instead of stacking
            # duplicates — the dashboard should always show the latest run.
            conn.execute(
                "UPDATE lane_proposals SET metrics=?, proposal=?, run_id=? "
                "WHERE id=?",
                (json.dumps(metrics), json.dumps(proposal), run_id,
                 exists["id"]))
            return exists["id"]
        cur = conn.execute(
            "INSERT INTO lane_proposals (lane, status, run_id, metrics, proposal) "
            "VALUES (?, 'pending', ?, ?, ?)",
            (lane, run_id, json.dumps(metrics), json.dumps(proposal)))
        return cur.lastrowid


def get_lane_proposals(status=None):
    """Proposals (newest first), optionally filtered by status; JSON parsed."""
    with get_conn() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM lane_proposals WHERE status=? ORDER BY id DESC",
                (status,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM lane_proposals ORDER BY id DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("metrics", "proposal"):
            try:
                d[k] = json.loads(d[k])
            except (TypeError, ValueError):
                d[k] = {}
        out.append(d)
    return out


def decide_lane_proposal(proposal_id, action):
    """Approve or deny a pending proposal.

    Approve → mark approved AND activate the lane override (arena_state
    'lane_overrides'); the arena picks it up within the hot-path cache TTL.
    Deny → just close it. Raises ValueError on bad input / already decided.
    """
    if action not in ("approve", "deny"):
        raise ValueError("action must be 'approve' or 'deny'")
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM lane_proposals WHERE id=?", (proposal_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"proposal {proposal_id} not found")
        if row["status"] != "pending":
            raise ValueError(f"proposal {proposal_id} already {row['status']}")
        status = "approved" if action == "approve" else "denied"
        conn.execute(
            "UPDATE lane_proposals SET status=?, decided_at=datetime('now') "
            "WHERE id=?", (status, proposal_id))
    if action == "approve":
        try:
            prop = json.loads(row["proposal"])
        except (TypeError, ValueError):
            prop = {}
        overrides = get_lane_overrides()
        overrides[row["lane"]] = {
            "enabled": True,
            "profile": prop.get("profile", {}),
            "approved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        set_arena_state("lane_overrides", json.dumps(overrides))
    return status


def get_lane_overrides() -> dict:
    """Approved lane overrides: lane -> {enabled, profile: {strategy: w}}."""
    raw = get_arena_state("lane_overrides")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def disable_lane_override(lane):
    """Kill an approved lane override (dashboard 'disable' safety hatch)."""
    overrides = get_lane_overrides()
    if lane in overrides:
        overrides[lane]["enabled"] = False
        set_arena_state("lane_overrides", json.dumps(overrides))
        return True
    return False


def get_auto_approve_lanes() -> bool:
    """Whether the promoter may auto-approve lane proposals (dashboard toggle).

    Stored in arena_state; falls back to config.AUTO_APPROVE_LANES_ENABLED as
    the boot default when the operator has never touched the switch.
    """
    raw = get_arena_state("auto_approve_lanes")
    if raw is None:
        return bool(getattr(config, "AUTO_APPROVE_LANES_ENABLED", True))
    return str(raw) == "1"


def set_auto_approve_lanes(enabled: bool):
    """Flip the auto-approve toggle (dashboard Signal Lab)."""
    set_arena_state("auto_approve_lanes", "1" if enabled else "0")


def annotate_lane_proposal(proposal_id, live: dict):
    """Attach live-attribution evidence to a proposal's metrics under 'live'.

    Lets the dashboard show live shadow numbers next to the harness metrics
    whether or not auto-approve is on. No-op if the proposal is gone.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT metrics FROM lane_proposals WHERE id=?", (proposal_id,)
        ).fetchone()
        if not row:
            return
        try:
            metrics = json.loads(row["metrics"])
        except (TypeError, ValueError):
            metrics = {}
        metrics["live"] = live
        conn.execute("UPDATE lane_proposals SET metrics=? WHERE id=?",
                     (json.dumps(metrics), proposal_id))


def _paper_pnl_and_reserved():
    """(realized paper P&L on resolved trades, reserved cost of open trades)."""
    with get_conn() as conn:
        realized = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades "
            "WHERE mode='paper' AND outcome IN ('win', 'loss')"
        ).fetchone()[0]
        open_cost = conn.execute(
            "SELECT COALESCE(SUM(amount + COALESCE(fee, 0)), 0) FROM trades "
            "WHERE mode='paper' AND outcome IS NULL"
        ).fetchone()[0]
    return (realized or 0.0), (open_cost or 0.0)


def get_open_exposure(market_id, side, mode="paper"):
    """Total OPEN (unresolved) cost committed to one (market, side) across all
    bots — the shared-pool concentration the per-bot Kelly sizing can't see
    (BUG #27). Same open-cost definition as ``_paper_pnl_and_reserved``."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount + COALESCE(fee, 0)), 0) FROM trades "
            "WHERE market_id=? AND side=? AND mode=? AND outcome IS NULL",
            (market_id, side, mode)
        ).fetchone()
    return row[0] or 0.0


def get_paper_pool_gross():
    """Gross paper pool (bankroll + realized P&L, BEFORE open-cost
    deductions) — the stable base for concentration caps; ``available``
    shrinks as trades open, which would make the cap self-tightening."""
    realized, _ = _paper_pnl_and_reserved()
    return get_paper_bankroll() + realized


def get_paper_available():
    """Available shared paper cash right now.

    cash = bankroll + realized paper P&L (resolved) - open paper cost (pending).
    All paper bots draw from this one pool.
    """
    realized, open_cost = _paper_pnl_and_reserved()
    return get_paper_bankroll() + realized - open_cost


def topup_paper_bankroll(target_available):
    """Top the shared paper pool up so *available cash* equals ``target_available``.

    The user enters the balance they want to see. Because
    ``available = bankroll + realized_pnl - reserved_open``, we solve for the
    underlying bankroll that yields the requested available cash *without*
    discarding trade history or un-reserving live positions::

        bankroll = target_available - realized_pnl + reserved_open

    So entering $200 when the pool is at $45 (after losses + open bets) sets
    available to exactly $200, and future resolutions still move it correctly.
    Returns the new available cash.
    """
    target = float(target_available)
    if target < 0:
        raise ValueError("Balance must be non-negative")
    realized, open_cost = _paper_pnl_and_reserved()
    set_arena_state("paper_bankroll", target - realized + open_cost)
    return get_paper_available()


def get_bot_mode(bot_name):
    """Get per-bot trading mode ('paper' or 'live'), cached briefly."""
    now = time.time()
    ttl = getattr(config, "BOT_MODE_CACHE_TTL_SEC", 3)
    with _bot_mode_lock:
        hit = _bot_mode_cache.get(bot_name)
        if hit and (now - hit[0]) < ttl:
            return hit[1]
    with get_conn() as conn:
        row = conn.execute(
            "SELECT trading_mode FROM bot_configs WHERE bot_name=? AND active=1",
            (bot_name,)
        ).fetchone()
    mode = (row["trading_mode"] or "paper") if row else "paper"
    with _bot_mode_lock:
        _bot_mode_cache[bot_name] = (now, mode)
    return mode


def set_bot_mode(bot_name, mode):
    """Set per-bot trading mode ('paper' or 'live')."""
    if mode not in ("paper", "live"):
        raise ValueError("Mode must be 'paper' or 'live'")
    with get_conn() as conn:
        conn.execute(
            "UPDATE bot_configs SET trading_mode=? WHERE bot_name=? AND active=1",
            (mode, bot_name)
        )
    with _bot_mode_lock:  # invalidate so the new mode is read immediately
        _bot_mode_cache.pop(bot_name, None)


init_db()

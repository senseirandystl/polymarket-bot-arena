# Regime Discovery, Context Attribution & Regime-Conditioned Control — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the arena a rich per-window market *context*, learn per-bot performance across it, discover named regimes from data, and let the existing portfolio allocator + core-lane tuner condition on the current regime.

**Architecture:** Three layers. (1) A pure `build_context()` composes existing signals into a context vector stamped on every trade. (2) An evolution-loop job (`arena/regime_map.py`) computes empirical-Bayes-shrunk per-bot edge over context, clusters + OOS-validates named regimes, and persists a `regime_map`. (3) The portfolio allocator (capital) and core-lane tuner (signal weights) read the map and apply bounded, floored, hysteretic tilts when a regime is validated and the toggle is on. One owner per knob is preserved.

**Tech Stack:** Python 3.14, SQLite (WAL), pytest. No new third-party dependencies (shrinkage/clustering are hand-rolled, matching repo style).

## Global Constraints

- Run Python only via `.venv/bin/python3`; tests via `.venv/bin/python3 -m pytest`.
- Immutability: return new dicts/lists; never mutate caller-owned structures in place.
- **Live resolved trades only** for attribution (`outcome IN ('win','loss','exit_tp','exit_sl')`).
- Every periodic job is **best-effort** and must never raise into the trading hot path (wrap the loop body; log and continue — match `arena/lane_monitor.py`).
- Layer 3 acts only when the current regime is `validated` **and** the toggle is on; otherwise it computes and persists *suggestions* only.
- Toggle default is **ON**: `config.REGIME_CONDITIONING_ENABLED = True` (paper mode, no real capital); dashboard-editable, stored in `arena_state`, read via `db.get_regime_conditioning()`.
- New files stay small and single-responsibility. Follow existing patterns (`get/set_auto_approve_lanes`, `core_lane_tuner.tune()`, `lane_monitor.check_lanes()`).
- Commit after every task (green tests).

---

### Task 1: Context vector (`signals/context.py`)

Pure composer of existing signals into one context dict — no network reads, no module state.

**Files:**
- Create: `signals/context.py`
- Test: `tests/unit/test_context.py`

**Interfaces:**
- Consumes: `signals/multiscale.py::compute(prices)`, `signals/volatility_regime.py::compute(prices)`, `arena/session_filter.py::_to_et(now_utc)`, `signals/macro_calendar.py::macro_caution(now)`.
- Produces:
  - `build_context(prices: list[float], signals: dict | None, now_utc: datetime) -> dict` with keys:
    continuous `vol, trend, flow, realized_vol, btc_mom_1m, btc_mom_5m, btc_mom_15m, btc_trend_slope`;
    categorical `weekday (0-6), hour_block (0-7), session ('asia'|'eu'|'us'|'overnight'), macro_prox (0|1|2)`;
    derived `vol_trend_regime (str)`.
  - `context_cell(ctx: dict) -> tuple` — the discretized key used for attribution grouping:
    `(vol_trend_regime, weekday, hour_block, session, macro_prox, btc_trend_bucket)` where
    `btc_trend_bucket` is `sign(btc_trend_slope)` bucketed to `-1|0|1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_context.py
from datetime import datetime, timezone
from signals.context import build_context, context_cell


def _rising_prices(n=40):
    return [100.0 + i * 0.05 for i in range(n)]


def test_build_context_has_all_keys():
    now = datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc)  # Wed
    ctx = build_context(_rising_prices(), signals=None, now_utc=now)
    for k in ("vol", "trend", "flow", "realized_vol",
              "btc_mom_1m", "btc_mom_5m", "btc_mom_15m", "btc_trend_slope",
              "weekday", "hour_block", "session", "macro_prox", "vol_trend_regime"):
        assert k in ctx, f"missing {k}"
    assert 0.0 <= ctx["vol"] <= 1.0
    assert ctx["weekday"] in range(7)
    assert ctx["hour_block"] in range(8)
    assert ctx["session"] in ("asia", "eu", "us", "overnight")
    assert ctx["macro_prox"] in (0, 1, 2)


def test_build_context_is_pure_and_deterministic():
    now = datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc)
    a = build_context(_rising_prices(), signals=None, now_utc=now)
    b = build_context(_rising_prices(), signals=None, now_utc=now)
    assert a == b


def test_build_context_empty_prices_safe():
    now = datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc)
    ctx = build_context([], signals=None, now_utc=now)
    assert ctx["vol_trend_regime"]  # still a string, no crash


def test_context_cell_is_hashable_tuple():
    now = datetime(2026, 7, 22, 14, 30, tzinfo=timezone.utc)
    ctx = build_context(_rising_prices(), signals=None, now_utc=now)
    cell = context_cell(ctx)
    assert isinstance(cell, tuple)
    hash(cell)  # must be hashable for dict grouping
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/unit/test_context.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'signals.context'`

- [ ] **Step 3: Write minimal implementation**

```python
# signals/context.py
"""Composes existing per-window signals into one structured *context* vector.

Pure function — no network reads (all inputs are already computed on the warm
path) and no module state, so it is safe on the 1s warm path, the offline
harness, and in tests. This is Layer 1 of the regime-discovery design
(docs/superpowers/specs/2026-07-24-regime-discovery-context-attribution-design.md):
the vector is stamped on every trade at decision time; Layer 2 attributes
per-bot performance to `context_cell(...)` groupings.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Sequence

_SESSIONS = ("asia", "eu", "us", "overnight")


def _session_for_hour_et(hour_et: int) -> str:
    # Coarse crypto session buckets in ET.
    if 3 <= hour_et < 9:
        return "eu"
    if 9 <= hour_et < 16:
        return "us"
    if 16 <= hour_et < 21:
        return "overnight"
    return "asia"


def _btc_trend_slope(prices: Sequence[float]) -> float:
    """Signed, bounded macro-trend slope from first vs last of a long window."""
    clean = [p for p in (prices or []) if p and p > 0]
    if len(clean) < 10:
        return 0.0
    import math
    span = clean[-1] - clean[0]
    base = clean[0] or 1.0
    return math.tanh((span / base) / 0.01)  # 1% move over the window ~ 0.76


def build_context(
    prices: Sequence[float],
    signals: Optional[dict] = None,
    now_utc: Optional[datetime] = None,
) -> dict[str, Any]:
    """Return the structured context vector for the current window."""
    from signals import multiscale, volatility_regime
    from arena.session_filter import _to_et
    from signals.macro_calendar import macro_caution

    now_utc = now_utc or datetime.now(tz=timezone.utc)
    clean = [p for p in (prices or []) if p and p > 0]

    vr = volatility_regime.compute(clean)
    ms = multiscale.compute(clean)

    sv = signals or {}
    flow = 0.0
    try:
        flow = 0.5 * (abs(float(sv.get("cvd", 0.0))) + abs(float(sv.get("obi", 0.0))))
    except (TypeError, ValueError):
        flow = 0.0

    et = _to_et(now_utc)
    hour_et = et.hour
    caution = macro_caution(now_utc)
    macro_prox = 2 if caution >= 0.75 else (1 if caution >= 0.25 else 0)

    return {
        # continuous
        "vol": max(0.0, min(1.0, float(vr.get("vol_score") or 0.0))),
        "trend": max(0.0, min(1.0, float(vr.get("trend_score") or 0.0))),
        "flow": max(0.0, min(1.0, flow)),
        "realized_vol": float(vr.get("realized_vol") or 0.0),
        "btc_mom_1m": float(ms.get("ms_mom_1m") or 0.0),
        "btc_mom_5m": float(ms.get("ms_mom_5m") or 0.0),
        "btc_mom_15m": float(ms.get("ms_mom_15m") or 0.0),
        "btc_trend_slope": _btc_trend_slope(clean),
        # categorical
        "weekday": int(et.weekday()),
        "hour_block": int(hour_et // 3),
        "session": _session_for_hour_et(hour_et),
        "macro_prox": int(macro_prox),
        # derived
        "vol_trend_regime": str(vr.get("regime") or vr.get("regime_id") or "unknown"),
    }


def context_cell(ctx: dict) -> tuple:
    """Discretized grouping key for attribution (hashable)."""
    slope = float(ctx.get("btc_trend_slope") or 0.0)
    trend_bucket = 1 if slope > 0.2 else (-1 if slope < -0.2 else 0)
    return (
        str(ctx.get("vol_trend_regime") or "unknown"),
        int(ctx.get("weekday") or 0),
        int(ctx.get("hour_block") or 0),
        str(ctx.get("session") or "asia"),
        int(ctx.get("macro_prox") or 0),
        trend_bucket,
    )
```

> Note: confirm the exact key names returned by `multiscale.compute` (`ms_mom_1m` etc. per its docstring) and `volatility_regime.compute` (`vol_score`, `trend_score`, `realized_vol`, `regime`) while implementing; adjust the `.get(...)` keys to match. The `.get` fallbacks keep it safe if a key is absent.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/unit/test_context.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add signals/context.py tests/unit/test_context.py
git commit -m "feat: context vector composer (regime discovery layer 1)"
```

---

### Task 2: Persist context on every trade (behavior-neutral)

Add a `context` JSON column, thread it through the venue engines to `db.log_trade`, stamp it in `make_decision`, and add a resolved-trades-with-context query.

**Files:**
- Modify: `db.py` (migration list ~line 255; `log_trade` ~line 294-314; add `get_resolved_trades_with_context`)
- Modify: `venues/paper.py` (`place` signature ~line 33; `log_trade` call ~line 118-130)
- Modify: `venues/live.py` (`place` signature ~line 29; `log_trade` call ~line 114-126)
- Modify: `bots/base_bot.py` (stamp context near the `regime:` stamp ~line 500; add `context` to the buy signal dict ~line 799-813; pass `context=signal.get("context")` in `execute` ~line 936-947)
- Test: `tests/unit/test_context_persistence.py`

**Interfaces:**
- Consumes: `signals.context.build_context`, `context_cell` (Task 1).
- Produces:
  - `db.log_trade(..., context: dict | None = None)` — stores JSON in the new column.
  - `db.get_resolved_trades_with_context(hours: float | None = None) -> list[dict]` — each row includes parsed `context` (dict) and `cell` (tuple) plus `bot_name, pnl, outcome, entry_price`.
  - venue `place(..., context: dict | None = None)` forwarding to `log_trade`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_context_persistence.py
import importlib
import db


def test_log_trade_stores_and_reads_context(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DB_PATH", str(tmp_path / "t.db"))
    importlib.reload(db)
    db.init_db()
    ctx = {"vol": 0.2, "trend": 0.1, "weekday": 2, "hour_block": 3,
           "session": "us", "macro_prox": 0, "vol_trend_regime": "low_vol_range",
           "btc_trend_slope": 0.0}
    rid = db.log_trade("momentum", "mkt1", "YES", 5.0, "paper", "paper",
                       context=ctx)
    db.resolve_trade(rid, "win", 1.5)
    rows = db.get_resolved_trades_with_context()
    assert len(rows) == 1
    assert rows[0]["context"]["session"] == "us"
    assert isinstance(rows[0]["cell"], tuple)
    assert rows[0]["pnl"] == 1.5


def test_context_column_migration_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DB_PATH", str(tmp_path / "t2.db"))
    importlib.reload(db)
    db.init_db()
    db.init_db()  # second call must not raise
    rid = db.log_trade("m", "mkt", "NO", 1.0, "paper", "paper", context=None)
    assert rid > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/unit/test_context_persistence.py -v`
Expected: FAIL — `TypeError: log_trade() got an unexpected keyword argument 'context'`

- [ ] **Step 3: Write minimal implementation**

In `db.py` migration list (after the `fee` migration, ~line 255):

```python
            "ALTER TABLE trades ADD COLUMN fee REAL DEFAULT 0",
            # Structured market-context vector stamped at decision time
            # (signals/context.py). JSON; NULL on legacy rows. Layer 1 of the
            # regime-discovery design — attribution groups on context_cell(...).
            "ALTER TABLE trades ADD COLUMN context TEXT",
```

Change `db.log_trade` signature and INSERT to include `context`:

```python
def log_trade(bot_name, market_id, side, amount, venue, mode, confidence=None,
              reasoning=None, market_question=None, trade_id=None, shares_bought=None,
              trade_features=None, fill_source=None, entry_price=None, fee=0.0,
              context=None):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO trades (bot_name, market_id, market_question, side, amount,
               confidence, reasoning, trade_features, venue, mode, trade_id,
               shares_bought, fill_source, entry_price, fee, context)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (bot_name, market_id, market_question, side, amount,
             confidence, reasoning,
             json.dumps(trade_features) if trade_features else None,
             venue, mode, trade_id, shares_bought, fill_source, entry_price, fee,
             json.dumps(context) if context else None)
        )
        return cur.lastrowid
```

Add the query helper (near `get_bot_trades`):

```python
def get_resolved_trades_with_context(hours=None):
    """Resolved trades that carry a stamped context vector (Layer 2 input)."""
    from signals.context import context_cell
    with get_conn() as conn:
        conds = ["outcome IN ('win','loss','exit_tp','exit_sl')", "context IS NOT NULL"]
        params = []
        if hours is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
            conds.append("created_at>=?")
            params.append(cutoff)
        where = " AND ".join(conds)
        rows = conn.execute(
            f"SELECT bot_name, side, pnl, outcome, entry_price, context, created_at "
            f"FROM trades WHERE {where} ORDER BY created_at DESC", params
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["context"] = json.loads(d["context"]) if d["context"] else None
        except (json.JSONDecodeError, TypeError):
            d["context"] = None
        d["cell"] = context_cell(d["context"]) if d["context"] else None
        if d["context"]:
            out.append(d)
    return out
```

In `venues/paper.py` `place` — add `context=None` to the keyword-only signature and forward it:

```python
    def place(self, *, bot_name, side, amount, market, mode,
              confidence=None, reasoning=None, features=None,
              expected_price=None, book=None, context=None):
```
```python
        row_id = db.log_trade(
            ...
            trade_features=features,
            context=context,
        )
```
(Apply the identical two edits to `venues/live.py`.)

In `bots/base_bot.py`, stamp context where `regime:` is stamped (~line 500). Immediately after the `features.append(f"regime_legacy:...")` block:

```python
            from signals.context import build_context
            try:
                ctx_vec = build_context(
                    self.price_history(signals), signals, __import__("datetime").datetime.now(
                        tz=__import__("datetime").timezone.utc))
            except Exception:
                ctx_vec = None
```

> While implementing, use the price series `make_decision` already has in scope (the closed-candle list used to build lanes) instead of a new `price_history` call — match the local variable name actually present. Assign `ctx_vec` there.

Add `context` to the buy signal dict (~line 799-813, the dict that carries `features`, `entry_price`, etc.):

```python
            "features": features,
            "context": ctx_vec,
```

Pass it through in `execute` (~line 936-947):

```python
        res = get_engine(mode).place(
            ...
            features=signal.get("features"),
            expected_price=expected,
            book=book,
            context=signal.get("context"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/unit/test_context_persistence.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Verify behavior-neutrality (decisions unchanged)**

Run: `.venv/bin/python3 -m pytest tests/unit/test_bot_decisions.py -v`
Expected: PASS — stamping context must not alter any decision.

- [ ] **Step 6: Commit**

```bash
git add db.py venues/paper.py venues/live.py bots/base_bot.py tests/unit/test_context_persistence.py
git commit -m "feat: stamp market context on every trade (regime discovery layer 1)"
```

---

### Task 3: Empirical-Bayes shrinkage attribution (`arena/regime_map.py` — part 1)

Pure attribution math: per-(bot, cell) edge shrunk toward parent/global.

**Files:**
- Create: `arena/regime_map.py` (functions only in this task)
- Test: `tests/unit/test_regime_map_shrinkage.py`

**Interfaces:**
- Consumes: `db.get_resolved_trades_with_context` (Task 2); `config.REGIME_SHRINKAGE_K` (Task 5 — read via `getattr` with default `40` so this task stands alone).
- Produces:
  - `shrink(cell_mean: float, cell_n: int, prior_mean: float, k: float) -> float` —
    `(cell_n*cell_mean + k*prior_mean) / (cell_n + k)`.
  - `attribute(trades: list[dict], k: float = 40.0) -> dict` — returns
    `{cell(tuple): {"n": int, "global_pnl": float, "bots": {bot: {"n": int, "pnl": float, "shrunk_pnl": float}}}}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_regime_map_shrinkage.py
from arena.regime_map import shrink, attribute


def test_shrink_thin_cell_pulls_to_prior():
    # 2 samples, k=40 -> estimate dominated by prior
    est = shrink(cell_mean=10.0, cell_n=2, prior_mean=0.0, k=40.0)
    assert abs(est - (2 * 10.0 + 40 * 0.0) / 42) < 1e-9
    assert est < 1.0  # strongly pulled toward prior


def test_shrink_rich_cell_trusts_itself():
    est = shrink(cell_mean=10.0, cell_n=400, prior_mean=0.0, k=40.0)
    assert est > 9.0  # mostly its own mean


def test_attribute_groups_by_cell_and_bot():
    trades = [
        {"bot_name": "a", "pnl": 2.0, "cell": ("r", 2, 3, "us", 0, 0)},
        {"bot_name": "a", "pnl": 4.0, "cell": ("r", 2, 3, "us", 0, 0)},
        {"bot_name": "b", "pnl": -1.0, "cell": ("r", 2, 3, "us", 0, 0)},
    ]
    out = attribute(trades, k=40.0)
    cell = ("r", 2, 3, "us", 0, 0)
    assert out[cell]["n"] == 3
    assert out[cell]["bots"]["a"]["n"] == 2
    assert "shrunk_pnl" in out[cell]["bots"]["a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_map_shrinkage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arena.regime_map'`

- [ ] **Step 3: Write minimal implementation**

```python
# arena/regime_map.py
"""Layer 2 of the regime-discovery design: per-bot performance attribution over
market context, empirical-Bayes shrunk toward coarser priors, plus discovery +
out-of-sample validation of named regimes. Persists arena_state['regime_map'].

Runs on the evolution loop (sibling of core_lane_tuner). Best-effort — never
raises into the trading path. Reads LIVE resolved trades only.
"""
from __future__ import annotations

from typing import Any, Sequence


def shrink(cell_mean: float, cell_n: int, prior_mean: float, k: float) -> float:
    """Empirical-Bayes shrinkage of a cell mean toward a prior mean."""
    denom = (cell_n + k) or 1.0
    return (cell_n * cell_mean + k * prior_mean) / denom


def attribute(trades: Sequence[dict], k: float = 40.0) -> dict[tuple, dict]:
    """Group resolved trades by context cell + bot; shrink each bot's mean PnL.

    Prior for a (cell, bot) is the cell's global mean PnL across all bots.
    """
    by_cell: dict[tuple, dict] = {}
    all_pnl: list[float] = []
    for t in trades:
        cell = t.get("cell")
        if cell is None:
            continue
        pnl = float(t.get("pnl") or 0.0)
        all_pnl.append(pnl)
        c = by_cell.setdefault(cell, {"pnls": [], "bots": {}})
        c["pnls"].append(pnl)
        b = c["bots"].setdefault(t["bot_name"], [])
        b.append(pnl)

    out: dict[tuple, dict] = {}
    for cell, c in by_cell.items():
        cell_pnls = c["pnls"]
        cell_n = len(cell_pnls)
        cell_mean = sum(cell_pnls) / cell_n if cell_n else 0.0
        bots = {}
        for bot, pnls in c["bots"].items():
            n = len(pnls)
            mean = sum(pnls) / n if n else 0.0
            bots[bot] = {
                "n": n,
                "pnl": round(mean, 4),
                "shrunk_pnl": round(shrink(mean, n, cell_mean, k), 4),
            }
        out[cell] = {
            "n": cell_n,
            "global_pnl": round(cell_mean, 4),
            "bots": bots,
        }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_map_shrinkage.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add arena/regime_map.py tests/unit/test_regime_map_shrinkage.py
git commit -m "feat: empirical-Bayes context attribution (regime discovery layer 2a)"
```

---

### Task 4: Discovery, OOS validation & persistence (`arena/regime_map.py` — part 2)

Promote cells to named regimes only when sample-count and out-of-sample stability pass; persist the map.

**Files:**
- Modify: `arena/regime_map.py` (add `validate_cell`, `rebuild`)
- Modify: `db.py` (add `get_regime_map`, `set_regime_map`)
- Test: `tests/unit/test_regime_map_discovery.py`

**Interfaces:**
- Consumes: `attribute` (Task 3); `db.get_resolved_trades_with_context`; `config.REGIME_MIN_SAMPLES` (default 60), `REGIME_SHRINKAGE_K` (40).
- Produces:
  - `validate_cell(train_trades, val_trades) -> bool` — True when the top bot by train shrunk-PnL is non-negative on validation.
  - `rebuild() -> dict` — builds the map, persists via `db.set_regime_map`, returns it.
  - `db.get_regime_map() -> dict`, `db.set_regime_map(payload: dict)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_regime_map_discovery.py
import importlib
import db
from arena import regime_map


def _mk(bot, pnl, cell, ts):
    return {"bot_name": bot, "pnl": pnl, "cell": cell, "created_at": ts}


def test_under_sampled_cell_not_promoted(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DB_PATH", str(tmp_path / "r.db"))
    importlib.reload(db)
    db.init_db()
    cell = ("r", 2, 3, "us", 0, 0)
    trades = [_mk("a", 1.0, cell, "2026-07-20 10:00:00") for _ in range(5)]
    monkeypatch.setattr(db, "get_resolved_trades_with_context", lambda hours=None: trades)
    monkeypatch.setattr(regime_map.config, "REGIME_MIN_SAMPLES", 60, raising=False)
    m = regime_map.rebuild()
    regimes = {r["cell"]: r for r in m["regimes"]}
    assert regimes[list(regimes)[0]]["validated"] is False


def test_well_sampled_consistent_cell_promoted(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DB_PATH", str(tmp_path / "r2.db"))
    importlib.reload(db)
    db.init_db()
    cell = ("r", 2, 3, "us", 0, 0)
    # 'a' consistently wins, 'b' consistently loses, 100 each
    trades = ([_mk("a", 2.0, cell, f"2026-07-20 10:{i:02d}:00") for i in range(60)]
              + [_mk("b", -2.0, cell, f"2026-07-20 11:{i:02d}:00") for i in range(60)])
    monkeypatch.setattr(db, "get_resolved_trades_with_context", lambda hours=None: trades)
    monkeypatch.setattr(regime_map.config, "REGIME_MIN_SAMPLES", 60, raising=False)
    m = regime_map.rebuild()
    reg = m["regimes"][0]
    assert reg["validated"] is True
    assert reg["bot_edges"]["a"]["shrunk_pnl"] > reg["bot_edges"]["b"]["shrunk_pnl"]
    # Persisted
    assert db.get_regime_map()["regimes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_map_discovery.py -v`
Expected: FAIL — `AttributeError: module 'arena.regime_map' has no attribute 'rebuild'`

- [ ] **Step 3: Write minimal implementation**

Add to `arena/regime_map.py`:

```python
import json
import logging

import config
import db

logger = logging.getLogger("arena.regime_map")

STATE_KEY = "regime_map"


def validate_cell(train_trades: Sequence[dict], val_trades: Sequence[dict],
                  k: float) -> bool:
    """OOS check: the train-best bot must not lose on the validation half."""
    if not train_trades or not val_trades:
        return False
    train = attribute(train_trades, k)
    if not train:
        return False
    cell = next(iter(train))
    bots = train[cell]["bots"]
    best = max(bots, key=lambda b: bots[b]["shrunk_pnl"])
    val_pnls = [float(t["pnl"] or 0.0) for t in val_trades if t["bot_name"] == best]
    if not val_pnls:
        return False
    return (sum(val_pnls) / len(val_pnls)) >= 0.0


def rebuild() -> dict:
    """Recompute the regime map from live resolved trades and persist it."""
    k = float(getattr(config, "REGIME_SHRINKAGE_K", 40))
    min_n = int(getattr(config, "REGIME_MIN_SAMPLES", 60))
    trades = db.get_resolved_trades_with_context()
    by_cell = attribute(trades, k)

    # Group raw trades per cell for the OOS split (chronological).
    raw: dict[tuple, list] = {}
    for t in trades:
        if t.get("cell") is not None:
            raw.setdefault(t["cell"], []).append(t)

    regimes = []
    for cell, agg in by_cell.items():
        cell_trades = sorted(raw.get(cell, []), key=lambda t: t.get("created_at") or "")
        validated = False
        if agg["n"] >= min_n:
            mid = len(cell_trades) // 2
            validated = validate_cell(cell_trades[:mid], cell_trades[mid:], k)
        regimes.append({
            "cell": list(cell),          # JSON-safe; matched back via tuple()
            "n": agg["n"],
            "validated": bool(validated),
            "bot_edges": agg["bots"],
        })

    regimes.sort(key=lambda r: r["n"], reverse=True)
    payload = {"regimes": regimes, "updated_at": __import__("time").time()}
    db.set_regime_map(payload)
    return payload


def edges_for_cell(cell: tuple) -> dict | None:
    """Validated per-bot shrunk edges for a cell, or None if not validated."""
    payload = db.get_regime_map()
    for r in payload.get("regimes", []):
        if tuple(r.get("cell") or []) == tuple(cell) and r.get("validated"):
            return r.get("bot_edges") or {}
    return None
```

Add to `db.py` (near the other arena_state helpers):

```python
def get_regime_map() -> dict:
    raw = get_arena_state("regime_map")
    if not raw:
        return {"regimes": []}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"regimes": []}


def set_regime_map(payload: dict):
    set_arena_state("regime_map", json.dumps(payload, default=str))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_map_discovery.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add arena/regime_map.py db.py tests/unit/test_regime_map_discovery.py
git commit -m "feat: regime discovery + OOS validation + persistence (layer 2b)"
```

---

### Task 5: Config knobs + dashboard toggle plumbing

**Files:**
- Modify: `config.py` (add constants near line 315-334; add env overrides in `_env`-style block near line 702; add invariants to the `model_validator` near line 749)
- Modify: `db.py` (add `get_regime_conditioning`, `set_regime_conditioning`)
- Test: `tests/unit/test_regime_toggle.py`

**Interfaces:**
- Produces: `db.get_regime_conditioning() -> bool`, `db.set_regime_conditioning(enabled: bool)`; config constants
  `REGIME_CONDITIONING_ENABLED=True, REGIME_MAP_INTERVAL_SEC=900, REGIME_MIN_SAMPLES=60, REGIME_SHRINKAGE_K=40, REGIME_RECENCY_HALFLIFE_DAYS=14, REGIME_ALLOC_MIN_WEIGHT=0.05, REGIME_ALLOC_MAX_TILT=0.25, REGIME_HOUR_BLOCK_HOURS=3`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_regime_toggle.py
import importlib
import config
import db


def test_config_defaults_present():
    importlib.reload(config)
    assert config.REGIME_CONDITIONING_ENABLED is True
    assert config.REGIME_MIN_SAMPLES == 60
    assert 0.0 < config.REGIME_ALLOC_MIN_WEIGHT < config.REGIME_ALLOC_MAX_TILT < 1.0


def test_toggle_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DB_PATH", str(tmp_path / "tg.db"))
    importlib.reload(db)
    db.init_db()
    assert db.get_regime_conditioning() is True   # default from config
    db.set_regime_conditioning(False)
    assert db.get_regime_conditioning() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_toggle.py -v`
Expected: FAIL — `AttributeError: module 'config' has no attribute 'REGIME_CONDITIONING_ENABLED'`

- [ ] **Step 3: Write minimal implementation**

In `config.py` (group with the other tuner toggles, ~line 334):

```python
# --- Regime discovery & conditioning (docs/.../2026-07-24-regime-...-design.md) ---
REGIME_CONDITIONING_ENABLED = True   # dashboard-editable; ON in paper mode
REGIME_MAP_INTERVAL_SEC = 900        # attribution/discovery cadence
REGIME_MIN_SAMPLES = 60              # promote a cell to a named regime
REGIME_SHRINKAGE_K = 40              # empirical-Bayes prior strength
REGIME_RECENCY_HALFLIFE_DAYS = 14    # decay for non-stationarity
REGIME_ALLOC_MIN_WEIGHT = 0.05       # explore floor per active bot
REGIME_ALLOC_MAX_TILT = 0.25         # max deviation from baseline weight
REGIME_HOUR_BLOCK_HOURS = 3          # ET time-of-day granularity
```

Add a pydantic invariant in the existing `@model_validator(mode="after")` block (~line 749) — follow the surrounding style:

```python
        if not (0.0 < self.REGIME_ALLOC_MIN_WEIGHT < self.REGIME_ALLOC_MAX_TILT < 1.0):
            raise ValueError("REGIME_ALLOC_MIN_WEIGHT < REGIME_ALLOC_MAX_TILT within (0,1)")
```

> If these constants are consumed only via `getattr(config, ...)` at call sites (as elsewhere), the Settings model may not need new fields; add them to the settings model only if the file validates every constant there. Match the file's existing convention.

In `db.py`:

```python
def get_regime_conditioning() -> bool:
    """Whether Layer-3 controllers may act on the regime map (dashboard toggle)."""
    raw = get_arena_state("regime_conditioning")
    if raw is None:
        return bool(getattr(config, "REGIME_CONDITIONING_ENABLED", True))
    return str(raw) == "1"


def set_regime_conditioning(enabled: bool):
    set_arena_state("regime_conditioning", "1" if enabled else "0")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_toggle.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add config.py db.py tests/unit/test_regime_toggle.py
git commit -m "feat: regime-conditioning config knobs + dashboard toggle store"
```

---

### Task 6: Regime-conditioned portfolio allocation

Blend each free bot's allocation score with its validated shrunk edge for the current regime, bounded by max-tilt and floored so no bot is starved.

**Files:**
- Modify: `arena/portfolio.py` (`allocate` ~line 365-425 to accept `regime_edges`; `_do_rebalance` ~line 548-573 to fetch + pass them)
- Test: `tests/unit/test_regime_allocation.py`

**Interfaces:**
- Consumes: `arena.regime_map.edges_for_cell`, `db.get_regime_conditioning`, `config.REGIME_ALLOC_MAX_TILT`, `REGIME_ALLOC_MIN_WEIGHT`.
- Produces: `allocate(..., regime_edges: dict[str,float] | None = None)` applying a bounded multiplicative tilt to free-bot scores before normalization; explore floor enforced post-normalization.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_regime_allocation.py
from arena.portfolio import apply_regime_tilt


def test_tilt_bounded_and_floored():
    scores = {"a": 1.0, "b": 1.0, "c": 1.0}
    edges = {"a": 5.0, "b": -5.0}          # c absent -> neutral
    out = apply_regime_tilt(scores, edges, max_tilt=0.25, min_weight=0.05)
    assert out["a"] > out["c"] > out["b"]  # winner up, loser down, neutral middle
    assert min(out.values()) >= 0.05 * max(out.values())  # explore floor kept relative
    # tilt magnitude bounded
    assert out["a"] <= scores["a"] * (1 + 0.25) + 1e-9
    assert out["b"] >= scores["b"] * (1 - 0.25) - 1e-9


def test_no_edges_is_identity():
    scores = {"a": 1.0, "b": 2.0}
    assert apply_regime_tilt(scores, None, max_tilt=0.25, min_weight=0.05) == scores
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_allocation.py -v`
Expected: FAIL — `ImportError: cannot import name 'apply_regime_tilt'`

- [ ] **Step 3: Write minimal implementation**

Add to `arena/portfolio.py` (module-level helper) and wire into `allocate`:

```python
def apply_regime_tilt(scores: dict[str, float],
                      regime_edges: dict[str, float] | None,
                      *, max_tilt: float, min_weight: float) -> dict[str, float]:
    """Multiplicatively tilt allocation scores by per-bot regime edge.

    Edge is mapped to a bounded multiplier in [1-max_tilt, 1+max_tilt] via the
    sign-scaled rank of edges. Bots absent from `regime_edges` are neutral. The
    explore floor keeps every score >= min_weight * max(score) so no bot is
    starved out of generating future data.
    """
    if not regime_edges:
        return dict(scores)
    vals = [v for v in regime_edges.values()]
    hi = max(vals) if vals else 0.0
    lo = min(vals) if vals else 0.0
    span = (hi - lo) or 1.0
    out = {}
    for bot, s in scores.items():
        e = regime_edges.get(bot)
        if e is None:
            out[bot] = s
            continue
        norm = 2.0 * (e - lo) / span - 1.0       # -1..1
        out[bot] = s * (1.0 + max_tilt * norm)
    if out:
        floor = min_weight * max(out.values())
        out = {b: max(v, floor) for b, v in out.items()}
    return out
```

In `allocate`, after `scores = _raw_scores(...)` (~line 409) and before `_normalize`:

```python
        scores = _raw_scores(method, {n: metrics[n] for n in free_names}, corr)
        scores = apply_regime_tilt(
            scores, regime_edges,
            max_tilt=float(getattr(config, "REGIME_ALLOC_MAX_TILT", 0.25)),
            min_weight=float(getattr(config, "REGIME_ALLOC_MIN_WEIGHT", 0.05)),
        )
```
Add `regime_edges: dict | None = None` to the `allocate` signature.

In `_do_rebalance` (~line 548, where `regime` is already computed), fetch validated edges and pass them — only when the toggle is on:

```python
    regime_edges = None
    try:
        if db.get_regime_conditioning():
            from signals.context import context_cell
            from arena.regime_map import edges_for_cell
            # current cell from the live detector's context snapshot
            payload = db.get_regime_map()
            cur_cell = tuple(payload.get("current_cell") or []) or None
            edges = edges_for_cell(cur_cell) if cur_cell else None
            if edges:
                regime_edges = {b: e["shrunk_pnl"] for b, e in edges.items()}
    except Exception:
        regime_edges = None
```
Then pass `regime_edges=regime_edges` into the `allocate(...)` call (~line 573).

> `current_cell` is written by `rebuild()` in Task 8's wiring (it stamps the live context cell). Until then `regime_edges` stays None (no-op), which is correct.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_allocation.py tests/unit/test_portfolio_allocation.py -v`
Expected: `test_regime_allocation` PASS (2); pre-existing `test_execute_scales_zone_bot_amount` failure is unrelated (documented).

- [ ] **Step 5: Commit**

```bash
git add arena/portfolio.py tests/unit/test_regime_allocation.py
git commit -m "feat: regime-conditioned portfolio allocation (bounded tilt + explore floor)"
```

---

### Task 7: Regime-condition the core-lane tuner

Restrict the tuner's live attribution to trades in the current regime when conditioning is on, so a lane is nudged for the regime it is (un)predictive in.

**Files:**
- Modify: `arena/core_lane_tuner.py` (`compute_core_attribution` ~line 64; `tune` ~line 107)
- Test: `tests/unit/test_core_tuner_regime.py`

**Interfaces:**
- Consumes: `db.get_regime_conditioning`, `db.get_regime_map` (`current_cell`), the existing per-strategy attribution.
- Produces: `compute_core_attribution(conn, deadband, *, cell_filter: tuple | None = None)` — when `cell_filter` is set, only trades whose stamped context matches the cell are counted.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_core_tuner_regime.py
import inspect
from arena import core_lane_tuner


def test_compute_core_attribution_accepts_cell_filter():
    sig = inspect.signature(core_lane_tuner.compute_core_attribution)
    assert "cell_filter" in sig.parameters


def test_tune_respects_conditioning_toggle(monkeypatch):
    # When conditioning is OFF, tuner must not pass a cell_filter (global behavior).
    captured = {}
    monkeypatch.setattr(core_lane_tuner.db, "get_regime_conditioning", lambda: False)
    orig = core_lane_tuner.compute_core_attribution

    def spy(conn, deadband, *, cell_filter=None):
        captured["cell_filter"] = cell_filter
        return {}
    monkeypatch.setattr(core_lane_tuner, "compute_core_attribution", spy)
    monkeypatch.setattr(core_lane_tuner.config, "CORE_TUNE_ENABLED", True, raising=False)
    core_lane_tuner.tune()
    assert captured.get("cell_filter") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/unit/test_core_tuner_regime.py -v`
Expected: FAIL — `cell_filter` not a parameter.

- [ ] **Step 3: Write minimal implementation**

- Add `*, cell_filter: tuple | None = None` to `compute_core_attribution`. Where it reads a trade's stamped lane reads, also parse `context` (already on the row) via `signals.context.context_cell`; when `cell_filter` is set, `continue` past trades whose cell != `cell_filter`.
- In `tune`, before calling `compute_core_attribution`:

```python
    cell_filter = None
    try:
        if db.get_regime_conditioning():
            payload = db.get_regime_map()
            cur = payload.get("current_cell")
            cell_filter = tuple(cur) if cur else None
    except Exception:
        cell_filter = None
    attribution = compute_core_attribution(conn, deadband, cell_filter=cell_filter)
```

> Keep the existing global behavior byte-for-byte when `cell_filter is None`. Persist `report["cell_filter"] = list(cell_filter) if cell_filter else None` so the dashboard shows which regime the nudges were computed in.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/unit/test_core_tuner_regime.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add arena/core_lane_tuner.py tests/unit/test_core_tuner_regime.py
git commit -m "feat: regime-condition the core-lane tuner attribution"
```

---

### Task 8: Schedule the regime_map job + stamp the current cell

Wire `rebuild()` into the evolution loop and record the live context cell so Layer 3 knows the current regime.

**Files:**
- Modify: `arena.py` (`_evolution_check_loop` ~line 476-581 — import `regime_map`, add interval + call)
- Modify: `arena/regime_map.py` (`rebuild` also writes `current_cell` from the live detector)
- Modify: `signals/regime_detector.py` or the warm path to expose the latest context cell (reuse the detector's live snapshot; simplest: `rebuild()` computes the cell from the most recent resolved/among-open trade's context, or from a live `build_context` call using the detector's last prices)
- Test: `tests/unit/test_regime_map_scheduling.py`

**Interfaces:**
- Consumes: `config.REGIME_MAP_INTERVAL_SEC`.
- Produces: `regime_map` payload gains `current_cell: list`. Evolution loop calls `regime_map.rebuild()` every `REGIME_MAP_INTERVAL_SEC`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_regime_map_scheduling.py
import importlib
import db
from arena import regime_map


def test_rebuild_records_current_cell(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DB_PATH", str(tmp_path / "sc.db"))
    importlib.reload(db)
    db.init_db()
    cell = ("low_vol_range", 2, 3, "us", 0, 0)
    trades = [{"bot_name": "a", "pnl": 1.0, "cell": cell,
               "created_at": f"2026-07-20 10:{i:02d}:00", "context": {"x": 1}}
              for i in range(3)]
    monkeypatch.setattr(db, "get_resolved_trades_with_context", lambda hours=None: trades)
    # Most-recent trade's cell becomes current_cell when no live snapshot.
    m = regime_map.rebuild()
    assert "current_cell" in m
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_map_scheduling.py -v`
Expected: FAIL — `current_cell` not in payload.

- [ ] **Step 3: Write minimal implementation**

In `regime_map.rebuild()`, before `db.set_regime_map(payload)`:

```python
    # Current cell: prefer a live warm-path context; fall back to the most
    # recent resolved trade's cell so Layer 3 has a regime to condition on.
    current_cell = None
    try:
        from signals.regime_detector import get_detector
        snap = get_detector().snapshot()
        # detector exposes vol/trend features; reuse most-recent trade cell as
        # the categorical part is time-derived and rebuild runs "now".
    except Exception:
        snap = None
    if trades:
        current_cell = list(trades[0]["cell"]) if trades[0].get("cell") else None
    payload["current_cell"] = current_cell
```

In `arena.py` `_evolution_check_loop`, add to the imports (~line 476):

```python
    from arena import lane_monitor, lane_promoter, core_lane_tuner, portfolio, regime_map
```
Add a timer near `last_portfolio_check` (~line 487):

```python
    regime_map_interval = float(getattr(config, "REGIME_MAP_INTERVAL_SEC", 900))
    last_regime_map_check = 0.0
```
Add a best-effort block after the lane pipeline block (~after line 584):

```python
        try:
            if time.time() - last_regime_map_check >= regime_map_interval:
                regime_map.rebuild()
                last_regime_map_check = time.time()
        except Exception as e:
            log_event(logger, logging.ERROR, f"Regime map rebuild error (caught): {e}",
                      exc_info=True, event_type="error", where="regime_map")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_map_scheduling.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the full affected suite**

Run: `.venv/bin/python3 -m pytest tests/unit/test_context.py tests/unit/test_context_persistence.py tests/unit/test_regime_map_shrinkage.py tests/unit/test_regime_map_discovery.py tests/unit/test_regime_toggle.py tests/unit/test_regime_allocation.py tests/unit/test_core_tuner_regime.py tests/unit/test_regime_map_scheduling.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add arena.py arena/regime_map.py tests/unit/test_regime_map_scheduling.py
git commit -m "feat: schedule regime_map rebuild on the evolution loop"
```

---

### Task 9: Dashboard — Regime Map card + Settings toggle

**Files:**
- Modify: `dashboard/server.py` (add `GET /api/regime-map`; add the toggle to the existing settings POST handler near line 785-811)
- Modify: `dashboard/index.html` (Regime Map card under the Market Regime section ~line 604; a Settings toggle near the Kelly/auto-approve toggles)
- Test: `tests/unit/test_regime_map_endpoint.py`

**Interfaces:**
- Consumes: `db.get_regime_map`, `db.get_regime_conditioning`, `db.set_regime_conditioning`.
- Produces: `GET /api/regime-map` → `{regimes, current_cell, conditioning_enabled}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_regime_map_endpoint.py
from fastapi.testclient import TestClient
import importlib, db
from dashboard import server


def test_regime_map_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("ARENA_DB_PATH", str(tmp_path / "e.db"))
    importlib.reload(db)
    db.init_db()
    db.set_regime_map({"regimes": [{"cell": ["r", 2, 3, "us", 0, 0], "n": 80,
                                    "validated": True, "bot_edges": {}}],
                       "current_cell": ["r", 2, 3, "us", 0, 0]})
    client = TestClient(server.app)
    r = client.get("/api/regime-map")
    assert r.status_code == 200
    body = r.json()
    assert body["regimes"][0]["validated"] is True
    assert "conditioning_enabled" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_map_endpoint.py -v`
Expected: FAIL — 404.

- [ ] **Step 3: Write minimal implementation**

In `dashboard/server.py`:

```python
@app.get("/api/regime-map")
def get_regime_map():
    payload = db.get_regime_map()
    payload["conditioning_enabled"] = db.get_regime_conditioning()
    return JSONResponse(payload)
```
In the settings POST handler, accept `regime_conditioning` like the existing toggles:

```python
    if "regime_conditioning" in body:
        db.set_regime_conditioning(bool(body["regime_conditioning"]))
```

In `dashboard/index.html`, add a card under the Market Regime section that fetches `/api/regime-map` and renders a table (cell signature · n · validated · top bot by shrunk edge), plus a Settings checkbox bound to `regime_conditioning`. Follow the existing `updateRegime`/fetch pattern (index.html ~line 2054) and the Settings toggle markup used for auto-approve.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python3 -m pytest tests/unit/test_regime_map_endpoint.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add dashboard/server.py dashboard/index.html tests/unit/test_regime_map_endpoint.py
git commit -m "feat: dashboard Regime Map card + conditioning toggle"
```

---

## Final verification

- [ ] Run the whole suite: `.venv/bin/python3 -m pytest -q` — only the pre-existing unrelated `test_execute_scales_zone_bot_amount` failure remains.
- [ ] Backtest neutrality of Layer 1: `.venv/bin/python3 -m backtest --days 2` still runs; context stamping did not change the decision path.
- [ ] Manual: start arena + dashboard, confirm `arena_state['regime_map']` populates within `REGIME_MAP_INTERVAL_SEC`, the dashboard Regime Map card fills, and toggling conditioning off in Settings makes Layer 3 suggestion-only.

## Self-review notes (author)

- **Spec coverage:** Layer 1 → Tasks 1–2; Layer 2 (shrinkage/discovery/validation/persistence) → Tasks 3–4, 8; toggle + knobs → Task 5; Layer 3 (allocator + core-lane tuner) → Tasks 6–7; dashboard → Task 9; safety (best-effort, live-only, bounds/floor/toggle) → Global Constraints + Tasks 4/6/8.
- **Deferred to implementation (from spec's open questions):** exact clustering method — this plan uses discrete `context_cell` grouping with sample + OOS gates as the discovery mechanism (the simplest form that satisfies the spec); a smoother online-clustering upgrade can follow once attribution is proven. Non-stationarity recency half-life knob is defined (`REGIME_RECENCY_HALFLIFE_DAYS`) and should be applied as a decay weight in `attribute()` as a fast-follow if the flat window proves too slow.
- **Type consistency:** `context_cell` returns a 6-tuple everywhere; `bot_edges[bot]["shrunk_pnl"]` is the field consumed by allocation and the dashboard; `current_cell` is a list in JSON, compared via `tuple(...)`.

# Regime Discovery, Context Attribution & Regime-Conditioned Control — Design

**Date:** 2026-07-24
**Status:** Approved (design) — pending implementation plan
**Author:** RJ + Claude

## Problem

The current regime detector (`signals/regime_detector.py`) classifies markets into a
fixed 2×2 taxonomy (vol × trend → 5 regimes + unknown). This is useful context but
incomplete:

- It ignores **time structure** (day-of-week, time-of-day/session, macro-release
  proximity) and **multi-scale BTC price trend**, all of which plausibly change which
  bots have edge.
- The taxonomy is **fixed** — there is no mechanism to *discover* regimes the data
  supports but no one hand-defined.
- Per-regime performance is tracked for **observability only** — nothing feeds it back
  into how signal weights or capital are allocated.

We want a system that (a) evaluates a **rich market context**, (b) **learns how each bot
performs** across that context, (c) **discovers** named regimes from data (including
combinations like "calm + choppy + Wednesday 09:00–12:00 ET"), and (d) **adjusts signal
strengths and capital allocation** accordingly on regime shifts.

## The core risk this design exists to defeat

Treating **every combination of context dimensions as its own discrete regime** fragments
the data into statistical noise. BTC 5-minute markets produce ~288 windows/day; a naive
grid (weekday × 8 time-blocks × vol × trend × BTC-macro × BTC-micro) is ~13,600 cells,
most of which see 0–3 trades ever. Per-bot rankings computed on such thin cells are pure
sampling noise, and a discovery engine scanning thousands of cells is a machine for
manufacturing false positives — exactly the "predictive but unprofitable" failure mode
this codebase has repeatedly hit (BUG #24 favorite-tilt; `pm_mom` 70% WR that lost money).

**The design separates three concepts the word "regime" currently conflates:**

1. **Context** — a rich per-window *feature vector* (cheap; ingredients already computed).
2. **Regime label** — a *learned clustering* of context space into a bounded set of
   **sample-gated, out-of-sample-validated** named regimes (this is "discovery", and it
   self-limits to what the data supports).
3. **Controller** — allocation/weight adjustment driven by a **hierarchical empirical-Bayes
   shrinkage** estimate of per-bot edge given context, so thin cells contribute a little
   signal instead of a wild swing. Discrete regimes are the *interpretability* layer; the
   control math stays smooth and regularized.

## Non-goals

- Not a new directional signal — this is context/allocation only.
- Not a replacement for the GA (roster), portfolio allocator (capital owner), core-lane
  tuner (signal-weight owner), or hybrid meta-learner. **One owner per knob** is preserved;
  the regime system makes existing owners context-aware.
- No live-capital exposure introduced. Paper mode only for now.

## Architecture

```
Warm path (1s) ──► build_context() ──► context vector
                                          │
Decision time  ──► stamp context on the trade row (+ regime label)
                                          │
Resolution     ──► trade resolves, PnL known, context already attached
                                          │
Evolution loop ──► regime_map job: shrinkage attribution + discovery + OOS validation
                     └► arena_state["regime_map"]
                                          │
Regime shift / ──► portfolio allocator (capital) + core-lane tuner (signal weights)
   timer            read regime_map; apply IF (toggle on AND regime validated),
                    bounded deviation + explore floor + hysteresis
                                          │
Dashboard      ──► Regime Map: per-bot per-regime edge, current regime, suggested vs applied
```

Each layer is an independently testable unit with a narrow interface:

### Layer 1 — Context vector (`signals/context.py`)

Pure function, no module state (safe for warm path, harness, tests):

```
build_context(prices, signals, now_utc) -> dict
```

Returns a structured dict:

- **Continuous** (0..1 unless noted): `vol`, `trend`, `flow`, `realized_vol`,
  `btc_mom_1m`, `btc_mom_5m`, `btc_mom_15m` (from `signals/multiscale.py`),
  `btc_trend_slope` (sign+magnitude of a longer EMA slope for macro trend).
- **Categorical:** `weekday` (0–6, ET), `hour_block` (3-hour ET blocks → 0–7),
  `session` (asia/eu/us/overnight, ET), `macro_prox` (0/1/2 = far / shoulder / at an
  08:30 or 14:00 ET slot, reusing `macro_calendar` logic).
- **Derived:** `vol_trend_regime` (the existing detector id, for continuity).

All ingredients already exist (`multiscale`, `volatility_regime`, `regime_detector`,
`session_filter`, `macro_calendar`); this function *composes* them — it does not add new
network reads.

**Storage.** New nullable `context` JSON column on `trades`, populated at decision time in
`bots/base_bot.make_decision` (alongside the existing `regime:` feature stamp). Migration
is idempotent in `db.init_db` (same pattern as prior column adds). Phase 1 is
**behavior-neutral**: stamping context must not change any decision — verified by backtest
replay producing identical trades.

### Layer 2 — Attribution + discovery (`arena/regime_map.py`)

A periodic job hosted on the evolution loop (sibling of `lane_monitor` / `lane_promoter`
/ `core_lane_tuner`), gated by `REGIME_MAP_INTERVAL_SEC`. Reads **resolved live trades
only** (`outcome IN ('win','loss','exit_tp','exit_sl')`) with a non-null `context`.

**Hierarchical empirical-Bayes shrinkage.** For each (bot, context-cell), estimate an
edge metric — both **avg PnL/trade** and **WR − avg entry** (the canonical break-even
gap) — shrunk up a hierarchy:

```
cell estimate  ->  parent (coarser cell) mean  ->  global mean
```

Shrinkage weight scales with the cell's sample count (Normal-Normal / Beta-Binomial
conjugate form; strength `REGIME_SHRINKAGE_K`). A 3-sample cell lands near its parent;
a 200-sample cell trusts itself. This is the mathematical core that makes fine context
usable without overfitting.

**Discovery.** Cluster observed continuous context vectors (extends the online-centroid
machinery already in `regime_detector`) crossed with the categorical splits into candidate
regimes. **Promote a candidate to a named regime only when:**

- `n >= REGIME_MIN_SAMPLES` resolved trades in the cell, **and**
- the per-bot performance signature is **stable across a train/validation split**
  (out-of-sample: the ranking/sign of top vs bottom bots must persist), **and**
- recency-weighted (older trades decayed via `REGIME_RECENCY_HALFLIFE_DAYS`) so a stale
  regime ages out.

**Persistence.** Writes `arena_state["regime_map"]`:

```json
{
  "regimes": [
    {"id": "...", "signature": {...}, "n": 213, "validated": true,
     "bot_edges": {"momentum": {"pnl": 0.03, "gap": 0.06, "n": 41, "shrunk": true}, ...}}
  ],
  "current_regime_id": "...",
  "updated_at": 1784...
}
```

This job **never** changes trading behavior — it is measurement + discovery only. It is
where the user first *sees* "bot A wins in calm-choppy-Wednesday-mornings" before any
dollar rides on it.

### Layer 3 — Regime-conditioned controllers (extend existing owners)

Both read `arena_state["regime_map"]`; both apply only when the current regime is
`validated` **and** the toggle is on. Enabled by default (paper mode).

- **Portfolio allocator** (`arena/portfolio.py`). Regime shift is *already* a rebalance
  trigger. Add a conditioning term: blend each bot's baseline weight with its **shrunk
  edge** for the current regime, subject to:
  - an **explore floor** `REGIME_ALLOC_MIN_WEIGHT` (never starve a bot to 0 — preserves
    exploration so a down-weighted bot keeps generating data), and
  - **bounded deviation** `REGIME_ALLOC_MAX_TILT` from the baseline weight, and
  - **hysteresis** so capital does not whipsaw on every transient shift.
- **Core-lane tuner** (`arena/core_lane_tuner.py`). Extend its live-attribution join so a
  lane is nudged based on its predictiveness **in the current regime** (drift/mom/strat),
  keeping the existing per-lane band + one-step-per-cycle bounds.

### Cross-cutting: safety, errors, config

- **Discipline:** live resolved trades only; OOS validation gates "active"; explore floor;
  recency weighting; hysteresis; one bounded step per cycle (mirrors `core_lane_tuner`).
- **Errors:** every periodic job is best-effort and never raises into the hot path (matches
  `lane_monitor`/`lane_promoter`). Missing/thin context → fall back to parent/global
  (cold-start safe). Toggle off → controllers compute + persist *suggestions* only.
- **Toggle:** `config.REGIME_CONDITIONING_ENABLED = True` (default ON — paper mode, no real
  capital). Stored in `arena_state` and editable from the **dashboard Settings tab**
  (like the Kelly Fraction / auto-approve-lanes toggles); picked up within seconds.
  OFF → Layer 3 becomes suggestion-only; Layers 1–2 always run.

### New / changed files (many small, per repo style)

| File | Change |
|---|---|
| `signals/context.py` | **new** — `build_context()` pure composer |
| `arena/regime_map.py` | **new** — shrinkage attribution + discovery + persistence |
| `arena/portfolio.py` | extend `allocate()` with regime-conditioning term |
| `arena/core_lane_tuner.py` | regime-condition the attribution join |
| `db.py` | `context` column migration; `regime_map` get/set helpers; resolved-trades-with-context query |
| `bots/base_bot.py` | stamp `context` on the trade at decision time |
| `arena.py` | schedule the `regime_map` job on the evolution loop |
| `dashboard/server.py` + `dashboard/index.html` | Regime Map card; Settings toggle |
| `config.py` | knobs + pydantic invariants (see below) |

### Config knobs (defaults)

```
REGIME_CONDITIONING_ENABLED   = True     # dashboard-editable toggle
REGIME_MAP_INTERVAL_SEC       = 900      # discovery/attribution cadence
REGIME_MIN_SAMPLES            = 60        # promote a cell to a named regime
REGIME_SHRINKAGE_K            = 40        # empirical-Bayes prior strength
REGIME_RECENCY_HALFLIFE_DAYS  = 14        # decay for non-stationarity
REGIME_ALLOC_MIN_WEIGHT       = 0.05      # explore floor per active bot
REGIME_ALLOC_MAX_TILT         = 0.25      # max deviation from baseline weight
REGIME_HOUR_BLOCK_HOURS       = 3         # ET time-of-day granularity
```

## Data flow (end to end)

1. Warm path builds `context` and carries it in combined signals.
2. `make_decision` stamps `context` onto the trade row at decision time.
3. Trade resolves; PnL is joined to the already-attached context.
4. Evolution-loop `regime_map` job recomputes shrunk per-bot edges, runs discovery +
   OOS validation, and writes `arena_state["regime_map"]`.
5. On regime shift or rebalance timer, the portfolio allocator and core-lane tuner read
   the map and apply bounded, floored, hysteretic adjustments **iff** validated + toggle on.
6. Dashboard renders the regime map, per-bot per-regime edge, and suggested vs applied
   adjustments.

## Testing

- **Unit:** `build_context` determinism + range; shrinkage pulls a thin cell toward its
  prior and a rich cell toward itself; discovery refuses to promote an under-sampled or
  OOS-unstable cell; allocator respects explore floor + max tilt + hysteresis; controllers
  no-op (suggestion-only) when toggle off.
- **Integration:** synthetic resolved-trade sets with known per-regime edge → `regime_map`
  recovers the expected ranking; a regime shift produces a bounded rebalance.
- **Backtest:** Phase 1 is behavior-neutral — replay resolved markets and assert the
  stamped context does not alter decisions.

## Rollout

All three layers are built now. Safety comes from *validation gating*, not from disabling
Layer 3: the controllers act only on regimes that clear the sample + OOS bar, so on a fresh
DB they are effectively no-ops until live paper trades accumulate. The dashboard toggle is
the single human override; demotion (a regime falling below the bar / aging out) is
automatic.

## Open questions deferred to the plan

- Exact clustering method for continuous-context discovery (online k-means vs extending the
  existing centroids) — decide during planning against the live feature distribution.
- Whether core-lane-tuner regime-conditioning ships in the same pass as the allocator or a
  fast-follow (both are Layer 3; sequencing is a plan detail).

# PLAN: Regime Adapt (Frequency-Stable) — Not Throttle

**Date:** 2026-08-05  
**Status:** Implemented (2026-08-05) — soak/validate live  
**Author:** RJ + Grok  
**Supersedes / extends:**  
- Philosophy of “sit flat / downsize in bad regimes” (`arena/regime_adapt.py` hard-skip + size_min 0.35)  
- Complements (does not replace) [regime discovery design](../specs/2026-07-24-regime-discovery-context-attribution-design.md)  
**Goal:** Keep **consistent trading frequency** across regimes by **adapting signal weights, strategy emphasis, and capital routing** to each regime — instead of primarily **skipping or downsizing** when live WR is weak.

---

## 0. Executive summary

### Problem (observed live)

- After profitable stretches, the arena can go **hours with 0 fills** while still logging thousands of **skips**.  
- Dominant label is often `low_vol_range`. Policy stack **raises bars** (edge mult, drift floors, mid-band floors), **damps mom**, **shrinks size** toward 0.35, and can **hard-skip** directionals.  
- Same decision blend is asked to work in every tape; when it fails, we **trade less** rather than **re-spec the blend**.

### Solution (target)

| Layer | Role |
|-------|------|
| **Relative multi-factor regime model** | Labels + continuous feature vector that stay meaningful as BTC vol base shifts |
| **Per-regime signal profiles** | Core lanes (and approved candidates) weighted **by regime × strategy** |
| **Capital / roster routing** | Portfolio + GA favor strategies that work *in this regime*; explore floor preserves diversity |
| **Throttle demotion** | Hard-skip off (or emergency-only); size mult near neutral; frequency soft-target optional |
| **Invariant safety** | High-price, consensus, book-sum, exposure, kill switch **never** regime-relaxed |

### Non-goals

- Not “trade every window” or buy 0.90 favorites.  
- Not removing dead-zone / high-price without separate evidence (those are R:R, not regime style).  
- Not replacing GA, core-lane tuner ownership, or portfolio ownership — **one owner per knob**, made **regime-conditional**.  
- Not live-capital rollout in this plan (paper first; same code paths).

### Success metrics (must hit before calling “done”)

| Metric | Target (paper, ≥5 trading days post full stack) |
|--------|--------------------------------------------------|
| Fills / hour by regime | Coefficient of variation of hourly fill rates across `low_vol_range`, `low_vol_trend`, `normal`, `high_vol_chop`, `high_vol_trend` **≤ 0.6** (baseline often ≫1) |
| No multi-hour total freeze | Max consecutive minutes with **0** directional evaluations that *could* clear guards **and** mid ∈ [0.35, 0.72] is ok; max consecutive minutes with **0 fills when mid-band opportunities exist** &lt; 90 min under healthy feed |
| Pool P&amp;L | Rolling 24h pool P&amp;L ≥ 0 **or** BE gap ≥ 0.03 on active directionals (same survival bar as evolution) |
| Regime label stability | Median dwell time per label ≥ 3 min; flap rate (switches/hour) not worse than pre-change ±20% |
| Relative vol | Under synthetic 2× vol regime, share of time in `high_vol_*` stays within ±15pp of baseline share (labels track relative, not absolute) |
| Safety | Zero trades with mid &gt; `HIGH_PRICE_GUARD` (unless learned softener explicitly on); exposure / book gates unchanged |

### Product locks (do not violate during implementation)

1. **Edge still model vs executable ask after fees** (BUG #24).  
2. **NO first-class**; two-sided selection unchanged.  
3. **Candidate lanes** stay kill-switched until harness + live shadow (BUG #26/#31).  
4. **Learning blend** stays off (`LEARNING_ENABLED=False`) unless separate redesign.  
5. **One owner per knob** (table below).

| Knob | Owner (after this work) |
|------|-------------------------|
| Discrete regime id + continuous features | `signals/regime_detector.py` (+ relative calibrator) |
| Per-regime lane weights | `arena/core_lane_tuner.py` extended + `lane_overrides` schema |
| Hard/emergency risk | `arena/risk_engine.py` + kill switch (not regime_adapt primary) |
| Capital weights | `arena/portfolio.py` + regime map conditioning |
| Roster / mutants | `evolution/ga.py` + type_alloc |
| Static style seeds | `arena/regime_adapt.py` priors (seeds only; live overrides win) |
| Hybrid sub-weights | `bots/meta_learner.py` / hybrid (existing) |

---

## 1. Current system map (implementer must read)

### 1.1 Files (load-bearing)

| Path | Role today |
|------|------------|
| `signals/volatility_regime.py` | Absolute `vol_score` via `VOL_TYPICAL=0.0006`; `trend_score` efficiency |
| `signals/regime.py` | Multi-horizon ER + choppiness (context features) |
| `signals/regime_detector.py` | EMA features, rule 2×2 labels, centroids, perf, persistence |
| `signals/lab.py` | `REGIME_LANE_DAMP` mom/strat damps by label |
| `signals/context.py` | Rich context vector for stamps / regime map |
| `arena/regime_adapt.py` | Size mult, edge mult, lane scales, mid-band floor, **hard_skip** |
| `arena/regime_map.py` | Cell discovery + EB shrinkage (capital / tuner conditioning) |
| `arena/core_lane_tuner.py` | Global (not per-regime) drift/mom/strat nudges |
| `arena/portfolio.py` | Kelly portfolio + regime tilt when conditioning ON |
| `bots/base_bot.py` | Consumes `adjustments()`, stamps `regime:id`, applies lane scales / hard-skip |
| `config.py` | All `REGIME_*`, `REGIME_ADAPT_*`, `REGIME_HARD_SKIP_*` knobs |
| `arena/decision_log.py` | Skip/buy telemetry (use for frequency + attribution) |
| Dashboard `index.html` / `server.py` | Regime map, Signal Lab, ops strip |

### 1.2 Current classification (absolute)

Features (approx [0,1]): `vol`, `trend`, `mom`, `flow` (+ `volume` unused in rules).  
Rules: `vol_hi=0.55`, `vol_lo=0.35`, `trend_hi=0.50`, `trend_lo=0.35` → five ids + unknown.  
Hysteresis: `REGIME_HOLD_TICKS`, `REGIME_SWITCH_MARGIN`.  
Centroids: online means for soft confidence (`REGIME_USE_CENTROIDS`).

### 1.3 Current policy (throttle-heavy)

- Live WR → `size_mult` ∈ [0.35, 1.15]  
- Toxic WR → `block_directional` hard-skip  
- Strategy×regime **priors** often **raise** `edge_mult` / drift floors (fewer trades)  
- Lab damps mom in quiet/chop  

### 1.4 Related prior design

`docs/superpowers/specs/2026-07-24-regime-discovery-context-attribution-design.md` already separates **context vs label vs controller**. This plan **aligns controller philosophy** with “adapt weights/capital” and **upgrades the label layer** to relative multi-factor features. Do not invent a second parallel taxonomy without merging names into `REGIME_IDS`.

---

## 2. Target architecture

```
Warm path (1s)
  price feed + orderflow + multiscale + session/macro
       │
       ▼
  Feature factory (absolute + relative)
       │
       ├─► continuous feature vector F (persisted on decisions/trades)
       └─► discrete regime_id (hysteresis) + confidence
       │
       ▼
  Policy stack (priority order)
       1. Invariant guards (price, book, exposure, kill, macro, session)
       2. Per-regime × strategy lane profile (overrides > seeds > class default)
       3. Continuous weight residual optional (Phase 5): Δw = B·F
       4. Portfolio capital weight for this bot in this regime
       5. Risk size_mult (drawdown) — NOT regime WR primary
       │
       ▼
  make_decision → venue
       │
  resolution → attribution
       │
  evolution loop:
       · per-regime core-lane tuner
       · regime map refresh
       · optional frequency soft-target report
       · dashboard state
```

### 2.1 Expanded informative feature set (classification + control)

All features must be: **causal at decision time**, **reproducible offline**, **bounded**, and (where scale-free needed) **relative**.

#### A. Tape microstructure / path (BTC)

| Feature key | Source | Absolute | Relative | Use |
|-------------|--------|----------|----------|-----|
| `realized_vol_1m` | `volatility_regime` | raw stdev | percentile 7d/30d | vol axis |
| `atr_pct` | same | raw | percentile | vol axis alt |
| `trend_eff_20` | efficiency 20m | [0,1] ramp | optional z vs hist | trend axis |
| `trend_eff_10` / `_30` | `signals/regime.py` | [0,1] | optional | multi-scale trend |
| `chop_14` | `regime.choppiness` | [0,1] | optional | chop vs trend disambiguation |
| `mom_1m_abs` | last 1m \|ret\| | sat | percentile | burstiness |
| `ms_mom_align` | sign(ms_mom_1m)×sign(ms_mom_5m) | {-1,0,1}→[0,1] | — | multi-horizon agreement |
| `vol_of_vol` | stdev of rolling vol | raw | percentile | regime instability |

#### B. Flow / book (when live; graceful zero if kill-switched)

| Feature | Source | Notes |
|---------|--------|-------|
| `flow_intensity` | mean \|cvd\|+\|obi\| | existing |
| `flow_align` | sign agreement flow vs mom | existing |
| `spread_score` | `micro_spread` / mid | wide books → different edge tax (already partial) |
| `book_sum_gap` | \|yes+no−1\| | instability; not directional |

#### C. Clock / calendar (non-directional)

| Feature | Source |
|---------|--------|
| `sess_*` | `session_features` (tod sin/cos, nyse proximity, weekend) |
| `macro_caution` | `macro_calendar` |
| `hour_block_et` | 3h blocks (align regime_map) |

#### D. Cross-asset / futures (shadow until validated)

| Feature | Source | Policy |
|---------|--------|--------|
| `xasset_abs` | cross_asset | context only until net edge |
| `fut_intensity` | \|taker_delta\| etc. | context only |

**Classifier core (Phase 1–2 minimum):** relative vol, relative/absolute trend efficiency, chop, multi-scale mom alignment, flow intensity.  
**Clock/macro:** used for **routing / soft priors**, not for inventing 1000 discrete labels (see discovery design anti-fragmentation).

### 2.2 Discrete taxonomy (stable API)

Keep public ids (dashboard, stamps, GA):

```
high_vol_trend | low_vol_range | high_vol_chop | low_vol_trend | normal | unknown
```

Optional later **sublabels** only if OOS-validated via regime_map discovery — not ad-hoc.

**Primary axes (relative):**

1. **Vol regime** — percentile of `realized_vol` (and optionally vol-of-vol)  
2. **Directionality** — blend of trend efficiency + (1−chop) + multi-scale mom align  

Secondary soft votes: flow intensity, clock priors (do not alone flip label without margin).

### 2.3 Relative calibration design

```
raw_vol_t  →  store in ring buffer / DB series
vol_score_rel = empirical_cdf(raw_vol_t; window=W, min_points=M)
vol_score_abs = sigmoid(raw_vol_t; VOL_TYPICAL)   # keep for risk/logging
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `REGIME_REL_WINDOW_DAYS` | 14 | primary percentile window |
| `REGIME_REL_WINDOW_DAYS_SLOW` | 60 | slow baseline for drift detection |
| `REGIME_REL_MIN_SAMPLES` | 500 | 1m bars; fail-open to abs score if cold |
| `REGIME_REL_UPDATE_HALFLIFE_DAYS` | 7 | optional exp-weight recent bars more |
| `REGIME_ABS_VOL_RISK_P99` | from slow window | absolute risk cap only |

**Cold start:** use absolute scores until min samples; log `calibration=absolute_fallback`.  
**Persist:** calibrator quantiles in `arena_state['regime_calibration']` every N minutes so restart doesn’t forget.  
**Tests:** synthetic constant vol → ~uniform labels over time; 2× vol shock → relative scores return to ~0.5 after window absorbs shock.

### 2.4 Policy: adapt not throttle

| Knob | Before | After (defaults) |
|------|--------|------------------|
| `REGIME_HARD_SKIP_ENABLED` | True | **False** (emergency path retained, bar much stricter) |
| `REGIME_ADAPT_SIZE_MIN` | 0.35 | **0.85** |
| `REGIME_ADAPT_SIZE_MAX` | 1.15 | 1.15 |
| Primary response to bad WR | skip / size | **reweight lanes + reallocate capital** |
| Edge mult priors | often &gt;1 (harder) | seeds only; **lane profile** carries style |
| Frequency | unconstrained collapse | soft monitor + optional tiny edge ease **only if** strategy has +EV in-regime |

**Emergency hard-skip (retain code path):** e.g. n≥80, WR≤0.38, pnl&lt;0, and **global** not just regime — or regime WR≤0.35 with n≥100. Must not fire on n=20 noise.

---

## 3. Phase plan (all six)

Each phase: **goal · design · files · tasks · tests · acceptance · rollback**.  
Implement in order; each phase should be shippable and toggleable.

---

### Phase 1 — Relative multi-factor feature factory + calibrated labels

**Goal:** Regime ids mean “high/low *for recent BTC*,” with richer features; dual absolute/relative scores.

#### Design

1. New module `signals/regime_calibration.py` (or package under `signals/regime/`):  
   - `RelativeCalibrator`: update(raw_vol, raw_trend_eff, …), `percentile(key, value) → [0,1]`  
   - Persistence via `db` arena_state  
   - Thread-safe; warm-path cheap (binary search on sorted reservoir or P² quantiles / t-digest lite)

2. Extend `volatility_regime.compute` outputs:  
   - keep `vol_score` as absolute  
   - add `vol_score_rel` when calibrator provided  

3. Extend `regime_detector.compute_features` / `RegimeDetector.update`:  
   - ingest multiscale + chop + session (optional)  
   - build feature dict with both abs and rel keys  
   - `classify_rules` uses **rel vol + directionality composite** by default when `REGIME_USE_RELATIVE=True`

4. Directionality composite (document formula in code):  
   ```
   direction = 0.45*trend_eff_rel + 0.35*(1-chop) + 0.20*ms_mom_align01
   ```  
   Thresholds on (vol_rel, direction) map to existing 2×2.

5. Stamp on trades / decision_events:  
   - `regime_id`, `regime_conf`  
   - compact feature snapshot JSON or selected floats in `context` / features list  

6. Config:  
   ```
   REGIME_USE_RELATIVE = True
   REGIME_REL_WINDOW_DAYS = 14
   REGIME_REL_WINDOW_DAYS_SLOW = 60
   REGIME_REL_MIN_SAMPLES = 500
   REGIME_CLASSIFY_VOL_HI = 0.70   # percentile
   REGIME_CLASSIFY_VOL_LO = 0.30
   REGIME_CLASSIFY_DIR_HI = 0.55
   REGIME_CLASSIFY_DIR_LO = 0.40
   ```

#### Files to touch

- `signals/regime_calibration.py` **(new)**  
- `signals/volatility_regime.py`  
- `signals/regime_detector.py`  
- `signals/context.py` (export new keys)  
- `arena/signals.py` (pass data into detector)  
- `config.py`  
- `tests/unit/test_regime_calibration.py` **(new)**  
- `tests/unit/test_regime_detector.py` (extend)  
- `CLAUDE.md` / `docs/signal-suite.md` short note  

#### Tasks

1. Implement calibrator with unit tests (cold start, update, percentile monotonicity, persist/load).  
2. Wire absolute + relative vol into detector.  
3. Add chop + multi-horizon trend/mom align to feature vector.  
4. Switch classifier to relative when enabled; keep absolute path for A/B flag.  
5. Persist calibration + detector state (existing STATE_KEY + new key).  
6. Dashboard: show `vol_rel` vs `vol_abs` on regime card (minimal).  
7. Offline: script or harness note — reconstruct relative scores from kline cache for backtests (Phase 1 can use absolute in backtest if relative history missing; document).

#### Tests

- Percentile of constant series → ~0.5 after fill.  
- Monotonic: higher raw vol → higher or equal percentile.  
- 2× vol after long window → rel score mean reverts.  
- classify_rules corners still map to five ids.  
- Hysteresis unchanged behavior under flapping synthetic features.  
- No regression: `get_detector().status()` shape for dashboard.

#### Acceptance

- `REGIME_USE_RELATIVE=True` default in paper.  
- Live soak: labels still flap-controlled; features table populated.  
- Unit tests green.

#### Rollback

- `REGIME_USE_RELATIVE=False` → old absolute sigmoid + old thresholds.

---

### Phase 2 — Policy flip: style switch instead of throttle

**Goal:** Bad live WR no longer starves frequency; priors become style seeds; hard-skip demoted.

#### Design

1. Config defaults:  
   ```
   REGIME_HARD_SKIP_ENABLED = False
   REGIME_HARD_SKIP_EMERGENCY_ONLY = True
   REGIME_HARD_SKIP_MIN_TRADES = 80
   REGIME_HARD_SKIP_WR = 0.38
   REGIME_ADAPT_SIZE_MIN = 0.85
   REGIME_ADAPT_SIZE_MAX = 1.15
   REGIME_ADAPT_PRIMARY = "style"   # vs "throttle"
   ```

2. Rewrite `adjustments()` behavior when primary=style:  
   - `size_mult` from regime WR only within [0.85, 1.15]  
   - `block_directional` only if emergency bar  
   - **Do not** raise `mid_band_drift_min` solely because size is depressed  
   - `edge_mult` priors: flatten toward 1.0 (max 1.15) — style via lane scales  
   - Keep `mom_lane_scale` / `strat_lane_scale` as **seed** until Phase 3 overrides exist  

3. Revisit `_REGIME_STRATEGY_PRIORS` for `low_vol_range`:  
   - Prefer **meanrev-friendly** scales (mom↓, strat↑) **without** huge edge_mult  
   - Document each prior as “seed until live profile”  

4. Align `REGIME_LANE_DAMP` in lab with same philosophy (damp mom in range/chop — OK; don’t double-count with regime_adapt scales — **single application site**: prefer lab OR adapt, not both ×).  
   - **Decision:** apply structural damps in **lab only**; regime_adapt lane scales only when no per-regime override (Phase 3).  

5. Logging: every skip reason `regime_hard_skip` must be rare; alert if emergency fires.

#### Files

- `arena/regime_adapt.py`  
- `config.py`  
- `signals/lab.py` (dedupe damps)  
- `bots/base_bot.py` (if adjust consumption changes)  
- `tests/unit/test_regime_adapt_adjustments.py`  
- `tests/unit/test_regime_hard_skip.py`  

#### Tasks

1. Change defaults + document migration.  
2. Implement emergency-only hard-skip.  
3. Flatten size range and edge mult.  
4. Deduplicate mom damp (lab vs adapt).  
5. Update tests for new defaults.  
6. Soak report field: size_mult histogram by regime.

#### Acceptance

- With identical markets, paper fill rate in `low_vol_range` rises vs pre-change (measure 24h).  
- Hard-skip count ≈ 0 under normal WR.  
- No increase in high-price or book-inconsistency fills.

#### Rollback

- Config restore old constants; redeploy.

---

### Phase 3 — Per-regime × strategy core-lane profiles

**Goal:** Main adaptation engine — drift/mom/strat (and later candidates) weights depend on **regime at decision time**.

#### Design

1. **Schema** for `lane_overrides` (backward compatible):  

   ```json
   {
     "drift": {
       "enabled": true,
       "core": true,
       "profile": { "momentum": 0.75, ... },
       "by_regime": {
         "low_vol_range": { "momentum": 0.80, "mean_reversion": 0.90, ... },
         "high_vol_trend": { "momentum": 0.60, ... }
       }
     },
     "mom": { "...": "..." },
     "ms_mom": {
       "enabled": true,
       "profile": { "momentum": 0.10 },
       "by_regime": { "high_vol_trend": { "momentum": 0.15 } }
     }
   }
   ```

   Resolution order in `SignalLab.weights_for` / `_lane_overrides` consumer:  
   ```
   by_regime[regime][strategy] > profile[strategy] > class default
   ```

2. **Hot path:** `BaseBot` / lab must know **current regime_id** when resolving weights (already have `regime_context`). Cache profile per (strategy, regime) for TTL `HOTPATH_CACHE_TTL_SEC`.

3. **`core_lane_tuner` extension:**  
   - Attribute accuracy **grouped by** `(strategy_type, regime_id, lane)`  
   - Source: `decision_events` (preferred) with regime column, else trade features `regime:id`  
   - Same bounds: `CORE_TUNE_STEP`, `CORE_TUNE_BAND`, min trades **per cell** (`CORE_TUNE_MIN_TRADES_REGIME`, default 40)  
   - Write `by_regime` subtrees; keep global `profile` as fallback / prior  
   - Toggle: `REGIME_PROFILE_ADAPT_ENABLED` (and respect auto_core_tune)  

4. **Seed profiles** (hand-set, evidence-backed starting points):  

   | Regime | momentum | mean_reversion | phantom | hybrid | sniper |
   |--------|----------|----------------|---------|--------|--------|
   | low_vol_range | drift↑ mom↓ strat mid | drift↑↑ mom0 strat↑ | drift↑ mom↓ | meanrev-lean | drift pure |
   | low_vol_trend | drift↑ mom mid | drift↑ mom0 | drift↑ mom mid | balanced | drift |
   | high_vol_trend | drift mid mom↑ | careful / lower size via capital | mom↑ | mom-lean | lag hunt |
   | high_vol_chop | drift mid mom↓↓ strat↓ | fade only strong | mom↓ | chop damp | selective |
   | normal | class defaults | defaults | defaults | defaults | defaults |

   Encode seeds in `bots/base_bot.py` or `arena/regime_profiles.py` as `REGIME_PROFILE_SEEDS`.

5. **Candidate lanes** (lag/ms_mom/xasset/…): same `by_regime` optional; only if override enabled.

6. **Dashboard Signal Lab:** show matrix strategy × regime for core lanes; “active regime” highlight.

#### Files

- `arena/core_lane_tuner.py`  
- `arena/regime_profiles.py` **(new)** seeds + resolve helper  
- `bots/base_bot.py` / `signals/lab.py` weight resolution  
- `db.py` if helpers for overrides needed  
- `dashboard/server.py` + `index.html`  
- `tests/unit/test_core_tuner_regime.py` (expand)  
- `tests/unit/test_regime_profiles.py` **(new)**  
- `tests/integration/test_lane_pipeline.py` if override shape changes  

#### Tasks

1. Spec + implement `resolve_lane_weight(lane, strategy, regime, overrides)`.  
2. Wire into lab blend path.  
3. Extend tuner SQL/aggregation by regime.  
4. Seeds + migration: on first boot, if `by_regime` empty, copy seeds (don’t wipe global).  
5. Dashboard matrix.  
6. Decision reasoning: log `prof=range` or effective weights briefly for debug (optional compact token).

#### Acceptance

- In live paper, `low_vol_range` effective mom weight &lt; `high_vol_trend` mom weight for momentum strategy after seeds.  
- Tuner only moves cells with n≥ min.  
- Hot path: no extra DB hit every tick (cache).

#### Rollback

- `REGIME_PROFILE_ADAPT_ENABLED=False` and ignore `by_regime` in resolver.

---

### Phase 4 — Strategy routing for frequency (capital + roster)

**Goal:** Same total directional risk budget; **who** gets it changes with regime.

#### Design

1. **Portfolio** (`arena/portfolio.py`):  
   - When `REGIME_CONDITIONING_ENABLED` and map validated: tilt weights using **per-regime bot edge** (existing map) **plus** live `regime_performance` by strategy.  
   - Explore floor unchanged (`REGIME_ALLOC_MIN_WEIGHT`).  
   - On regime flip: rebalance (already `PORTFOLIO_REBALANCE_ON_REGIME`).  
   - **Frequency-aware:** if a bot’s regime WR is good but capital weight was floored, allow tilt toward it faster (document hysteresis).

2. **Strategy eligibility soft scores** (new helper `arena/regime_router.py`):  
   ```
   score(strategy, regime) = f(live_WR, live_pnl, n, seed_fit)
   ```  
   Used by portfolio and optional **decision-time size share** (not skip).

3. **GA / type_alloc:**  
   - When spawning replacements, boost probability of types with high `score(type, current_regime)` and positive regime fitness.  
   - Do not delete types; maintain diversity cap (`GA_MAX_PER_TYPE_PER_CYCLE`).

4. **Default slate:** consider promoting `lag_residual` into optional slate when `low_vol_range` dwell is high (config flag `REGIME_ROUTE_LAG_RESIDUAL=True`) — menu already exists.

5. **Makers:** exempt from directional hard-skip (already); routing may **increase** maker capital in range if their live cell works (makers still zone-gated).

6. **Frequency soft-target (optional Phase 4b):**  
   ```
   REGIME_FREQ_TARGET_FILLS_PER_HOUR = 4   # slate-level directional
   REGIME_FREQ_EDGE_EASE_MAX = 0.15        # max *reduction* of effective min_edge
   ```  
   Only if: strategy in-regime net edge &gt; 0 over n≥30; never ease high-price/consensus.

#### Files

- `arena/portfolio.py`  
- `arena/regime_router.py` **(new)**  
- `arena/regime_map.py` (consume scores)  
- `evolution/type_alloc.py` / `evolution/ga.py`  
- `config.py`  
- `tests/unit/test_regime_allocation.py`  
- `tests/unit/test_regime_router.py` **(new)**  

#### Tasks

1. Implement score function + unit tests.  
2. Wire portfolio tilt.  
3. Wire GA type prior.  
4. Optional frequency ease behind flag default **off** until Phase 3 profiles stable.  
5. Dashboard: capital bar by regime recommendation vs applied.

#### Acceptance

- Capital weight of meanrev in `low_vol_range` ≥ weight of pure momentum when meanrev regime WR higher (given enough n).  
- No bot weight below explore floor.  
- Fill rate CV across regimes improves vs Phase 0 baseline.

#### Rollback

- Disable regime conditioning; GA type stickiness only.

---

### Phase 5 — Continuous feature residual (smooth weight surface)

**Goal:** Avoid cliff effects when features sit near thresholds; smooth adaptation within a label.

#### Design

1. After discrete profile `w0(lane, strategy, regime)`:  
   ```
   w = clip(w0 + B[lane,strategy] · F_rel, lo, hi)
   ```  
   `F_rel` = standardized vector e.g. `[vol_rel-0.5, dir-0.5, chop-0.5, flow-0.5]`  

2. `B` matrices:  
   - Initialize **0**  
   - Online update: correlation of feature with signed correctness of lane (ridge / small gradient)  
   - Caps: ‖B‖ small so residual ≤ `REGIME_CONTINUOUS_MAX_DELTA` (e.g. 0.08 weight)  

3. Only enable when `REGIME_CONTINUOUS_BLEND=True` and n_global ≥ 200 resolved attributions.

4. Persistence: `arena_state['regime_continuous_B']`.

5. Safety: residual cannot zero drift; cannot invent candidate lane weight if kill-switched.

#### Files

- `arena/regime_continuous.py` **(new)**  
- `signals/lab.py` or weight resolver  
- `config.py`  
- `tests/unit/test_regime_continuous.py` **(new)**  

#### Tasks

1. Implement residual apply + learn update on evolution loop.  
2. Feature standardization using calibrator.  
3. Dashboard: optional “continuous delta” debug.  
4. A/B: compare flap and PnL with flag on/off.

#### Acceptance

- Weight changes smoothly as vol_rel moves 0.45→0.55 without requiring label flip.  
- Residual magnitude stays within cap.  
- No regression in unit tests for discrete path when flag off.

#### Rollback

- `REGIME_CONTINUOUS_BLEND=False`.

---

### Phase 6 — Observability, measurement, gates, docs, long-term robustness

**Goal:** Operable system; no silent failure; long-term performance maintained.

#### Design

1. **Ops / dashboard**  
   - “Regime control” card: current id, conf, vol_abs, vol_rel, direction, chop  
   - Active profile weights for each strategy  
   - Fills/hour last 6h by regime  
   - Hard-skip emergency counter  
   - Calibrator health (n samples, last update)  
   - Frequency CV metric  

2. **Alerts** (`arena/alerts.py`)  
   - `regime_calibrator_stale` if no update &gt; 1h while arena live  
   - `regime_frequency_collapse` if 0 fills for 2h **and** mid-band decision opportunities &gt; N  
   - `regime_emergency_skip` when emergency hard-skip fires  

3. **Health checks**  
   - Calibrator loaded; detector live; profile resolve non-throwing  

4. **Offline validation**  
   - Extend `tools/validate_signals.py` / backtest to report **per-regime net edge** for rules  
   - `python -m backtest` regime split uses **relative** features when history available  

5. **Walk-forward protocol** (document + optional script `tools/regime_walkforward.py`):  
   - Train profiles on days 1..D-2  
   - Freeze  
   - Score day D-1..D  
   - Compare to global profile baseline  

6. **Docs**  
   - Update `CLAUDE.md` regime section  
   - Update `docs/signal-suite.md`  
   - Short operator runbook: toggles, expected frequency, how to freeze profiles  

7. **Long-term robustness checklist**  
   - Quarterly: re-estimate absolute `VOL_TYPICAL` from slow window median (or auto-slow-adapt with max 5%/week)  
   - Cap number of `by_regime` cells written  
   - Demote `by_regime` cell if live accuracy &lt; 0.48 after n≥60 (mirror lane_monitor)  
   - Never let continuous residual exceed discrete seed importance  

8. **Config inventory (final)** — all new flags listed in `config.py` with comments + env overrides where needed.

#### Files

- `dashboard/*`  
- `arena/alerts.py`  
- `arena/health.py`  
- `arena/ops_snapshot.py`  
- `tools/regime_walkforward.py` **(optional new)**  
- `CLAUDE.md`, `docs/signal-suite.md`, this PLAN status → Implemented  
- Tests for alerts/health  

#### Acceptance

- Operator can answer “why flat?” from dashboard in &lt;30s.  
- Walk-forward script runs on DB.  
- Docs match code.  
- Full success metrics from §0 measured and recorded in a soak note.

#### Rollback

- Master kill:  
  ```
  REGIME_USE_RELATIVE=False
  REGIME_PROFILE_ADAPT_ENABLED=False
  REGIME_CONTINUOUS_BLEND=False
  REGIME_HARD_SKIP_ENABLED=True   # old
  REGIME_ADAPT_SIZE_MIN=0.35
  ```

---

## 4. Cross-cutting implementation details

### 4.1 Hot-path performance

- Calibrator update: O(1) or O(log n) only.  
- No Gamma/network in regime path.  
- Profile resolve: memory dict after cache warm.  
- Budget: detector update stays ≪ 1ms typical.

### 4.2 Decision / trade stamping (required for all later phases)

Ensure every decision_event and trade has:

- `regime` / `regime_id`  
- optional JSON `regime_features` (or subset in `context`)  
- strategy_type (already)

Backfill not required; forward-only.

### 4.3 Interaction with invariant guards

| Guard | Regime interaction |
|-------|-------------------|
| HIGH_PRICE_GUARD 0.72 | **None** (no ease via regime) |
| CONSENSUS_GUARD 0.35 | **None** |
| DEAD_ZONE | Optional later: *regime-specific drift floor only* if evidence; default keep global |
| BOOK_SUM / exposure | **None** |
| Session / macro skip | **None** (orthogonal) |
| MIN_EDGE | Soft frequency ease only under Phase 4b flag |

### 4.4 Interaction with existing loops

| Loop | Change |
|------|--------|
| Evolution | Host regime tuner + optional continuous update |
| Lane monitor / promoter | Unchanged; candidates still live-shadow |
| Core tuner | Becomes multi-key (regime) |
| Learned rules | May consume regime cells later; out of scope except non-conflict |
| Hybrid meta | Keep; ensure bucket map still valid |

### 4.5 Testing strategy

| Level | Coverage |
|-------|----------|
| Unit | calibrator, classify relative, profile resolve, emergency skip, continuous residual |
| Integration | make_decision with by_regime weights; portfolio tilt on regime flip |
| Soak | 48h+ paper; metrics §0 |
| Regression | existing `test_regime_*`, `test_core_lane*`, `test_contrarian*` must pass |

### 4.6 Migration / deploy sequence

1. Deploy Phase 1 behind `REGIME_USE_RELATIVE` (default on after tests).  
2. Deploy Phase 2 config flip (expect fill rate up).  
3. Deploy Phase 3 seeds first (no tuner writes) → enable tuner.  
4. Phase 4 capital routing.  
5. Phase 5 continuous off by default → on after 200 samples.  
6. Phase 6 dashboards/alerts anytime after Phase 1.

Each step: restart arena; watch `/healthz` + regime card.

### 4.7 Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Relative window too short → chase noise | 14d primary + 60d slow; min samples |
| Relative window too long → stale | exp half-life 7d option |
| Thin per-regime cells | min n; shrink to global profile |
| Double-damping mom | single site (lab vs adapt) |
| Frequency ease → bad fills | flag off default; require +EV in-regime |
| Fragmentation | keep 5 labels; discovery cells separate |
| Dashboard process stale detector | already read-through DB; keep that pattern for calibrator |

---

## 5. Suggested implementation order (session checklist)

Use this as the future-session runbook:

- [ ] **P1.1** `regime_calibration.py` + tests  
- [ ] **P1.2** Wire relative vol/trend into detector + classify  
- [ ] **P1.3** Add chop, multi-scale align, stamp features  
- [ ] **P1.4** Persist calibrator; dashboard abs vs rel  
- [ ] **P2.1** Config defaults (hard-skip off, size min 0.85)  
- [ ] **P2.2** `adjustments()` style mode + emergency skip  
- [ ] **P2.3** Dedupe lab/adapt damps  
- [ ] **P3.1** `resolve_lane_weight` + seeds module  
- [ ] **P3.2** Wire blend path  
- [ ] **P3.3** Tuner by_regime writes  
- [ ] **P3.4** Signal Lab matrix UI  
- [ ] **P4.1** `regime_router` scores  
- [ ] **P4.2** Portfolio + GA hooks  
- [ ] **P4.3** (optional) frequency ease flag  
- [ ] **P5.1** Continuous residual module  
- [ ] **P5.2** Evolution-loop learning of B  
- [ ] **P6.1** Ops card + alerts + health  
- [ ] **P6.2** Docs + walkforward tool  
- [ ] **P6.3** Multi-day soak; fill metrics table; ship  

---

## 6. Config reference (target final set)

```python
# --- Relative calibration ---
REGIME_USE_RELATIVE = True
REGIME_REL_WINDOW_DAYS = 14
REGIME_REL_WINDOW_DAYS_SLOW = 60
REGIME_REL_MIN_SAMPLES = 500
REGIME_REL_UPDATE_HALFLIFE_DAYS = 7.0
REGIME_CLASSIFY_VOL_HI = 0.70
REGIME_CLASSIFY_VOL_LO = 0.30
REGIME_CLASSIFY_DIR_HI = 0.55
REGIME_CLASSIFY_DIR_LO = 0.40

# --- Policy ---
REGIME_ADAPT_ENABLED = True
REGIME_ADAPT_PRIMARY = "style"  # "style" | "throttle"
REGIME_ADAPT_SIZE_MIN = 0.85
REGIME_ADAPT_SIZE_MAX = 1.15
REGIME_HARD_SKIP_ENABLED = False
REGIME_HARD_SKIP_EMERGENCY_ONLY = True
REGIME_HARD_SKIP_MIN_TRADES = 80
REGIME_HARD_SKIP_WR = 0.38
REGIME_HARD_SKIP_CLEAR_WR = 0.48

# --- Per-regime profiles ---
REGIME_PROFILE_ADAPT_ENABLED = True
REGIME_PROFILE_SEEDS_ENABLED = True
CORE_TUNE_MIN_TRADES_REGIME = 40

# --- Continuous residual ---
REGIME_CONTINUOUS_BLEND = False  # enable after sample mass
REGIME_CONTINUOUS_MAX_DELTA = 0.08
REGIME_CONTINUOUS_MIN_SAMPLES = 200

# --- Frequency (optional) ---
REGIME_FREQ_TARGET_ENABLED = False
REGIME_FREQ_TARGET_FILLS_PER_HOUR = 4.0
REGIME_FREQ_EDGE_EASE_MAX = 0.15
```

---

## 7. Explicit out-of-scope (future)

- Full unsupervised discovery of new discrete ids beyond five (use regime_map cells instead).  
- Live trading enablement checklist.  
- Re-enabling `LEARNING_ENABLED`.  
- Changing HIGH_PRICE_GUARD (separate experiment).  
- True maker limit posting.

---

## 8. Definition of done (full program)

1. All six phases implemented or explicitly deferred with flags.  
2. Unit + integration tests green; no drop in existing regime tests.  
3. Paper soak ≥5 days with success metrics §0 recorded in a short soak note under `docs/` or logs.  
4. `CLAUDE.md` regime section rewritten to “adapt weights/capital; relative features; hard-skip emergency-only.”  
5. Operator can freeze profiles and revert to throttle mode via config without code revert.

---

## 9. Open decisions (resolve at implementation start if needed)

| # | Decision | Recommendation |
|---|----------|----------------|
| D1 | Percentile method (reservoir vs P²) | Reservoir of 1m vols (cap 20k) simple + testable |
| D2 | Apply structural mom damp in lab or adapt | **Lab only** |
| D3 | Frequency ease default | **Off** until Phase 3 stable |
| D4 | Include session in discrete label | **No** — routing only |
| D5 | Absolute VOL_TYPICAL slow-adapt | Phase 6 optional auto 5%/week toward median |

Record choices in this file’s changelog when implementing.

---

## 10. Changelog

| Date | Note |
|------|------|
| 2026-08-05 | Initial comprehensive PLAN (all 6 phases) for adapt-not-throttle regime system |
| 2026-08-05 | Implemented Phases 1–6 in tree: calibrator, style policy, profiles, router, continuous (flag off), alerts/ops/health/docs |

# Strategy Root-Cause Improvements — Design

**Date:** 2026-07-15
**Status:** Approved (directions), pending implementation
**Basis:** Overnight paper run — 655 trades, 559 resolved, overall P&L **−75.16**, ~62% resolved WR.

## Evidence (from `bot_arena.db`)

Decomposition by outcome: win +929.35 / loss −830.63 / **exit_sl −173.88** = −75.16.

Per-signal predictiveness (460 directional trades, "confirms chosen side → WR"):

| Signal | confirms | contradicts | verdict |
|---|---|---|---|
| CVD | 66.9% | 52.4% | **real edge (+14.5pp)** — underweighted |
| BTC mom | 63.8% | 57.4% | weak + |
| PM mom | 62.6% | 56.1% | weak + |
| OBI | 58.1% | 66.7% | **inverted (−8.6pp)** |
| learning bias | 53.5% | 77.6% | **inverted (−24pp)** |
| composite `edge` | 60.6% | 61.0% | **non-predictive** (good cancelled by bad) |

Inversions robust in both price zones (learning FAV 58.1 vs 80.9; OBI FAV 59.4 vs 77.4; CVD FAV 70.5 vs 59.5).

WR vs entry price: coin-flip **45–55¢ loses** (48.4% WR, −10.7, 157 trades); **55–75¢ is the edge** (+110); ≥75¢ break-even after fees.

Confidence anti-calibrated: conf 0.5–0.7 → 73% WR but ~$0 P&L (avg entry 0.73); conf ≥0.7 → loses.

Differentiation absent: per-strategy `analyze()` signal fires in only **37/564 trades (6.6%)** — all directional bots trade the same base stack.

Stop-loss counterfactual: holding to resolution (−150.7) barely beats stopping (−172.3); mean-rev entries are break-even even held.

## Root causes and fixes

### R1 — Anti-predictive signals poison the composite (signal weighting)
- **OBI** (resting-depth fade) is inverted → `SIGNAL_WEIGHT_OBI = 0.0`.
- **CVD** (executed aggression) is the one real flow edge → raise `SIGNAL_WEIGHT_CVD` 0.10 → **0.25**.
- **Learning bias** is inverted (−24pp) and conceptually double-counts price + overfits → **disabled in live decisions** (`config.LEARNING_ENABLED = False`); outcomes still recorded. Replaced by R5.
- BTC/PM momentum kept modest.

### R2 — Confidence manufactures fake edge at extremes (confidence calc)
`_compute_fair_yes` applies `price_tilt = (mid−0.5)·aggression·K_TILT`, which grows unbounded toward the extremes and inflates `edge`→`confidence` where the market is actually efficient (≥75¢ break-even). The empirical favorite underpricing is ~flat +4–6¢ across 55–85¢, not proportional. **Cap the tilt** at `±config.FAVORITE_EDGE_CAP` (default **0.06**) so it models the real, bounded favorite edge and stops overbetting extremes. Confidence stays `edge × EDGE_TO_CONFIDENCE`; once R1 makes `edge` predictive and R2 caps the fake edge, confidence tracks the profitable 55–75¢ band. Coin-flip suppression then falls out of the existing per-strategy `MIN_EDGE` gate applied to a *now-real* edge (no bucket ban).

### R3 — Stop-loss is the wrong tool for fully-resolving 5-min markets (core logic)
Remove the stop-loss exit: `mean_reversion_sl` variants set `exit_strategy = None` (hold to resolution). Risk is managed at entry (edge gate), not by exiting on intra-window noise. `position_monitor` then has no stop bots to act on.

### R4 — Strategies are clones (core logic / differentiation)
Each `analyze()` gates itself off (fires 6.6%) and is weighted at only 0.15. Make them **fire often and matter**: lower the internal gating so momentum / mean_reversion / sentiment / hybrid emit a directional lean on most ticks, and raise `config.STRATEGY_SIGNAL_WEIGHT` 0.15 → **0.30**. Each strategy must still be individually sound (mean_reversion stays near-neutral per prior findings). Goal: evolution selects among genuinely different theses, not clones.

### R5 — Edge-calibrated learning (redesign; separate module)
Rebuild learning to bias only where a feature bucket shows a **fee-adjusted realized edge** (`WR − avg_entry − avg_fee`) with enough samples, not raw YES-WR. Disabled behind R1 until built and validated; wired behind `config.LEARNING_ENABLED`. Own TDD cycle after R1–R4 land.

## Implementation phases (each committed, TDD)

- **Phase 1 (R1+R2+R3):** config weights + `_compute_fair_yes` tilt cap + alpha drop of OBI/learning + boost CVD + remove stop-loss. Highest leverage, best-supported, lowest risk. Tests: alpha excludes OBI/learning; tilt capped; SL bots hold.
- **Phase 2 (R4):** per-strategy `analyze()` fires often + weight up; assert each type produces distinct leans on the same input.
- **Phase 3 (R5):** new edge-calibrated learning module; enable via flag once validated.

## Validation

After Phase 1 accumulates ~50+ resolved trades, re-run the per-signal predictiveness and price-bucket queries (in this doc). Success = composite `edge` becomes predictive (confirms WR ≫ contradicts), coin-flip volume drops via the edge gate, and overall P&L turns positive. Constants (`SIGNAL_WEIGHT_CVD`, `FAVORITE_EDGE_CAP`, `STRATEGY_SIGNAL_WEIGHT`, `EDGE_TO_CONFIDENCE`) are starting points, tunable in one place.

## Out of scope
- Makers (+14.8) and arbitrage (+10.9) — already profitable, untouched.
- Numeric re-tuning beyond sane starting values (follow-up once data accumulates).

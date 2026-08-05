# Expanded Signal Suite & Validation Harness (2026-07-23)

> Parent overview: [README.md](../README.md) · promotion pipeline also in
> [strategy.md §7](../strategy.md) and [CLAUDE.md](../CLAUDE.md).

## What was added

### New pure feature modules (`signals/`)

All new modules are **pure and deterministic**: explicit inputs (price lists,
book dicts, trade tapes, datetimes) in, bounded floats out — no clocks, no
network, no module state inside `compute()`. The offline harness therefore
validates exactly the code that would ship.

| Module | Outputs | Kind |
|---|---|---|
| `signals/multiscale.py` | `ms_mom_{1,3,5,15}m` (tanh momentum per horizon), `ms_rvol_{5,15,30}m` (realized vol), `ms_atr_5m`, `ms_vol_ratio` (vol expansion) | momentum = directional candidates; vol = context |
| `signals/microstructure.py` | `micro_obi_w` (distance-weighted book imbalance), `micro_cross` (Up-vs-Down bid-support), `micro_spread`, `micro_spread_score`, `micro_depth` | imbalance = directional candidates; spread/depth = context |
| `signals/flow.py` | `flow_cvd_decay` (time-decayed, volume-floored CVD), `flow_whale` (large-print delta), `flow_rate` (tape activity) | CVD/whale = directional candidates; rate = context |
| `signals/session_features.py` | `sess_tod_sin/cos`, `sess_dow_sin/cos`, `sess_label`, `sess_nyse_prox`, `sess_weekend` | context only |
| `signals/regime.py` | `regime_trend_{10,30}`, `regime_trend`, `regime_chop` | context only |

**None of these carry live weight by default.** House rule (validate-before-weighting)
stands: directional candidates must show positive net edge in the harness AND
survive live shadow attribution via the existing lane-proposal pipeline before
any lane weight moves off zero.

### Wired candidate lanes (2026-08 audit)

These now flow through `SignalLab` + `cand(...)` logging for live shadow:

| Lane key | Source | Config kill-switch |
|---|---|---|
| `lag` | drift-implied P − YES mid residual | `SIGNAL_WEIGHT_LAG` |
| `ms_mom` | `multiscale.ms_mom_1m` | `SIGNAL_WEIGHT_MS_MOM` |
| `flow_decay` | `flow.flow_cvd_decay` | `SIGNAL_WEIGHT_FLOW_DECAY` |

Spread context (`micro_spread`) taxes `min_edge` when books are wide
(`SPREAD_EDGE_MULT_ENABLED`) — non-directional size/skip, not a side-picker.

Book/flow features cannot be fully backfilled (historical books/tape are not
archived), so their primary validation path is live `cand(...)` attribution.

### Expanded validation harness (`tools/`)

`tools/signal_validation.py` (pure, unit-tested) gained:

- **`information_coefficient`** — point-biserial correlation between signal
  value and the Up outcome. Uses magnitude, not just sign, so a lane whose
  size means something scores above a same-sign coin flip.
- **`net_edge(..., slippage=)`** — per-share EV after the canonical taker fee
  **plus a flat slippage penalty** on the (stale, optimistic) PM history mid.
  A signal that only survives at zero slippage has no real edge.
- **`decay_analysis`** — follow-WR + IC split into chronological thirds by
  market recency. Strong-old/flat-recent = the market adapted; don't weight
  it off the pooled number.
- **`regime_split`** — follow-WR / IC / slip-adjusted EV per tercile of any
  context feature (trend strength, realized vol, …): where does a signal
  actually earn?
- **`rank_signals`** — the ranked scorecard combining all of the above,
  sorted by slippage-adjusted EV.

`tools/lane_candidates.py` gained `attach_features` (backfills the multiscale
/ regime / session features onto every harness decision point using the same
Binance series the candidate lanes already fetch) plus
`FEATURE_DIRECTIONAL_KEYS` / `FEATURE_CONTEXT_KEYS`.

### New command

```bash
.venv/bin/python3 tools/validate_signals.py --markets 300 --rank
# options: --slippage 0.005 (default), --report path/to/report.md
```

`--rank` re-validates **every** signal (live lanes, candidate lanes, expanded
feature suite) against recent resolved markets and prints a ranked scorecard
(IC, follow-WR, EV at mid, EV after slippage, recent-slice WR, verdict), a
performance-decay table, and regime splits — then writes a markdown report
(default `logs/signal_report.md`). It writes **nothing** to `bot_arena.db`
(promotion still goes through `--propose` + Signal Lab + live attribution).

## Validation results (300 resolved markets, 2026-07-23)

See `logs/signal_report.md` for the full machine-written scorecard of this
run. Summary and weight decisions below.

Run: `--markets 300 --rank` → 300 markets, 1,200 decision samples, slippage
assumption 0.5c/share on top of the taker fee.

### Ranked scorecard (top of table; EV per share after fee + slippage)

| signal | n | IC | follow-WR | EV@slip | recent-third WR | verdict |
|---|---|---|---|---|---|---|
| `ms_mom_1m` | 1176 | +0.32 | 65.4% | +10.95c | 67.0% | positive edge |
| `mom2` (live mom lane) | 1145 | +0.31 | 65.2% | +10.94c | 67.3% | positive edge |
| `xasset` | 1197 | +0.29 | 62.4% | +9.36c | 64.5% | positive edge |
| `drift_prod` (live drift lane) | 1195 | **+0.54** | 73.3% | +8.25c | 78.2% | positive edge |
| `tech_mtf` | 1200 | +0.41 | 68.3% | +7.61c | 68.2% | positive edge |
| `ms_mom_3m` | 1198 | +0.41 | 68.4% | +5.27c | 71.2% | positive edge |
| `ms_mom_5m` | 1195 | +0.32 | 64.5% | +2.43c | 61.4% | positive edge |
| `tech_bb` | 1200 | +0.30 | 60.5% | +1.28c | 62.5% | positive edge |
| `pm_mom` | 886 | +0.37 | 70.1% | +0.86c | 75.0% | thin — priced in |
| `ms_mom_15m` | 1200 | +0.22 | 58.1% | +0.63c | 58.8% | thin |
| `fut_funding` | 1200 | −0.02 | 52.3% | +0.05c | 44.0% | no signal |
| `tech_macd` | 1200 | +0.20 | 54.8% | −3.43c | 51.5% | **no edge** |
| `fut_oi` | 1200 | +0.00 | 48.7% | −3.64c | 55.0% | **no edge** |
| `fut_taker` | 1200 | −0.20 | 41.0% | −8.61c | 39.0% | **INVERTED** |

### Performance decay

No decay on any keeper — every top signal is *stronger* in the recent third
than the oldest third (drift 78.2% recent vs 66.3% oldest; mom 67.3% vs
61.3%; xasset 64.5% vs 60.5%). Nothing is being kept on stale evidence.

### Regime-specific value

- `drift_prod` earns everywhere but is best in **trending** (+10.6c vs +6.5c
  mid-trend) and **high-vol** tape (+10.4c vs +4.9c low-vol) — consistent
  with the drift-conditional dead-zone gate (flat drift + coin-flip price =
  skip).
- `mom2` is remarkably regime-stable (+9.7c to +12.0c across trend terciles),
  best in high-vol (+13.6c) — supporting the existing quiet-regime damp
  (`MOM_QUIET_REGIME_DAMP`) rather than contradicting it: low-vol is its
  weakest vol bucket (+7.7c).

### Interpretation caveats

- Harness EVs are **optimistic upper bounds** (stale PM history mids;
  adverse selection invisible). The 24h v5 run proved this concretely: the
  harness read tech at 74–80% follow-WR, live it scored 51.7% and was
  auto-demoted; fut likewise (52.6%). Use the table for *ordering and sign*;
  live shadow attribution remains the judge.
- `fut_taker` reads **inverted** even in the harness now — its kill-switch
  and live auto-demotion are both confirmed. `tech_macd` is predictive but
  loses money after costs (the classic pm_mom trap).
- The harness would nominate `tech` and `xasset` for proposals; xasset's
  live attribution has stayed healthy (57.2%), tech's has not (51.7%). The
  auto-validation scheduler + lane promoter already run this loop — no
  manual weight change is warranted from this run.

## SignalLab (`signals/lab.py`, 2026-07-23)

Central signal service every bot now goes through (distinct from the
dashboard's "Signal Lab" tab, which is the human approval UI for lane
proposals — the class is the runtime engine behind the same pipeline):

- **Consistent fetch + cache** — `SignalLab.compute_lanes(market, signals)`
  turns one combined-signals dict into the normalized lanes
  (drift/mom/pm/cvd/obi/fut/tech/xasset) with kill-switches, approved-lane
  overrides and the quiet-regime momentum damp applied. Cached one warmer
  tick on a **value key** (never object identity — a recycled dict id must
  not serve stale lanes), so all 8 bots deciding on a tick share one
  computation and see identical values. `raw` output preserves
  pre-kill-switch reads for feature extraction and the `cand(...)`
  validation log.
- **Dynamic weighting** — `blend(strategy_type, lanes, profile)` merges the
  per-strategy profile with DB lane overrides (the tuner/promoter closed
  loop = the performance-based half) and `REGIME_LANE_DAMP` (regime-based
  half). `set_model_hook(fn)` is the documented seam for a light ML model
  later: `fn(strategy_type, lanes, weights) -> prob | None`, linear
  attribution still logged either way, exceptions can never stall a tick.
- **Validation gating** — `gated_lanes()` reads the live lane-monitor report
  (arena_state `lane_monitor`) and zero-weights any lane whose verdict is
  `disabled` or whose live accuracy sits under the demotion bar at full
  sample — defense-in-depth on top of `db.disable_lane_override`.
- **Clean API** — bots read signals through `SignalView` (typed accessors,
  still a `Mapping` so dict-passing tests/callers are untouched) and
  probabilities through `BaseBot._model_prob_yes` → `lab.blend`. All
  ad-hoc `signals.get(...)` access was removed from every strategy bot
  (momentum, mean-rev, phantom, sentiment, sniper, hybrid, all three
  makers).
- **Contribution logging** — every blend logs per-lane `weight × value` at
  debug and `make_decision` embeds `P=0.xxx[drift=+0.150 ...]` in the
  persisted trade reasoning; buy decisions also carry a
  `lane_contributions` dict. The `drift=`/`mom=`/`strat=`/`cand(...)`
  reasoning tokens parsed by `arena/core_lane_tuner.py` and
  `arena/lane_monitor.py` are unchanged (load-bearing contract).
- **Hybrid** — its meta-learner now scores sub-strategy performance via the
  shared, pure `SignalLab.score_perf_tilts` (fetch + per-instance cache stay
  in the bot so its tests keep stubbing `bots.bot_hybrid.db`).

Decision POLICY (guards, gates, Kelly sizing, side selection) deliberately
stays in `BaseBot.make_decision` — the lab only answers *what the signals
say and how much each counts for this strategy*.

## Weight policy applied

- **drift (`drift_prod`) and BTC momentum (`mom2`)** remain the only weighted
  price lanes — they are the only directional signals clearing positive
  net edge after fees + slippage with a healthy recent slice.
- **Every other directional signal stays at weight 0** (kill-switched or
  candidate): promotion happens only through the lane-proposal pipeline
  (harness nominates → live shadow attribution judges → auto/human approve),
  never directly from a harness run.
- Context features (regime/vol/session) carry no direction by construction;
  their job is conditioning (e.g. the regime splits above justify the
  existing drift-conditional dead-zone gate and quiet-regime momentum damp).

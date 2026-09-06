# Strategy Lab cycle (formerly Desk / Trading Floor)

> **Phase 3 (2026-09-05):** The `desk/` package and Floor UI (`/floor`,
> `dashboard/desk_*`) were deleted. The invent → gate → graduate loop lives in
> **`signals/strategy_pipeline/`** and the dashboard **Lab → Strategies** pane
> (`/api/lab/pipeline/*`).
>
> **Phase 4:** Learning is unified in `learning_spine.py` (durable constraints
> in arena_state `lab_learning_spine`). Optional LLM assist lives in `llm.py`.
> See `_refs/PHASE4_NOTES.md`.

This is the six-stage loop mapped onto the existing arena.

The hot path (1s trader tick, fills, risk, kill switch) stays deterministic.
LLMs (via `STRATEGY_LAB_LLM_PROVIDER`) are optional *research labor*. They never
place orders and never emit Python that is `exec`'d. With provider `none` the
Lab loop is fully self-sufficient (analyse → test → learn → improve).

## Stages

| Stage | Owner | What happens |
|-------|--------|----------------|
| 1 Research | `signals/strategy_pipeline/research.py` | Load spine `get_constraints()`, recent trades, regimes, autopsies, universe. Emit `StrategySpec` JSON (new **params genomes**, not lane weights). Skip dead fingerprints; bias mutate away from avoid bands; prefer positive factor cells. |
| 2 Code | `signals/strategy_pipeline/compiler.py` | Bind spec to an existing primitive (`momentum`, `mean_reversion`, `sniper`, …) + params. Lane weights on the spec are thesis notes only. |
| 3 Backtest | `signals/strategy_pipeline/cycle.py` + `evolution/backtest_gate.py` | Replay resolved history. Must clear Lab backtest mins (`STRATEGY_LAB_BACKTEST_MIN_*`). Replay-without-crash is not a pass. |
| 4 Paper / shadow | existing paper venue | Candidate trades real books with `trading_mode=paper` when paper slots allow. |
| 4b Ready | Lab **Promote** | After promote bars (`STRATEGY_LAB_PROMOTE_MIN_*`) and paper P&L > 0, the spec sits in `ready`. Auto-live stays off unless `STRATEGY_LAB_AUTO_PROMOTE`. |
| 5 Live | existing live venue + GA | Graduated bots compete. GA still culls losers; culls write spine autopsies. Founders stay protected. |
| 6 Post-mortem + fine-tune | `postmortem.py` → `learning_spine.py` | Every death writes a structured autopsy (fingerprint, factor cells, avoid constraints). Research loads those constraints next tick. Optional LLM narrative is storage-only. |

## Learning spine vs Learned Trade Rules

| Surface | State key | Role |
|---------|-----------|------|
| Lab spine | `lab_learning_spine` | Genome avoid fingerprints / param bands + prefer/avoid factor cells for research |
| Lab Signals rules | `learned_trade_rules` | Hot-path skip/go/size from decision_events |

Shared cell vocabulary: `regime|price_band|drift_band|side[|strategy_type]`
(see `_refs/PHASE4_NOTES.md`). Spine folds rules via `fold_learned_rules()`
without changing the existing mine path.

## Ownership (do not let three writers share fields)

| Writer | Owns | Must not write |
|--------|------|----------------|
| Lab (`strategy_pipeline`) | New genomes (`strategy_params` on a new bot) | Live `lane_overrides` / core-lane blend |
| Core-lane tuner | Live lane mix per `strategy_type` | Strategy params / new bot names |
| GA / evolution | Mutate/cull of non-founder roster params | Spec `lane_weights` as live overlay |

Spec `lane_weights` are a research note. They are not applied at compile or
restart. That is the three-writer rule under Lab naming.

## Config

See `STRATEGY_LAB_*` in `config.py` / `.env.example`. Empty roster always uses
`arena.startup.build_default_bots()` (founders / DEFAULT_INDICES) — Lab does not
own the lean fallback path.

Optional LLM: `STRATEGY_LAB_LLM_PROVIDER=none|ollama|grok` plus `OLLAMA_*` or
`XAI_API_KEY` / `XAI_MODEL`. Overlay from Settings → Strategy Lab wins.

## Host

`LabHost` starts from `arena.py` when `STRATEGY_LAB_ENABLED` is true.
Dashboard: Lab → Strategies. API: `/api/lab/pipeline/*`.

Orphan SQLite tables `desk_hypotheses` / `desk_events` may remain; Lab uses
`lab_hypotheses` / `lab_events` only.

# Desk cycle — research → code → backtest → paper → live → post-mortem → fine-tune

This is the six-stage loop from the antpalkin map, mapped onto the existing
arena instead of a cloud agent swarm.

The hot path (1s trader tick, fills, risk, kill switch) stays deterministic.
LLMs (local Ollama or xAI Grok API) are optional *research labor*. They never
place orders and never emit Python that is `exec`'d.

## Stages

| Stage | Owner | What happens |
|-------|--------|----------------|
| 1 Research | `desk/research.py` | Read recent trades, regimes, autopsies, universe. Emit `StrategySpec` JSON. |
| 2 Code | `desk/compiler.py` | Bind spec to an existing primitive (`momentum`, `mean_reversion`, `sniper`, …) + params + lane weights. |
| 3 Backtest | `desk/cycle.py` + `evolution/backtest_gate.py` | Replay resolved history. Losers die here and get an autopsy. |
| 4 Paper / shadow | existing paper venue | Candidate trades real books with `trading_mode=paper`. Promote after **100 resolved trades OR 7 days**, with a floor of 30 trades. |
| 5 Live | existing live venue + GA | Graduated bots compete. GA still culls losers. |
| 6 Post-mortem + fine-tune | `desk/postmortem.py` | Every death (backtest / paper / GA) writes an autopsy. Research reads the graph so the next spec does not repeat the same failure. |

## Why primitives, not generated source

5-minute binary markets punish wrong strike, mid-priced decisions, and
uncapped size. Those bugs are already encoded in `bots/` + guards. A generated
`analyze()` that bypasses them would reintroduce paid-for loss modes.

A `StrategySpec` is therefore a **parameterized primitive**, not a new language.
Future work can add a sandboxed rule DSL; it still has to call `make_decision`.

## Factory mode vs lean-6

`DESK_FACTORY_MODE=True` (default off until you flip it):

- Empty DB does **not** auto-launch momentum/meanrev/sniper/hybrid/arb/sweeper.
- The desk host researches, backtests, and deploys paper candidates via the
  existing mid-run deploy queue.
- Continue-from-DB still resumes whatever is already active.

Flip it only after you have watched one paper desk cycle.

## Market universe (stepwise)

`CRYPTO_UNIVERSE_PHASE` in `config.py`:

1. Current: Polymarket BTC 5m + Kalshi BTC 15m.
2. Major-coin short windows (BTC/ETH/SOL/XRP × 5m/15m/1h where the series exists).
3. All crypto binary prediction markets on both venues (discovery widen).

Phase 2–3 change *which windows are discovered*. They do not change edge math.
Each new series needs a correct strike/settlement adapter before it is traded.

## LLM providers

```
DESK_LLM_PROVIDER=none|ollama|grok
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1
XAI_API_KEY=...
XAI_MODEL=grok-4
```

`none` is a full heuristic researcher (regime + autopsy + primitive catalog).
Use that on the $200 stack. Point `ollama` at the Jetson/Umbrel box when you
want richer theses. Grok API is optional and off the execution path.

## Floor UI

Dashboard tab **Floor** (`/api/desk/floor`) shows the seven roles and the
hypothesis pipeline in real time. Roles are *views* of this process, not
separate cloud computers.

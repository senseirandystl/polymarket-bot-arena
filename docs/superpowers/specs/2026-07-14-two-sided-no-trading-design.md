# Two-Sided (YES/NO) Net-Edge Side Selection — Design

**Date:** 2026-07-14
**Status:** Approved (design), pending implementation
**Scope:** `bots/base_bot.py` (`make_decision`), `config.py`, tests

## Problem

Every directional bot currently trades **YES only**. `make_decision` collapses
the signal stack into a single scalar `combined`, sets `side = "yes" if combined
> 0 else "no"`, and then a blanket **NO ban** (`if side == "no": skip`) discards
every NO lean. This forfeits the entire NO side of the book — a large amount of
opportunity, especially during market upswings where the NO side may be the
better cost-adjusted buy.

The NO decision machinery already exists and is genuinely strategy-driven:
- Each strategy's `analyze()` natively returns NO on its own logic (momentum on
  `pct_change < 0`, mean_reversion on overbought RSI > 70, sentiment on bearish,
  hybrid on a negative weighted vote).
- `venues/paper.py` / `venues/live.py` already fill against the NO token/book at
  the NO price when `side == "no"`.
- `learning.record_outcome` already handles `side == "no"` (a NO win increments
  the YES-loss counter, keeping the learned bias a coherent YES-probability);
  `get_learned_bias` returns a YES-bias that leans NO when < 0.5.
- `market["no_price"]` is populated by the warmer (`arena/market_data.py`) and
  `polymarket_markets.refresh_price`, and threaded onto the hot-path market dict
  by `arena/trader.py:127-129`.

Only `make_decision` blocks NO, and its guards/sizing are YES-centric.

## Goal

Allow all directional bots to trade **either** side, choosing per-market via a
full **two-sided net-edge comparison** grounded in each side's real book price
and fee — never random, never a mechanical mirror of the YES decision. A
directional bot takes at most one side per market; **arbitrage** remains the only
two-legged bot and is untouched.

## Design

### 1. Fair value from the existing signal stack (remap, not rewrite)

Keep every existing signal and weight; reinterpret their sum as a probability:

```
price_tilt = (yes_mid − 0.5) × aggression × K_TILT      # favorite-following (the 0.50 lane)
alpha      = momentum_signal + pm_momentum_signal
           + strategy_signal + obi_signal + cvd_signal
           + learning_signal × learning_weight          # secondary lanes (unchanged weights)
fair_yes   = clamp(yes_mid + price_tilt + alpha, 0.02, 0.98)
```

- `aggression` = existing `MARKET_PRICE_AGGRESSION[strategy_type]` (unchanged).
- `K_TILT` (new `config` constant) scales the favorite-following tilt. It
  replaces the old hard-coded `price_edge × 0.50` lane weight; chosen so default
  behavior stays close to today's tuning. **Default `K_TILT = 0.5`** so that at
  `aggression = 1.0`, `price_tilt = (yes_mid − 0.5) × 0.5` — the same magnitude
  as the old `price_edge × 0.50` term.
- `alpha` reuses the already-clamped secondary-lane signals verbatim; no lane is
  discarded. (Their previous outer weights — `momentum_signal × 0.15`, etc. —
  are folded in: the individual signals are already clamped to small bands, so
  summing them directly keeps `alpha` in a comparable range. See §7 for the
  exact reconciliation the implementer must preserve.)

`price_tilt` retains the "market price is the strongest signal / favorites are
underpriced" behavior the arena is tuned around: it pushes `fair_yes` *further*
from 0.5 in the market's own direction, scaled per-strategy.

### 2. Two-sided edge and side selection

```
yes_price = yes_mid                       # market["current_price"] (coalesced to 0.5 if None)
no_price  = market["no_price"]            # real NO book mid; falls back to 1 − yes only upstream

fee_yes = polymarket_fills.taker_fee(1.0, yes_price)   # per-share fee, single source of truth
fee_no  = polymarket_fills.taker_fee(1.0, no_price)

edge_yes = fair_yes        − yes_price − fee_yes
edge_no  = (1 − fair_yes)  − no_price  − fee_no

if edge_yes >= edge_no:  side, side_price, chosen_edge = "yes", yes_price, edge_yes
else:                    side, side_price, chosen_edge = "no",  no_price,  edge_no
```

- Fee is per-share via `taker_fee(1.0, price)` — never re-derived (BUG_HISTORY #17).
- When the two mids are exactly complementary (`no_price == 1 − yes_price`) the
  comparison reduces to the directional sign — correct for a binary market. The
  two-sided value materializes precisely when the NO book mid diverges from
  `1 − yes_mid`.
- `argmax` picks exactly one side; a directional bot never holds both sides.

### 3. Minimum-edge gate (replaces the confidence floor for entry)

```
min_edge = MIN_EDGE.get(strategy_type, MIN_EDGE_DEFAULT)
if chosen_edge < min_edge:  → skip ("no edge")
```

`MIN_EDGE` is a new per-strategy `config` map (mirrors the shape of
`MIN_TRADE_CONFIDENCE`). Edge is in probability/price units (e.g. `0.02` = 2¢ of
cost-adjusted edge). The old `MIN_TRADE_CONFIDENCE` gate is superseded for entry;
`confidence` is still computed (see §5) for sizing and reporting.

### 4. Symmetric guards (keyed on the chosen side's price)

The current guards reference the YES price only. Reframe to `side_price` so they
protect both sides identically:

- **High-price guard:** `side_price > HIGH_PRICE_GUARD` (default `0.72`) → skip
  (pay ~75¢ to win ~25¢ — bad risk/reward on either side).
- **Consensus guard:** `side_price < CONSENSUS_GUARD` (default `0.35`) → skip
  (don't fight strong market consensus on either side).

Net: a bot only buys a side priced ~`0.35`–`0.72`. The **blanket NO ban is
deleted**. Both thresholds become `config` constants.

### 5. Sizing, confidence, entry price (side-aware)

- `confidence = min(0.95, chosen_edge × EDGE_TO_CONFIDENCE)` (new constant;
  default chosen so a ~0.10 edge → ~0.45 sizing confidence). Late-window boost
  (`time_rem < 60 → × 1.25`) unchanged.
- Shares-first sizing block: `price` becomes **`side_price`** (not always
  `market_price`). `max_shares`, `target_shares`, `amount`, and `entry_price`
  are all computed against the side actually being bought. Cap at 0.45 for
  sizing, 5-share minimum floor — all unchanged.
- The returned signal's `side`, `entry_price`, `suggested_amount`,
  `target_shares`, and `confidence` all reflect the chosen side.

### 6. Downstream (already correct — no change)

- `venues/paper.py` / `venues/live.py`: pick token/book by `side`, walk that
  book, apply the slippage limit against `entry_price` — already side-correct.
- `learning`: `record_outcome` / `get_learned_bias` already handle NO;
  `features` continue to use `yes_mid` as market state (a market fact,
  side-independent).
- `arena/trader.py`: already threads `no_price` onto the hot-path market dict.

### 7. Signal-reconciliation constraint (implementer MUST preserve)

The old model: `combined = price_edge×0.50 + momentum×0.15 + pm×0.10 +
strategy×0.15 + obi×W_obi + cvd×W_cvd + learning×w`. In the new model these same
terms are partitioned into `price_tilt` (the `price_edge` lane) and `alpha` (the
rest). The implementer must keep each secondary lane's contribution to `alpha`
at the **same effective weight it has today** (i.e. carry the `× 0.15`, `× 0.10`,
`× SIGNAL_WEIGHT_OBI`, `× SIGNAL_WEIGHT_CVD`, `× learning_weight` factors into
`alpha`), so lifting NO does not silently re-tune the YES behavior. Only the
price lane changes representation (`× 0.50` → `× K_TILT` inside `price_tilt`),
and `K_TILT = 0.5` keeps it numerically equivalent at `aggression = 1.0`.

## New config constants

| Constant | Default | Purpose |
|---|---|---|
| `K_TILT` | `0.5` | Favorite-following tilt scale (replaces the `price_edge × 0.50` lane) |
| `MIN_EDGE` | `{momentum:0.015, mean_reversion:0.02, sentiment:0.02, hybrid:0.02, ...}` | Per-strategy minimum cost-adjusted edge to trade |
| `MIN_EDGE_DEFAULT` | `0.02` | Fallback min edge |
| `EDGE_TO_CONFIDENCE` | `4.5` | Maps edge → sizing confidence (≈0.10 edge → 0.45) |
| `HIGH_PRICE_GUARD` | `0.72` | Skip a side priced above this |
| `CONSENSUS_GUARD` | `0.35` | Skip a side priced below this |

(Defaults are starting points, tunable in one place; final values validated
against live data after accumulation.)

## Testing (TDD)

New/updated tests in `tests/` (pytest):

1. **NO chosen when NO book underpriced** — `no_price` set well below
   `1 − yes_price` so `edge_no > edge_yes` → `side == "no"`.
2. **YES still chosen on favorite/upswing** — high `yes_mid` + bullish alpha →
   `side == "yes"` (no regression of the tuned favorite-following behavior).
3. **Both-negative → skip** — mids complementary, no alpha → `chosen_edge <
   min_edge` → `action == "skip"`.
4. **NO sized/priced against the NO book** — a NO decision's `entry_price` and
   `suggested_amount` derive from `no_price`, not `yes_price`.
5. **Symmetric guards fire on the NO price** — `no_price > HIGH_PRICE_GUARD`
   skips; `no_price < CONSENSUS_GUARD` skips.
6. **Complementary mids reduce to the sign decision** — with
   `no_price == 1 − yes_price`, chosen side matches `sign(fair_yes − yes_price)`.
7. **NO ban is gone** — a strong NO lean with a mid-band NO price produces
   `action == "buy", side == "no"`.
8. **YES-behavior parity** — a representative YES scenario yields the same side
   (and comparable sizing) as before the change, guarding the §7 reconciliation.

## Out of scope

- Arbitrage bot (two-legged; untouched).
- Maker bots (override `make_decision`; untouched).
- Evolution, dashboard, discovery, venues, learning internals (already
  NO-capable).
- Retuning of the numeric defaults beyond sane starting values (a follow-up once
  live NO data accumulates).

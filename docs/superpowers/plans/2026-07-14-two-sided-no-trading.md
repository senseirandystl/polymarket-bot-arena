# Two-Sided (YES/NO) Net-Edge Side Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let every directional bot trade YES *or* NO, chosen per-market by a full two-sided net-edge comparison grounded in each side's real book price and fee.

**Architecture:** Reinterpret the existing signal stack as a fair YES probability (`fair_yes = yes_mid + combined`, where `combined` is the current signal sum). Compute cost-adjusted edge on each side using that side's own mid price and per-share taker fee, take the side with the larger positive edge, and gate on a per-strategy minimum edge. Guards and sizing key off the chosen side's price. All changes live in `bots/base_bot.make_decision` (via two new pure helpers) plus `config.py`; strategies, venues, and learning already support NO.

**Tech Stack:** Python 3, pytest, SQLite. Fee math via `polymarket_fills.taker_fee` (single source of truth).

## Global Constraints

- Fee math MUST come from `polymarket_fills.taker_fee(shares, price)` — never re-derived (BUG_HISTORY #17).
- No signal or weight may be silently re-tuned: each secondary lane keeps its current effective weight inside `alpha`, and `K_TILT = 0.5` keeps the price lane numerically equal to the old `price_edge × 0.50` at `aggression = 1.0`. Identity to preserve: `price_tilt + alpha == combined_old`.
- A directional bot takes at most one side per market. Arbitrage (two-legged) and maker bots (override `make_decision`) are untouched.
- Per-strategy tunables follow the existing pattern: a class-attribute dict on `BaseBot` (like `MIN_TRADE_CONFIDENCE`); global scalars live in `config.py`.
- Prices/edges are in probability units (0–1). `market["current_price"]` may be `None` → coalesce to `0.5`; `market["no_price"]` may be `None` → coalesce to `1 - yes_price`.

---

### Task 1: Config constants + pure edge/fair-value helpers

**Files:**
- Modify: `config.py` (add constants near `SIGNAL_WEIGHT_CVD`, line ~114)
- Modify: `bots/base_bot.py` (add `import polymarket_fills`; add `MIN_EDGE` class dict; add `_compute_fair_yes` and `_side_net_edges` methods)
- Test: `tests/test_two_sided.py` (new)

**Interfaces:**
- Consumes: `config.K_TILT`, `config.POLYMARKET_TAKER_FEE_RATE`, `polymarket_fills.taker_fee(shares, price)`.
- Produces:
  - `config.K_TILT: float`, `config.MIN_EDGE_DEFAULT: float`, `config.EDGE_TO_CONFIDENCE: float`, `config.HIGH_PRICE_GUARD: float`, `config.CONSENSUS_GUARD: float`
  - `BaseBot.MIN_EDGE: dict[str, float]`
  - `BaseBot._compute_fair_yes(self, yes_mid: float, aggression: float, alpha: float) -> float` — returns `clamp(yes_mid + (yes_mid-0.5)*aggression*K_TILT + alpha, 0.02, 0.98)`
  - `BaseBot._side_net_edges(self, fair_yes: float, yes_price: float, no_price: float) -> tuple[float, float]` — returns `(edge_yes, edge_no)`, each `= prob - price - taker_fee(1.0, price)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_two_sided.py`:

```python
import polymarket_fills
from bots.bot_momentum import MomentumBot


def _bot():
    # MomentumBot constructs with a name/generation; adjust if the ctor differs.
    return MomentumBot(name="momentum-test", generation=0)


def test_compute_fair_yes_identity_with_combined():
    # fair_yes = yes_mid + price_tilt + alpha; with aggression=1, K_TILT=0.5,
    # price_tilt = (0.60-0.5)*1*0.5 = 0.05. alpha=0.02 → fair = 0.60+0.05+0.02 = 0.67
    bot = _bot()
    fair = bot._compute_fair_yes(0.60, 1.0, 0.02)
    assert abs(fair - 0.67) < 1e-9


def test_compute_fair_yes_clamped():
    bot = _bot()
    assert bot._compute_fair_yes(0.98, 2.0, 0.5) <= 0.98
    assert bot._compute_fair_yes(0.02, 2.0, -0.5) >= 0.02


def test_side_net_edges_complementary_is_mirror():
    # When no_price == 1 - yes_price, the pre-fee edges are exact negatives.
    bot = _bot()
    yes_price, no_price = 0.55, 0.45
    fair_yes = 0.62
    edge_yes, edge_no = bot._side_net_edges(fair_yes, yes_price, no_price)
    fee_y = polymarket_fills.taker_fee(1.0, yes_price)
    fee_n = polymarket_fills.taker_fee(1.0, no_price)
    assert abs(edge_yes - (fair_yes - yes_price - fee_y)) < 1e-9
    assert abs(edge_no - ((1 - fair_yes) - no_price - fee_n)) < 1e-9
    # pre-fee mirror
    assert abs((fair_yes - yes_price) + ((1 - fair_yes) - no_price)) < 1e-9


def test_side_net_edges_no_book_divergence_favors_no():
    # NO book underpriced vs 1-yes → NO edge beats YES edge even at fair_yes>0.5
    bot = _bot()
    fair_yes = 0.55            # market mildly favors YES
    yes_price = 0.55           # YES fairly priced → ~zero edge before fee
    no_price = 0.38            # NO cheap (1-yes would be 0.45) → real NO edge
    edge_yes, edge_no = bot._side_net_edges(fair_yes, yes_price, no_price)
    assert edge_no > edge_yes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_two_sided.py -v`
Expected: FAIL — `AttributeError: 'MomentumBot' object has no attribute '_compute_fair_yes'` (and `_side_net_edges`). If the `MomentumBot(...)` ctor signature differs, fix `_bot()` first using the real signature (grep `def __init__` in `bots/base_bot.py`) — the ctor is not the unit under test.

- [ ] **Step 3: Add config constants**

In `config.py`, immediately after the `SIGNAL_WEIGHT_CVD = 0.10` line (~114):

```python
# --- Two-sided (YES/NO) net-edge side selection ---
# Favorite-following tilt scale. Replaces the old hard-coded price_edge * 0.50
# lane weight; 0.5 keeps it numerically identical at aggression == 1.0.
K_TILT = 0.5
# Fallback minimum cost-adjusted edge (probability units) to place a trade.
MIN_EDGE_DEFAULT = 0.02
# Maps the chosen side's edge → sizing confidence (~0.10 edge → 0.45 cap).
EDGE_TO_CONFIDENCE = 4.5
# A bot never buys a side priced above HIGH_PRICE_GUARD (bad risk/reward) or
# below CONSENSUS_GUARD (fighting strong market consensus). Symmetric per side.
HIGH_PRICE_GUARD = 0.72
CONSENSUS_GUARD = 0.35
```

- [ ] **Step 4: Add import + MIN_EDGE dict + helpers to `bots/base_bot.py`**

Add the import near the other top-level imports (after `import learning`, line ~15):

```python
import polymarket_fills
```

Add the `MIN_EDGE` class dict directly after the `MIN_TRADE_CONFIDENCE` dict (~line 72), mirroring its shape:

```python
    # Minimum cost-adjusted edge (probability units) to place a trade. Two-sided
    # selection buys the side with the larger positive edge above this floor.
    MIN_EDGE = {
        "momentum": 0.015,
        "mean_reversion": 0.02,
        "mean_reversion_sl": 0.02,
        "mean_reversion_tp": 0.02,
        "sniper": 0.02,
        "phantom": 0.015,
        "sentiment": 0.02,
        "hybrid": 0.02,
    }
```

Add the two helper methods to `BaseBot` (place them just above `make_decision`):

```python
    def _compute_fair_yes(self, yes_mid: float, aggression: float,
                          alpha: float) -> float:
        """Fair YES probability from the signal stack.

        fair = yes_mid + price_tilt + alpha, where price_tilt is the
        favorite-following lane ((yes_mid-0.5) * aggression * K_TILT) and alpha
        is the summed secondary lanes. Clamped to [0.02, 0.98].
        """
        price_tilt = (yes_mid - 0.5) * aggression * config.K_TILT
        return max(0.02, min(0.98, yes_mid + price_tilt + alpha))

    def _side_net_edges(self, fair_yes: float, yes_price: float,
                        no_price: float) -> tuple:
        """Cost-adjusted edge on each side: prob - price - per-share fee.

        Fee is the canonical taker fee for one share at that side's price.
        """
        edge_yes = fair_yes - yes_price - polymarket_fills.taker_fee(1.0, yes_price)
        edge_no = (1.0 - fair_yes) - no_price - polymarket_fills.taker_fee(1.0, no_price)
        return edge_yes, edge_no
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/test_two_sided.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add config.py bots/base_bot.py tests/test_two_sided.py
git commit -m "feat(bots): add two-sided fair-value + net-edge helpers and config"
```

---

### Task 2: Rewrite `make_decision` side selection, guards, and sizing

**Files:**
- Modify: `bots/base_bot.py` — `make_decision`, lines ~191-315 (from the `# --- Combine all signals ---` block through the returned buy signal)
- Test: `tests/test_two_sided.py` (extend)

**Interfaces:**
- Consumes: `_compute_fair_yes`, `_side_net_edges` (Task 1); `MIN_EDGE`, `config.MIN_EDGE_DEFAULT`, `config.EDGE_TO_CONFIDENCE`, `config.HIGH_PRICE_GUARD`, `config.CONSENSUS_GUARD`.
- Produces: `make_decision(market, signals) -> dict` now returns `side` ∈ {`"yes"`,`"no"`} with `entry_price`/`suggested_amount`/`target_shares`/`confidence` all computed against the chosen side's price. The blanket NO ban is deleted.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_two_sided.py`. Build a minimal `market`/`signals` and drive `make_decision` directly:

```python
def _market(yes=0.55, no=None):
    return {
        "id": "mkt-1",
        "current_price": yes,
        "no_price": (1 - yes) if no is None else no,
        "polymarket_token_id": "yes-tok",
        "polymarket_no_token_id": "no-tok",
        "time_remaining_seconds": 180,
    }


def _signals(**over):
    base = {
        "prices": [100.0, 100.0], "latest": 100.0, "orderflow": {},
        "pm_momentum": 0.0, "obi": 0.0, "cvd": 0.0,
    }
    base.update(over)
    return base


def test_no_ban_is_gone_strong_no_lean_buys_no():
    # Bearish alpha (negative pm/obi/cvd) at a mid-band NO price → buy NO.
    bot = _bot()
    m = _market(yes=0.55, no=0.44)          # NO in tradeable band
    s = _signals(pm_momentum=-0.15, obi=-1.0, cvd=-1.0)
    d = bot.make_decision(m, s)
    assert d["action"] == "buy"
    assert d["side"] == "no"


def test_no_trade_sizes_against_no_price():
    bot = _bot()
    m = _market(yes=0.55, no=0.44)
    s = _signals(pm_momentum=-0.15, obi=-1.0, cvd=-1.0)
    d = bot.make_decision(m, s)
    assert d["side"] == "no"
    assert abs(d["entry_price"] - 0.44) < 1e-6   # priced off NO book, not YES


def test_high_price_guard_fires_on_no_price():
    bot = _bot()
    m = _market(yes=0.20, no=0.80)          # NO expensive
    s = _signals(pm_momentum=-0.15, obi=-1.0, cvd=-1.0)
    d = bot.make_decision(m, s)
    assert d["action"] == "skip"


def test_consensus_guard_fires_on_low_side_price():
    bot = _bot()
    m = _market(yes=0.80, no=0.20)          # NO cheap = fighting consensus
    s = _signals(pm_momentum=-0.15, obi=-1.0, cvd=-1.0)
    d = bot.make_decision(m, s)
    assert d["action"] == "skip"


def test_no_edge_skips():
    # Complementary mids, neutral alpha → edge below floor → skip.
    bot = _bot()
    m = _market(yes=0.50)
    s = _signals()
    d = bot.make_decision(m, s)
    assert d["action"] == "skip"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/test_two_sided.py -k "no_ban or sizes_against or guard or no_edge" -v`
Expected: FAIL — current code still bans NO / sizes off `market_price` (e.g. `test_no_ban_is_gone...` returns `action == "skip"`).

- [ ] **Step 3: Replace the combine/side/guards block**

In `make_decision`, replace everything from `# --- Combine all signals ---` (line ~191) down to and including the `min_conf`/low-confidence skip block (line ~256) with:

```python
        # --- Fair value: reinterpret the signal stack as a YES probability ---
        # alpha keeps each secondary lane at its existing effective weight; the
        # price lane moves into _compute_fair_yes via K_TILT. Identity preserved:
        # price_tilt + alpha == the old `combined`.
        alpha = (
            momentum_signal * 0.15 +
            pm_momentum_signal * 0.10 +
            strategy_signal * 0.15 +
            obi_signal * config.SIGNAL_WEIGHT_OBI +
            cvd_signal * config.SIGNAL_WEIGHT_CVD +
            learning_signal * learning_weight
        )
        fair_yes = self._compute_fair_yes(market_price, aggression, alpha)

        # --- Two-sided net edge: buy the side with the larger positive edge ---
        yes_price = market_price
        no_price = market.get("no_price")
        if no_price is None:
            no_price = round(1.0 - yes_price, 4)
        edge_yes, edge_no = self._side_net_edges(fair_yes, yes_price, no_price)
        if edge_yes >= edge_no:
            side, side_price, chosen_edge = "yes", yes_price, edge_yes
        else:
            side, side_price, chosen_edge = "no", no_price, edge_no

        confidence = min(0.95, max(0.0, chosen_edge) * config.EDGE_TO_CONFIDENCE)

        # --- Minimum-edge gate (no edge = no bet) ---
        min_edge = self.MIN_EDGE.get(self.strategy_type, config.MIN_EDGE_DEFAULT)
        if chosen_edge < min_edge:
            return {
                "action": "skip",
                "side": side,
                "confidence": confidence,
                "reasoning": (
                    f"No edge: {side} edge={chosen_edge:+.3f} < {min_edge:.3f} "
                    f"| fair={fair_yes:.2f} yes={yes_price:.2f} no={no_price:.2f}"
                ),
                "suggested_amount": 0,
                "features": features,
            }

        # --- Symmetric guards (keyed on the chosen side's price) ---
        if side_price > config.HIGH_PRICE_GUARD:
            return {
                "action": "skip",
                "side": side,
                "confidence": confidence,
                "reasoning": (
                    f"High-price guard: {side} price={side_price:.2f} "
                    f">{config.HIGH_PRICE_GUARD:.2f}, bad risk/reward"
                ),
                "suggested_amount": 0,
                "features": features,
            }
        if side_price < config.CONSENSUS_GUARD:
            return {
                "action": "skip",
                "side": side,
                "confidence": confidence,
                "reasoning": (
                    f"Consensus guard: {side} price={side_price:.2f} "
                    f"<{config.CONSENSUS_GUARD:.2f}, fighting consensus"
                ),
                "suggested_amount": 0,
                "features": features,
            }
```

- [ ] **Step 4: Make sizing/entry_price/reasoning side-aware**

In the sizing block (lines ~264-305), change the price basis and reasoning to use the chosen side. Replace `price = max(market_price, 0.01)` with:

```python
        price = max(side_price, 0.01)
```

Replace the `reasoning = (...)` assignment with:

```python
        reasoning = (
            f"fair={fair_yes:.2f} yes={yes_price:.2f} no={no_price:.2f} "
            f"=> {side} edge={chosen_edge:+.3f} "
            f"mom={momentum_signal:+.3f} pm={pm_momentum_signal:+.3f} "
            f"of(obi={obi_signal:+.3f} cvd={cvd_signal:+.3f}) "
            f"strat={strategy_signal:+.3f} learn={learning_signal:+.3f} "
            f"{target_shares:.2f}sh conf={confidence:.2f}"
        )
```

(The `entry_price` in the returned dict is already `round(price, 4)`, which now resolves to the chosen side's price — no further change needed there. Delete the now-unused `min_conf` lookup if it remains.)

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `.venv/bin/python3 -m pytest tests/test_two_sided.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add bots/base_bot.py tests/test_two_sided.py
git commit -m "feat(bots): two-sided net-edge side selection; remove NO ban"
```

---

### Task 3: YES-parity regression, full suite, and docs

**Files:**
- Test: `tests/test_two_sided.py` (extend)
- Modify: `CLAUDE.md` (Signal Hierarchy + Safeguards sections), `BUG_HISTORY.md` (new entry)

**Interfaces:**
- Consumes: the full `make_decision` from Task 2.
- Produces: no new code interfaces — regression coverage + documentation.

- [ ] **Step 1: Write the YES-parity + favorite tests**

Append to `tests/test_two_sided.py`:

```python
def test_favorite_upswing_still_buys_yes():
    # Strong YES favorite with bullish alpha → still YES (no regression).
    bot = _bot()
    m = _market(yes=0.62, no=0.38)
    s = _signals(pm_momentum=0.15, obi=1.0, cvd=1.0)
    d = bot.make_decision(m, s)
    assert d["action"] == "buy"
    assert d["side"] == "yes"
    assert abs(d["entry_price"] - 0.62) < 1e-6


def test_complementary_mids_reduce_to_sign():
    # With no_price == 1 - yes_price, chosen side == sign(fair_yes - yes_price).
    bot = _bot()
    m = _market(yes=0.58)                       # no_price defaults to 0.42
    s = _signals(pm_momentum=0.15, obi=1.0, cvd=1.0)  # bullish → fair > yes
    d = bot.make_decision(m, s)
    if d["action"] == "buy":
        assert d["side"] == "yes"
```

- [ ] **Step 2: Run parity tests**

Run: `.venv/bin/python3 -m pytest tests/test_two_sided.py -v`
Expected: PASS.

- [ ] **Step 3: Run the FULL suite (regression gate)**

Run: `.venv/bin/python3 -m pytest -q`
Expected: All pass. If any pre-existing test asserted the NO ban or YES-only behavior, update it to the two-sided semantics (grep tests for `"NO ban"`, `side == "yes"` assertions that assumed the ban). Fix the test to reflect intended new behavior, not the implementation.

- [ ] **Step 4: Update `CLAUDE.md`**

In the **Signal Hierarchy** section, add a note under the `make_decision` block:

```markdown
The combined signal is reinterpreted as a fair YES probability
(`fair_yes = yes_mid + combined`). Bots then compute a cost-adjusted **net edge**
on BOTH sides (`edge = prob − side_price − per_share_fee`) and buy whichever side
has the larger positive edge above a per-strategy `MIN_EDGE` floor. YES and NO
are evaluated on their own book prices/fees — NO is a first-class decision, not a
mirror. A directional bot takes at most one side per market (arbitrage is the
only two-legged bot).
```

In **Safeguards**, replace the "Market consensus guard" bullet with:

```markdown
- **Symmetric side guards:** A bot never buys a side priced above
  `config.HIGH_PRICE_GUARD` (0.72 — bad risk/reward) or below
  `config.CONSENSUS_GUARD` (0.35 — fighting consensus). Both key off the CHOSEN
  side's price, so YES and NO are protected identically. (Replaces the old
  YES-only NO-ban + one-sided consensus guard.)
```

- [ ] **Step 5: Add `BUG_HISTORY.md` entry**

Append a new numbered entry (use the next number after the current highest):

```markdown
### #20 — YES-only NO ban forfeited the entire NO side

**Symptom:** 100% of directional trades were YES; zero NO fills ever recorded.

**Cause:** `make_decision` had a blanket `if side == "no": return skip` (inherited
from the forked codebase), plus guards/sizing that referenced only the YES price.
The NO decision machinery (strategy `analyze()`, venue NO-book fills,
`learning.record_outcome` NO handling, `market["no_price"]`) all existed and were
correct — only `make_decision` blocked NO.

**Fix:** Replaced the sign-then-ban logic with a two-sided net-edge comparison:
`fair_yes = yes_mid + combined`, then `edge_side = prob − side_price −
taker_fee(1, side_price)` for each side; buy the larger positive edge above a
per-strategy `MIN_EDGE`. Guards and shares-first sizing now key off the chosen
side's price. `K_TILT = 0.5` keeps YES behavior numerically identical at
`aggression = 1.0` (`price_tilt + alpha == old combined`). See design/plan under
`docs/superpowers/`.
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_two_sided.py CLAUDE.md BUG_HISTORY.md
git commit -m "test/docs: YES-parity regression + document two-sided NO trading"
```

---

## Self-Review

**Spec coverage:**
- §1 fair value → Task 1 `_compute_fair_yes` + Task 2 `alpha` assembly + §7 identity preserved (K_TILT=0.5).
- §2 two-sided edge/selection → Task 1 `_side_net_edges` + Task 2 argmax.
- §3 min-edge gate → Task 2 `MIN_EDGE` gate.
- §4 symmetric guards → Task 2 guard block.
- §5 side-aware sizing/confidence/entry → Task 2 Step 4.
- §6 downstream unchanged → asserted via full suite (Task 3 Step 3).
- §7 reconciliation → Global Constraints + Task 2 Step 3 `alpha` weights.
- Config table → Task 1 Step 3 + `MIN_EDGE` dict.
- Tests 1-8 → distributed across Tasks 1-3.

**Placeholder scan:** none — all steps carry real code/commands.

**Type consistency:** `_compute_fair_yes(yes_mid, aggression, alpha) -> float` and `_side_net_edges(fair_yes, yes_price, no_price) -> (edge_yes, edge_no)` used identically in Tasks 1-3. `MIN_EDGE` dict keys match `strategy_type` values used elsewhere. `config` constant names consistent throughout.

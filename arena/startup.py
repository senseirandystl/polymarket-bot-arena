"""Interactive startup: continue-vs-fresh and default-vs-manual bot selection.

Only runs for TERMINAL launches (``sys.stdin.isatty()``). Under launchd / any
non-interactive parent there is no tty, so the whole flow is skipped and the
arena resumes its previous DB configuration silently — the service must never
block on a prompt.

Flow (terminal only):

    1. If a previous run left data → ask **Continue** or **Start fresh**.
         • Continue  → resume exactly as it was (return None; caller loads the
                       existing DB slate).
         • Fresh     → wipe DB + logs, then fall through to step 2.
    2. Ask **Default** bots (Enter) or **Manual** selection.
         • Default   → the 8 canonical bots (incl. arbitrage, sniper + makers).
         • Manual    → show every strategy, accept a list/range like
                       ``1,3,5`` or ``1-6`` (or a mix) → launch exactly those.

``interactive_startup`` returns the bot list to launch, or ``None`` meaning
"use the existing DB configuration" (continue / non-interactive).
"""

import logging
import sys

import config
import db
from bots.bot_momentum import MomentumBot
from bots.bot_mean_rev import MeanRevBot
from bots.bot_meanrev_sl import MeanRevSLBot
from bots.bot_meanrev_tp import MeanRevTPBot
from bots.bot_sniper import SniperBot
from bots.bot_phantom import PhantomBot
from bots.bot_sentiment import SentimentBot
from bots.bot_hybrid import HybridBot
from bots.bot_arbitrage import ArbitrageBot
from bots.bot_late_window_maker import LateWindowMakerBot
from bots.bot_fee_zone_maker import FeeZoneMakerBot

logger = logging.getLogger("arena.startup")

# Ordered menu of every launchable strategy: (class, default_name, blurb).
# The list index (1-based) is what the user selects in manual mode. The maker
# bots (late-window / fee-zone) are first-class members of the slate: they run
# on the discovery-cycle (maker) cadence rather than the 1s trader tick, but
# they are selectable here and included in the default lineup so the Active Bots
# roster always matches what was launched.
STRATEGY_MENU = [
    (MomentumBot,        "momentum-v1",     "Momentum — rides short-term price trend"),
    (MeanRevBot,         "meanrev-v1",      "Mean reversion — drift anchor + buy-the-dip fade"),
    (MeanRevTPBot,       "meanrev-tp-v1",   "Mean reversion + take-profit exit"),
    (SniperBot,          "sniper-v1",       "Sniper — price-zone strike, drift-confirmed"),
    (PhantomBot,         "phantom-v1",      "Phantom — EMA trend + breakout follower"),
    (SentimentBot,       "sentiment-v1",    "Sentiment — in-market repricing + flow"),
    (HybridBot,          "hybrid-v1",       "Hybrid — blended signal stack"),
    (ArbitrageBot,       "arbitrage-v1",    "Arbitrage — market-neutral YES+NO (fees-aware)"),
    (LateWindowMakerBot, "late-window-maker-v1", "Late-window maker — final-150s drift-conviction entry"),
    (FeeZoneMakerBot,    "fee-zone-maker-v1", "Fee-zone maker — 56-86¢ zone, drift-backed quoting"),
]
# (The old separate meanrev-sl25 menu entry is gone: with the stop-loss
# removed it was byte-identical to the base meanrev bot. MeanRevSLBot stays
# importable for pre-migration DB rows; db.init_db renames those to
# meanrev-v1 / mean_reversion.)
_ = MeanRevSLBot  # retained for legacy strategy_type resolution

# The 8 default bots (1-based indices into STRATEGY_MENU): momentum, phantom,
# arbitrage, meanrev, hybrid, sniper, and the two maker bots (2026-07-18
# roster update — sniper promoted into the default slate).
DEFAULT_INDICES = [1, 5, 8, 2, 7, 4, 9, 10]


def build_default_bots() -> list:
    """The canonical default slate (8 bots incl. arbitrage, sniper + both makers)."""
    return _build_from_indices(DEFAULT_INDICES)


def _build_from_indices(indices) -> list:
    bots = []
    for i in indices:
        cls, name, _ = STRATEGY_MENU[i - 1]
        bots.append(cls(name=name, generation=0))
    return bots


def parse_selection(text: str, n: int) -> list:
    """Parse ``"1,3,5"`` / ``"1-6"`` / ``"1-3,5"`` → ordered unique 1..n indices.

    Raises ``ValueError`` on any non-numeric or out-of-range token so the caller
    can re-prompt with a clear message.
    """
    picks: list = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            picks.extend(range(lo, hi + 1))
        else:
            picks.append(int(part))

    seen: set = set()
    out: list = []
    for i in picks:
        if not (1 <= i <= n):
            raise ValueError(f"{i} is out of range (1-{n})")
        if i not in seen:
            seen.add(i)
            out.append(i)
    if not out:
        raise ValueError("no strategies selected")
    return out


# ---------------------------------------------------------------------------
# Previous-run detection + fresh wipe
# ---------------------------------------------------------------------------

def has_previous_run() -> bool:
    """True if the DB holds data from an earlier run (trades or bot configs)."""
    try:
        with db.get_conn() as conn:
            for tbl in ("trades", "bot_configs", "evolution_events"):
                if conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]:
                    return True
    except Exception as e:
        logger.debug(f"has_previous_run check failed: {e}")
    return False


def start_fresh() -> None:
    """Wipe DB rows + truncate log files for a clean slate."""
    tables = db.wipe_all()
    cleared = _clear_logs()
    logger.info(f"Started fresh: wiped {tables} DB tables, truncated {cleared} log files")


def _clear_logs() -> int:
    cleared = 0
    for path in config.LOG_DIR.glob("*.log"):
        try:
            path.open("w").close()
            cleared += 1
        except OSError as e:
            logger.debug(f"could not truncate {path}: {e}")
    return cleared


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _prompt_continue_or_fresh() -> str:
    print("\nA previous arena run was found in the database.")
    while True:
        ans = input("  [C]ontinue previous run, or start [F]resh? (C): ").strip().lower()
        if ans in ("", "c", "continue"):
            return "continue"
        if ans in ("f", "fresh"):
            return "fresh"
        print("  Please enter C or F.")


def _prompt_default_or_manual() -> str:
    print("\nBot selection:")
    while True:
        ans = input("  [D]efault bots (press Enter), or [M]anually select? (D): ").strip().lower()
        if ans in ("", "d", "default"):
            return "default"
        if ans in ("m", "manual"):
            return "manual"
        print("  Please enter D or M.")


def _prompt_manual_selection() -> list:
    print("\nAvailable bot strategies:")
    for idx, (_, name, blurb) in enumerate(STRATEGY_MENU, start=1):
        print(f"  {idx}. {name:<18} {blurb}")
    print("\nEnter the strategies to launch — e.g. '1,3,5' or '1-6' or '1-3,9'.")
    n = len(STRATEGY_MENU)
    while True:
        raw = input(f"  Selection (1-{n}): ").strip()
        try:
            indices = parse_selection(raw, n)
        except ValueError as e:
            print(f"  Invalid selection: {e}. Try again.")
            continue
        bots = _build_from_indices(indices)
        print("  Launching: " + ", ".join(b.name for b in bots))
        return bots


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def interactive_startup():
    """Run the terminal startup flow. Returns a bot list, or ``None`` to mean
    'use the existing DB configuration' (continue / non-interactive)."""
    if not sys.stdin.isatty():
        return None

    if has_previous_run():
        if _prompt_continue_or_fresh() == "continue":
            return None
        start_fresh()

    if _prompt_default_or_manual() == "default":
        bots = build_default_bots()
        print("  Launching defaults: " + ", ".join(b.name for b in bots))
        return bots
    return _prompt_manual_selection()

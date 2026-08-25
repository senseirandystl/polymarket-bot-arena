"""Apply the small arena/dashboard hooks that are awkward to push as 80KB rewrites.

Run from the repo root on this branch:

    python3 -m desk.wire

Idempotent. Does not touch strategy code or the 1s trader tick.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SERVER_NEEDLE = "from arena.market_utils import is_5min_market"
SERVER_INSERT = (
    SERVER_NEEDLE + "\n"
    "from dashboard.desk_routes import register_desk_routes  # desk cycle Floor API\n"
)

SERVER_TAIL_NEEDLE = "if __name__ == \"__main__\":"
SERVER_TAIL_INSERT = (
    "try:\n"
    "    register_desk_routes(app, verify_auth=verify_auth)\n"
    "except Exception:\n"
    "    logging.getLogger(__name__).exception(\"desk routes failed to register\")\n\n"
    "if __name__ == \"__main__\":"
)

ARENA_FALLBACK_OLD = (
    "    # First-run fallback (empty DB, non-interactive): lean 6 default slate.\n"
    "    from arena import startup\n"
    "    return startup.build_default_bots()"
)
ARENA_FALLBACK_NEW = (
    "    # First-run fallback (empty DB, non-interactive): lean 6 default slate\n"
    "    # unless factory mode — then the desk cycle researches the opening roster.\n"
    "    from arena import startup\n"
    "    if getattr(config, \"DESK_FACTORY_MODE\", False):\n"
    "        logger.info(\"DESK_FACTORY_MODE: no lean-6 fallback; desk will propose specs\")\n"
    "        return []\n"
    "    return startup.build_default_bots()"
)

ARENA_START_OLD = (
    "    trader.set_bots(trader_bots)\n"
    "    trader.start()"
)
ARENA_START_NEW = (
    "    trader.set_bots(trader_bots)\n"
    "    trader.start()\n"
    "\n"
    "    desk_host = None\n"
    "    if getattr(config, \"DESK_CYCLE_ENABLED\", True):\n"
    "        try:\n"
    "            from desk.cycle import get_host\n"
    "            desk_host = get_host()\n"
    "            desk_host.start()\n"
    "        except Exception:\n"
    "            logger.exception(\"desk cycle host failed to start\")"
)

ARENA_STOP_OLD = (
    "        for w in (trader, resolver, discovery, warmer, pos_monitor, maker_thread):\n"
    "            w.stop()"
)
ARENA_STOP_NEW = (
    "        for w in (trader, resolver, discovery, warmer, pos_monitor, maker_thread):\n"
    "            w.stop()\n"
    "        if desk_host is not None:\n"
    "            try:\n"
    "                desk_host.stop()\n"
    "            except Exception:\n"
    "                pass"
)

CONFIG_NEEDLE = "PAPER_BANKROLL_DEFAULT = 200.0"
CONFIG_INSERT = '''PAPER_BANKROLL_DEFAULT = 200.0

# Desk cycle (research → code → backtest → paper → live → autopsy)
DESK_FACTORY_MODE = False
DESK_CYCLE_ENABLED = True
DESK_CYCLE_INTERVAL_SEC = 300.0
DESK_MAX_OPEN_SPECS = 8
DESK_MAX_NEW_PER_TICK = 2
DESK_PROMOTE_MIN_TRADES = 100
DESK_PROMOTE_MIN_DAYS = 7
DESK_PROMOTE_TRADE_FLOOR = 30
DESK_AUTO_LIVE = False
DESK_LLM_PROVIDER = "none"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.1"
XAI_API_KEY = ""
XAI_MODEL = "grok-4"
CRYPTO_UNIVERSE_PHASE = 1
'''

DB_NEEDLE = "        # Data migration (idempotent): the meanrev slate bot dropped its"
DB_INSERT = '''        try:
            from desk.store import SCHEMA as _DESK_SCHEMA
            conn.executescript(_DESK_SCHEMA)
        except Exception:
            pass

        # Data migration (idempotent): the meanrev slate bot dropped its'''


def _patch(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new.strip() in text and old not in text:
        print(f"already wired: {path.relative_to(ROOT)}")
        return False
    if old not in text:
        print(f"SKIP (needle missing): {path.relative_to(ROOT)}")
        return False
    path.write_text(text.replace(old, new, 1))
    print(f"patched {path.relative_to(ROOT)}")
    return True


def main() -> None:
    _patch(ROOT / "dashboard" / "server.py", SERVER_NEEDLE, SERVER_INSERT)
    _patch(ROOT / "dashboard" / "server.py", SERVER_TAIL_NEEDLE, SERVER_TAIL_INSERT)
    _patch(ROOT / "arena.py", ARENA_FALLBACK_OLD, ARENA_FALLBACK_NEW)
    _patch(ROOT / "arena.py", ARENA_START_OLD, ARENA_START_NEW)
    _patch(ROOT / "arena.py", ARENA_STOP_OLD, ARENA_STOP_NEW)
    _patch(ROOT / "config.py", CONFIG_NEEDLE, CONFIG_INSERT)
    _patch(ROOT / "db.py", DB_NEEDLE, DB_INSERT)
    print("done. restart arena + dashboard, then open /floor")


if __name__ == "__main__":
    main()

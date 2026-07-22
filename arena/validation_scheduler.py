"""In-arena auto-validation scheduler for the lane-promotion pipeline.

Runs ``tools/validate_signals.py --markets N --propose`` from inside the
arena on a market-count cadence, so the measurement half of the pipeline is
fully hands-off — proposals land in the dashboard Signal Lab and the human
touch stays exactly one thing: approve/deny.

Cadence is expressed in MARKETS (the user's mental model), converted to
wall-clock internally: BTC 5-min windows are strictly one per 5 minutes, so
``AUTO_VALIDATE_EVERY_MARKETS = 100`` means one run every ~8.3h. Counting
elapsed windows beats counting DB rows — the trades table only sees markets
some bot traded (~150 of 288 windows/day in the v5 run), which would make
the cadence drift with bot activity.

State: arena_state key 'last_auto_validation_time' (epoch seconds), so the
cadence survives restarts like the evolution timer does. One subprocess at
a time; output goes to logs/lane_validation.log — the same file the
dashboard's Run Validation button uses, so Signal Lab's status poll shows
auto-runs identically to manual ones.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

import config
import db

logger = logging.getLogger("arena.validation_scheduler")

_STATE_KEY = "last_auto_validation_time"
_MARKET_SEC = 300  # one BTC market per 5-min window, by construction


class ValidationScheduler:
    """Call check() on any cadence; it spawns the harness when a run is due."""

    def __init__(self):
        self._proc = None
        saved = db.get_arena_state(_STATE_KEY)
        if saved:
            self._last_run = float(saved)
        else:
            # First boot: anchor to now so a fresh arena doesn't immediately
            # burn a run before any live trades exist to compare against.
            self._last_run = time.time()
            db.set_arena_state(_STATE_KEY, str(self._last_run))

    def _running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def due(self, now: float = None) -> bool:
        if not getattr(config, "AUTO_VALIDATE_ENABLED", False):
            return False
        if self._running():
            return False
        every = getattr(config, "AUTO_VALIDATE_EVERY_MARKETS", 100)
        return ((now or time.time()) - self._last_run) >= every * _MARKET_SEC

    def check(self) -> bool:
        """Spawn a harness run if one is due. Returns True when spawned."""
        if not self.due():
            # Reap a finished run so its exit status gets logged once.
            if self._proc is not None and self._proc.poll() is not None:
                rc = self._proc.poll()
                (logger.info if rc == 0 else logger.warning)(
                    f"Auto-validation run finished with exit code {rc} "
                    f"(see {config.LOG_DIR / 'lane_validation.log'})"
                )
                self._proc = None
            return False

        markets = getattr(config, "AUTO_VALIDATE_WINDOW_MARKETS", 300)
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "tools" / "validate_signals.py"
        log_path = config.LOG_DIR / "lane_validation.log"
        try:
            log = open(log_path, "w")  # one run per log, matches dashboard
            self._proc = subprocess.Popen(
                [sys.executable, str(script),
                 "--markets", str(markets), "--propose"],
                cwd=str(repo_root), stdout=log, stderr=subprocess.STDOUT,
            )
        except OSError as e:
            logger.error(f"Auto-validation spawn failed: {e}")
            return False

        self._last_run = time.time()
        db.set_arena_state(_STATE_KEY, str(self._last_run))
        logger.info(
            f"Auto-validation started: --markets {markets} --propose "
            f"(cadence: every {getattr(config, 'AUTO_VALIDATE_EVERY_MARKETS', 100)}"
            f" markets); proposals will appear in Signal Lab"
        )
        return True

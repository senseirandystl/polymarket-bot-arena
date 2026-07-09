"""Trader thread — runs every ``config.TRADE_LOOP_INTERVAL_SEC`` seconds.

This is the lean *hot path* of the arena.  It performs **zero network IO**
per tick (the only network call is ``bot.execute()``, and only when a
bot actually places a trade):

  1. Snapshot ``discovery.current_market`` (deep-copied under the
     discovery thread's lock; once we have the copy we don't need to
     coordinate on any further reads).
  2. Freshen ``time_remaining_seconds`` on the snapshot so the staleness
     guard sees live numbers even between discovery cycles.
  3. If the snapshot is missing or its remaining time has dropped below
     1 s, skip this tick — we're either between windows or right at the
     rollover boundary (per the user's "swap only on actual rollover"
     policy; never speculatively hop to the next window).
  4. Build ``combined_signals`` from the cached price / sentiment /
     Polymarket-momentum feeds for the snapshot's market.
  5. For each taker bot: skip if (bot, market) already traded this
     window, else call ``make_decision`` → ``execute`` exactly once.

Evolution pause/resume: ``set_bots()`` swaps the bot list under a lock
that the run-loop also takes for the duration of one tick only as a
list copy.  Bots that appear in the list at the start of a tick remain
in scope for the whole tick — we don't churn mid-iteration.
"""

import logging
import threading
from datetime import datetime, timezone

import db
from bots.base_bot import BaseBot
from config import TRADE_LOOP_INTERVAL_SEC
from arena.market_utils import compute_time_remaining_seconds
from arena.signals import build_combined_signals
from arena.state import SharedArenaState

# `datetime` is imported above at module top so the hot-path 1s tick loop
# doesn't pay for a fresh import every second.

logger = logging.getLogger("arena.trader")


class Trader(threading.Thread):
    def __init__(
        self,
        discovery,
        state: SharedArenaState,
        price_feed,
        sentiment_feed,
        polymarket_price_feed,
    ) -> None:
        super().__init__(daemon=True, name="trader")
        self._stop_event = threading.Event()
        self._discovery = discovery
        self._state = state
        self._price_feed = price_feed
        self._sentiment_feed = sentiment_feed
        self._pm_price_feed = polymarket_price_feed

        # Snapshot of the bots list, swap-safe.  The coordinator calls
        # ``set_bots()`` after each evolution cycle; the run-loop copies
        # the list under this lock at the top of every tick.
        self._bots_lock = threading.Lock()
        self._bots: list = []

    def set_bots(self, bots) -> None:
        """Called by the coordinator after evolution. Atomic swap."""
        with self._bots_lock:
            self._bots = list(bots)

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info(f"Trader started (interval={TRADE_LOOP_INTERVAL_SEC}s)")
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Trader tick error: {e}")
            self._stop_event.wait(TRADE_LOOP_INTERVAL_SEC)
        logger.info("Trader stopped")

    # ------------------------------------------------------------------

    def _tick(self) -> None:
        market = self._discovery.current_market_snapshot()
        if market is not None:
            market["time_remaining_seconds"] = compute_time_remaining_seconds(
                market, datetime.now(timezone.utc)
            )
        # Per the user's "swap only on actual rollover" policy: stop
        # trading the moment the current market's residual drops below
        # 1 s.  No speculative hop to the next window — that trade is
        # bounded by the live market only.
        if market is None or market.get("time_remaining_seconds", 0) < 1:
            return

        market_id = market.get("id") or market.get("market_id")
        if not market_id:
            return

        with self._bots_lock:
            bots = list(self._bots)

        combined_signals = build_combined_signals(
            self._price_feed,
            self._sentiment_feed,
            self._pm_price_feed,
            market,
        )

        new_trades = 0
        for bot in bots:
            key = (bot.name, market_id)
            if self._state.is_traded(key):
                continue
            try:
                signal = bot.make_decision(market, combined_signals)
                if signal.get("action") == "skip":
                    self._state.mark_traded(key)
                    bot_mode = db.get_bot_mode(bot.name)
                    if bot_mode == "live":
                        logger.info(
                            f"[{bot.name}] SKIP "
                            f"price={market.get('current_price', 0):.3f} | "
                            f"{signal.get('reasoning', '')}"
                        )
                    else:
                        logger.debug(
                            f"[{bot.name}] skip | {signal.get('reasoning', '')}"
                        )
                    continue

                result = bot.execute(signal, market)
                self._state.mark_traded(key)
                if result.get("success"):
                    new_trades += 1
                    logger.info(
                        f"[{bot.name}] {signal['side'].upper()} "
                        f"${signal['suggested_amount']:.2f} "
                        f"(conf={signal['confidence']:.2f}) on "
                        f"{market.get('question', '')[:50]}"
                    )
                else:
                    logger.warning(
                        f"[{bot.name}] Trade failed on {market_id}: "
                        f"{result.get('reason')}"
                    )
            except Exception as e:
                logger.error(f"[{bot.name}] Error on {market_id}: {e}")
                self._state.mark_traded(key)

        if new_trades > 0:
            logger.debug(
                f"Trader tick: {new_trades} new trades on {market_id[:12]}..."
            )

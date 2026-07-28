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

import json
import logging
import threading
import time
from datetime import datetime, timezone

import db
import polymarket_markets
from bots.base_bot import BaseBot
from config import TRADE_LOOP_INTERVAL_SEC
from arena import market_data
from arena.market_utils import compute_time_remaining_seconds
from arena.session_filter import session_skip
from arena.log_setup import log_event
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
        # Skip tally is flushed to arena_state at most every 30s so the
        # dashboard (a separate process) can surface why the arena sat flat.
        self._last_skip_flush = 0.0

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
                log_event(logger, logging.ERROR, f"Trader tick error: {e}",
                          exc_info=True, event_type="error", where="trader_run")
            self._stop_event.wait(TRADE_LOOP_INTERVAL_SEC)
        logger.info("Trader stopped")

    # ------------------------------------------------------------------

    def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        market = self._discovery.current_market_snapshot()
        if market is not None:
            market["time_remaining_seconds"] = compute_time_remaining_seconds(
                market, now
            )
        # Stop trading the moment the current market's residual drops below 1s.
        if market is None or market.get("time_remaining_seconds", 0) < 1:
            return

        market_id = market.get("id") or market.get("market_id")
        if not market_id:
            return

        # Global kill switch — cheapest possible gate (cached ~2s). When armed
        # (dashboard / API / logs/KILL_SWITCH file) the whole taker loop sits
        # flat; risk_engine logs the arming event separately.
        try:
            from arena.risk_engine import is_killed
            if is_killed():
                self._state.note_skip("kill_switch")
                return
        except Exception:
            pass

        # Session-timing gate — 'build the skip, default state is flat'. Sit out
        # high-flip session handovers (NYSE open/close) entirely, one check for
        # all taker bots. Cheap and off the per-bot path.
        skip_reason = session_skip(now)
        if skip_reason is not None:
            self._state.note_skip("session")
            logger.debug(f"Session skip ({skip_reason}) — no taker trades this tick")
            return

        # FRESH data every tick with ZERO network on the hot path: the
        # market-data warmer refreshes YES+NO prices, both books, OBI, CVD and
        # PM momentum every ~1s into a shared warm cache. Lay mids, asks AND
        # full books onto the market so make_decision and paper fill share one
        # snapshot (slippage path A). Fall back to a direct price fetch only
        # until the warmer has primed this market.
        warm = market_data.store().get(market_id)
        if warm is not None:
            market_data.lay_warm_onto_market(market, warm)
        else:
            polymarket_markets.refresh_price(market)

        with self._bots_lock:
            bots = list(self._bots)

        combined_signals = build_combined_signals(
            self._price_feed,
            self._sentiment_feed,
            self._pm_price_feed,
            market,
            warm=warm,
        )

        import config as _cfg
        slip_cd = float(getattr(_cfg, "SLIPPAGE_RETRY_COOLDOWN_SEC", 10.0))
        slip_reasons = frozenset({"slippage_band", "slippage_exceeded"})

        new_trades = 0
        for bot in bots:
            key = (bot.name, market_id)
            # Once a bot has an open position on this market it's done for the
            # window; otherwise it RE-EVALUATES every tick (a skip is not sticky)
            # so it enters the moment its edge appears mid-window.
            if self._state.is_traded(key):
                continue
            # Post-slippage backoff: don't re-hammer a whipping book every 1s.
            if self._state.is_slippage_cooling(key):
                self._state.note_skip("slippage_cooldown")
                continue
            try:
                signal = bot.make_decision(market, combined_signals)
                if signal.get("action") == "skip":
                    # Do NOT mark traded — re-evaluate next tick. Skip is a
                    # first-class outcome; tally it so runs are explainable.
                    # Classify coarse reason from reasoning for skip-storm
                    # diagnostics (dead-zone was historically the largest leak).
                    why = (signal.get("reasoning") or "").lower()
                    if "dead-zone" in why or "dead zone" in why:
                        skip_bucket = "dead_zone"
                    elif "macro-release" in why or "macro" in why:
                        skip_bucket = "macro"
                    elif "consensus" in why:
                        skip_bucket = "consensus"
                    elif "high-price" in why:
                        skip_bucket = "high_price"
                    elif "model lean" in why:
                        skip_bucket = "weak_lean"
                    elif "no edge" in why:
                        skip_bucket = "no_edge"
                    else:
                        skip_bucket = "skip"
                    self._state.note_skip(skip_bucket)
                    # Counterfactual log (throttled): skips still carry lane
                    # reads for offline fine-tuning. Hot path = queue only.
                    try:
                        from arena.decision_log import enqueue as _dec_enqueue
                        _dec_enqueue(
                            bot_name=bot.name,
                            strategy_type=bot.strategy_type,
                            market_id=market_id,
                            signal=signal,
                        )
                    except Exception:
                        pass
                    log_event(
                        logger, logging.DEBUG,
                        f"[{bot.name}] skip | {signal.get('reasoning', '')}",
                        event_type="decision", outcome="skip",
                        bot=bot.name, strategy=bot.strategy_type,
                        market_id=market_id, side=signal.get("side"),
                        reason=signal.get("reasoning"),
                    )
                    continue

                result = bot.execute(signal, market)
                if result.get("success"):
                    self._state.mark_traded(key)  # one position per market
                    new_trades += 1
                    try:
                        from arena.decision_log import enqueue as _dec_enqueue
                        _dec_enqueue(
                            bot_name=bot.name,
                            strategy_type=bot.strategy_type,
                            market_id=market_id,
                            signal=signal,
                            trade_id=result.get("trade_id"),
                            force=True,
                        )
                    except Exception:
                        pass
                    log_event(
                        logger, logging.INFO,
                        f"[{bot.name}] {signal['side'].upper()} "
                        f"${signal['suggested_amount']:.2f} "
                        f"(conf={signal['confidence']:.2f}) on "
                        f"{market.get('question', '')[:50]}",
                        event_type="trade", outcome="placed",
                        bot=bot.name, strategy=bot.strategy_type,
                        market_id=market_id, side=signal.get("side"),
                        amount=round(float(signal.get("suggested_amount", 0.0)), 4),
                        confidence=round(float(signal.get("confidence", 0.0)), 4),
                        entry_price=signal.get("entry_price"),
                        target_shares=signal.get("target_shares"),
                        trade_id=result.get("trade_id"),
                        fill_source=result.get("fill_source"),
                        mode=bot.trading_mode,
                    )
                else:
                    reason = result.get("reason")
                    if reason in slip_reasons:
                        self._state.mark_slippage_reject(key, slip_cd)
                        self._state.note_skip("slippage")
                    # Live-mode fill anomalies: slippage rejects, venue errors,
                    # naked arb legs — notify operators (debounced in alerts).
                    if (getattr(bot, "trading_mode", "paper") or "paper") == "live":
                        try:
                            from arena.alerts import alert_live_fill
                            alert_live_fill(
                                bot.name, str(reason or "not_placed"),
                                side=str(signal.get("side") or ""),
                                market_id=str(market_id),
                            )
                        except Exception:
                            pass
                    # Transient (no book / bankroll dry / cooldown) — don't mark
                    # traded; retry next eligible tick.
                    log_event(
                        logger, logging.DEBUG,
                        f"[{bot.name}] trade not placed on {market_id[:12]}…: "
                        f"{reason}",
                        event_type="trade", outcome="not_placed",
                        bot=bot.name, strategy=bot.strategy_type,
                        market_id=market_id, side=signal.get("side"),
                        reason=reason,
                    )
            except Exception as e:
                log_event(
                    logger, logging.ERROR,
                    f"[{bot.name}] Error on {market_id}: {e}",
                    exc_info=True,
                    event_type="error", where="trader_tick",
                    bot=bot.name, market_id=market_id,
                )

        if new_trades > 0:
            logger.debug(
                f"Trader tick: {new_trades} new trades on {market_id[:12]}..."
            )

        # Periodically persist the skip tally (cross-process observability).
        now_ts = time.time()
        if now_ts - self._last_skip_flush >= 30:
            self._last_skip_flush = now_ts
            try:
                db.set_arena_state("skip_counts", json.dumps(self._state.skip_snapshot()))
            except Exception as e:
                logger.debug(f"skip_counts flush failed: {e}")

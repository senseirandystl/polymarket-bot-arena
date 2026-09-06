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

# Dashboard toggles change rarely; cache 1-3s (BOT_MODE_CACHE style) so we
# do not hit SQLite 3x per market per 1Hz tick.
_TOGGLE_CACHE: dict = {"ts": 0.0, "one": None, "hybrid": None, "lock": None}
_TOGGLE_CACHE_LOCK = threading.Lock()


def _cached_trade_toggles(cfg_mod):
    """Return (one_per_tick, hybrid_yield, window_lock) with short TTL."""
    now = time.time()
    ttl = float(getattr(cfg_mod, "BOT_MODE_CACHE_TTL_SEC", 3) or 3)
    ttl = max(1.0, min(ttl, 3.0))
    with _TOGGLE_CACHE_LOCK:
        hit = _TOGGLE_CACHE
        if (
            hit["one"] is not None
            and (now - float(hit["ts"])) < ttl
        ):
            return bool(hit["one"]), bool(hit["hybrid"]), bool(hit["lock"])
    try:
        one = bool(db.get_one_trade_per_tick())
    except Exception:
        one = bool(getattr(cfg_mod, "ONE_TRADE_PER_TICK", False))
    try:
        hybrid = bool(db.get_hybrid_yield())
    except Exception:
        hybrid = bool(getattr(cfg_mod, "HYBRID_YIELD_ENABLED", False))
    try:
        lock = bool(db.get_directional_window_lock())
    except Exception:
        lock = bool(getattr(cfg_mod, "DIRECTIONAL_WINDOW_LOCK", False))
    with _TOGGLE_CACHE_LOCK:
        _TOGGLE_CACHE["ts"] = now
        _TOGGLE_CACHE["one"] = one
        _TOGGLE_CACHE["hybrid"] = hybrid
        _TOGGLE_CACHE["lock"] = lock
    return one, hybrid, lock



def directional_buy_score(signal: dict) -> float:
    """Rank competing buys by dollar EV: edge / (1 − ask). Ignores conf/weight."""
    try:
        edge = float(signal.get("edge") or 0.0)
    except (TypeError, ValueError):
        edge = 0.0
    if edge != edge or edge == float("-inf"):
        edge = 0.0
    try:
        ask = float(signal.get("entry_price") or 0.5)
    except (TypeError, ValueError):
        ask = 0.5
    ask = min(0.99, max(0.01, ask))
    return edge / max(1e-6, 1.0 - ask)


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
            t0 = time.perf_counter()
            try:
                self._tick()
            except Exception as e:
                log_event(logger, logging.ERROR, f"Trader tick error: {e}",
                          exc_info=True, event_type="error", where="trader_run")
            # Deadline-based sleep: remain near 1 Hz under light load.
            elapsed = time.perf_counter() - t0
            remain = max(0.0, float(TRADE_LOOP_INTERVAL_SEC) - elapsed)
            if elapsed > float(TRADE_LOOP_INTERVAL_SEC) * 1.5:
                logger.warning(
                    f"Trader tick slow: {elapsed*1000:.0f}ms "
                    f"(budget {TRADE_LOOP_INTERVAL_SEC*1000:.0f}ms)"
                )
            self._stop_event.wait(remain)
        logger.info("Trader stopped")

    # ------------------------------------------------------------------

    def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        try:
            from arena.risk_engine import is_killed
            if is_killed():
                self._state.note_skip("kill_switch")
                return
        except Exception as e:
            logger.warning("trader kill-switch check failed: %s", e)
        markets = {}
        if hasattr(self._discovery, "current_markets_snapshot"):
            try:
                markets = self._discovery.current_markets_snapshot() or {}
            except Exception as e:
                logger.warning("trader markets snapshot failed: %s", e)
                markets = {}
        if not markets:
            market = self._discovery.current_market_snapshot()
            if market:
                markets = {"polymarket": market}
        with self._bots_lock:
            bots = list(self._bots)
        for exch, market in markets.items():
            try:
                peers = {k: v for k, v in markets.items() if k != exch}
                self._tick_one_market(market, bots, now, peers=peers)
            except Exception as e:
                log_event(
                    logger, logging.ERROR, f"Trader market tick error: {e}",
                    exc_info=True, event_type="error", where="trader_run",
                )
        now_ts = time.time()
        if now_ts - self._last_skip_flush >= 30:
            self._last_skip_flush = now_ts
            try:
                db.set_arena_state("skip_counts", json.dumps(self._state.skip_snapshot()))
            except Exception as e:
                logger.debug(f"skip_counts flush failed: {e}")

    def _tick_one_market(self, market: dict, bots: list, now, peers: dict | None = None) -> None:
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

        # Session-timing gate — NYSE open/close. Arb is market-neutral and
        # exempt; directionals and sweeper sit flat.
        session_reason = session_skip(now)

        # FRESH data every tick with ZERO network on the hot path: the
        # market-data warmer refreshes YES+NO prices, both books, OBI, CVD and
        # PM momentum every ~1s into a shared warm cache. Lay mids, asks AND
        # full books onto the market so make_decision and paper fill share one
        # snapshot (slippage path A).
        #
        # Never block the tick on a cold refresh_price (15s timeout trap).
        # Missing/stale warm → skip with warm_stale reason.
        warm = market_data.store().get(market_id)
        if not market_data.is_warm_fresh(warm):
            self._state.note_skip("warm_stale")
            logger.debug(
                "Trader skip warm_stale market=%s age=%s",
                str(market_id)[:12],
                market_data.warm_age_sec(warm),
            )
            return
        market_data.lay_warm_onto_market(market, warm)

        combined_signals = build_combined_signals(
            self._price_feed,
            self._sentiment_feed,
            self._pm_price_feed,
            market,
            warm=warm,
        )
        # Warm age for diagnostics / size context (never network).
        combined_signals["warm_age_sec"] = market_data.warm_age_sec(warm)
        # Cross-venue peer mids for menu-only cross_venue_lag (and diagnostics).
        if peers:
            try:
                peer_ex, peer_m = next(iter(peers.items()))
                combined_signals["cross_venue"] = {
                    "peer_exchange": peer_ex,
                    "peer_yes_mid": peer_m.get("current_price")
                    or peer_m.get("yes_price"),
                    "peer_window_sec": peer_m.get("window_sec"),
                    "local_exchange": market.get("exchange")
                    or market.get("venue"),
                    "local_yes_mid": market.get("current_price")
                    or market.get("yes_price"),
                    "local_window_sec": market.get("window_sec"),
                }
            except Exception as e:
                logger.warning("cross_venue attach failed: %s", e)

        import config as _cfg
        slip_cd = float(getattr(_cfg, "SLIPPAGE_RETRY_COOLDOWN_SEC", 10.0))
        slip_reasons = frozenset({"slippage_band", "slippage_exceeded"})
        one_per_tick, hybrid_yield_on, window_lock = _cached_trade_toggles(_cfg)
        # Live fuse: paper-eval open cluster must not follow a bot onto CLOB.
        any_live = any(
            (getattr(b, "trading_mode", "paper") or "paper") == "live"
            for b in bots
        )
        if any_live:
            one_per_tick = True
            hybrid_yield_on = True
        exempt_types = {
            str(t).lower()
            for t in (getattr(_cfg, "ONE_TRADE_PER_TICK_EXEMPT", ()) or ())
        }
        lock_exempt = {
            str(t).lower()
            for t in (getattr(_cfg, "DIRECTIONAL_WINDOW_LOCK_EXEMPT", ()) or ())
        }

        def _is_exempt(bot, exempt_set) -> bool:
            return (getattr(bot, "strategy_type", "") or "").lower() in exempt_set

        def _note_decision(bot, signal, *, force_buy_log=False):
            if signal.get("action") == "skip":
                try:
                    from arena.decision_log import classify_skip_reason
                    skip_bucket = classify_skip_reason(
                        signal.get("reasoning"),
                        explicit=signal.get("skip_reason"),
                    ) or "skip"
                except Exception:
                    skip_bucket = "skip"
                self._state.note_skip(skip_bucket)
                try:
                    from arena.decision_log import enqueue as _dec_enqueue
                    _dec_enqueue(
                        bot_name=bot.name,
                        strategy_type=bot.strategy_type,
                        market_id=market_id,
                        signal=signal,
                    )
                except Exception as e:
                    logger.warning("swallowed exception: %s", e)
                log_event(
                    logger, logging.DEBUG,
                    f"[{bot.name}] skip | {signal.get('reasoning', '')}",
                    event_type="decision", outcome="skip",
                    bot=bot.name, strategy=bot.strategy_type,
                    market_id=market_id, side=signal.get("side"),
                    reason=signal.get("reasoning"),
                )
            elif force_buy_log:
                try:
                    from arena.decision_log import enqueue as _dec_enqueue
                    _dec_enqueue(
                        bot_name=bot.name,
                        strategy_type=bot.strategy_type,
                        market_id=market_id,
                        signal=signal,
                    )
                except Exception as e:
                    logger.warning("swallowed exception: %s", e)

        def _execute_one(bot, signal) -> bool:
            """Place one trade; return True on success."""
            key = (bot.name, market_id)
            try:
                result = bot.execute(signal, market)
            except Exception as e:
                log_event(
                    logger, logging.ERROR,
                    f"[{bot.name}] Error on {market_id}: {e}",
                    exc_info=True,
                    event_type="error", where="trader_tick",
                    bot=bot.name, market_id=market_id,
                )
                return False
            if result.get("success"):
                self._state.mark_traded(key)
                if not _is_exempt(bot, lock_exempt):
                    self._state.mark_directional_lock(market_id)
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
                except Exception as e:
                    logger.warning("swallowed exception: %s", e)
                log_event(
                    logger, logging.INFO,
                    f"[{bot.name}] {signal['side'].upper()} "
                    f"${signal.get('suggested_amount', 0):.2f} "
                    f"(conf={float(signal.get('confidence') or 0):.2f}) on "
                    f"{market.get('question', '')[:50]}",
                    event_type="trade", outcome="placed",
                    bot=bot.name, strategy=bot.strategy_type,
                    market_id=market_id, side=signal.get("side"),
                    amount=round(float(signal.get("suggested_amount", 0.0) or 0), 4),
                    confidence=round(float(signal.get("confidence", 0.0) or 0), 4),
                    entry_price=signal.get("entry_price"),
                    target_shares=signal.get("target_shares"),
                    trade_id=result.get("trade_id"),
                    fill_source=result.get("fill_source"),
                    mode=bot.trading_mode,
                )
                return True
            reason = result.get("reason")
            # One live GTC per (bot, market). A resting limit must not be
            # re-posted every tick or we stack orphan orders on the CLOB.
            if isinstance(reason, str) and reason.startswith("limit_resting"):
                self._state.mark_traded(key)
            if reason in slip_reasons:
                self._state.mark_slippage_reject(key, slip_cd)
                self._state.note_skip("slippage")
            if (getattr(bot, "trading_mode", "paper") or "paper") == "live":
                try:
                    from arena.alerts import alert_live_fill
                    alert_live_fill(
                        bot.name, str(reason or "not_placed"),
                        side=str(signal.get("side") or ""),
                        market_id=str(market_id),
                    )
                except Exception as e:
                    logger.warning("swallowed exception: %s", e)
            log_event(
                logger, logging.DEBUG,
                f"[{bot.name}] trade not placed on {str(market_id)[:12]}…: "
                f"{reason}",
                event_type="trade", outcome="not_placed",
                bot=bot.name, strategy=bot.strategy_type,
                market_id=market_id, side=signal.get("side"),
                reason=reason,
            )
            return False

        new_trades = 0
        # Collect competing directional buys; structural bots execute immediately.
        pending_buys: list[tuple] = []  # (score, bot, signal)

        for bot in bots:
            key = (bot.name, market_id)
            if self._state.is_traded(key):
                continue
            if self._state.is_slippage_cooling(key):
                self._state.note_skip("slippage_cooldown")
                continue
            if (
                session_reason is not None
                and (getattr(bot, "strategy_type", "") or "").lower()
                != "arbitrage"
            ):
                self._state.note_skip("session")
                continue
            # Window lock: after any directional fill, other directionals sit out
            if (
                window_lock
                and self._state.is_directional_locked(market_id)
                and not _is_exempt(bot, lock_exempt)
            ):
                self._state.note_skip("window_lock")
                continue
            try:
                signal = bot.make_decision(market, combined_signals)
            except Exception as e:
                log_event(
                    logger, logging.ERROR,
                    f"[{bot.name}] Error on {market_id}: {e}",
                    exc_info=True,
                    event_type="error", where="trader_tick",
                    bot=bot.name, market_id=market_id,
                )
                continue

            if signal.get("action") == "skip":
                _note_decision(bot, signal)
                continue

            # Structural / exempt bots execute immediately (arb, sweeper, …)
            if _is_exempt(bot, exempt_types) or not one_per_tick:
                if _execute_one(bot, signal):
                    new_trades += 1
                continue

            # Directional buy: rank and pick one winner after the loop
            signal = dict(signal)
            signal["_bot_name"] = bot.name
            pending_buys.append((directional_buy_score(signal), bot, signal))

        if pending_buys:
            if hybrid_yield_on:
                occupied = {
                    str(sig.get("side") or "")
                    for _sc, b, sig in pending_buys
                    if (getattr(b, "strategy_type", "") or "").lower() != "hybrid"
                }
                yielded = []
                kept = []
                for item in pending_buys:
                    _sc, b, sig = item
                    if (
                        (getattr(b, "strategy_type", "") or "").lower() == "hybrid"
                        and str(sig.get("side") or "") in occupied
                    ):
                        yielded.append(item)
                    else:
                        kept.append(item)
                for _sc, bot, signal in yielded:
                    sup = dict(signal)
                    sup["action"] = "skip"
                    sup["skip_reason"] = "hybrid_yield"
                    sup["reasoning"] = (
                        f"Hybrid yield: dedicated directional already pending "
                        f"{signal.get('side')}"
                    )
                    _note_decision(bot, sup)
                    self._state.note_skip("hybrid_yield")
                pending_buys = kept
            if pending_buys:
                pending_buys.sort(key=lambda t: t[0], reverse=True)
            try:
                max_dir = int(__import__("db").get_market_side_max_bots())
            except Exception:
                try:
                    max_dir = int(getattr(_cfg, "MARKET_SIDE_MAX_BOTS", 0) or 0)
                except (TypeError, ValueError):
                    max_dir = 0
            if any_live and max_dir <= 0:
                max_dir = 1
            filled_side: dict[str, int] = {}
            winner_name = pending_buys[0][1].name if pending_buys else ""
            winner_score = pending_buys[0][0] if pending_buys else 0.0
            for score, bot, signal in pending_buys:
                side = str(signal.get("side") or "")
                n_side = int(filled_side.get(side, 0))
                if max_dir > 0 and n_side >= max_dir:
                    sup = dict(signal)
                    sup["action"] = "skip"
                    sup["skip_reason"] = "superseded_by_peer"
                    sup["reasoning"] = (
                        f"Superseded by {winner_name} "
                        f"(score {winner_score:.4f} > {score:.4f}; "
                        f"max {max_dir} bot(s)/side)"
                    )
                    _note_decision(bot, sup)
                    self._state.note_skip("superseded_by_peer")
                    continue
                signal = dict(signal)
                # Same-tick peer count for pile-in (DB may not see the
                # fill we just wrote this tick).
                signal["_pilein_extra_peers"] = n_side
                if _execute_one(bot, signal):
                    filled_side[side] = n_side + 1
                    new_trades += 1
                else:
                    # execute() refused (pile-in / exposure) — count as skip
                    self._state.note_skip("execute_refused")

        if new_trades > 0:
            logger.debug(
                f"Trader tick: {new_trades} new trades on {str(market_id)[:12]}..."
            )

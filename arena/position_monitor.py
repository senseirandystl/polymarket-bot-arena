"""Position monitor thread — ~1s SL/TP exit engine for bots that carry
an ``exit_strategy``.

Prefers mids from the market_data warm store; falls back to CLOB/Kalshi
HTTP only when warm is missing/stale. Looks at each bot's open positions,
exits at the
configured stop-loss / take-profit threshold, writes outcome=exit_sl / exit_tp
on the trade row, and feeds the outcome back into the learning system.

Kept separate from the ``Trader`` (1s) and ``TradeResolver`` (60s)
threads so the SL/TP engine can stay hard-realtime without slowing
down trade evaluation, and so it can have its own polling cadence
without forcing every other worker to share its interval.
"""

import json
import logging
import threading
import time

import config
import db
import polymarket_markets
from learning import extract_features_from_reasoning, record_outcome


# SL/TP poll rate. Prefer warm-store mids; HTTP fallback is still throttled
# via PositionMonitorThread._PRICE_TTL so a cold warm path cannot hammer CLOB.
FAST_POLL_INTERVAL = float(
    getattr(config, "POSITION_MONITOR_INTERVAL_SEC", 1.0) or 1.0
)

logger = logging.getLogger("arena.position_monitor")


class PositionMonitorThread(threading.Thread):
    # Refresh open-position prices from the CLOB at most this often (seconds).
    _PRICE_TTL = 3.0

    def __init__(self) -> None:
        super().__init__(daemon=True, name="position-monitor")
        self._bots: dict = {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._price_cache: dict = {}
        self._price_ts: float = 0.0

    def update_bots(self, bots) -> None:
        """Called by the coordinator after each evolution cycle."""
        with self._lock:
            self._bots = {
                b.name: b for b in bots if getattr(b, "exit_strategy", None)
            }

    def stop(self) -> None:
        self._stop_event.set()

    def _fetch_market_prices(self) -> dict:
        """{market_id: current_yes_price} for markets with OPEN positions.

        Prefer MarketDataWarmer mids (fresh, in-memory). HTTP to Polymarket
        CLOB / Kalshi only when warm is missing or stale, and then still
        throttled by ``_PRICE_TTL``.
        """
        now = time.time()
        try:
            with db.get_conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT market_id FROM trades WHERE outcome IS NULL"
                ).fetchall()
        except Exception:
            return self._price_cache

        prices = {}
        need_http = []
        try:
            from arena import market_data
        except Exception:
            market_data = None  # type: ignore

        for r in rows:
            mid = r["market_id"]
            p = None
            if market_data is not None:
                try:
                    warm = market_data.store().get(mid)
                    if market_data.is_warm_fresh(warm):
                        p = warm.get("yes_price")
                except Exception:
                    p = None
            if p is not None:
                prices[mid] = p
            else:
                need_http.append(mid)

        if need_http:
            # Throttle HTTP fallback; reuse prior cache entries when still fresh.
            http_ok = (now - self._price_ts) >= self._PRICE_TTL or not self._price_cache
            for mid in need_http:
                if not http_ok and mid in self._price_cache:
                    prices[mid] = self._price_cache[mid]
                    continue
                p = None
                if str(mid).startswith("kalshi:"):
                    try:
                        import kalshi_markets
                        from exchanges import native_market_id
                        p = kalshi_markets.current_up_price(
                            native_market_id(str(mid)) or str(mid)
                        )
                    except Exception:
                        p = None
                else:
                    p = polymarket_markets.current_up_price(mid)
                if p is not None:
                    prices[mid] = p
            if http_ok:
                self._price_ts = now

        self._price_cache = prices
        return prices


    def _check_positions(self, price_map: dict) -> None:
        with self._lock:
            exit_bots = dict(self._bots)
        if not exit_bots:
            return

        bot_names = list(exit_bots.keys())
        if not bot_names:
            return
        with db.get_conn() as conn:
            placeholders = ",".join("?" for _ in bot_names)
            rows = conn.execute(
                f"SELECT id, bot_name, market_id, side, amount, shares_bought, "
                f"trade_features, reasoning FROM trades "
                f"WHERE outcome IS NULL AND bot_name IN ({placeholders})",
                bot_names,
            ).fetchall()
        if not rows:
            return

        for trade in rows:
            market_id = trade["market_id"]
            current_yes_price = price_map.get(market_id)
            if current_yes_price is None:
                continue
            bot = exit_bots.get(trade["bot_name"])
            if not bot:
                continue
            side = trade["side"]
            amount = trade["amount"]
            try:
                shares = trade["shares_bought"] or 0
            except (KeyError, IndexError):
                shares = 0
            if shares <= 0:
                continue
            entry_price = amount / shares
            if entry_price <= 0:
                continue

            if side == "yes":
                current_share_price = current_yes_price
            else:
                current_share_price = 1.0 - current_yes_price

            pnl_pct = (current_share_price - entry_price) / entry_price

            exit_reason = None
            exit_pnl = None
            if bot.exit_strategy == "stop_loss" and pnl_pct <= -bot.stop_loss_pct:
                exit_pnl = (current_share_price - entry_price) * shares
                exit_reason = f"exit_sl ({pnl_pct:+.1%})"
            if bot.exit_strategy == "take_profit" and pnl_pct >= bot.take_profit_pct:
                exit_pnl = (current_share_price - entry_price) * shares
                exit_reason = f"exit_tp ({pnl_pct:+.1%})"

            if exit_reason and exit_pnl is not None:
                outcome = "exit_tp" if "tp" in exit_reason else "exit_sl"
                db.resolve_trade(trade["id"], outcome, exit_pnl)
                logger.info(
                    f"[{trade['bot_name']}] EARLY EXIT: {exit_reason} on "
                    f"{market_id[:12]}... entry=${entry_price:.3f} "
                    f"now=${current_share_price:.3f} pnl=${exit_pnl:+.2f}"
                )

                try:
                    stored = trade["trade_features"]
                    if stored:
                        features = json.loads(stored)
                    else:
                        try:
                            features = extract_features_from_reasoning(
                                trade["reasoning"]
                            )
                        except (KeyError, IndexError):
                            features = None
                except (KeyError, json.JSONDecodeError):
                    features = None
                if features:
                    won = exit_pnl > 0
                    record_outcome(trade["bot_name"], features, side, won)

    def run(self) -> None:
        logger.info(f"Position monitor started (polling every {FAST_POLL_INTERVAL}s)")
        consecutive_errors = 0
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    has_bots = bool(self._bots)
                if has_bots:
                    price_map = self._fetch_market_prices()
                    if price_map:
                        self._check_positions(price_map)
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                if consecutive_errors > 10:
                    self._stop_event.wait(5)
                elif consecutive_errors > 3:
                    self._stop_event.wait(2)
                else:
                    self._stop_event.wait(FAST_POLL_INTERVAL)
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                consecutive_errors += 1
                self._stop_event.wait(2)

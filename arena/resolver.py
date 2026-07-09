"""Trade resolution thread.

Ticks once every ``config.RESOLVE_INTERVAL_SEC`` (60s by default).  Checks
Simmer's resolved-markets endpoint, matches each pending trade row in our
``trades`` table to its now-resolved market, and writes the outcome + P&L.
Also sweeps stale-pending trades that fell off Simmer's 200-deep resolved
window (5-min markets that never re-surface get expired after 1h, which
keeps them off the "Pending forever" list in the dashboard).
"""

import json
import logging
import threading

import requests

import config
import db
from learning import (
    extract_features_from_reasoning,
    record_outcome,
)

logger = logging.getLogger("arena.resolver")


class TradeResolver(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="trade-resolver")
        self._stop_event = threading.Event()

    def run(self) -> None:
        logger.info(
            f"TradeResolver started (interval={config.RESOLVE_INTERVAL_SEC}s)"
        )
        while not self._stop_event.is_set():
            try:
                self._do_resolution_cycle()
            except Exception as e:
                logger.error(f"Resolver scan error: {e}")
            self._stop_event.wait(config.RESOLVE_INTERVAL_SEC)
        logger.info("TradeResolver stopped")

    def stop(self) -> None:
        self._stop_event.set()

    def _do_resolution_cycle(self) -> None:
        """Hot-reload credentials on every cycle, then resolve.

        Credentials saved via the dashboard Settings tab take effect on
        the next 60s boundary without restarting the arena.  Multi-account
        mode (≥ ``NUM_BOTS`` bot keys configured) resolves under each
        distinct per-slot key once; single-account mode resolves under
        the default key once.
        """
        api_key = config.get_credential("simmer_api_key")
        bot_keys: dict = {}
        raw = config.get_credential("simmer_bot_keys")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    bot_keys = parsed
            except (json.JSONDecodeError, TypeError):
                pass

        if len(bot_keys) >= config.NUM_BOTS:
            for k in set(bot_keys.values()):
                self._resolve_pending(k)
        elif api_key:
            self._resolve_pending(api_key)

        self._expire_stale_trades()

    def _resolve_pending(self, api_key: str) -> int:
        """Resolve one Simmer account's pending trades. Returns count."""
        if not api_key:
            return 0
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            with db.get_conn() as conn:
                pending = conn.execute(
                    "SELECT id, market_id, bot_name, side, amount, "
                    "shares_bought, trade_features, reasoning "
                    "FROM trades WHERE outcome IS NULL"
                ).fetchall()
            if not pending:
                return 0

            market_ids = list({t["market_id"] for t in pending})

            resp = requests.get(
                f"{config.SIMMER_BASE_URL}/api/sdk/markets",
                headers=headers,
                params={"status": "resolved", "limit": 200},
                timeout=15,
            )
            if resp.status_code != 200:
                return 0

            data = resp.json()
            markets_list = data if isinstance(data, list) else data.get("markets", [])
            resolved_map = {
                (m.get("id") or m.get("market_id")): m
                for m in markets_list
                if (m.get("id") or m.get("market_id")) in market_ids
            }
            if not resolved_map:
                return 0

            count = 0
            for trade in pending:
                market_id = trade["market_id"]
                if market_id not in resolved_map:
                    continue
                market = resolved_map[market_id]
                market_outcome = market.get("outcome")
                if market_outcome is None:
                    continue

                side = trade["side"]
                amount = trade["amount"]
                try:
                    shares = trade["shares_bought"] or 0
                except (IndexError, KeyError):
                    shares = 0

                # Trades whose shares_bought was never recorded (legacy
                # rows or failed POSTs) are left pending — the 1h stale
                # expiry sweep will catch them.  Writing outcome=win/loss
                # with pnl=0 here would surface as a misleading zero-P&L
                # win on the Dashboard's Recent Trades table.
                if shares <= 0:
                    logger.warning(
                        f"[{trade['bot_name']}] Skipping resolution: "
                        f"shares_bought absent or zero on "
                        f"{market_id[:12]}... — leaving pending for "
                        f"stale-expiry sweep."
                    )
                    continue

                if side == "yes":
                    won = market_outcome is True
                else:
                    won = market_outcome is False

                outcome = "win" if won else "loss"
                pnl = (shares - amount) if won else -amount

                db.resolve_trade(trade["id"], outcome, pnl)

                # Wire the outcome into learning so future bots bias on it.
                try:
                    stored = trade["trade_features"]
                    if stored:
                        features = json.loads(stored)
                    else:
                        try:
                            reasoning = trade["reasoning"]
                        except (KeyError, IndexError):
                            reasoning = None
                        features = extract_features_from_reasoning(reasoning)
                except (KeyError, json.JSONDecodeError):
                    features = None

                if features:
                    record_outcome(trade["bot_name"], features, side, won)

                count += 1

            if count > 0:
                logger.info(
                    f"Resolved {count} trades "
                    f"({sum(1 for t in pending if resolved_map.get(t['market_id']))} "
                    f"pending matched {len(resolved_map)} resolved markets)"
                )
            return count
        except Exception as e:
            logger.error(f"_resolve_pending error: {e}")
            return 0

    def _expire_stale_trades(self) -> None:
        """Move >1h-pending trades to outcome='expired' so they fall off
        the dashboard's Pending list with a correct label."""
        with db.get_conn() as conn:
            count = conn.execute(
                "UPDATE trades SET outcome = 'expired', pnl = 0, "
                "resolved_at = datetime('now') "
                "WHERE outcome IS NULL AND created_at < datetime('now', '-1 hour')"
            ).rowcount
        if count > 0:
            logger.info(f"Expired {count} stale trades (>1h old, never resolved)")

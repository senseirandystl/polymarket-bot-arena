"""Trade resolution thread.

Ticks once every ``config.RESOLVE_INTERVAL_SEC`` (60s by default). For each
pending trade it looks the market up directly (``GET /api/sdk/markets/{id}``)
and, once the market has resolved, writes the outcome + P&L. Resolution is
account-independent, so both local-sim paper fills and Simmer/Polymarket fills
settle against the same real market outcome.

There is intentionally **no stale-expiry sweep**: a pending trade stays pending
until its market actually resolves (Simmer can take up to a day). Every cycle
retries every pending trade, so nothing is lost by waiting.
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
        the next 60s boundary without restarting the arena.

        Market-outcome lookups are **account-independent**: any authenticated
        Simmer key can read any market's resolved outcome. Resolution must
        therefore NOT be routed solely through per-bot keys, which are often
        placeholder values (e.g. an unset ``slot_0`` that returns HTTP 401).
        Routing all resolution through an invalid per-bot key silently expired
        every trade at $0 — the failure mode that produced the "$0.00 P&L"
        rows on the dashboard. Instead we build an ordered candidate list
        (default key first, then distinct per-bot keys) and resolve with the
        first key that actually authenticates.
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

        candidates: list = []
        for k in [api_key, *bot_keys.values()]:
            if k and k not in candidates:
                candidates.append(k)
        if not candidates:
            return

        for key in candidates:
            _count, auth_ok = self._resolve_pending(key)
            # A valid key can see every market's outcome, so once one key
            # authenticates there's nothing to gain from trying the rest.
            if auth_ok:
                break

        # NOTE: no stale-expiry sweep. Pending trades stay pending until the
        # market actually resolves — Simmer can take up to a day to settle a
        # market, and prematurely marking trades 'expired'/$0 threw away real
        # outcomes. The per-market lookup above retries every pending trade on
        # every cycle, so a trade resolves as soon as its market does.

    def _settle_trade(self, trade, market_outcome: bool) -> bool:
        """Write win/loss + P&L + learning for one resolved trade.

        Returns True if the trade was settled, False if it was skipped
        (missing share count). Shared by the normal resolution loop and the
        final resolve-before-expiry pass so both paths stay consistent.
        """
        side = trade["side"]
        amount = trade["amount"]
        try:
            shares = trade["shares_bought"] or 0
        except (IndexError, KeyError):
            shares = 0

        # Trades whose shares_bought was never recorded (legacy rows or failed
        # POSTs) are left pending — the 1h stale expiry sweep will catch them.
        # Writing outcome=win/loss with pnl=0 here would surface as a
        # misleading zero-P&L win on the dashboard's Recent Trades table.
        if shares <= 0:
            logger.warning(
                f"[{trade['bot_name']}] Skipping resolution: shares_bought "
                f"absent or zero on {str(trade['market_id'])[:12]}... — "
                f"leaving pending for stale-expiry sweep."
            )
            return False

        won = (market_outcome is True) if side == "yes" else (market_outcome is False)
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
        return True

    def _resolve_pending(self, api_key: str):
        """Resolve one Simmer key's view of pending trades.

        Returns ``(resolved_count, auth_ok)``. ``auth_ok`` is False only when
        the key was rejected (HTTP 401/403), which signals the caller to try
        the next candidate key.
        """
        if not api_key:
            return (0, True)
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            with db.get_conn() as conn:
                pending = conn.execute(
                    "SELECT id, market_id, bot_name, side, amount, "
                    "shares_bought, trade_features, reasoning "
                    "FROM trades WHERE outcome IS NULL"
                ).fetchall()
            if not pending:
                return (0, True)

            # Resolve each DISTINCT market by direct per-id lookup rather than
            # scanning the general ``?status=resolved`` list. The BTC 5-min
            # markets are tagged ``fast-5m`` and never appear in that general
            # list (the same reason discovery must pass ``tags=fast-5m``), so
            # the old list scan matched zero pending trades and every trade
            # expired at $0. ``GET /api/sdk/markets/{id}`` returns the resolved
            # outcome for these markets reliably. See resolve-troubleshooting
            # note: Simmer leaves ``resolved_at`` null even after resolution,
            # so ``status == 'resolved'`` + a non-null boolean ``outcome`` is
            # the authoritative resolution signal.
            resolved_map = {}
            for market_id in {t["market_id"] for t in pending}:
                state, outcome = self._fetch_market_outcome(headers, market_id)
                if state == "auth_error":
                    logger.warning(
                        "Resolver key rejected (401/403); trying next candidate key"
                    )
                    return (0, False)
                if state == "resolved":
                    resolved_map[market_id] = outcome
            if not resolved_map:
                return (0, True)

            count = 0
            for trade in pending:
                market_id = trade["market_id"]
                if market_id not in resolved_map:
                    continue
                if self._settle_trade(trade, resolved_map[market_id]):
                    count += 1

            if count > 0:
                logger.info(
                    f"Resolved {count} trades "
                    f"({sum(1 for t in pending if t['market_id'] in resolved_map)} "
                    f"pending matched {len(resolved_map)} resolved markets)"
                )
            return (count, True)
        except Exception as e:
            logger.error(f"_resolve_pending error: {e}")
            return (0, True)

    def _fetch_market_outcome(self, headers: dict, market_id: str):
        """Look a single market up by id and return ``(state, outcome)``.

        ``state`` is one of:
          * ``"resolved"`` — ``outcome`` is the boolean Up/Down result.
          * ``"pending"``  — market still active / not yet decided.
          * ``"auth_error"`` — key rejected (HTTP 401/403); caller should
            retry with a different key rather than expire the trade.
          * ``"error"``    — transport / non-auth HTTP error; treat as pending.

        Uses the direct ``/api/sdk/markets/{id}`` endpoint because fast-5m
        markets are absent from the general market list.
        """
        try:
            resp = requests.get(
                f"{config.SIMMER_BASE_URL}/api/sdk/markets/{market_id}",
                headers=headers,
                timeout=15,
            )
            if resp.status_code in (401, 403):
                return ("auth_error", None)
            if resp.status_code != 200:
                return ("error", None)
            data = resp.json()
            # The SDK wraps the payload as {"market": {...}}; older/list
            # responses return the market object directly.
            market = data.get("market", data) if isinstance(data, dict) else {}
            if market.get("status") != "resolved":
                return ("pending", None)
            outcome = market.get("outcome")
            # Only a concrete boolean counts as resolved; null/other means
            # "not yet decided" — leave the trade pending.
            if isinstance(outcome, bool):
                return ("resolved", outcome)
            return ("pending", None)
        except Exception as e:
            logger.debug(
                f"market outcome fetch failed for {market_id[:12]}...: {e}"
            )
            return ("error", None)

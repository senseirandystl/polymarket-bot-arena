"""Trade resolution thread.

Ticks once every ``config.RESOLVE_INTERVAL_SEC`` (60s by default). Each cycle it
builds one ``condition_id -> outcome`` map from Polymarket's recently-resolved
markets (``polymarket_markets.recent_resolutions``) and settles every pending
trade whose market has resolved, writing win/loss + fee-aware P&L.

There is intentionally **no stale-expiry sweep**: a pending trade stays pending
until its market actually resolves. Every cycle re-checks all pending trades, so
nothing is lost by waiting.
"""

import json
import logging
import threading

import config
import db
import polymarket_markets
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
        """Settle pending trades against Polymarket outcomes.

        Primary path: bulk map of recently *closed* series markets.
        Fallback: per-``market_id`` Gamma lookup for markets that already
        show extreme outcomePrices but are still ``closed=false`` (Gamma
        lag — these never appear in the closed bulk page and used to sit
        pending forever, firing resolver_stuck alerts).
        """
        with db.get_conn() as conn:
            pending = conn.execute(
                "SELECT id, market_id, bot_name, side, amount, shares_bought, "
                "fee, trade_features, reasoning FROM trades WHERE outcome IS NULL"
            ).fetchall()
        if not pending:
            return

        resolved = dict(polymarket_markets.recent_resolutions() or {})

        # Direct lookup for any pending market missing from the closed map.
        missing = {
            t["market_id"] for t in pending
            if t["market_id"] and t["market_id"] not in resolved
        }
        fallback_hits = 0
        for mid in missing:
            outcome = polymarket_markets.fetch_market_outcome(mid)
            if outcome is not None:
                resolved[mid] = outcome
                fallback_hits += 1
        if fallback_hits:
            logger.info(
                "Resolver fallback: settled outcomes for %d market(s) via "
                "direct Gamma lookup (closed-map miss / defacto resolve)",
                fallback_hits,
            )

        if not resolved:
            return

        count = 0
        matched = 0
        for trade in pending:
            outcome = resolved.get(trade["market_id"])
            if outcome is None:
                continue
            matched += 1
            if self._settle_trade(trade, outcome):
                count += 1
        if count > 0:
            logger.info(
                f"Resolved {count} trades ({matched} pending matched "
                f"{len(resolved)} resolved markets)"
            )
        # Stamp decision_events (incl. skips) with the same market outcomes
        # so offline rollups can score lanes without a placed trade.
        try:
            from arena.decision_log import resolve_from_resolution_map, flush
            flush()
            resolve_from_resolution_map(resolved)
        except Exception as e:
            logger.debug("decision_events resolve failed: %s", e)

    def _settle_trade(self, trade, market_outcome: bool) -> bool:
        """Write win/loss + fee-aware P&L + learning for one resolved trade.

        ``market_outcome`` is True when Up won, False when Down won. A YES/Up
        bet wins on True; a NO/Down bet wins on False. P&L nets the entry cost
        and the taker fee:

            win  → shares * $1  - amount - fee
            loss → -amount - fee
        """
        side = trade["side"]
        amount = trade["amount"] or 0.0
        try:
            shares = trade["shares_bought"] or 0
        except (IndexError, KeyError):
            shares = 0
        try:
            fee = trade["fee"] or 0.0
        except (IndexError, KeyError):
            fee = 0.0

        # A trade with no shares never really filled — nothing to settle.
        if shares <= 0:
            logger.warning(
                f"[{trade['bot_name']}] Skipping resolution: no shares on "
                f"{str(trade['market_id'])[:12]}… — leaving pending."
            )
            return False

        won = (market_outcome is True) if side == "yes" else (market_outcome is False)
        outcome = "win" if won else "loss"
        pnl = (shares - amount - fee) if won else (-amount - fee)
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
        # Online regime performance: attribute this P&L to the regime the
        # trade was opened in (stamped in trade_features as regime:<id>).
        try:
            rid = None
            if isinstance(features, list):
                for f in features:
                    if isinstance(f, str) and f.startswith("regime:") and not f.startswith("regime_legacy:"):
                        rid = f.split(":", 1)[1]
                        break
            if rid:
                from signals.regime_detector import get_detector
                get_detector().record_outcome(rid, pnl, won=won)
        except Exception:
            pass
        return True

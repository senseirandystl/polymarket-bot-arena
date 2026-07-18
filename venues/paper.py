"""Paper trading — full simulation against real Polymarket order books.

No order is ever submitted. Each "fill" is computed by walking the live CLOB
asks for the side's token (:mod:`polymarket_fills`), so slippage, depth and
Polymarket's taker fee are all modelled exactly as live trading would see them.
The only difference from :mod:`venues.live` is that live actually posts the order.

All paper bots share ONE virtual USDC bankroll (``db.get_paper_bankroll`` /
``get_paper_available``), set by the user in the dashboard Settings tab. A bot
cannot spend cash the shared pool does not have.
"""

import logging

import config
import db
import polymarket_fills
import polymarket_markets
from venues import TradeResult

logger = logging.getLogger("venues.paper")


class PaperEngine:
    _instance = None

    @classmethod
    def instance(cls) -> "PaperEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def place(self, *, bot_name, side, amount, market, mode,
              confidence=None, reasoning=None, features=None,
              target_shares=None, limit_price=None, expected_price=None,
              book=None) -> TradeResult:
        market_id = market.get("id") or market.get("market_id")
        token = (
            market.get("polymarket_token_id") if side == "yes"
            else market.get("polymarket_no_token_id")
        )
        if not token:
            return TradeResult(success=False, reason="missing_token_id")

        # Order book for THIS side's token. Normally a fresh read (never a cached
        # price), but callers that must fill ATOMICALLY on a snapshot they already
        # validated (the arbitrage bot's two legs) pass ``book`` so decision and
        # fill can't drift apart. See config.MAX_FILL_SLIPPAGE / BUG_HISTORY.
        if book is None:
            book = polymarket_markets.get_order_book(token)
        if not book.get("valid"):
            logger.debug(f"[{bot_name}] No order book for {str(market_id)[:12]}… — skip")
            return TradeResult(success=False, reason="no_book")

        # Shared bankroll gate: can't spend more cash than the pool holds.
        available = db.get_paper_available()
        if available <= 0:
            logger.info(f"[{bot_name}] Paper bankroll exhausted (${available:.2f}) — skip")
            return TradeResult(success=False, reason="insufficient_bankroll")

        # Two sizing modes:
        #  * share-matched (``target_shares`` set, used by the arbitrage bot):
        #    fill an EXACT share count so both legs stay balanced. The pair must
        #    fill in full — a partial share fill would unbalance the arb — and it
        #    must be affordable from the shared pool.
        #  * USD-budget (default): spend up to ``amount`` (capped by the pool).
        if target_shares is not None:
            fill = polymarket_fills.simulate_fill_shares(book, target_shares)
            if not fill["filled"] or not fill["full"]:
                logger.debug(
                    f"[{bot_name}] Insufficient depth for {target_shares:.2f} sh "
                    f"on {str(market_id)[:12]}… — skip (share-matched)"
                )
                return TradeResult(success=False, reason="insufficient_depth")
            if fill["cost"] + fill["fee"] > available:
                logger.info(
                    f"[{bot_name}] Paper bankroll ${available:.2f} < arb leg cost "
                    f"${fill['cost'] + fill['fee']:.2f} — skip"
                )
                return TradeResult(success=False, reason="insufficient_bankroll")
        else:
            spend = min(amount, available)
            # Simulate the fill by walking the real book (depth + slippage).
            fill = polymarket_fills.simulate_fill(book, spend)
        min_size = book.get("min_order_size", 0) or 0
        if not fill["filled"] or fill["shares"] < min_size:
            logger.debug(
                f"[{bot_name}] Fill too small on {str(market_id)[:12]}… "
                f"(shares={fill['shares']:.2f} < min {min_size}) — skip"
            )
            return TradeResult(success=False, reason="below_min_size")

        # Slippage guard: the book may have moved between the bot's decision and
        # this fill. If the realized avg BUY price drifted above the caller's
        # limit, reject rather than fill into a worse-than-expected price. This
        # is what keeps thin edges (esp. arbitrage) from filling at a loss.
        if limit_price is not None and fill["avg_price"] > limit_price + 1e-9:
            logger.info(
                f"[{bot_name}] Slippage guard: fill {fill['avg_price']:.3f} > "
                f"limit {limit_price:.3f} on {str(market_id)[:12]}… — reject"
            )
            return TradeResult(success=False, reason="slippage_exceeded")

        # Symmetric band (BUG #28): a fill far BELOW expectation is not a
        # bargain — it means the book moved materially since the decision and
        # the inputs are stale (live: 9 fills >5c under the decision ask, one
        # at 0.06 seconds before expiry; 22% WR). Reject in both directions.
        if (expected_price is not None
                and abs(fill["avg_price"] - expected_price)
                > config.MAX_FILL_SLIPPAGE + 1e-9):
            logger.info(
                f"[{bot_name}] Slippage guard: fill {fill['avg_price']:.3f} vs "
                f"expected {expected_price:.3f} (±{config.MAX_FILL_SLIPPAGE:.2f}) "
                f"on {str(market_id)[:12]}… — reject (stale data)"
            )
            return TradeResult(success=False, reason="slippage_band")

        row_id = db.log_trade(
            bot_name=bot_name,
            market_id=market_id,
            market_question=market.get("question"),
            side=side,
            amount=fill["cost"],          # USDC actually spent on shares
            venue="polymarket",
            mode=mode,
            confidence=confidence,
            reasoning=reasoning,
            trade_id=None,                # no real order in paper mode
            shares_bought=fill["shares"],
            trade_features=features,
            fill_source="paper_sim",
            entry_price=fill["avg_price"],
            fee=fill["fee"],
        )
        logger.info(
            f"[{bot_name}] Paper fill: {side} ${fill['cost']:.2f} @ "
            f"{fill['avg_price']:.3f} ({fill['shares']:.2f} sh, fee ${fill['fee']:.3f}"
            f"{'' if fill['full'] else ', PARTIAL'}) on "
            f"{str(market.get('question', ''))[:40]}"
        )
        return TradeResult(
            success=True, trade_id=str(row_id), fill_source="paper_sim",
            shares=fill["shares"], entry_price=fill["avg_price"],
        )

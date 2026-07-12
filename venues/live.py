"""Live trading on the Polymarket CLOB.

Fully wired but only exercised when a bot's ``trading_mode`` is ``'live'``
(the arena starts in paper and every bot defaults to paper). All order
placement goes through :mod:`polymarket_client`, which uses the CLOB
``create_market_order`` path (auto-resolves tick size, neg-risk and fee rate).

Kept deliberately separate from :mod:`venues.paper` so live/Polymarket and
paper/Simmer code never share a path.
"""

import logging

import db
from venues import TradeResult

logger = logging.getLogger("venues.live")


class LiveEngine:
    _instance = None

    @classmethod
    def instance(cls) -> "LiveEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def place(self, *, bot_name, side, amount, market, mode,
              confidence=None, reasoning=None, features=None) -> TradeResult:
        import polymarket_client

        market_id = market.get("id") or market.get("market_id")
        token_id = (
            market.get("polymarket_token_id") if side == "yes"
            else market.get("polymarket_no_token_id")
        )
        if not token_id:
            logger.error(
                f"[{bot_name}] No Polymarket token id for side={side} on "
                f"{str(market.get('question', ''))[:50]}"
            )
            return TradeResult(success=False, reason="missing_token_id")

        neg_risk = bool(market.get("polymarket_neg_risk"))
        result = polymarket_client.place_market_order(
            token_id=token_id, side=side, amount=amount, neg_risk=neg_risk,
        )
        if not result.get("success"):
            logger.error(f"[{bot_name}] LIVE order failed: {result.get('error')}")
            return TradeResult(success=False, reason=result.get("error"))

        import polymarket_fills

        price = float(result.get("price") or 0.0)
        shares = float(result.get("size") or (amount / price if price else 0.0))
        # Polymarket charges the taker fee on-chain; record our estimate so
        # paper and live P&L are computed the same way.
        fee = polymarket_fills.taker_fee(shares, price)
        row_id = db.log_trade(
            bot_name=bot_name,
            market_id=market_id,
            market_question=market.get("question"),
            side=side,
            amount=amount,
            venue="polymarket",
            mode=mode,
            confidence=confidence,
            reasoning=reasoning,
            trade_id=result.get("order_id"),
            shares_bought=shares,
            trade_features=features,
            fill_source="polymarket",
            entry_price=price,
            fee=fee,
        )
        logger.info(
            f"[{bot_name}] LIVE fill: {side} ${amount:.2f} @ {price} "
            f"({shares} sh) on {str(market.get('question', ''))[:40]}"
        )
        return TradeResult(
            success=True, trade_id=str(row_id), fill_source="polymarket",
            shares=shares, entry_price=price,
        )

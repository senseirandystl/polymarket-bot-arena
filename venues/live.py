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
              confidence=None, reasoning=None, features=None,
              target_shares=None, limit_price=None, expected_price=None,
              book=None, context=None) -> TradeResult:
        # ``target_shares`` (share-matched arb sizing) is accepted for a uniform
        # engine signature. The CLOB market-order path is USD-denominated, so we
        # convert to a USD budget when a share target is supplied and no explicit
        # amount was set — the on-chain match still fills at book prices.
        # ``book`` (a validated snapshot) is accepted for signature parity with
        # the paper engine and used only to estimate slippage below; a live order
        # always executes on-chain against the real book.
        import polymarket_client
        import polymarket_fills
        import polymarket_markets

        book_side = (
            market.get("polymarket_token_id") if side == "yes"
            else market.get("polymarket_no_token_id")
        )
        if target_shares is not None and not amount:
            try:
                est = polymarket_fills.simulate_fill_shares(
                    book if book is not None
                    else polymarket_markets.get_order_book(book_side),
                    target_shares,
                )
                amount = est["cost"] + est["fee"]
            except Exception:
                pass

        # Fail-safe slippage guard: estimate the fill avg from the current book
        # and refuse to submit if it already exceeds the caller's limit — or,
        # when the caller supplied its decision price, deviates from it in
        # EITHER direction by more than MAX_FILL_SLIPPAGE (a fill far below
        # expectation means the book moved and the inputs are stale — BUG #28).
        # Can only PREVENT an order, never place a larger one.
        if limit_price is not None or expected_price is not None:
            try:
                import config
                probe = book if book is not None \
                    else polymarket_markets.get_order_book(book_side)
                est = (polymarket_fills.simulate_fill_shares(probe, target_shares)
                       if target_shares is not None
                       else polymarket_fills.simulate_fill(probe, amount or 0.0))
                if est.get("filled"):
                    over_limit = (limit_price is not None
                                  and est["avg_price"] > limit_price + 1e-9)
                    out_of_band = (expected_price is not None
                                   and abs(est["avg_price"] - expected_price)
                                   > config.MAX_FILL_SLIPPAGE + 1e-9)
                    if over_limit or out_of_band:
                        logger.info(
                            f"[{bot_name}] LIVE slippage guard: est "
                            f"{est['avg_price']:.3f} vs limit={limit_price} "
                            f"expected={expected_price} — refuse to submit"
                        )
                        return TradeResult(success=False, reason="slippage_exceeded")
            except Exception:
                pass

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
        import config
        use_limit = (
            getattr(config, "ORDER_STYLE", "limit") == "limit"
            and target_shares is None
        )
        if use_limit:
            # Size in shares; price from caller limit or book-derived mode.
            mid = (market.get("current_price") if side == "yes"
                   else market.get("no_price"))
            probe = book if book is not None else polymarket_markets.get_order_book(
                book_side)
            lim = limit_price
            if lim is None:
                lim = polymarket_fills.limit_buy_price(probe, mid=mid)
            if lim is None or lim <= 0:
                return TradeResult(success=False, reason="no_limit_price")
            shares_req = (float(target_shares) if target_shares is not None
                          else (float(amount or 0.0) / lim))
            if shares_req < getattr(config, "POLYMARKET_MIN_SHARES", 5):
                shares_req = float(getattr(config, "POLYMARKET_MIN_SHARES", 5))
            result = polymarket_client.place_limit_order(
                token_id=token_id,
                side="buy",
                size=shares_req,
                price=float(lim),
                order_type="GTC",
                neg_risk=neg_risk,
            )
            if not result.get("success"):
                logger.error(
                    f"[{bot_name}] LIVE limit order failed: {result.get('error')}")
                return TradeResult(success=False, reason=result.get("error"))
            status = (result.get("status") or "").lower()
            # Only book a position when the CLOB reports an immediate match.
            # Resting orders need a fill watcher (future); do not invent PnL.
            if status not in ("matched", "filled"):
                logger.info(
                    f"[{bot_name}] LIVE limit resting ({status}) "
                    f"{shares_req:.2f}sh @ {lim:.3f} — not logged as fill"
                )
                return TradeResult(
                    success=False, reason=f"limit_resting:{status or 'live'}")
            price = float(result.get("price") or lim)
            shares = float(result.get("size") or shares_req)
            # Matched at our limit without walking asks → treat as maker.
            is_maker = abs(price - float(lim)) <= 1e-6
            fee = polymarket_fills.trading_fee(shares, price, is_maker=is_maker)
            amount_out = shares * price
        else:
            result = polymarket_client.place_market_order(
                token_id=token_id, side=side, amount=amount, neg_risk=neg_risk,
            )
            if not result.get("success"):
                logger.error(
                    f"[{bot_name}] LIVE order failed: {result.get('error')}")
                return TradeResult(success=False, reason=result.get("error"))
            price = float(result.get("price") or 0.0)
            shares = float(result.get("size") or (amount / price if price else 0.0))
            fee = polymarket_fills.taker_fee(shares, price)
            amount_out = amount

        row_id = db.log_trade(
            bot_name=bot_name,
            market_id=market_id,
            market_question=market.get("question"),
            side=side,
            amount=amount_out,
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
            context=context,
        )
        logger.info(
            f"[{bot_name}] LIVE fill: {side} ${amount_out:.2f} @ {price} "
            f"({shares} sh, fee ${fee:.3f}) on "
            f"{str(market.get('question', ''))[:40]}"
        )
        return TradeResult(
            success=True, trade_id=str(row_id), fill_source="polymarket",
            shares=shares, entry_price=price,
        )

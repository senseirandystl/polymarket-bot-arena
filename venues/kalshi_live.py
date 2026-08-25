"""Live Kalshi order placement. Never used unless mode=live and exchange=kalshi."""

from __future__ import annotations

import logging

import config
import db
from venues import TradeResult

logger = logging.getLogger("venues.kalshi_live")


class KalshiLiveEngine:
    _instance = None

    @classmethod
    def instance(cls) -> "KalshiLiveEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def place(self, *, bot_name, side, amount, market, mode,
              confidence=None, reasoning=None, features=None,
              target_shares=None, limit_price=None, expected_price=None,
              book=None, context=None) -> TradeResult:
        if mode != "live":
            return TradeResult(success=False, reason="not_live")
        try:
            from exchanges import exchange_enabled, KALSHI
            if not exchange_enabled(KALSHI):
                return TradeResult(success=False, reason="kalshi_disabled")
        except Exception:
            logger.warning("[%s] Kalshi live blocked — toggle check failed", bot_name)
            return TradeResult(success=False, reason="kalshi_toggle_error")
        try:
            import kalshi_client
            if not kalshi_client.has_auth():
                logger.warning("[%s] Kalshi live skipped — missing API keys", bot_name)
                return TradeResult(success=False, reason="kalshi_no_credentials")
        except Exception as e:
            return TradeResult(success=False, reason=f"kalshi_client:{e}")

        from exchanges import native_market_id
        ticker = native_market_id(
            str(market.get("ticker") or market.get("native_id") or market.get("id") or "")
        )
        if not ticker:
            return TradeResult(success=False, reason="missing_ticker")
        yes = side == "yes"
        # Count = contracts. Prefer target_shares; else amount / price.
        price = float(expected_price or (market.get("yes_ask") if yes else market.get("no_ask")) or 0.5)
        price = min(0.99, max(0.01, price))
        count = float(target_shares or 0.0)
        if count <= 0 and amount:
            count = float(amount) / price
        count = max(1, int(round(count)))

        # Pre-submit slippage probe — refuse to POST if the book already
        # disagrees with the decision price (BUG #28 analog). Never retries
        # the order POST (a 5xx after ack can double-fill).
        if limit_price is not None or expected_price is not None:
            try:
                import polymarket_fills
                probe = book
                if not (probe and probe.get("valid")):
                    import kalshi_markets
                    both = kalshi_markets.get_order_book(ticker)
                    probe = (both or {}).get("yes" if yes else "no")
                if not (probe and probe.get("valid")):
                    logger.info(
                        "[%s] Kalshi live slippage guard: no book — refuse",
                        bot_name,
                    )
                    return TradeResult(success=False, reason="slippage_no_book")
                est = (
                    polymarket_fills.simulate_fill_shares(probe, target_shares)
                    if target_shares is not None
                    else polymarket_fills.simulate_fill(probe, amount or 0.0)
                )
                if est.get("filled"):
                    over_limit = (
                        limit_price is not None
                        and est["avg_price"] > limit_price + 1e-9
                    )
                    out_of_band = (
                        expected_price is not None
                        and abs(est["avg_price"] - expected_price)
                        > config.MAX_FILL_SLIPPAGE + 1e-9
                    )
                    if over_limit or out_of_band:
                        logger.info(
                            "[%s] Kalshi live slippage guard: est "
                            "%.3f vs limit=%s expected=%s — refuse",
                            bot_name, est["avg_price"], limit_price,
                            expected_price,
                        )
                        return TradeResult(
                            success=False, reason="slippage_exceeded",
                        )
            except Exception as e:
                logger.warning(
                    "[%s] Kalshi live slippage probe failed (%s) — refuse",
                    bot_name, e,
                )
                return TradeResult(success=False, reason="slippage_probe_error")

        # Kalshi has no true market order. Cap at decision + slippage as FOK
        # limit so a thin book cannot walk past the probe (venues/live analog).
        if limit_price is not None:
            cap = float(limit_price)
        elif expected_price is not None:
            cap = float(expected_price) + float(config.MAX_FILL_SLIPPAGE)
        else:
            return TradeResult(success=False, reason="no_limit_price")
        cap = min(0.99, max(0.01, cap))
        cap_cents = max(1, min(99, int(round(cap * 100.0))))

        import uuid
        body = {
            "ticker": ticker,
            "side": "yes" if yes else "no",
            "action": "buy",
            "count": count,
            "type": "limit",
            "time_in_force": "fill_or_kill",
            "client_order_id": str(uuid.uuid4()),
        }
        if yes:
            body["yes_price"] = cap_cents
        else:
            body["no_price"] = cap_cents
        try:
            resp = kalshi_client.request(
                "POST", "/portfolio/orders", json_body=body, timeout=15,
                auth=True, retries=0,
            )
        except Exception as e:
            logger.error("[%s] Kalshi order failed: %s", bot_name, e)
            return TradeResult(success=False, reason="kalshi_http")
        if resp is None or resp.status_code >= 400:
            logger.warning(
                "[%s] Kalshi order HTTP %s %s",
                bot_name, getattr(resp, "status_code", None),
                (getattr(resp, "text", "") or "")[:180],
            )
            return TradeResult(success=False, reason="kalshi_rejected")
        data = {}
        try:
            data = resp.json() or {}
        except Exception:
            data = {}
        order = data.get("order") or data
        def _px(*keys):
            for k in keys:
                try:
                    v = float(order.get(k))
                except (TypeError, ValueError):
                    continue
                if v > 1.0 + 1e-9:
                    v = v / 100.0
                if 0.0 < v < 1.0:
                    return v
            return None
        fill_px = _px(
            "yes_price_dollars" if yes else "no_price_dollars",
            "avg_price_dollars", "avg_price",
            "yes_price" if yes else "no_price",
        )
        if fill_px is None:
            fill_px = price
        fill_px = min(0.99, max(0.01, float(fill_px)))
        status = str(order.get("status") or "").lower()
        if status in ("canceled", "cancelled", "rejected"):
            return TradeResult(success=False, reason="kalshi_rejected")
        fill_count = order.get("fill_count")
        if fill_count is None:
            fill_count = order.get("filled_count")
        remaining = order.get("remaining_count")
        shares = None
        if fill_count is not None:
            try:
                shares = float(fill_count)
            except (TypeError, ValueError):
                shares = None
        if shares is None and remaining is not None:
            try:
                shares = max(0.0, float(count) - float(remaining))
            except (TypeError, ValueError):
                shares = None
        if shares is None and status in ("executed", "filled"):
            try:
                shares = float(order.get("count") or count)
            except (TypeError, ValueError):
                shares = float(count)
        if shares is None or shares <= 0:
            logger.info(
                "[%s] Kalshi order not filled (status=%s remaining=%s)",
                bot_name, status or "?", remaining,
            )
            return TradeResult(success=False, reason="kalshi_unfilled")
        cost = shares * fill_px
        from kalshi_markets import kalshi_taker_fee
        fee = kalshi_taker_fee(shares, fill_px)
        oid = str(order.get("order_id") or order.get("id") or "")
        row_id = db.log_trade(
            bot_name=bot_name,
            market_id=market.get("id") or ticker,
            market_question=market.get("question"),
            side=side,
            amount=cost,
            venue="kalshi",
            mode="live",
            confidence=confidence,
            reasoning=reasoning,
            trade_id=oid or None,
            shares_bought=shares,
            trade_features=features,
            fill_source="kalshi",
            entry_price=fill_px,
            fee=fee,
            context=context,
        )
        logger.info(
            "[%s] Kalshi live fill: %s %s @ %.3f (%s sh) %s",
            bot_name, side, ticker, fill_px, shares, oid,
        )
        return TradeResult(
            success=True, trade_id=str(row_id), fill_source="kalshi",
            shares=shares, entry_price=fill_px,
        )

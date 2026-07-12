"""Paper trading — local simulated fills (primary) + optional Simmer mirror.

Why local fills: Simmer's free SDK tier caps at **50 buys/day per account**.
Past that, ``POST /api/sdk/trade`` returns HTTP 200 with ``success:false`` and
no ``trade_id``; the old code logged those as real trades and fabricated a
share count, inflating P&L with "phantom" fills. See BUG_HISTORY.md.

Instead we price each paper fill LOCALLY from the real market snapshot
(``shares = amount / entry_price``) and let the resolver settle it against the
real market outcome. This makes paper trading an honest, UNLIMITED simulation
that is independent of Simmer's cap — ideal for strategy evaluation. Simmer is
used only as an opt-in real-account cross-check (``config.SIMMER_MIRROR_ENABLED``).
"""

import logging

import requests

import config
import db
from venues import TradeResult

logger = logging.getLogger("venues.paper")


def entry_price_for(market: dict, side: str):
    """Per-share fill price for ``side`` from the market's YES price.

    Returns ``None`` when the market has no usable price (so the caller skips
    rather than logging a 0-share phantom).
    """
    yes = market.get("current_price")
    try:
        yes = float(yes)
    except (TypeError, ValueError):
        return None
    if not (0.0 < yes < 1.0):
        return None
    return (1.0 - yes) if side == "no" else yes


class PaperEngine:
    _instance = None

    @classmethod
    def instance(cls) -> "PaperEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def place(self, *, bot_name, side, amount, market, mode,
              confidence=None, reasoning=None, features=None) -> TradeResult:
        market_id = market.get("id") or market.get("market_id")
        price = entry_price_for(market, side)
        if price is None:
            logger.warning(
                f"[{bot_name}] No usable market price for "
                f"{str(market_id)[:12]}… — skipping paper fill (no phantom)."
            )
            return TradeResult(success=False, reason="no_market_price")

        shares = amount / price
        fill_source = "local_sim"
        trade_id = None

        if config.SIMMER_MIRROR_ENABLED:
            mirror = self._mirror_to_simmer(
                bot_name, market_id, side, amount, reasoning,
            )
            if mirror is not None:
                fill_source = "simmer"
                trade_id = mirror.get("trade_id")
                if mirror.get("shares"):
                    shares = float(mirror["shares"])
                if mirror.get("price"):
                    price = float(mirror["price"])

        row_id = db.log_trade(
            bot_name=bot_name,
            market_id=market_id,
            market_question=market.get("question"),
            side=side,
            amount=amount,
            venue="simmer",
            mode=mode,
            confidence=confidence,
            reasoning=reasoning,
            trade_id=trade_id,
            shares_bought=shares,
            trade_features=features,
            fill_source=fill_source,
            entry_price=price,
        )
        logger.info(
            f"[{bot_name}] Paper fill ({fill_source}): {side} ${amount:.2f} "
            f"@ {price:.4f} ({shares:.4f} sh) on "
            f"{str(market.get('question', ''))[:40]}"
        )
        return TradeResult(
            success=True, trade_id=str(row_id), fill_source=fill_source,
            shares=shares, entry_price=price,
        )

    def _mirror_to_simmer(self, bot_name, market_id, side, amount, reasoning):
        """Best-effort real Simmer trade for cross-checking.

        Returns a dict on a CONFIRMED fill (``success:true`` + ``trade_id``),
        else ``None`` (rate-limited, rejected, or transport error) — in which
        case the caller keeps the local-sim fill.
        """
        api_key = config.get_credential("simmer_api_key")
        if not api_key:
            return None
        try:
            resp = requests.post(
                f"{config.SIMMER_BASE_URL}/api/sdk/trade",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "market_id": market_id,
                    "side": side,
                    "amount": amount,
                    "venue": "sim",
                    "source": f"arena:{bot_name}",
                    "reasoning": reasoning or "",
                },
                timeout=30,
            )
            if resp.status_code not in (200, 201):
                return None
            data = resp.json()
            # HTTP 200 with success:false is Simmer's rate-limit rejection.
            if not data.get("success") or not data.get("trade_id"):
                logger.debug(
                    f"[{bot_name}] Simmer mirror not filled: {data.get('error')}"
                )
                return None
            return {
                "trade_id": data.get("trade_id"),
                "shares": data.get("shares_bought") or data.get("shares"),
                "price": data.get("avg_price"),
            }
        except Exception as e:
            logger.debug(f"[{bot_name}] Simmer mirror error: {e}")
            return None

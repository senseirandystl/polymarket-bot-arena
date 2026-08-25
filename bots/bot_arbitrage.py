"""Arbitrage bot — market-neutral cross-book arb on a single Polymarket window.

The classic Polymarket arb: on ONE market, buy both YES and NO whenever the two
best asks sum to less than $1.00 with enough margin to cover both legs' taker
fees. At resolution exactly one side pays $1/share, so a *matched* pair of shares
locks in::

    profit_per_pair = 1.0 - (yes_ask + no_ask + fee_yes + fee_no)

regardless of which way BTC moves — this bot has no directional view. Because the
edge is small and fleeting, the bot is fast (evaluates every trader tick),
precise (reads the real best asks, prices fees per leg), and only fires when the
net edge clears ``config.ARBITRAGE_MIN_MARGIN``.

Design notes:
  * This bot **overrides** ``make_decision`` and ``execute`` — it does NOT use the
    directional signal stack, NO-bet ban, or consensus guards in ``BaseBot``
    (those are all directional; arb is not).
  * It places TWO legs. Both must fill for the position to be a true arb; a
    one-legged fill is naked directional risk, so ``execute`` reports success
    only when both legs fill and logs a warning otherwise.
  * It is excluded from evolution (see ``arena.EVOLUTION_EXEMPT_TYPES``) — a
    market-neutral bot should not be culled by a directional win-rate threshold.
"""

import logging
import threading
import time

import config
import db
import polymarket_markets
from bots.base_bot import BaseBot, strategy_decision
from polymarket_fills import simulate_fill_shares
from venues import get_engine

logger = logging.getLogger("bots.arbitrage")

DEFAULT_PARAMS = {
    "min_margin": config.ARBITRAGE_MIN_MARGIN,
    "target_shares": config.ARBITRAGE_TARGET_SHARES,
}


class ArbitrageBot(BaseBot):
    def __init__(self, name="arbitrage-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="arbitrage",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )
        # Micro-cache so re-evaluating the same market on back-to-back 1s ticks
        # doesn't double-hit the CLOB /book endpoint. Fresh enough for arb.
        self._book_cache: dict[str, tuple[float, dict]] = {}
        self._book_lock = threading.Lock()

    # ``analyze`` is abstract on BaseBot but unused here — arb overrides the
    # whole decision path. Return a neutral hold so nothing calls into it.
    def analyze(self, market: dict, signals: dict) -> dict:
        return strategy_decision(
            "hold", reasoning="arbitrage uses make_decision override")

    def _book(self, token_id: str) -> dict:
        """Fetch a token's book with a sub-second micro-cache."""
        now = time.time()
        with self._book_lock:
            hit = self._book_cache.get(token_id)
            if hit and (now - hit[0]) < config.ARBITRAGE_BOOK_CACHE_SEC:
                return hit[1]
        book = polymarket_markets.get_order_book(token_id)
        with self._book_lock:
            self._book_cache[token_id] = (now, book)
        return book

    @staticmethod
    def _skip(reason: str, edge: float = 0.0, signals: dict | None = None) -> dict:
        return strategy_decision("skip", edge=edge, reasoning=reason,
                                 signals=signals or {})

    def make_decision(self, market: dict, signals: dict) -> dict:
        from exchanges import KALSHI, exchange_of
        is_kalshi = exchange_of(market) == KALSHI
        yes_tok = market.get("polymarket_token_id")
        no_tok = market.get("polymarket_no_token_id")
        if not is_kalshi and (not yes_tok or not no_tok):
            return self._skip("arb: missing token ids")

        # Prefer the shared warm cache (refreshed <=1s by the market-data
        # warmer) so the arb bot makes ZERO network calls on the hot path. Fall
        # back to a direct (micro-cached) book read only if the warmer has not
        # primed this market yet. Lazy import avoids any import-order coupling.
        yes_book = no_book = None
        from arena.market_data import store
        market_id = market.get("id") or market.get("market_id")
        warm = store().get(market_id)
        if warm is not None:
            yes_book, no_book = warm.get("yes_book"), warm.get("no_book")
        if not (yes_book and yes_book.get("valid")
                and no_book and no_book.get("valid")):
            if is_kalshi:
                try:
                    import kalshi_markets
                    both = kalshi_markets.get_order_book(
                        market.get("ticker") or market_id
                    )
                    yes_book = (both or {}).get("yes")
                    no_book = (both or {}).get("no")
                except Exception:
                    yes_book = no_book = None
            else:
                yes_book = self._book(yes_tok)
                no_book = self._book(no_tok)
        if not yes_book or not no_book:
            return self._skip("arb: no book on one side")
        if not yes_book.get("valid") or not no_book.get("valid"):
            return self._skip("arb: no book on one side")

        if yes_book.get("best_ask") is None or no_book.get("best_ask") is None:
            return self._skip("arb: missing ask on one side")

        # --- Depth-aware, share-matched sizing -----------------------------
        # The old bug: edge was measured on the top-of-book best_ask (one share)
        # but the position was sized to ~20 shares, so walking deeper into the
        # book made the REALIZED cost exceed $1/pair — every "arb" lost. And the
        # two legs were sized in dollars independently, so they filled DIFFERENT
        # share counts, leaving naked directional risk.
        #
        # The fix: size BOTH legs to the SAME share count, and compute the edge
        # from the actual VWAP of walking each book to that depth (not best_ask).
        target = self.strategy_params.get(
            "target_shares", config.ARBITRAGE_TARGET_SHARES
        )
        max_pos = config.get_max_position()

        # 1) Depth cap: can't take more shares than either book actually offers.
        fy = simulate_fill_shares(yes_book, target)
        fn = simulate_fill_shares(no_book, target)
        shares = min(fy["shares"], fn["shares"])
        if shares <= 0:
            return self._skip("arb: no fillable depth on one side")

        # 2) Per-leg position cap: neither leg's USD cost may exceed max_pos.
        #    Re-solve at the matched share count, then scale down if a leg is
        #    over budget (using its VWAP as a linear approximation).
        fy = simulate_fill_shares(yes_book, shares)
        fn = simulate_fill_shares(no_book, shares)
        for f in (fy, fn):
            if f["cost"] > max_pos and f["avg_price"] > 0:
                shares = min(shares, max_pos / f["avg_price"])
        if shares < config.POLYMARKET_MIN_SHARES:
            return self._skip(
                f"arb: matched size {shares:.2f}sh < min "
                f"{config.POLYMARKET_MIN_SHARES}"
            )
        fy = simulate_fill_shares(yes_book, shares)
        fn = simulate_fill_shares(no_book, shares)
        shares = min(fy["shares"], fn["shares"])

        # 3) Edge from REALIZED cost. At resolution one leg pays $1/share on
        #    ``shares`` shares; net = shares - (yes_cost + no_cost + fees).
        net_cost = fy["cost"] + fn["cost"] + fy["fee"] + fn["fee"]
        edge_total = shares - net_cost
        edge = edge_total / shares  # locked-in profit per matched pair

        # Fee attribution: the edge above is already NET of both legs' taker
        # fees (fees are inside net_cost) — the threshold below therefore
        # compares fee-adjusted lock-in against the margin floor, never a
        # gross yes+no<1 gap that fees would eat.
        fee_per_pair = (fy["fee"] + fn["fee"]) / shares
        arb_signals = {"yes_vwap": fy["avg_price"], "no_vwap": fn["avg_price"],
                       "fee_per_pair": fee_per_pair, "shares": shares,
                       "net_cost": net_cost}

        min_margin = self.strategy_params.get("min_margin", config.ARBITRAGE_MIN_MARGIN)
        if edge < min_margin:
            return self._skip(
                f"arb: no edge (yes_vwap={fy['avg_price']:.3f}+"
                f"no_vwap={fn['avg_price']:.3f}, edge={edge:+.4f}/pair"
                f"<{min_margin:.3f}, fees={fee_per_pair:.4f}/pair) "
                f"@ {shares:.1f}sh",
                edge=edge, signals=arb_signals,
            )

        reasoning = (
            f"ARB edge={edge:+.4f}/pair x{shares:.1f}sh "
            f"(yes_vwap={fy['avg_price']:.3f} no_vwap={fn['avg_price']:.3f} "
            f"fees={fee_per_pair:.4f}/pair net=${net_cost:.2f} "
            f"lock=${edge_total:.2f})"
        )
        return strategy_decision(
            "buy", "yes",  # nominal side — this is a two-legged trade (see 'legs')
            edge=edge,
            confidence=min(0.95, edge * 10),
            reasoning=reasoning,
            signals=arb_signals,
            suggested_amount=fy["cost"] + fn["cost"],
            legs=[
                {"side": "yes", "shares": shares, "amount": fy["cost"],
                 "vwap": fy["avg_price"]},
                {"side": "no", "shares": shares, "amount": fn["cost"],
                 "vwap": fn["avg_price"]},
            ],
            features=None,
        )

    def execute(self, signal: dict, market: dict) -> dict:
        """Place BOTH legs. Both must fill for a genuine (neutral) arb.

        The decision that produced ``signal`` was made on an earlier book
        snapshot. Before committing capital we RE-READ both books, re-simulate
        the matched fill, and re-check that the combined edge still clears the
        margin — the fleeting sub-$1 window the decision saw is often gone by
        now (YES+NO rests just above $1). If the edge holds we fill each leg
        against the EXACT snapshot we just validated (passed to the engine via
        ``book=``) plus a per-leg ``limit_price``, so the fill can't drift.
        """
        if self._paused:
            return {"success": False, "reason": "bot_paused"}
        if signal.get("action") != "buy" or not signal.get("legs"):
            return {"success": False, "reason": "no_legs"}

        # Risk engine gate (kill switch / portfolio pause / daily loss).
        try:
            from arena.risk_engine import pre_trade
            mode_hint = db.get_bot_mode(self.name)
            risk = pre_trade(self.name, mode=mode_hint,
                             amount=float(signal.get("suggested_amount") or 0))
            if not risk.allow:
                if risk.action in ("pause", "kill"):
                    self._paused = True
                logger.warning(f"[{self.name}] ARB risk block: {risk.reason}")
                return {"success": False, "reason": risk.reason}
        except Exception as e:
            logger.warning(f"[{self.name}] ARB risk check failed (continuing): {e}")

        from exchanges import KALSHI, exchange_of
        is_kalshi = exchange_of(market) == KALSHI
        yes_tok = market.get("polymarket_token_id")
        no_tok = market.get("polymarket_no_token_id")
        if not is_kalshi and (not yes_tok or not no_tok):
            return {"success": False, "reason": "missing_token_ids"}

        # --- Atomic re-validation on a fresh snapshot ----------------------
        yes_book = no_book = None
        if is_kalshi:
            market_id = market.get("id") or market.get("market_id")
            try:
                from arena.market_data import store
                warm = store().get(market_id)
                if warm is not None:
                    yes_book, no_book = warm.get("yes_book"), warm.get("no_book")
            except Exception:
                yes_book = no_book = None
            if not (yes_book and yes_book.get("valid")
                    and no_book and no_book.get("valid")):
                try:
                    import kalshi_markets
                    both = kalshi_markets.get_order_book(
                        market.get("ticker") or market_id
                    )
                    yes_book = (both or {}).get("yes")
                    no_book = (both or {}).get("no")
                except Exception:
                    yes_book = no_book = None
        else:
            yes_book = self._book(yes_tok)
            no_book = self._book(no_tok)
        if not yes_book or not no_book:
            return {"success": False, "reason": "arb_book_gone"}
        if not yes_book.get("valid") or not no_book.get("valid"):
            return {"success": False, "reason": "arb_book_gone"}

        target = min(leg.get("shares", 0) for leg in signal["legs"])
        fy = simulate_fill_shares(yes_book, target)
        fn = simulate_fill_shares(no_book, target)
        shares = min(fy["shares"], fn["shares"])
        min_sh = 1 if is_kalshi else config.POLYMARKET_MIN_SHARES
        if shares < min_sh:
            return {"success": False, "reason": "arb_depth_gone"}
        # Re-match at the achievable share count on both legs.
        fy = simulate_fill_shares(yes_book, shares)
        fn = simulate_fill_shares(no_book, shares)
        shares = min(fy["shares"], fn["shares"])
        fy = simulate_fill_shares(yes_book, shares)
        fn = simulate_fill_shares(no_book, shares)

        net_cost = fy["cost"] + fn["cost"] + fy["fee"] + fn["fee"]
        edge = (shares - net_cost) / shares if shares > 0 else -1.0
        min_margin = self.strategy_params.get("min_margin", config.ARBITRAGE_MIN_MARGIN)
        if edge < min_margin:
            logger.info(
                f"[{self.name}] ARB edge gone at fill "
                f"(yes_vwap={fy['avg_price']:.3f}+no_vwap={fn['avg_price']:.3f}, "
                f"edge={edge:+.4f}/pair<{min_margin:.3f}) — abort, no legs placed"
            )
            return {"success": False, "reason": "arb_edge_gone"}

        self.trading_mode = db.get_bot_mode(self.name)
        mode = self.trading_mode
        from exchanges import exchange_of
        engine = get_engine(mode, exchange=exchange_of(market))

        # --- BOTH legs must be affordable BEFORE leg 1 is placed ------------
        # The pool is shared with the directional bots; if it can cover leg 1
        # but not leg 2, the "arb" becomes a naked one-legged position (live
        # bug: leg 1 filled $13.00, leg 2 rejected at $0.94 available -> the
        # naked leg lost -$13.32, wiping out every paired gain of the session).
        if mode == "paper":
            available = db.get_paper_available()
            if available < net_cost:
                logger.info(
                    f"[{self.name}] ARB skipped: pool ${available:.2f} cannot "
                    f"cover BOTH legs (${net_cost:.2f}) — no naked leg"
                )
                return {"success": False, "reason": "arb_insufficient_bankroll"}

        # Fill each leg against the SAME validated snapshot, share-matched, with
        # a tight per-leg slippage limit as a belt-and-suspenders guard.
        legs = [
            {"side": "yes", "book": yes_book, "vwap": fy["avg_price"]},
            {"side": "no", "book": no_book, "vwap": fn["avg_price"]},
        ]
        results = []
        for leg in legs:
            res = engine.place(
                bot_name=self.name,
                side=leg["side"],
                amount=0.0,
                market=market,
                mode=mode,
                confidence=signal.get("confidence"),
                reasoning=signal.get("reasoning"),
                features=signal.get("features"),
                # Fill an EXACT share count so both legs stay balanced (true arb).
                target_shares=shares,
                book=leg["book"],
                limit_price=leg["vwap"] + config.MAX_FILL_SLIPPAGE,
            )
            results.append(res)
            if not res.success:
                # Second leg won't help if the first failed — stop to avoid a
                # naked one-legged position beyond what already filled.
                break

        filled = [r for r in results if r.success]
        if len(filled) == len(legs):
            legs_desc = "/".join(
                "{}={:.1f}sh@{:.3f}".format(leg["side"], shares, leg["vwap"])
                for leg in legs
            )
            logger.info(
                f"[{self.name}] ARB filled both legs on "
                f"{str(market.get('question', ''))[:40]}: {legs_desc} "
                f"(edge={edge:+.4f}/pair)"
            )
            return {"success": True, "trade_id": filled[0].trade_id,
                    "reason": "arb_pair_filled", "fill_source": filled[0].fill_source}

        if filled:
            # One leg filled, the other didn't — naked leg. Not an arb; surface it.
            logger.warning(
                f"[{self.name}] ARB one-legged fill (naked risk) on "
                f"{str(market.get('id', ''))[:12]}…: filled "
                f"{filled[0].fill_source} — other leg failed"
            )
            try:
                if mode == "live":
                    from arena.alerts import alert_live_fill
                    alert_live_fill(
                        self.name, "naked_arb_leg",
                        market_id=str(market.get("id") or ""),
                        detail={"filled_legs": len(filled),
                                "fill_source": filled[0].fill_source},
                    )
            except Exception:
                pass
            return {"success": False, "reason": "naked_arb_leg"}
        return {"success": False, "reason": "arb_leg_unfilled"}

"""In-memory broker for backtests — shared bankroll, fills, resolution.

Mirrors venues/paper.py semantics (shared pool, depth-walked fills, taker
fee, slippage band, shared-pool concentration cap) but keeps every trade in
memory: nothing is written to bot_arena.db.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import config
import polymarket_fills
from backtest.books import synth_book

logger = logging.getLogger("backtest.broker")


@dataclass(frozen=True)
class BacktestTrade:
    """One filled (and eventually resolved) backtest trade."""
    bot_name: str
    strategy_type: str
    market_id: str
    side: str                    # "yes" | "no"
    shares: float
    cost: float                  # USDC spent on shares
    entry_price: float           # avg fill price (ask-walked)
    fee: float
    confidence: float
    entered_at: float            # epoch seconds of the decision tick
    time_remaining: float        # seconds to window close at entry
    context: dict                # lane/signal snapshot at decision time
    outcome: str | None = None   # "win" | "loss" | None while open
    pnl: float | None = None

    def resolved(self, yes_won: bool) -> "BacktestTrade":
        won = yes_won if self.side == "yes" else (not yes_won)
        payout = self.shares if won else 0.0
        return dataclasses.replace(
            self, outcome="win" if won else "loss",
            pnl=payout - self.cost - self.fee)


class BacktestBroker:
    """Shared virtual USDC pool + open/resolved trade ledger."""

    def __init__(self, bankroll: float | None = None, compound: bool = False):
        self.initial_bankroll = float(
            bankroll if bankroll is not None else config.BACKTEST_BANKROLL)
        # Kelly sizing base: fixed notional by default so per-trade sizes stay
        # comparable across the run and the P&L reads as edge, not compounding
        # (a replay's optimistic WR would otherwise snowball the bankroll and
        # dominate every dollar figure). ``compound=True`` sizes off the live
        # pool like the real arena.
        self.compound = bool(compound)
        self.realized_pnl = 0.0
        self.open_trades: list = []       # [BacktestTrade]
        self.resolved_trades: list = []   # [BacktestTrade]
        self.rejects: dict = {}           # reason -> count

    # -- bankroll (same shape as db.get_paper_available) --------------------
    @property
    def reserved(self) -> float:
        return sum(t.cost + t.fee for t in self.open_trades)

    @property
    def available(self) -> float:
        return self.initial_bankroll + self.realized_pnl - self.reserved

    @property
    def gross_pool(self) -> float:
        return self.initial_bankroll + self.realized_pnl

    def sizing_bankroll(self) -> float:
        """What Kelly sizing sees: fixed notional, or the live pool if compounding."""
        if self.compound:
            return max(0.0, self.available)
        return min(self.initial_bankroll, max(0.0, self.available))

    def _reject(self, reason: str) -> None:
        self.rejects[reason] = self.rejects.get(reason, 0) + 1

    # -- exposure cap (BUG #27, shared-pool concentration) ------------------
    def _side_headroom(self, market_id: str, side: str) -> float:
        base = self.gross_pool if self.compound else self.initial_bankroll
        cap = config.MARKET_SIDE_EXPOSURE_CAP * max(base, 0.0)
        open_cost = sum(t.cost for t in self.open_trades
                        if t.market_id == market_id and t.side == side)
        return cap - open_cost

    # -- fills ---------------------------------------------------------------
    def place(self, *, bot, market_id: str, side: str, side_mid: float,
              amount: float, expected_price: float | None,
              confidence: float, entered_at: float, time_remaining: float,
              context: dict) -> BacktestTrade | None:
        """Fill a BUY by walking a synthetic book. Returns the trade or None."""
        headroom = self._side_headroom(market_id, side)
        if headroom <= 0:
            self._reject("exposure_cap")
            return None
        spend = min(amount, headroom, max(self.available, 0.0))
        if spend <= 0:
            self._reject("insufficient_bankroll")
            return None

        book = synth_book(side_mid)
        fill = polymarket_fills.simulate_fill(book, spend)
        min_size = book.get("min_order_size", 0) or 0
        if not fill["filled"] or fill["shares"] < min_size:
            self._reject("below_min_size")
            return None
        # Same symmetric slippage band as venues/paper.py (BUG #28).
        if (expected_price is not None
                and abs(fill["avg_price"] - expected_price)
                > config.MAX_FILL_SLIPPAGE + 1e-9):
            self._reject("slippage_band")
            return None
        if fill["cost"] + fill["fee"] > self.available + 1e-9:
            self._reject("insufficient_bankroll")
            return None

        trade = BacktestTrade(
            bot_name=bot.name, strategy_type=bot.strategy_type,
            market_id=market_id, side=side,
            shares=fill["shares"], cost=fill["cost"],
            entry_price=fill["avg_price"], fee=fill["fee"],
            confidence=confidence, entered_at=entered_at,
            time_remaining=time_remaining, context=dict(context))
        self.open_trades.append(trade)
        return trade

    # -- resolution ----------------------------------------------------------
    def resolve_market(self, market_id: str, yes_won: bool) -> list:
        """Settle every open trade on ``market_id`` against the true outcome."""
        settled = []
        still_open = []
        for t in self.open_trades:
            if t.market_id == market_id:
                r = t.resolved(yes_won)
                self.realized_pnl += r.pnl
                self.resolved_trades.append(r)
                settled.append(r)
            else:
                still_open.append(t)
        self.open_trades = still_open
        return settled

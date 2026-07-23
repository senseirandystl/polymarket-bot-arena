"""Regression tests for the secondary (maker) bot section in arena.py.

Guards the bug where a HOLD returned by a time-gated maker bot was added to
the (bot, market) dedup set, permanently locking the market out so the bot
could never re-evaluate during its actual entry window (=> zero trades for
LateWindowMaker). See ``_run_maker_section`` and ``BUG_HISTORY.md``.
"""

import importlib.util
import pathlib

import pytest

from arena.state import SharedArenaState

# ``import arena`` resolves to the ``arena/`` package, which shadows the
# top-level ``arena.py`` script that owns ``_run_maker_section``. Load the
# script explicitly by path so we can exercise the maker-section logic.
_ARENA_PY = pathlib.Path(__file__).resolve().parents[2] / "arena.py"
_spec = importlib.util.spec_from_file_location("arena_main", _ARENA_PY)
arena = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arena)


class _StubMakerBot:
    """Minimal maker-bot stand-in: always holds."""

    def __init__(self, name="stub-maker"):
        self.name = name
        self.trading_mode = "paper"

    def analyze(self, market, signals):
        return {
            "action": "hold",
            "side": "yes",
            "confidence": 0.0,
            "reasoning": "stub: holding",
            "maker_bid": 0.48,
            "maker_ask": 0.52,
            "maker_mid": 0.50,
            "maker_side": "both",
        }


def test_hold_does_not_lock_market_out_of_dedup():
    """A maker HOLD must leave the (bot, market) pair un-marked so the next
    discovery cycle re-evaluates it (critical for time-gated makers)."""
    state = SharedArenaState()
    bot = _StubMakerBot(name="late-window-maker-v1")
    market = {"id": "0xcond", "question": "BTC 9:00PM-9:05PM Up or Down",
              "current_price": 0.50, "time_remaining_seconds": 1200}

    placed = arena._run_maker_section(bot, market, {}, state)

    assert placed is False
    # The pair must NOT be in the dedup set — otherwise the bot is locked out
    # for the rest of the market's life, including its real entry window.
    assert not state.is_traded((bot.name, "0xcond"))


def test_late_window_maker_holds_early_trades_late():
    """The bot itself gates on the final entry window: PRE-WINDOW -> hold,
    final window with drift conviction + confirming momentum + price -> buy."""
    from bots.bot_late_window_maker import LateWindowMakerBot

    bot = LateWindowMakerBot()
    # up-drift conviction + non-contradicting up momentum
    signals = {"prices": [100.0, 100.0, 100.2], "btc_drift": 0.5}

    early = {"current_price": 0.65, "time_remaining_seconds": 1200}
    assert bot.analyze(early, signals)["action"] == "hold"

    late = {"current_price": 0.65, "time_remaining_seconds": 45}
    assert bot.analyze(late, signals)["action"] == "buy"


def test_late_window_maker_requires_drift_conviction():
    """Weak drift -> hold even in-window; momentum contradicting drift -> hold."""
    from bots.bot_late_window_maker import LateWindowMakerBot

    bot = LateWindowMakerBot()
    late = {"current_price": 0.65, "time_remaining_seconds": 45}
    weak = {"prices": [100.0, 100.0, 100.2], "btc_drift": 0.1}
    assert bot.analyze(late, weak)["action"] == "hold"
    contra = {"prices": [100.4, 100.2, 100.0], "btc_drift": 0.5}
    assert bot.analyze(late, contra)["action"] == "hold"


def test_fee_zone_taker_fee_matches_canonical_polymarket_formula():
    """The fee-zone gate must use the SAME fee formula as settled P&L
    (official Polymarket: feeRate × shares × p × (1-p), crypto tier). Guards
    against re-introducing the bogus quadratic. See BUG_HISTORY #17."""
    import polymarket_fills
    from bots import bot_fee_zone_maker as fzm
    for p in (0.50, 0.60, 0.70, 0.82, 0.90):
        assert fzm.taker_fee(p) == polymarket_fills.taker_fee(1.0, p)
    # Sanity: crypto tier peaks at $1.75 per 100 shares at 50¢.
    assert polymarket_fills.taker_fee(100.0, 0.50) == pytest.approx(1.75)

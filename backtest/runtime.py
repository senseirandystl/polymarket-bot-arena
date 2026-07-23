"""Isolate the bots' runtime DB reads during a backtest.

``BaseBot.make_decision`` reaches into live state in four places: the sizing
bankroll (paper pool), the Kelly fraction, the learned bias, and the approved
lane overrides — all backed by bot_arena.db. This context manager swaps those
module-level hooks for backtest-local implementations so a backtest (a) never
touches the live DB and (b) sizes off the backtest broker's own bankroll,
then restores everything on exit.
"""

from __future__ import annotations

import contextlib
import time

import config
import learning
from bots import base_bot


@contextlib.contextmanager
def patched_runtime(broker, kelly_fraction: float | None = None,
                    lane_overrides: dict | None = None):
    """Patch base_bot/learning runtime hooks onto ``broker`` for the duration.

    ``lane_overrides`` lets a backtest replay an approved-lane configuration
    (same shape as db.get_lane_overrides) without the DB — default: none.
    """
    kf = float(kelly_fraction if kelly_fraction is not None
               else config.KELLY_FRACTION)
    overrides = dict(lane_overrides or {})

    saved = (base_bot._sizing_bankroll, base_bot._kelly_fraction,
             base_bot._lane_overrides, learning.get_learned_bias)

    base_bot._sizing_bankroll = lambda mode: broker.sizing_bankroll()
    base_bot._kelly_fraction = lambda: kf
    base_bot._lane_overrides = lambda: overrides
    # Learning is disabled live (config.LEARNING_ENABLED False) and its DB
    # cache must not leak live history into a replay — bias = prior.
    learning.get_learned_bias = lambda bot_name, features, prior=0.5: prior
    try:
        yield
    finally:
        (base_bot._sizing_bankroll, base_bot._kelly_fraction,
         base_bot._lane_overrides, learning.get_learned_bias) = saved


def silence_perf_cache(bots: list) -> None:
    """Pin each bot's resolved-count cache so make_decision never queries db.

    The learning-weight ramp is 0 with LEARNING_ENABLED False, so the count
    is inert — pinning it just prevents the DB read on the first tick.
    """
    for bot in bots:
        bot._perf_cache = (time.time() + 1e12, 0)

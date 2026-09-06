"""Mean Reversion bot with 2x take-profit via intra-window tick tracking.

Inherits MeanRevBot entry logic (including skips when signals are weak).
When a buy is placed, the position monitor polls every 0.5s and closes
early if the position reaches 100% profit (2x the initial bet). If it
never hits 2x, the position holds until the window resolves normally.

Entry: same mean-reversion signals as MeanRevBot (may skip).
Exit: early close at 2x via PositionMonitorThread, otherwise hold.
"""

import config
from bots.bot_mean_rev import MeanRevBot, DEFAULT_PARAMS


class MeanRevTPBot(MeanRevBot):
    exit_strategy = "take_profit"
    take_profit_pct = 1.0  # 100% = 2x the initial bet

    def __init__(self, name="meanrev-tp2x-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )
        self.strategy_type = "mean_reversion_tp"

    def make_decision(self, market, signals):
        """TP bot: enter when base logic says buy, monitor for 2x exit.

        Respects all base class logic (two-sided net-edge side selection,
        symmetric price guards), including NO-side entries. Only adds TP
        monitoring annotation.
        """
        decision = super().make_decision(market, signals)

        if decision.get("action") == "buy":
            decision["reasoning"] += " [TP: monitoring for 2x exit @0.5s]"

        return decision

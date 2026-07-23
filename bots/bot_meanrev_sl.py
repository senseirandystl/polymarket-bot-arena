"""Mean Reversion bot — formerly a 25% stop-loss variant.

Stop-loss removed (2026-07-15 root-cause analysis, spec R3): in fully-resolving
5-min binary markets a stop-loss is net-harmful — the held-to-resolution
counterfactual (-150.7) beat stopping (-172.3), because a -25% stop just converts
intra-window price noise into locked losses. Risk is managed at ENTRY (the edge
gate), not by exiting mid-window.

With the stop-loss gone, the old 1.5x position oversizing (justified only by the
capped downside) is also removed — it would otherwise be a full-downside bot
betting oversized. This bot now holds to resolution like the base mean-rev bot;
the distinct ``strategy_type`` is retained for DB/evolution continuity.
"""

from bots.bot_mean_rev import MeanRevBot, DEFAULT_PARAMS


class MeanRevSLBot(MeanRevBot):
    # Hold to resolution — no early stop-loss exit (see module docstring).
    exit_strategy = None
    stop_loss_pct = 0.0

    def __init__(self, name="meanrev-sl25-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )
        self.strategy_type = "mean_reversion_sl"

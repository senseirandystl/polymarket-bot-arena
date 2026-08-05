"""REMOVED — Sentiment bot (2026-08 audit).

This strategy depended on kill-switched pm/cvd lanes and was never properly
configured for 5m BTC Up/Down. Hybrid no longer embeds a sentiment
sub-analyzer. Importing this module raises ImportError so stale DB configs
and tests fail loudly rather than trading a dead thesis.
"""

raise ImportError(
    "SentimentBot was removed (2026-08). Re-run startup with a default slate "
    "or manually select bots; migrate any bot_configs with strategy_type="
    "'sentiment' to hybrid/momentum."
)

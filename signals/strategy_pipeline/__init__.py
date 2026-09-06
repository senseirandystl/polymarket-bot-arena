"""Strategy Lab pipeline: research -> compile -> backtest -> (paper) -> ready -> live.

Owns the invent/gate/graduate loop (successor to the deleted Trading Floor / desk).
"""

from signals.strategy_pipeline.cycle import LabHost, get_host
from signals.strategy_pipeline.store import HypothesisStore

__all__ = ["LabHost", "get_host", "HypothesisStore"]

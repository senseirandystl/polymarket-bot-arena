"""Desk cycle: research → compile → backtest → paper → live → autopsy."""

from desk.roles import ROLES, FloorSnapshot, get_floor
from desk.store import HypothesisStore
from desk.universe import phase_universe, UniverseSlot

__all__ = [
    "ROLES",
    "FloorSnapshot",
    "get_floor",
    "HypothesisStore",
    "phase_universe",
    "UniverseSlot",
]

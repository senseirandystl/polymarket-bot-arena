"""Stepwise crypto prediction-market universe for Lab research.

Phase 1 is what the arena already trades. Phase 2-3 widen discovery later;
they do not invent settlement math. A slot without a settlement adapter
stays tradable=False.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class UniverseSlot:
    slot_id: str
    venue: str
    asset: str
    window_label: str
    window_sec: int
    series_key: str
    settlement: str
    tradable: bool
    phase: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_SLOTS: tuple[UniverseSlot, ...] = (
    UniverseSlot(
        "polymarket:btc_5m", "polymarket", "btc", "5m", 300,
        "series:btc_5m_updown", "chainlink_twap60", True, 1,
    ),
    UniverseSlot(
        "kalshi:btc_15m", "kalshi", "btc", "15m", 900,
        "KXBTC15M", "brti_last60", True, 1,
    ),
    UniverseSlot(
        "polymarket:eth_5m", "polymarket", "eth", "5m", 300,
        "series:eth_5m_updown", "chainlink_twap60", False, 2,
    ),
    UniverseSlot(
        "polymarket:sol_5m", "polymarket", "sol", "5m", 300,
        "series:sol_5m_updown", "chainlink_twap60", False, 2,
    ),
    UniverseSlot(
        "polymarket:xrp_5m", "polymarket", "xrp", "5m", 300,
        "series:xrp_5m_updown", "chainlink_twap60", False, 2,
    ),
    UniverseSlot(
        "polymarket:btc_15m", "polymarket", "btc", "15m", 900,
        "series:btc_15m_updown", "chainlink_twap60", False, 2,
    ),
    UniverseSlot(
        "kalshi:eth_15m", "kalshi", "eth", "15m", 900,
        "KXETH15M", "unknown", False, 2,
    ),
    UniverseSlot(
        "kalshi:btc_1h", "kalshi", "btc", "1h", 3600,
        "KXBTC1H", "unknown", False, 2,
    ),
)


def phase_universe(phase: int | None = None) -> list[UniverseSlot]:
    if phase is None:
        phase = 1
    phase = max(1, min(int(phase), 3))
    if phase >= 3:
        return list(_SLOTS)
    return [s for s in _SLOTS if s.phase <= phase]


def tradable_slots(phase: int | None = None) -> list[UniverseSlot]:
    return [s for s in phase_universe(phase) if s.tradable]


def slot_by_id(slot_id: str) -> UniverseSlot | None:
    for s in _SLOTS:
        if s.slot_id == slot_id:
            return s
    return None

"""Smoke tests for cross-venue lag prototype (menu-only)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from arena.startup import DEFAULT_INDICES, STRATEGY_MENU, build_default_bots
from bots.bot_cross_venue_lag import CrossVenueLagBot, DEFAULT_PARAMS


def test_cross_venue_not_on_default_slate():
    types = [b.strategy_type for b in build_default_bots()]
    assert "cross_venue_lag" not in types
    assert DEFAULT_INDICES == [4, 7, 13]
    menu_types = []
    for cls, _name, _blurb in STRATEGY_MENU:
        st = getattr(cls, "strategy_type", None)
        if not isinstance(st, str):
            st = cls(name=_name, generation=0).strategy_type
        menu_types.append(st)
    assert "cross_venue_lag" in menu_types


def test_make_decision_skips_without_peer():
    bot = CrossVenueLagBot()
    market = {"current_price": 0.45, "no_price": 0.55, "window_sec": 300}
    signals = {"btc_drift": 0.25, "btc_implied_yes": 0.62}
    d = bot.make_decision(market, signals)
    assert d["action"] == "skip"
    assert d.get("skip_reason") in ("no_cross_venue", "cross_gap_thin", "weak_drift")


def test_make_decision_buy_shape_with_dummy_peer():
    bot = CrossVenueLagBot(params={
        **DEFAULT_PARAMS,
        "min_mid_gap": 0.03,
        "min_residual": 0.01,
        "min_edge": 0.01,
        "min_drift": 0.05,
        "min_confidence": 0.05,
    })
    market = {
        "current_price": 0.42,
        "no_price": 0.58,
        "yes_ask": 0.43,
        "window_sec": 300,
        "exchange": "polymarket",
    }
    signals = {
        "btc_drift": 0.30,
        "btc_implied_yes": 0.70,
        "btc_drift_z": 1.2,
        "cross_venue": {
            "peer_exchange": "kalshi",
            "peer_yes_mid": 0.55,
            "peer_window_sec": 900,
            "local_exchange": "polymarket",
            "local_yes_mid": 0.42,
            "local_window_sec": 300,
        },
    }
    d = bot.make_decision(market, signals)
    assert d["action"] == "buy"
    assert d["side"] == "yes"
    assert d.get("edge", 0) > 0
    assert "cross_venue" in (d.get("reasoning") or "")

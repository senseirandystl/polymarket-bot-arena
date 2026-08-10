"""Tests for interactive startup: selection parsing + default slate."""

import pytest

from arena import startup


# --- parse_selection --------------------------------------------------------

def test_parse_comma_list():
    assert startup.parse_selection("1,3,5", 9) == [1, 3, 5]


def test_parse_range():
    assert startup.parse_selection("1-6", 9) == [1, 2, 3, 4, 5, 6]


def test_parse_mixed_range_and_singletons():
    assert startup.parse_selection("1-3,5,9", 9) == [1, 2, 3, 5, 9]


def test_parse_dedupes_preserving_order():
    assert startup.parse_selection("3,1,3,1", 9) == [3, 1]


def test_parse_tolerates_spaces_and_reversed_range():
    assert startup.parse_selection(" 6 - 4 ", 9) == [4, 5, 6]


def test_parse_out_of_range_raises():
    with pytest.raises(ValueError):
        startup.parse_selection("1,99", 9)


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        startup.parse_selection("", 9)


def test_parse_non_numeric_raises():
    with pytest.raises(ValueError):
        startup.parse_selection("abc", 9)


# --- default slate ----------------------------------------------------------

def test_default_slate_includes_hybrid_and_sweeper():
    """Default: mom / meanrev / sniper / hybrid / arb / sweeper."""
    bots = startup.build_default_bots()
    assert len(bots) == 6
    types = [b.strategy_type for b in bots]
    assert types == [
        "momentum",
        "mean_reversion",
        "sniper",
        "hybrid",
        "arbitrage",
        "sweeper",
    ]
    # Explicitly not on the default slate (menu-only / mid-run deploy).
    assert "phantom" not in types
    assert "late_window_maker" not in types
    assert "fee_zone_maker" not in types


def test_manual_selection_builds_exactly_chosen():
    # Indices: 1=momentum, 7=arbitrage (sentiment removed; menu renumbered).
    bots = startup._build_from_indices([1, 7])
    assert [b.strategy_type for b in bots] == ["momentum", "arbitrage"]

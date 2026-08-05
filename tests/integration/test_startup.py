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

def test_default_slate_is_eight_bots_with_arbitrage_sniper_and_makers():
    bots = startup.build_default_bots()
    assert len(bots) == 8
    types = {b.strategy_type for b in bots}
    assert "arbitrage" in types
    # The directional defaults (meanrev is the plain mean_reversion bot
    # since the sl25 rename) + the sniper (promoted 2026-07-18).
    assert {"momentum", "phantom", "mean_reversion", "hybrid", "sniper"} <= types
    # Both maker bots are now first-class members of the default lineup.
    assert {"late_window_maker", "fee_zone_maker"} <= types


def test_manual_selection_builds_exactly_chosen():
    # Indices: 1=momentum, 7=arbitrage (sentiment removed; menu renumbered).
    bots = startup._build_from_indices([1, 7])
    assert [b.strategy_type for b in bots] == ["momentum", "arbitrage"]

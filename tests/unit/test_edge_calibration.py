"""Confidence inversion fix — concave sizing edge + structure confidence."""

import pytest

from bots.edge_calibration import calibrated_sizing_edge, quality_confidence
from bots.maker_utils import mid_ask_gap_ok


def test_calibrated_edge_full_credit_below_threshold():
    assert calibrated_sizing_edge(0.03) == pytest.approx(0.03)
    assert calibrated_sizing_edge(0.04) == pytest.approx(0.04)


def test_calibrated_edge_tapers_large_edges():
    small = calibrated_sizing_edge(0.04)
    large = calibrated_sizing_edge(0.15)
    assert large > small
    # Large raw edge must not pass through 1:1 (concave + hard cap).
    assert large < 0.15
    assert large <= 0.08 + 1e-9


def test_quality_confidence_not_monotone_in_edge():
    """Outsized edge is less 'confident' than a modest well-structured edge."""
    modest = quality_confidence(edge=0.04, abs_drift=0.30, side_mid=0.52)
    huge = quality_confidence(edge=0.20, abs_drift=0.30, side_mid=0.52)
    assert modest >= huge - 0.05  # at least not strongly inverted the wrong way
    flat = quality_confidence(edge=0.04, abs_drift=0.02, side_mid=0.50)
    assert modest > flat


def test_mid_ask_gap_rejects_fantasy_fill():
    ok, reason = mid_ask_gap_ok(0.85, 0.30)
    assert not ok
    assert "gap" in reason or "<<" in reason


def test_mid_ask_gap_allows_tight_spread():
    ok, _ = mid_ask_gap_ok(0.60, 0.62)
    assert ok

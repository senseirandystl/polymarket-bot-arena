"""Smooth scoring curves (signals/curves.py)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from signals.curves import soft_saturate, sigmoid, gaussian_zone, smooth_ramp


class TestSoftSaturate:
    def test_zero_at_zero(self):
        assert soft_saturate(0.0, 0.002) == 0.0

    def test_bounded(self):
        # tanh of huge args rounds to exactly +/-1.0 in float — bound inclusive.
        assert -1.0 <= soft_saturate(-100.0, 0.002) < -0.99
        assert 0.99 < soft_saturate(100.0, 0.002) <= 1.0

    def test_linear_near_zero(self):
        # tanh(x) ~ x for small x: slope near zero matches the hard clamp's.
        assert abs(soft_saturate(0.0002, 0.002) - 0.1) < 0.01

    def test_antisymmetric(self):
        assert soft_saturate(0.5, 1.0) == -soft_saturate(-0.5, 1.0)

    def test_zero_scale_safe(self):
        assert soft_saturate(1.0, 0.0) == 0.0

    def test_smoother_than_clamp_past_saturation(self):
        # A 2x-scale input must NOT read identically to a 10x-scale input
        # (the old hard clamp pinned both at exactly 1.0).
        assert soft_saturate(0.004, 0.002) < soft_saturate(0.02, 0.002)


class TestSigmoid:
    def test_half_at_center(self):
        assert abs(sigmoid(0.5, center=0.5, steepness=10) - 0.5) < 1e-9

    def test_monotonic(self):
        vals = [sigmoid(x, center=0.0, steepness=2.0) for x in (-2, -1, 0, 1, 2)]
        assert vals == sorted(vals)

    def test_extreme_inputs_no_overflow(self):
        assert sigmoid(1e6, steepness=100) == 1.0
        assert sigmoid(-1e6, steepness=100) == 0.0


class TestGaussianZone:
    def test_peak_at_center(self):
        assert gaussian_zone(0.44, 0.44, 0.04) == 1.0

    def test_decays_symmetrically(self):
        lo = gaussian_zone(0.40, 0.44, 0.04)
        hi = gaussian_zone(0.48, 0.44, 0.04)
        assert abs(lo - hi) < 1e-12
        assert abs(lo - math.exp(-0.5)) < 1e-9

    def test_zero_width_safe(self):
        assert gaussian_zone(0.5, 0.5, 0.0) == 1.0
        assert gaussian_zone(0.4, 0.5, 0.0) == 0.0


class TestSmoothRamp:
    def test_clamps_outside(self):
        assert smooth_ramp(-5.0, 0.0, 1.0) == 0.0
        assert smooth_ramp(5.0, 0.0, 1.0) == 1.0

    def test_midpoint(self):
        assert abs(smooth_ramp(0.5, 0.0, 1.0) - 0.5) < 1e-9

    def test_degenerate_interval(self):
        assert smooth_ramp(1.0, 1.0, 1.0) == 1.0
        assert smooth_ramp(0.9, 1.0, 1.0) == 0.0

    def test_late_window_semantics(self):
        # Negated-time trick used for the late-window boost: 90s+ remaining
        # -> 0; 30s or less -> 1; in between rises smoothly.
        assert smooth_ramp(-120.0, -90.0, -30.0) == 0.0
        assert smooth_ramp(-30.0, -90.0, -30.0) == 1.0
        assert 0.0 < smooth_ramp(-60.0, -90.0, -30.0) < 1.0

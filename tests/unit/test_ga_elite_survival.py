"""GA elites must pass survival bar; early deep-red cull; recency blend."""

import config
from evolution.ga import _survives_legacy_bar, _effective_survival_pnl


def test_effective_survival_pnl_blend(monkeypatch):
    monkeypatch.setattr(config, "GA_SURVIVAL_RECENCY_WEIGHT", 0.5)
    # 50% long (−10) + 50% recent (−30) = −20
    assert _effective_survival_pnl(
        {"pnl": -10.0, "recent_pnl": -30.0}
    ) == -20.0


def test_early_cull_deep_red(monkeypatch):
    # Baseline cull path (paper gates off). n=20 < 40 but catastrophic.
    monkeypatch.setattr(config, "PAPER_GATE_PROFILE", "off")
    assert not _survives_legacy_bar({
        "trades": 20, "pnl": -20.0, "be_gap": -0.15, "generation": 1,
    })


def test_early_cull_not_triggered_mild(monkeypatch):
    monkeypatch.setattr(config, "PAPER_GATE_PROFILE", "off")
    # n=20, only mildly red → still immune
    assert _survives_legacy_bar({
        "trades": 20, "pnl": -5.0, "be_gap": -0.05, "generation": 1,
    })


def test_founder_protect_decays_after_cycles(monkeypatch):
    monkeypatch.setattr(config, "GA_PROTECT_FOUNDERS", True)
    monkeypatch.setattr(config, "GA_FOUNDER_PROTECT_MAX_CYCLES", 5)
    # Moderate loss would be founder-protected when cycle < max
    assert _survives_legacy_bar({
        "trades": 50, "pnl": -15.0, "be_gap": -0.01, "generation": 0,
    }, cycle_number=3)
    # After max cycles, same loss is replaceable (pnl between −12 and −20
    # still hits soft floor... need deeper red past −12 and founder floors)
    # Soft floor: pnl > −12 survives. Need ≤ −12 and fail founder.
    # With protect expired: pnl=−15, gap=−0.05 → soft floor: −15 < −12 so
    # not soft-survived; founder protect off → replaceable.
    assert not _survives_legacy_bar({
        "trades": 50, "pnl": -15.0, "be_gap": -0.05, "generation": 0,
    }, cycle_number=25)

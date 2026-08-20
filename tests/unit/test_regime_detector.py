"""Tests for the robust regime detector + propagation hooks."""

from __future__ import annotations

import json
import math
import random
import time

import pytest

import config
from bots.base_bot import BaseBot
from bots.meta_learner import bucket_for
from evolution.fitness import multi_objective_fitness, regime_breakdown
from signals.lab import REGIME_LANE_DAMP, SignalLab, SignalView
from signals.regime_detector import (
    REGIME_IDS,
    classify_rules,
    compute_features,
    detect_once,
    directionality,
    get_detector,
    legacy_label,
    meta_bucket,
    reset_detector,
)


def _trending_prices(n=40, start=100_000.0, step=40.0):
    """Monotonic uptrend with small noise — high efficiency."""
    rng = random.Random(1)
    p = start
    out = []
    for _ in range(n):
        p += step + rng.uniform(-5, 5)
        out.append(p)
    return out


def _chop_prices(n=40, start=100_000.0, amp=80.0):
    """Oscillating path — high path, low net move."""
    rng = random.Random(2)
    out = []
    for i in range(n):
        out.append(start + amp * math.sin(i * 1.3) + rng.uniform(-10, 10))
    return out


def _quiet_range(n=40, start=100_000.0):
    rng = random.Random(3)
    return [start + rng.uniform(-8, 8) for _ in range(n)]


# ---------------------------------------------------------------------------
# Pure classification
# ---------------------------------------------------------------------------

def test_compute_features_bounded():
    feats = compute_features(_trending_prices())
    for k in ("vol", "trend", "mom", "flow"):
        assert 0.0 <= feats[k] <= 1.0
    # volume is a separate feature (activity); missing series → 0
    assert feats["volume"] == 0.0
    # volume != volatility: explicit volume series moves only volume
    with_vol = compute_features(
        _trending_prices(),
        volumes=[100, 100, 100, 100, 100, 300, 300, 300, 300, 300],
    )
    assert with_vol["volume"] > 0.5
    assert "vol" in with_vol  # volatility key still present


def test_high_vol_trend_classification():
    # Force high vol + high trend via explicit scores
    rid, conf = classify_rules({"vol": 0.85, "trend": 0.80, "mom": 0.5, "flow": 0.3})
    assert rid == "high_vol_trend"
    assert conf > 0.5


def test_low_vol_range_classification():
    rid, conf = classify_rules({"vol": 0.15, "trend": 0.15, "mom": 0.1, "flow": 0.1})
    assert rid == "low_vol_range"


def test_high_vol_chop_classification():
    rid, conf = classify_rules({"vol": 0.80, "trend": 0.15, "mom": 0.2, "flow": 0.5})
    assert rid == "high_vol_chop"


def test_low_vol_trend_classification():
    rid, conf = classify_rules({"vol": 0.20, "trend": 0.75, "mom": 0.4, "flow": 0.2})
    assert rid == "low_vol_trend"


def test_legacy_and_meta_maps():
    assert legacy_label("high_vol_trend") == "trending"
    assert legacy_label("low_vol_range") == "quiet"
    assert legacy_label("high_vol_chop") == "volatile"
    assert meta_bucket("high_vol_trend") == "trending"
    assert meta_bucket("low_vol_range") == "ranging"
    assert meta_bucket("high_vol_chop") == "chop"


def test_detect_once_on_real_series():
    snap = detect_once(_trending_prices(), cvd=0.4, obi=0.3)
    assert snap["regime_id"] in REGIME_IDS
    assert "features" in snap
    assert snap["known"] is True


def test_insufficient_prices_classify_unknown():
    """Cold start must not look like a quiet range just because scores are 0."""
    feats = compute_features([100_000.0, 100_001.0])
    rid, conf = classify_rules(feats)
    assert rid == "unknown"
    assert conf == 0.0
    assert feats.get("sample_ok", 1.0) < 0.5
    snap = detect_once([])
    assert snap["regime_id"] == "unknown"
    assert snap["known"] is False


def test_live_zero_scores_on_short_series_stay_unknown():
    """arena/signals always passes vol_score=0.0 (not None) on a cold feed."""
    feats = compute_features(
        [100_000.0, 100_001.0], vol_score=0.0, trend_score=0.0, realized_vol=0.0,
    )
    assert feats.get("sample_ok", 1.0) < 0.5
    rid, conf = classify_rules(feats)
    assert rid == "unknown"
    assert conf == 0.0


def test_snapshot_actionable_requires_hold_and_conf(monkeypatch):
    monkeypatch.setattr(config, "REGIME_HOLD_TICKS", 1, raising=False)
    monkeypatch.setattr(config, "REGIME_EMA_ALPHA", 1.0, raising=False)
    monkeypatch.setattr(config, "REGIME_ACTION_MIN_CONF", 0.50, raising=False)
    monkeypatch.setattr(config, "REGIME_ACTION_MIN_HOLD_SEC", 20.0, raising=False)
    det = reset_detector()
    monkeypatch.setattr(det, "_persist", lambda force=False: None)
    monkeypatch.setattr(det, "_ensure_loaded", lambda: None)
    snap = det.update(_trending_prices(), vol_score=0.80, trend_score=0.75)
    assert "actionable" in snap
    assert "held_sec" in snap
    assert snap["held_sec"] < 20.0
    assert snap["actionable"] is False
    det._last_change_ts = time.time() - 30.0
    det._confidence = 0.80
    later = det.snapshot()
    assert later["actionable"] is True


def test_load_missing_last_change_ts_seeds_held(monkeypatch):
    """Old arena_state without last_change_ts must not freeze actionable off."""
    det = reset_detector()
    monkeypatch.setattr(det, "_persist", lambda force=False: None)
    now = time.time()

    class _DB:
        @staticmethod
        def get_arena_state(key, default=None):
            if key == "regime_detector":
                return json.dumps({
                    "regime": "low_vol_trend",
                    "confidence": 0.72,
                    "ema": {},
                    "last_features": {"sample_ok": 1.0, "vol": 0.3, "trend": 0.6},
                    "updated_at": now - 120.0,
                })
            return default

    monkeypatch.setattr("db.get_arena_state", _DB.get_arena_state, raising=False)
    det._loaded = False
    det._ensure_loaded()
    snap = det.snapshot()
    assert snap["regime_id"] == "low_vol_trend"
    assert snap["held_sec"] >= 20.0
    assert snap["actionable"] is True


def test_directionality_preserves_zero_chop_and_align():
    """0.0 is a real extreme (straight tape / scale disagreement), not 'missing'."""
    strong = directionality({"trend": 0.80, "chop": 0.0, "ms_mom_align": 1.0})
    muted = directionality({"trend": 0.80, "chop": 0.5, "ms_mom_align": 0.5})
    disagree = directionality({"trend": 0.80, "chop": 0.0, "ms_mom_align": 0.0})
    assert strong > muted
    assert disagree < strong
    # Missing keys still default to 0.5
    assert directionality({"trend": 0.80}) == pytest.approx(muted)


def test_directionality_uses_chop_and_align():
    """Live classifier axis is the composite, not raw trend alone."""
    high_dir = compute_features(_trending_prices())
    chop_dir = compute_features(_chop_prices())
    assert "direction" in high_dir and "chop" in high_dir
    assert high_dir["direction"] > chop_dir["direction"]
    assert chop_dir["chop"] > high_dir["chop"]


# ---------------------------------------------------------------------------
# Online detector (hysteresis, continuous update)
# ---------------------------------------------------------------------------

def test_online_detector_updates_and_hysteresis(monkeypatch):
    monkeypatch.setattr(config, "REGIME_HOLD_TICKS", 2, raising=False)
    monkeypatch.setattr(config, "REGIME_EMA_ALPHA", 0.5, raising=False)
    # Avoid DB
    det = reset_detector()
    monkeypatch.setattr(det, "_persist", lambda force=False: None)
    monkeypatch.setattr(det, "_ensure_loaded", lambda: None)

    # Seed with quiet range
    for _ in range(5):
        snap = det.update(_quiet_range(), cvd=0.0, obi=0.0,
                          vol_score=0.15, trend_score=0.15)
    assert snap["regime_id"] in ("low_vol_range", "normal", "unknown", "low_vol_trend")

    # Suddenly inject high-vol chop features for several ticks
    for _ in range(6):
        snap = det.update(_chop_prices(), cvd=0.6, obi=-0.5,
                          vol_score=0.85, trend_score=0.12)
    # After hold_ticks it should commit (or at least not crash)
    assert snap["regime_id"] in REGIME_IDS
    assert snap["confidence"] >= 0.0


def test_record_outcome_updates_performance(monkeypatch):
    det = reset_detector()
    monkeypatch.setattr(det, "_persist", lambda force=False: None)
    monkeypatch.setattr(det, "_ensure_loaded", lambda: None)
    det.record_outcome("high_vol_trend", 2.5, won=True)
    det.record_outcome("high_vol_trend", -1.0, won=False)
    perf = det.performance_snapshot()
    assert perf["high_vol_trend"]["n"] == 2
    assert perf["high_vol_trend"]["wins"] == 1
    assert perf["high_vol_trend"]["pnl"] == pytest.approx(1.5)


def test_market_rollover_soft_note_retains_state(monkeypatch, caplog):
    """P2: window change logs a soft note; EMA/regime are NOT reset."""
    import logging
    det = reset_detector()
    monkeypatch.setattr(det, "_persist", lambda force=False: None)
    monkeypatch.setattr(det, "_ensure_loaded", lambda: None)

    for _ in range(4):
        det.update(_quiet_range(), cvd=0.0, obi=0.0,
                   vol_score=0.15, trend_score=0.15, market_id="mkt-a")
    snap_before = det.snapshot()
    ema_before = dict(det._ema)
    regime_before = snap_before["regime_id"]
    ticks_before = snap_before["ticks"]

    with caplog.at_level(logging.INFO, logger="signals.regime_detector"):
        det.update(_quiet_range(), cvd=0.1, obi=0.0,
                   vol_score=0.15, trend_score=0.15, market_id="mkt-b")

    assert any("REGIME MARKET ROLLOVER" in r.message for r in caplog.records)
    assert any("mkt-a -> mkt-b" in r.message for r in caplog.records)
    snap_after = det.snapshot()
    # Soft note only — continuous state retained (no partial reset).
    assert snap_after["market_id"] == "mkt-b"
    assert snap_after["ticks"] == ticks_before + 1
    assert snap_after["regime_id"] == regime_before or snap_after["regime_id"] in REGIME_IDS
    assert det._ema  # still populated
    # Same quiet inputs → EMA should still be present for feature keys
    for k in ema_before:
        assert k in det._ema

    # Same market again → no second rollover log
    caplog.clear()
    det.note_market("mkt-b")
    assert not any("ROLLOVER" in r.message for r in caplog.records)


def test_live_snapshot_keeps_classifier_axes(monkeypatch):
    """EMA path must retain direction/chop/trend_sign — not drop them."""
    monkeypatch.setattr(config, "REGIME_HOLD_TICKS", 1, raising=False)
    monkeypatch.setattr(config, "REGIME_EMA_ALPHA", 1.0, raising=False)
    det = reset_detector()
    monkeypatch.setattr(det, "_persist", lambda force=False: None)
    monkeypatch.setattr(det, "_ensure_loaded", lambda: None)
    snap = det.update(_trending_prices(), cvd=0.2, obi=0.1)
    feats = snap["features"]
    for key in ("direction", "chop", "trend_sign", "ms_mom_align", "vol_rel"):
        assert key in feats, f"missing {key} on live snapshot"
    assert snap["trend_side"] in ("yes", "no", "flat")
    # A clear uptrend should not stamp flat just because trend_sign was dropped.
    assert snap["trend_side"] == "yes"
    assert 0.0 <= feats["direction"] <= 1.0


def test_pm_state_damps_confidence_not_regime_id(monkeypatch):
    """Polymarket book quality is sidecar context — never flips the BTC grid."""
    monkeypatch.setattr(config, "REGIME_HOLD_TICKS", 1, raising=False)
    monkeypatch.setattr(config, "REGIME_EMA_ALPHA", 1.0, raising=False)

    def _fresh():
        d = reset_detector()
        monkeypatch.setattr(d, "_persist", lambda force=False: None)
        monkeypatch.setattr(d, "_ensure_loaded", lambda: None)
        return d

    prices = _trending_prices()
    clean = _fresh().update(
        prices, vol_score=0.80, trend_score=0.75,
        pm_state={"spread_score": 0.85, "yes_price": 0.52, "no_price": 0.48},
    )
    broken = _fresh().update(
        prices, vol_score=0.80, trend_score=0.75,
        pm_state={"spread_score": 0.10, "yes_price": 0.40, "no_price": 0.40},
    )
    assert clean["regime_id"] == broken["regime_id"]
    assert broken["confidence"] < clean["confidence"]
    assert "pm_book_quality" in broken["features"]
    assert broken["features"]["pm_book_quality"] < clean["features"]["pm_book_quality"]

    # Damp must still apply after centroids have warmed (production path).
    warm = _fresh()
    for _ in range(8):
        warm.update(
            prices, vol_score=0.80, trend_score=0.75,
            pm_state={"spread_score": 0.85, "yes_price": 0.52, "no_price": 0.48},
        )
    warmed_clean = warm.snapshot()["confidence"]
    warmed_broken = warm.update(
        prices, vol_score=0.80, trend_score=0.75,
        pm_state={"spread_score": 0.10, "yes_price": 0.40, "no_price": 0.40},
    )["confidence"]
    assert warmed_broken < warmed_clean


# ---------------------------------------------------------------------------
# Propagation: SignalView, BaseBot, Lab, meta-learner
# ---------------------------------------------------------------------------

def test_signal_view_prefers_rich_regime_label():
    sv = SignalView({
        "vol_regime": {"regime": "quiet", "trend_score": 0.2},
        "market_regime": {"regime_id": "low_vol_range", "label": "low_vol_range"},
    })
    assert sv.regime_label == "low_vol_range"
    assert sv.market_regime["regime_id"] == "low_vol_range"


def test_regime_context_exposes_rich_fields():
    signals = {
        "vol_regime": {
            "regime": "quiet", "regime_id": "low_vol_range",
            "trend_score": 0.2, "vol_score": 0.15, "confidence": 0.7,
            "features": {"vol": 0.15, "trend": 0.2},
            "meta_bucket": "ranging",
        },
        "market_regime": {
            "regime_id": "low_vol_range", "label": "low_vol_range",
            "legacy": "quiet", "trend_score": 0.2, "vol_score": 0.15,
            "confidence": 0.7, "known": True, "meta_bucket": "ranging",
            "features": {"vol": 0.15, "trend": 0.2},
        },
    }
    ctx = BaseBot.regime_context(signals)
    assert ctx["label"] == "low_vol_range"
    assert ctx["legacy"] == "quiet"
    assert ctx["known"] is True
    assert ctx["ranging"] is True
    assert ctx["vol_score"] == pytest.approx(0.15)


def test_lab_regime_damps_quiet_and_chop():
    assert "mom" in REGIME_LANE_DAMP["quiet"]       # legacy label still damps
    # low_vol_range / low_vol_trend mom damp moved to regime_adapt (P1c:
    # single-authority mom_lane_scale, no double-damp with lab.py).
    assert "low_vol_range" in REGIME_LANE_DAMP
    assert REGIME_LANE_DAMP["low_vol_range"] == {}
    assert "low_vol_trend" in REGIME_LANE_DAMP
    assert REGIME_LANE_DAMP["low_vol_trend"] == {}
    assert "mom" in REGIME_LANE_DAMP["high_vol_chop"]
    assert "strat" in REGIME_LANE_DAMP["high_vol_chop"]

    lab = SignalLab()
    damps = lab.regime_damps_for({
        "market_regime": {"regime_id": "high_vol_chop", "legacy": "volatile"},
        "vol_regime": {"regime": "volatile", "regime_id": "high_vol_chop"},
    })
    assert damps["mom"] < 1.0
    assert damps["strat"] < 1.0


def test_lab_blend_applies_strat_damp_under_chop():
    lab = SignalLab()
    lanes = {"drift": 0.2, "mom": 0.1, "strat": 0.5, "pm": 0, "cvd": 0,
             "obi": 0, "fut": 0, "tech": 0, "xasset": 0, "learn": 0}
    profile = {k: 1.0 for k in lanes}
    signals = {
        "market_regime": {"regime_id": "high_vol_chop", "legacy": "volatile"},
        "vol_regime": {"regime": "volatile", "regime_id": "high_vol_chop"},
    }
    with_damp = lab.blend("momentum", lanes, profile, signals=signals)
    without = lab.blend("momentum", lanes, profile, signals=None)
    # Strat contribution should be smaller under chop damp
    assert abs(with_damp.contributions["strat"]) < abs(without.contributions["strat"])


def test_meta_bucket_for_uses_regime_id():
    assert bucket_for(0.5, regime_id="high_vol_chop") == "chop"
    assert bucket_for(0.5, regime_id="high_vol_trend") == "trending"
    assert bucket_for(0.2, regime_id=None) == "ranging"


# ---------------------------------------------------------------------------
# Evolution fitness: regime-conditioned
# ---------------------------------------------------------------------------

def _trade(pnl, regime, outcome=None):
    if outcome is None:
        outcome = "win" if pnl >= 0 else "loss"
    return {
        "pnl": pnl,
        "outcome": outcome,
        "created_at": "2026-07-20 12:00:00",
        "trade_features": [f"regime:{regime}", "price_neutral"],
    }


def test_regime_breakdown_groups_trades():
    trades = [
        _trade(2.0, "high_vol_trend"),
        _trade(1.0, "high_vol_trend"),
        _trade(-3.0, "high_vol_chop"),
        _trade(-1.0, "high_vol_chop"),
    ]
    bd = regime_breakdown(trades)
    assert bd["high_vol_trend"]["pnl"] == pytest.approx(3.0)
    assert bd["high_vol_chop"]["pnl"] == pytest.approx(-4.0)


def test_regime_conditioned_fitness_penalizes_single_regime_bleed(monkeypatch):
    monkeypatch.setattr(config, "GA_REGIME_CONDITION", True, raising=False)
    monkeypatch.setattr(config, "GA_REGIME_MIN_TRADES", 2, raising=False)

    # Strong overall P&L but toxic in chop
    toxic = [
        _trade(5.0, "high_vol_trend"),
        _trade(5.0, "high_vol_trend"),
        _trade(-8.0, "high_vol_chop"),
        _trade(-8.0, "high_vol_chop"),
    ]
    # Balanced positive
    robust = [
        _trade(2.0, "high_vol_trend"),
        _trade(2.0, "high_vol_trend"),
        _trade(1.5, "high_vol_chop"),
        _trade(1.5, "high_vol_chop"),
    ]
    t = multi_objective_fitness(toxic)
    r = multi_objective_fitness(robust)
    assert r["regime_robustness"] > t["regime_robustness"]
    # Overall P&L: toxic is more negative actually (-6 vs +7)
    # Make toxic look good on raw P&L but bad on robustness
    toxic2 = (
        [_trade(4.0, "high_vol_trend")] * 6
        + [_trade(-3.0, "high_vol_chop")] * 2
    )
    robust2 = (
        [_trade(1.5, "high_vol_trend")] * 4
        + [_trade(1.5, "high_vol_chop")] * 4
    )
    t2 = multi_objective_fitness(toxic2)
    r2 = multi_objective_fitness(robust2)
    assert t2["pnl"] > r2["pnl"]  # toxic has higher raw P&L
    assert r2["regime_robustness"] > t2["regime_robustness"]

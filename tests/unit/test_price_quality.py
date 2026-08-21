"""Mid-band price-quality filter — skip 50–58¢ unless cheap-ask + strong drift."""

import pytest

from bots.base_bot import apply_xasset_confirm, price_quality_ok
from bots.bot_sniper import SniperBot
from bots.bot_momentum import MomentumBot


def test_price_quality_fails_closed_on_bad_inputs():
    assert not price_quality_ok(side_mid=None, side_ask=0.51, signed_drift=0.40)
    assert not price_quality_ok(side_mid=0.51, side_ask="x", signed_drift=0.40)


def test_xasset_confirm_does_not_mutate_input():
    src = {"drift": 0.40, "xasset": -0.80}
    out = apply_xasset_confirm(src)
    assert src["xasset"] == -0.80
    assert out["xasset"] == 0.0
    assert out is not src


def test_price_quality_passes_outside_band():
    assert price_quality_ok(side_mid=0.34, side_ask=0.35, signed_drift=0.50)
    assert price_quality_ok(side_mid=0.62, side_ask=0.63, signed_drift=0.70)


def test_price_quality_blocks_mid_band_no_residual_lag():
    # Honest Φ ~0.58 vs 56¢ ask is not 12¢ of lag.
    assert not price_quality_ok(
        side_mid=0.56, side_ask=0.56, signed_drift=0.40,
        implied_side=0.58,
    )


def test_price_quality_allows_strong_lag_in_band():
    # This session: 55–58¢ + |d|≥0.40 won; do not require ask≤0.52.
    assert price_quality_ok(side_mid=0.56, side_ask=0.56, signed_drift=0.65)


def test_sniper_skips_56c_lag():
    bot = SniperBot(name="sniper-pq")
    market = {
        "current_price": 0.56,
        "no_price": 0.44,
        "yes_ask": 0.56,
        "no_ask": 0.45,
        "time_remaining_seconds": 120,
    }
    signals = {
        "btc_drift": 0.65,
        "btc_drift_pct": 0.0008,
        "btc_strike": 100000.0,
        "btc_now": 100080.0,
        "btc_implied_yes": 0.57,
        "prices": [100.0, 100.1, 100.2],
        "orderflow": {},
        "regime": {"label": "normal", "known": True, "vol_score": 0.5},
    }
    d = bot.make_decision(market, signals)
    assert d["action"] == "skip"


def test_sniper_stamps_regime_feature():
    bot = SniperBot(name="sniper-reg")
    market = {
        "current_price": 0.45,
        "no_price": 0.55,
        "yes_ask": 0.46,
        "no_ask": 0.56,
        "time_remaining_seconds": 180,
    }
    signals = {
        "btc_drift": 0.50,
        "btc_drift_pct": 0.0016,
        "btc_drift_z": 0.50,
        "prices": [100.0, 100.1],
        "orderflow": {},
        "market_regime": {
            "regime_id": "normal", "label": "normal",
            "legacy": "normal", "known": True, "confidence": 0.8,
            "features": {},
        },
        "vol_regime": {"regime": "normal", "regime_id": "normal"},
    }
    d = bot.make_decision(market, signals)
    feats = d.get("features") or []
    assert any(str(f).startswith("regime:") for f in feats)


def test_momentum_skips_mid_band_weak_drift():
    bot = MomentumBot(name="mom-pq")
    market = {
        "current_price": 0.56,
        "yes_price": 0.56,
        "no_price": 0.44,
        "yes_ask": 0.56,
        "no_ask": 0.45,
        "time_remaining_seconds": 180,
        "resolves_at": None,
    }
    signals = {
        "btc_drift": 0.25,
        "btc_drift_pct": 0.00025,
        "btc_strike": 100000.0,
        "btc_implied_yes": 0.58,
        "prices": [100.0] * 20 + [100.1],
        "orderflow": {},
        "market_regime": {
            "regime_id": "normal", "label": "normal",
            "legacy": "normal", "known": True, "confidence": 0.8,
            "features": {"vol": 0.4, "trend": 0.5},
        },
    }
    d = bot.make_decision(market, signals)
    assert d["action"] == "skip"


def test_xasset_confirm_zeros_when_fights_drift():
    out = apply_xasset_confirm({"drift": 0.40, "xasset": -0.80})
    assert out["xasset"] == 0.0


def test_xasset_confirm_keeps_agreeing():
    out = apply_xasset_confirm({"drift": 0.40, "xasset": 0.50})
    assert out["xasset"] == pytest.approx(0.50)


def test_xasset_confirm_zeros_when_drift_flat():
    out = apply_xasset_confirm({"drift": 0.01, "xasset": 0.80})
    assert out["xasset"] == 0.0


def test_sniper_dual_gate_skip_stamps_entry_and_reason():
    bot = SniperBot(name="sniper-dg")
    market = {
        "current_price": 0.45,
        "no_price": 0.55,
        "yes_ask": 0.46,
        "no_ask": 0.56,
        "time_remaining_seconds": 200,
    }
    signals = {
        "btc_drift": 0.02,
        "btc_drift_pct": 0.00005,
        "btc_strike": 100000.0,
        "btc_now": 100005.0,
        "prices": [100.0, 100.1],
        "orderflow": {},
        "regime": {"label": "normal", "known": True, "vol_score": 0.5},
    }
    d = bot.make_decision(market, signals)
    assert d["action"] == "skip"
    assert d.get("skip_reason") == "drift_dual_gate"
    assert d.get("entry_price") is not None

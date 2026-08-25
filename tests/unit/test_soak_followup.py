"""Soak follow-up (2026-08-23): late-momentum skip, 58¢ cap, sniper 8bp
pocket, gate-tuner mid-band, hybrid single-mom in normal, TWAP pullback
meanrev, mom+tech mid-band shadow, portfolio on by default.
"""

from unittest.mock import patch

import config
from arena import combo_explorer as ce
from arena import gate_tuner
from arena.live_scorecard import _gate_stats
from bots.base_bot import BaseBot
from bots.bot_hybrid import HybridBot
from bots.bot_mean_rev import MeanRevBot, strike_pullback
from bots.bot_momentum import MomentumBot
from bots.bot_sniper import SniperBot
from tests.conftest import make_market, make_signals


def test_momentum_skips_last_80s_even_with_consecutive_bars():
    bot = MomentumBot(name="mom-late")
    prices = [100_000.0 * (1.001 ** i) for i in range(30)]
    sig = bot.analyze(
        make_market(time_remaining=80),
        make_signals(prices=prices, latest=prices[-1], btc_drift=0.25),
    )
    assert sig["action"] == "hold"
    assert "late-window" in (sig.get("reasoning") or "").lower()


def test_momentum_still_fires_mid_window():
    bot = MomentumBot(name="mom-mid")
    prices = [100_000.0 * (1.001 ** i) for i in range(30)]
    sig = bot.analyze(
        make_market(time_remaining=150),
        make_signals(prices=prices, latest=prices[-1], btc_drift=0.25),
    )
    assert sig["action"] == "buy" and sig["side"] == "yes"


def test_momentum_and_hybrid_max_mid_is_058():
    assert BaseBot.STRATEGY_MAX_SIDE_PRICE["momentum"] == 0.58
    assert BaseBot.STRATEGY_MAX_SIDE_PRICE["hybrid"] == 0.58
    assert BaseBot.STRATEGY_MAX_SIDE_PRICE["hybrid"] <= \
        BaseBot.STRATEGY_MAX_SIDE_PRICE["momentum"]


def _sniper_mkt(yes=0.54, tr=150):
    return {
        "current_price": yes,
        "no_price": round(1.0 - yes, 4),
        "yes_ask": round(yes + 0.01, 4),
        "no_ask": round(1.0 - yes + 0.01, 4),
        "time_remaining_seconds": tr,
    }


def _sniper_sig(*, d_pct, z, drift=0.40, implied=0.66):
    return {
        "btc_drift": drift,
        "btc_drift_pct": d_pct,
        "btc_drift_z": z,
        "btc_implied_yes": implied,
        "btc_strike": 100000.0,
        "btc_now": 100000.0 * (1.0 + d_pct),
        "prices": [100.0, 100.1, 100.2],
        "orderflow": {},
        "regime": {"label": "normal", "known": True, "vol_score": 0.5},
    }


def test_sniper_allows_8bp_in_50_58_pocket():
    bot = SniperBot(name="snp-pocket")
    d = bot.make_decision(
        _sniper_mkt(yes=0.54),
        _sniper_sig(d_pct=0.00085, z=0.45, drift=0.40, implied=0.66),
    )
    assert d["action"] == "buy"
    assert d["side"] == "yes"


def test_sniper_still_needs_15bp_outside_pocket():
    bot = SniperBot(name="snp-out")
    d = bot.make_decision(
        _sniper_mkt(yes=0.45),
        _sniper_sig(d_pct=0.00085, z=0.45, drift=0.40, implied=0.62),
    )
    assert d["action"] in ("skip", "hold")
    why = (d.get("reasoning") or "").lower()
    assert d.get("skip_reason") == "sniper_conviction" or "15bp" in why or "0.0015" in why


def test_gate_stats_split_by_price_band():
    rows = [
        {"action": "skip", "skip_reason": "drift_dual_gate",
         "entry_price": 0.54, "would_win": 1, "hyp_pnl": 0.04},
        {"action": "skip", "skip_reason": "drift_dual_gate",
         "entry_price": 0.93, "would_win": 1, "hyp_pnl": 0.004},
    ]
    gates = _gate_stats(rows)
    dg = gates["drift_dual_gate"]
    assert "by_band" in dg
    assert dg["by_band"]["mid_50_58"]["n_hyp"] == 1
    assert dg["by_band"]["expensive"]["n_hyp"] == 1
    assert dg["avg_entry"] > 0.70  # blend is still expensive


def test_gate_tuner_loosens_on_midband_not_expensive_blend(monkeypatch):
    report = {
        "gates": {
            "drift_dual_gate": {
                "markets": 658, "n_hyp": 658, "wr": 0.91,
                "avg_hyp_pnl": 0.006, "avg_entry": 0.90,
                "by_band": {
                    "mid_50_58": {
                        "markets": 40, "n_hyp": 40, "wr": 0.70,
                        "avg_hyp_pnl": 0.04, "avg_entry": 0.54,
                    },
                    "expensive": {
                        "markets": 600, "n_hyp": 600, "wr": 0.93,
                        "avg_hyp_pnl": 0.004, "avg_entry": 0.95,
                    },
                },
            }
        }
    }
    monkeypatch.setattr(gate_tuner, "_load_scorecard", lambda hours=None: report)
    out = gate_tuner.suggest(apply=False)
    z = out["suggestions"]["DRIFT_MIN_ABS_Z"]
    assert z["action"] == "loosen"
    assert z["suggested"] < float(config.DRIFT_MIN_ABS_Z)
    assert "0.54" in z["why"] or "mid" in z["why"]


def test_hybrid_single_momentum_in_normal_buys():
    bot = HybridBot(name="hy-norm")
    hold = {"action": "hold", "side": "yes", "confidence": 0.0, "reasoning": ""}
    buy = {"action": "buy", "side": "yes", "confidence": 0.6, "reasoning": "mom"}
    with patch.object(bot, "_perf_tilts",
                      return_value={"momentum": 1.0, "mean_rev": 1.0,
                                    "phantom": 1.0}):
        with patch.object(bot, "_cached_sub_analyze", return_value={
            "momentum": buy, "mean_rev": hold, "phantom": hold,
        }):
            sig = bot.analyze(
                make_market(),
                make_signals(
                    btc_drift=0.25,
                    vol_regime={"regime": "normal", "trend_score": 0.5, "known": True},
                    market_regime={"regime": "normal", "label": "normal"},
                ),
            )
    assert sig["action"] == "buy"
    assert sig["side"] == "yes"
    assert abs(sig["signals"]["votes"].get("momentum", 0)) > 0


def test_hybrid_unknown_still_needs_two_sub_agreement():
    bot = HybridBot(name="hy-unk")
    hold = {"action": "hold", "side": "yes", "confidence": 0.0, "reasoning": ""}
    buy = {"action": "buy", "side": "yes", "confidence": 0.6, "reasoning": "mom"}
    with patch.object(bot, "_cached_sub_analyze", return_value={
        "momentum": buy, "mean_rev": hold, "phantom": hold,
    }):
        sig = bot.analyze(
            make_market(),
            make_signals(btc_drift=0.2, vol_regime={}, market_regime={}),
        )
    assert sig["action"] == "hold"
    assert sig.get("meta_token")


def test_strike_pullback_yes_dip():
    # TWAP above strike, retraced 50% from the window high toward PTB.
    frac, sign, extreme = strike_pullback(100_250.0, 100_000.0,
                                          [100_200.0, 100_500.0, 100_250.0])
    assert sign == 1
    assert extreme == 100_500.0
    assert abs(frac - 0.5) < 1e-9


def test_meanrev_buys_twap_pullback_not_four_bar_z():
    bot = MeanRevBot(name="mr-pb")
    # Window-local: bounce toward strike while still below it (NO winning).
    # 60s remaining → 4 closed 1m bars. Down-drift backs the NO side.
    prices = [99_400.0, 99_400.0, 99_400.0, 99_400.0, 99_850.0]
    late = make_market(time_remaining=60)
    d = bot.analyze(
        late,
        make_signals(
            prices=prices, latest=prices[-1],
            btc_now=99_850.0, btc_strike=100_000.0, btc_drift=-0.30,
        ),
    )
    assert d["action"] == "buy" and d["side"] == "no"
    why = d.get("reasoning") or ""
    assert "pullback" in why.lower() or "pb=" in why.lower()
    assert "z=" not in why.split("|")[0] or "pullback" in why.lower()


def test_meanrev_holds_at_window_extreme():
    """Chasing the window high is momentum's job — meanrev needs a dip."""
    bot = MeanRevBot(name="mr-ext")
    prices = [100_100.0, 100_200.0, 100_300.0, 100_400.0]
    d = bot.analyze(
        make_market(time_remaining=60),
        make_signals(
            prices=prices, latest=prices[-1],
            btc_now=100_400.0, btc_strike=100_000.0, btc_drift=0.30,
        ),
    )
    assert d["action"] == "hold"


def test_meanrev_holds_when_twap_now_mixes_spot_path():
    bot = MeanRevBot(name="mr-mix")
    d = bot.analyze(
        make_market(time_remaining=60),
        make_signals(
            prices=[70_000.0, 70_100.0, 70_300.0, 70_300.0],
            latest=70_300.0,
            btc_now=70_040.0, btc_strike=70_000.0, btc_drift=-0.30,
        ),
    )
    assert d["action"] == "hold"
    assert "mix" in (d.get("reasoning") or "").lower()


def test_combo_mom_tech_midband_is_shadow_only():
    rows = []
    for i in range(25):
        rows.append({
            "market_id": f"m{i}", "side": "yes", "entry_price": 0.54,
            "drift": 0.20, "mom": 0.30, "tech": 0.28, "xasset": 0.01,
            "strat": 0.05, "market_up": True, "action": "buy",
        })
    report = ce.build_combo_report(rows)
    rule = report["rules"]["mom_tech_midband"]
    assert rule["markets"] == 25
    assert rule["verdict"] == "earned"
    names = {e["name"] for e in report["earned"]}
    assert "mom_tech_midband" not in names
    assert config.COMBO_CONFIRM_APPLY is False
    assert ce.try_confirm(
        {"mom": 0.30, "tech": 0.28, "drift": 0.20},
        yes_mid=0.54, no_mid=0.46, yes_ask=0.55, no_ask=0.47,
    ) is None


def test_portfolio_config_default_is_on():
    assert config.PORTFOLIO_ALLOCATION_ENABLED is True
    assert config.GATE_TUNE_APPLY is False
    assert config.COMBO_CONFIRM_APPLY is False

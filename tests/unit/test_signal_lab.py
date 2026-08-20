"""Tests for the central SignalLab (signals/lab.py): consistent lane
computation + caching, dynamic weighting (profiles / overrides / regime),
live-validation gating, the ML hook seam, perf tilts, and the contribution
logging contract that make_decision embeds in trade reasoning.
"""

import config
import pytest

from signals.lab import BlendResult, SignalLab, SignalView, get_lab

PROFILE = {"drift": 0.5, "mom": 0.3, "strat": 0.2,
           "pm": 0.0, "cvd": 0.0, "obi": 0.0,
           "fut": 0.0, "tech": 0.0, "xasset": 0.0}


def _signals(**over):
    base = {"prices": [100.0, 100.1], "latest": 100.1, "orderflow": {},
            "pm_momentum": 0.0, "obi": 0.2, "cvd": -0.3, "btc_drift": 0.4,
            "vol_regime": {"regime": "normal"},
            "futures": {"taker_delta": 0.5},
            "technicals": {"mtf_score": -0.6}, "xasset": 0.7}
    base.update(over)
    return base


def _lab(overrides=None, monitor=None) -> SignalLab:
    return SignalLab(overrides_provider=lambda: dict(overrides or {}),
                     monitor_provider=lambda: dict(monitor or {}))


class TestSignalView:
    def test_mapping_compatibility(self):
        sv = SignalView({"prices": [1.0], "custom": 7})
        assert sv["custom"] == 7
        assert sv.get("custom") == 7
        assert "prices" in sv
        assert len(sv) == 2

    def test_typed_defaults_on_empty(self):
        sv = SignalView(None)
        assert sv.prices == []
        assert sv.btc_drift == 0.0
        assert sv.vol_regime == {}
        assert sv.regime_label is None
        assert sv.macro_caution == 0.0

    def test_of_is_idempotent(self):
        sv = SignalView({"latest": 5.0})
        assert SignalView.of(sv) is sv
        assert SignalView.of({"latest": 5.0}).latest == 5.0


class TestComputeLanes:
    def test_kill_switched_lanes_zero_but_raw_preserved(self):
        lanes, raw = _lab().compute_lanes({"id": "m"}, _signals())
        # pm/cvd/obi/fut/tech stay kill-switched; xasset is confirm-only live.
        for lane in ("pm", "cvd", "obi", "fut", "tech"):
            assert lanes[lane] == 0.0, lane
        assert lanes["xasset"] == pytest.approx(0.7)
        # Raw pre-kill-switch reads survive for the validation dataset.
        assert raw["fut_taker"] == 0.5
        assert raw["tech_mtf"] == -0.6
        assert raw["xasset"] == 0.7

    def test_drift_passthrough_and_clamp(self):
        lanes, _ = _lab().compute_lanes({"id": "m"}, _signals(btc_drift=0.4))
        assert lanes["drift"] == pytest.approx(0.4)
        lanes, _ = _lab().compute_lanes({"id": "m"}, _signals(btc_drift=3.0))
        assert lanes["drift"] == 1.0

    def test_momentum_from_candles_and_fallback(self):
        from signals.drift_scale import reset_drift_scale_estimator
        reset_drift_scale_estimator()  # cold → MOM_SCALE_PRIOR
        lanes, raw = _lab().compute_lanes(
            {"id": "m"}, _signals(prices=[100.0, 100.2], latest=0.0))
        assert raw["price_momentum"] == pytest.approx(0.002)
        # Cold adaptive scale ≈ 0.002 prior → soft_saturate(0.002) ≈ 0.76
        assert lanes["mom"] > 0.55
        # Single candle + live price falls back to latest-vs-candle.
        lanes2, raw2 = _lab().compute_lanes(
            {"id": "m2"}, _signals(prices=[100.0], latest=100.1))
        assert raw2["price_momentum"] == pytest.approx(0.001)
        assert lanes2["mom"] > 0.0

    def test_quiet_regime_damps_momentum(self):
        sig_normal = _signals(prices=[100.0, 100.2],
                              vol_regime={"regime": "normal"})
        sig_quiet = _signals(prices=[100.0, 100.2],
                             vol_regime={"regime": "quiet"})
        lanes_n, _ = _lab().compute_lanes({"id": "a"}, sig_normal)
        lanes_q, _ = _lab().compute_lanes({"id": "b"}, sig_quiet)
        damp = getattr(config, "MOM_QUIET_REGIME_DAMP", 0.5)
        assert lanes_q["mom"] == pytest.approx(lanes_n["mom"] * damp)

    def test_override_enables_killed_lane(self):
        overrides = {"fut": {"enabled": True, "profile": {"momentum": 0.1}}}
        lanes, _ = _lab(overrides).compute_lanes({"id": "m"}, _signals())
        assert lanes["fut"] == pytest.approx(0.5)   # kill-switch lifted

    def test_cache_shares_within_tick_and_distinguishes_values(self):
        lab = _lab()
        lanes_a, _ = lab.compute_lanes({"id": "m"}, _signals(btc_drift=0.4))
        lanes_b, _ = lab.compute_lanes({"id": "m"}, _signals(btc_drift=0.4))
        assert lanes_a is lanes_b   # identical inputs -> shared computation
        # Same market id, different values (e.g. dict reallocated at the same
        # address next tick) must NOT hit the cache.
        lanes_c, _ = lab.compute_lanes({"id": "m"}, _signals(btc_drift=-0.4))
        assert lanes_c["drift"] == pytest.approx(-0.4)


class TestWeightingAndGating:
    def test_profile_weights_applied(self):
        lab = _lab()
        lanes = {"drift": 0.4, "mom": 0.2, "strat": -0.5}
        res = lab.blend("momentum", lanes, PROFILE)
        expected = 0.5 + 0.5 * (0.5 * 0.4 + 0.3 * 0.2 + 0.2 * -0.5)
        assert res.prob == pytest.approx(expected)
        assert res.contributions["drift"] == pytest.approx(0.2)

    def test_unknown_lane_carries_weight_in_value(self):
        # The learn-lane convention: lanes absent from the profile get w=1.0.
        res = _lab().blend("momentum", {"learn": 0.1}, PROFILE)
        assert res.prob == pytest.approx(0.55)

    def test_override_profile_beats_static_profile(self):
        overrides = {"fut": {"enabled": True,
                             "profile": {"momentum": 0.2, "hybrid": 0.0}}}
        lab = _lab(overrides)
        res_mom = lab.blend("momentum", {"fut": 1.0}, PROFILE)
        assert res_mom.prob == pytest.approx(0.6)   # 0.5 + 0.5*0.2
        # A strategy the override profile doesn't name stays at 0.
        res_mr = lab.blend("mean_reversion", {"fut": 1.0}, PROFILE)
        assert res_mr.prob == pytest.approx(0.5)

    def test_monitor_disabled_verdict_gates_lane(self):
        overrides = {"fut": {"enabled": True, "profile": {"momentum": 0.2}}}
        monitor = {"fut": {"n": 80, "accuracy": 0.44, "verdict": "disabled",
                           "min_trades": 50, "min_accuracy": 0.53}}
        res = _lab(overrides, monitor).blend("momentum", {"fut": 1.0}, PROFILE)
        assert res.prob == pytest.approx(0.5)
        assert "fut" in res.gated

    def test_failing_accuracy_gates_even_before_verdict_flip(self):
        overrides = {"tech": {"enabled": True, "profile": {"momentum": 0.2}}}
        monitor = {"tech": {"n": 120, "accuracy": 0.48, "verdict": "healthy",
                            "min_trades": 50, "min_accuracy": 0.53}}
        res = _lab(overrides, monitor).blend("momentum", {"tech": 1.0}, PROFILE)
        assert res.prob == pytest.approx(0.5)
        assert res.gated == ("tech",)

    def test_healthy_or_collecting_lane_not_gated(self):
        overrides = {"xasset": {"enabled": True, "profile": {"momentum": 0.2}}}
        monitor = {"xasset": {"n": 120, "accuracy": 0.57, "verdict": "healthy",
                              "min_trades": 50, "min_accuracy": 0.53},
                   "fut": {"n": 10, "accuracy": 0.40, "verdict": "collecting",
                           "min_trades": 50, "min_accuracy": 0.53}}
        lab = _lab(overrides, monitor)
        assert lab.gated_lanes() == frozenset()
        res = lab.blend("momentum", {"xasset": 1.0}, PROFILE)
        assert res.prob == pytest.approx(0.6)

    def test_prob_clamped_to_config_bounds(self):
        res = _lab().blend("momentum", {"drift": 1.0, "mom": 1.0, "strat": 1.0},
                           PROFILE)
        assert res.prob == config.MODEL_PROB_MAX
        res = _lab().blend("momentum",
                           {"drift": -1.0, "mom": -1.0, "strat": -1.0}, PROFILE)
        assert res.prob == config.MODEL_PROB_MIN


class TestModelHook:
    def test_hook_replaces_probability(self):
        lab = _lab()
        lab.set_model_hook(lambda strat, lanes, weights: 0.72)
        res = lab.blend("momentum", {"drift": 0.0}, PROFILE)
        assert res.prob == pytest.approx(0.72)

    def test_hook_none_and_out_of_range_keep_linear_blend(self):
        lab = _lab()
        lab.set_model_hook(lambda strat, lanes, weights: None)
        assert lab.blend("momentum", {"drift": 0.4},
                         PROFILE).prob == pytest.approx(0.6)
        lab.set_model_hook(lambda strat, lanes, weights: 7.0)
        assert lab.blend("momentum", {"drift": 0.4},
                         PROFILE).prob == pytest.approx(0.6)

    def test_broken_hook_never_stalls_a_decision(self):
        lab = _lab()
        lab.set_model_hook(lambda *a: 1 / 0)
        assert lab.blend("momentum", {"drift": 0.4},
                         PROFILE).prob == pytest.approx(0.6)


class TestPerfTilts:
    def test_score_perf_tilts_pure(self):
        perf = {"momentum-v1": {"total_trades": 30, "wins": 21},
                "meanrev-v1": {"total_trades": 30, "wins": 9}}
        tilts = SignalLab.score_perf_tilts(
            perf, {"momentum": "momentum", "mean_rev": "meanrev",
                   "phantom": "phantom"})
        assert tilts["momentum"] > 1.0
        assert tilts["mean_rev"] < 1.0
        assert tilts["phantom"] == 1.0    # no data -> neutral

    def test_small_sample_damped(self):
        perf = {"momentum-v1": {"total_trades": 2, "wins": 2}}
        tilts = SignalLab.score_perf_tilts(perf, {"momentum": "momentum"})
        assert 1.0 < tilts["momentum"] < 1.06


class TestBlendResultLogging:
    def test_log_str_carries_contributions_and_gate(self):
        res = BlendResult(prob=0.61, weights={"drift": 0.5},
                          contributions={"drift": 0.22, "mom": 0.0},
                          gated=("tech",))
        s = res.log_str()
        assert s.startswith("P=0.610[")
        assert "drift=+0.220" in s
        assert "mom=" not in s          # zero contributions omitted
        assert "gated=tech" in s


class TestMakeDecisionIntegration:
    """The lab is wired through BaseBot: reasoning keeps the parsed tokens
    (cand(...), drift=, mom=, strat=) AND now carries the contribution log;
    the decision dict exposes per-lane attribution."""

    def _decide(self, monkeypatch, drift=0.9):
        import db
        from bots.bot_momentum import MomentumBot
        monkeypatch.setattr(db, "get_paper_available", lambda: 200.0)
        monkeypatch.setattr(db, "get_kelly_fraction", lambda: 0.25)
        bot = MomentumBot(name="momentum-lab-test")
        bot._perf_cache = (9e12, 0)     # skip the resolved-count DB read
        market = {"id": "m", "current_price": 0.50, "no_price": 0.50,
                  "yes_ask": 0.51, "no_ask": 0.51,
                  "time_remaining_seconds": 60}
        signals = {"prices": [100.0, 100.4], "latest": 100.4,
                   "orderflow": {}, "pm_momentum": 0.0, "obi": 0.0,
                   "cvd": 0.0, "btc_drift": drift,
                   # Dual-gate needs real moneyness (not only z-score).
                   "btc_drift_pct": 0.001 if drift > 0 else -0.001,
                   "drift_vol_scale": 0.0022,
                   "btc_strike": 100000.0, "btc_now": 100100.0}
        return bot.make_decision(market, signals)

    def test_buy_reasoning_contains_contract_tokens(self, monkeypatch):
        d = self._decide(monkeypatch)
        assert d["action"] == "buy"
        assert "cand(fut=" in d["reasoning"]        # lane_monitor contract
        assert "drift=+" in d["reasoning"]          # core_lane_tuner contract
        assert "P=" in d["reasoning"]               # contribution log
        assert "lane_contributions" in d
        assert d["lane_contributions"]["drift"] > 0.0

    def test_default_lab_singleton_has_override_provider(self):
        import bots.base_bot  # noqa: F401 — importing wires the provider
        assert get_lab().overrides_provider is not None

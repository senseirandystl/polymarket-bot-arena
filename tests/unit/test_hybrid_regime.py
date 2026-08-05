"""HybridBot regime-switching meta-learner (bots/bot_hybrid.py)."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bots.bot_hybrid import HybridBot, SUBS


def _bot():
    return HybridBot(name="hybrid-test")


def _sigs(trend_score=0.5, regime="normal"):
    return {"vol_regime": {"trend_score": trend_score, "vol_score": 0.5,
                           "regime": regime}}


class TestDynamicWeights:
    def test_weights_normalized(self):
        w = _bot()._dynamic_weights(_sigs())
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert set(w) == {s for s, *_ in SUBS}

    def test_trending_regime_upweights_trend_followers(self):
        bot = _bot()
        with patch.object(bot, "_perf_tilts", return_value={s: 1.0 for s, *_ in SUBS}):
            chop = bot._dynamic_weights(_sigs(trend_score=0.0))
            trend = bot._dynamic_weights(_sigs(trend_score=1.0))
        assert trend["momentum"] > chop["momentum"]
        assert trend["phantom"] > chop["phantom"]
        assert trend["mean_rev"] < chop["mean_rev"]

    def test_tilt_is_continuous_not_a_switch(self):
        bot = _bot()
        with patch.object(bot, "_perf_tilts", return_value={s: 1.0 for s, *_ in SUBS}):
            mids = [bot._dynamic_weights(_sigs(trend_score=t))["momentum"]
                    for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert mids == sorted(mids)          # monotonic in trendiness
        assert len(set(mids)) == len(mids)   # strictly — no plateau/bucket

    def test_missing_regime_neutral(self):
        bot = _bot()
        with patch.object(bot, "_perf_tilts", return_value={s: 1.0 for s, *_ in SUBS}):
            w_missing = bot._dynamic_weights({})
            w_neutral = bot._dynamic_weights(_sigs(trend_score=0.5))
        assert w_missing == w_neutral


class TestPerfTilts:
    def test_db_failure_falls_back_to_neutral(self):
        bot = _bot()
        with patch("bots.bot_hybrid.db.get_all_bots_performance",
                   side_effect=RuntimeError("db down")):
            tilts = bot._perf_tilts()
        assert tilts == {s: 1.0 for s, *_ in SUBS}

    def test_winning_substrategy_upweighted(self):
        bot = _bot()
        perf = {"momentum-v1": {"total_trades": 30, "wins": 21, "losses": 9,
                                "total_pnl": 12.0, "win_rate": 0.7},
                "meanrev-v1": {"total_trades": 30, "wins": 9, "losses": 21,
                               "total_pnl": -12.0, "win_rate": 0.3}}
        with patch("bots.bot_hybrid.db.get_all_bots_performance",
                   return_value=perf):
            tilts = bot._perf_tilts()
        assert tilts["momentum"] > 1.0
        assert tilts["mean_rev"] < 1.0
        assert tilts["phantom"] == 1.0  # no data -> neutral

    def test_small_samples_barely_move(self):
        bot = _bot()
        perf = {"momentum-v1": {"total_trades": 2, "wins": 2, "losses": 0,
                                "total_pnl": 2.0, "win_rate": 1.0}}
        with patch("bots.bot_hybrid.db.get_all_bots_performance",
                   return_value=perf):
            tilts = bot._perf_tilts()
        assert 1.0 < tilts["momentum"] < 1.06


class TestAnalyze:
    def test_all_hold_holds(self):
        bot = _bot()
        hold = {"action": "hold", "side": "yes", "confidence": 0, "reasoning": ""}
        for a in bot._analyzers.values():
            patch.object(a, "analyze", return_value=hold).start()
        try:
            out = bot.analyze({}, _sigs())
        finally:
            patch.stopall()
        assert out["action"] == "hold"

    def test_agreement_produces_buy_with_regime_label(self):
        bot = _bot()
        buy_yes = {"action": "buy", "side": "yes", "confidence": 0.5,
                   "reasoning": "x"}
        with patch.object(bot, "_perf_tilts",
                          return_value={s: 1.0 for s, *_ in SUBS}):
            for a in bot._analyzers.values():
                patch.object(a, "analyze", return_value=buy_yes).start()
            try:
                out = bot.analyze({}, _sigs(trend_score=1.0, regime="trending"))
            finally:
                patch.stopall()
        assert out["action"] == "buy"
        assert out["side"] == "yes"
        assert "trending" in out["reasoning"]

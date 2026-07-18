"""New signal modules: volatility regime, technicals, cross-asset, macro,
futures meta (offline paths only — no network in tests)."""

import datetime
import sys
import zoneinfo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from signals import cross_asset, technicals, volatility_regime
from signals.futures_meta import FuturesMetaFeed
from signals.macro_calendar import macro_caution

ET = zoneinfo.ZoneInfo("America/New_York")


def _trend_prices(n=30, step=0.001):
    """Monotonic up-trend closes."""
    p, out = 100000.0, []
    for _ in range(n):
        p *= 1 + step
        out.append(p)
    return out


def _chop_prices(n=30, step=0.001):
    """Alternating chop closes with zero net drift."""
    p, out = 100000.0, []
    for i in range(n):
        p *= (1 + step) if i % 2 == 0 else 1 / (1 + step)
        out.append(p)
    return out


class TestVolatilityRegime:
    def test_empty_and_short_inputs_safe(self):
        assert volatility_regime.compute([])["regime"] == "unknown"
        assert volatility_regime.compute([1, 2])["regime"] == "unknown"
        assert volatility_regime.compute([0, 0, 0, 0, 0, 0])["regime"] == "unknown"

    def test_trend_reads_trendier_than_chop(self):
        trend = volatility_regime.compute(_trend_prices())
        chop = volatility_regime.compute(_chop_prices())
        assert trend["trend_score"] > chop["trend_score"]
        assert trend["trend_score"] > 0.9  # straight line ~ max efficiency
        assert chop["trend_score"] < 0.1

    def test_vol_score_monotonic_in_move_size(self):
        quiet = volatility_regime.compute(_chop_prices(step=0.0001))
        wild = volatility_regime.compute(_chop_prices(step=0.003))
        assert quiet["vol_score"] < 0.5 < wild["vol_score"]

    def test_scores_bounded(self):
        r = volatility_regime.compute(_trend_prices())
        assert 0.0 <= r["vol_score"] <= 1.0
        assert 0.0 <= r["trend_score"] <= 1.0


class TestTechnicals:
    def test_insufficient_data_zeroes(self):
        t = technicals.compute([100.0] * 5)
        assert t == {"macd_score": 0.0, "bb_score": 0.0, "mtf_score": 0.0}

    def test_uptrend_scores_positive(self):
        t = technicals.compute(_trend_prices(n=60))
        assert t["macd_score"] > 0
        assert t["bb_score"] > 0
        assert t["mtf_score"] > 0

    def test_downtrend_scores_negative(self):
        prices = list(reversed(_trend_prices(n=60)))
        t = technicals.compute(prices)
        assert t["macd_score"] < 0
        assert t["mtf_score"] < 0

    def test_flat_prices_safe(self):
        t = technicals.compute([100.0] * 60)
        assert t["macd_score"] == 0.0
        assert t["bb_score"] == 0.0  # zero stdev guard

    def test_bounded(self):
        t = technicals.compute(_trend_prices(n=60, step=0.05))
        for v in t.values():
            assert -1.0 <= v <= 1.0

    def test_bad_values_filtered(self):
        t = technicals.compute([None, 0, -5] + _trend_prices(n=60))
        assert -1.0 <= t["macd_score"] <= 1.0


class _FakeFeed:
    def __init__(self, sigs):
        self._sigs = sigs

    def get_signals(self, sym):
        return self._sigs.get(sym, {"prices": [], "latest": 0})


class TestCrossAsset:
    def test_none_feed(self):
        assert cross_asset.compute(None) == {"xasset_score": 0.0}

    def test_majors_up_positive(self):
        feed = _FakeFeed({
            "eth": {"prices": [3000.0, 3010.0], "latest": 3012.0, "stale": False},
            "sol": {"prices": [150.0, 150.6], "latest": 150.7, "stale": False},
        })
        assert cross_asset.compute(feed)["xasset_score"] > 0

    def test_stale_peers_read_zero(self):
        feed = _FakeFeed({
            "eth": {"prices": [3000.0, 3100.0], "latest": 3100.0, "stale": True},
            "sol": {"prices": [], "latest": 0, "stale": False},
        })
        assert cross_asset.compute(feed)["xasset_score"] == 0.0


class TestMacroCalendar:
    def test_peak_at_cpi_slot(self):
        now = datetime.datetime(2026, 7, 17, 8, 30, tzinfo=ET)  # Friday
        assert macro_caution(now) > 0.99

    def test_decays_off_slot(self):
        near = macro_caution(datetime.datetime(2026, 7, 17, 8, 34, tzinfo=ET))
        far = macro_caution(datetime.datetime(2026, 7, 17, 11, 0, tzinfo=ET))
        assert 0.0 < near < 1.0
        assert far < 0.01

    def test_weekend_zero(self):
        now = datetime.datetime(2026, 7, 18, 8, 30, tzinfo=ET)  # Saturday
        assert macro_caution(now) == 0.0

    def test_fomc_slot(self):
        assert macro_caution(datetime.datetime(2026, 7, 15, 14, 0, tzinfo=ET)) > 0.99


class TestFuturesMetaFeed:
    def test_unfetched_snapshot_is_neutral_and_stale(self):
        feed = FuturesMetaFeed()
        s = feed.get_signals()
        assert s["stale"] is True
        assert s["funding"] == s["oi_delta"] == s["taker_delta"] == 0.0

    def test_stale_snapshot_reports_zeros(self):
        feed = FuturesMetaFeed()
        feed._snapshot = {"funding": 0.5, "oi_delta": 0.2, "taker_delta": 0.1}
        feed._snapshot_ts = 0.0  # epoch: ancient
        s = feed.get_signals()
        assert s["stale"] is True
        assert s["funding"] == 0.0

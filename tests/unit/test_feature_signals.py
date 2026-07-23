"""Tests for the expanded signal suite (multiscale / microstructure / flow /
session / regime) and the expanded validation metrics (IC, slippage-adjusted
net edge, decay, regime splits, ranking).

Every signal module is pure and deterministic: same inputs -> same outputs,
bounded values, graceful zeros on empty/short inputs.
"""

import datetime
import math

import pytest

from signals import flow, microstructure, multiscale, regime, session_features
from tools.signal_validation import (
    Sample, decay_analysis, information_coefficient, net_edge,
    rank_signals, regime_split,
)


def _flat_fee(shares, price):
    return 0.0


# ---------------------------------------------------------------------------
# multiscale
# ---------------------------------------------------------------------------

class TestMultiscale:
    def test_deterministic(self):
        prices = [100.0 + i * 0.05 for i in range(40)]
        assert multiscale.compute(prices) == multiscale.compute(prices)

    def test_uptrend_positive_momentum_all_horizons(self):
        prices = [100.0 * (1.001 ** i) for i in range(40)]
        out = multiscale.compute(prices)
        for h in multiscale.MOM_HORIZONS:
            assert 0.0 < out[f"ms_mom_{h}m"] < 1.0

    def test_downtrend_negative_momentum(self):
        prices = [100.0 * (0.999 ** i) for i in range(40)]
        out = multiscale.compute(prices)
        assert out["ms_mom_5m"] < 0.0

    def test_short_input_zeros(self):
        out = multiscale.compute([100.0])
        assert out["ms_mom_1m"] == 0.0
        assert out["ms_rvol_5m"] == 0.0
        assert out["ms_vol_ratio"] == 0.0

    def test_vol_ratio_sign(self):
        # Calm for 30 candles then violent recently -> expansion (positive).
        calm = [100.0 + 0.001 * (i % 2) for i in range(30)]
        wild = [100.0 + (2.0 if i % 2 else -2.0) for i in range(8)]
        assert multiscale.compute(calm + wild)["ms_vol_ratio"] > 0.0

    def test_realized_vol_scales_with_amplitude(self):
        small = [100.0 + 0.01 * (i % 2) for i in range(31)]
        big = [100.0 + 1.0 * (i % 2) for i in range(31)]
        assert (multiscale.realized_vol(big, 30)
                > multiscale.realized_vol(small, 30))


# ---------------------------------------------------------------------------
# microstructure
# ---------------------------------------------------------------------------

def _book(bids, asks, valid=True):
    return {"valid": valid, "bids": bids, "asks": asks}


class TestMicrostructure:
    def test_invalid_book_zeroes(self):
        out = microstructure.compute(_book([], [], valid=False))
        assert out["micro_obi_w"] == 0.0
        assert out["micro_depth"] == 0.0

    def test_bid_heavy_book_positive(self):
        book = _book([(0.50, 500), (0.49, 400)], [(0.52, 50), (0.53, 40)])
        assert microstructure.weighted_imbalance(book) > 0.5

    def test_distance_weighting_discounts_far_walls(self):
        # Identical top-of-book; the big ask wall sits near the touch in one
        # book and far from it in the other. Same total sizes, same mid —
        # only the wall's distance differs, and the far wall must count less.
        near_wall = _book([(0.50, 100)], [(0.51, 50), (0.53, 500)])
        far_wall = _book([(0.50, 100)], [(0.51, 50), (0.60, 500)])
        assert (microstructure.weighted_imbalance(far_wall)
                > microstructure.weighted_imbalance(near_wall))

    def test_spread_and_score(self):
        tight = _book([(0.50, 10)], [(0.51, 10)])
        wide = _book([(0.45, 10)], [(0.55, 10)])
        assert microstructure.spread_pct(wide) > microstructure.spread_pct(tight)
        t = microstructure.compute(tight)
        w = microstructure.compute(wide)
        assert t["micro_spread_score"] > w["micro_spread_score"]

    def test_cross_book_pressure_sign_and_bounds(self):
        yes = _book([(0.50, 900)], [(0.52, 100)])
        no = _book([(0.48, 100)], [(0.50, 100)])
        v = microstructure.cross_book_pressure(yes, no)
        assert 0.0 < v <= 1.0

    def test_deterministic(self):
        yes = _book([(0.50, 100), (0.48, 50)], [(0.52, 80)])
        no = _book([(0.47, 60)], [(0.50, 90)])
        assert (microstructure.compute(yes, no)
                == microstructure.compute(yes, no))


# ---------------------------------------------------------------------------
# flow
# ---------------------------------------------------------------------------

def _trade(side, outcome, size, ts):
    return {"side": side, "outcome": outcome, "size": size, "timestamp": ts}


class TestFlow:
    NOW = 1_000_000.0

    def test_empty_tape_zero(self):
        out = flow.compute([], self.NOW)
        assert out == {"flow_cvd_decay": 0.0, "flow_whale": 0.0,
                       "flow_rate": 0.0}

    def test_buy_up_positive(self):
        trades = [_trade("BUY", "Up", 300, self.NOW - 5)]
        assert flow.decayed_cvd(trades, self.NOW) > 0.0

    def test_sell_up_and_buy_down_negative(self):
        trades = [_trade("SELL", "Up", 200, self.NOW - 5),
                  _trade("BUY", "Down", 200, self.NOW - 5)]
        assert flow.decayed_cvd(trades, self.NOW) < 0.0

    def test_recent_outweighs_old(self):
        trades = [_trade("BUY", "Up", 300, self.NOW - 2),
                  _trade("SELL", "Up", 300, self.NOW - 290)]
        # Equal sizes, but the buy is fresh and the sell is ~5 half-lives old.
        assert flow.decayed_cvd(trades, self.NOW) > 0.0

    def test_volume_floor_damps_thin_tape(self):
        thin = [_trade("BUY", "Up", 20, self.NOW - 1)]
        assert abs(flow.decayed_cvd(thin, self.NOW)) < 0.2

    def test_whale_ignores_small_prints(self):
        trades = [_trade("SELL", "Up", 10, self.NOW - 1) for _ in range(20)]
        trades.append(_trade("BUY", "Up", 400, self.NOW - 1))
        assert flow.whale_delta(trades, self.NOW) > 0.0

    def test_trade_rate_bounds(self):
        quiet = [_trade("BUY", "Up", 5, self.NOW - 10)]
        busy = [_trade("BUY", "Up", 200, self.NOW - i) for i in range(10)]
        assert 0.0 <= flow.trade_rate(quiet, self.NOW) < flow.trade_rate(
            busy, self.NOW) <= 1.0

    def test_deterministic(self):
        trades = [_trade("BUY", "Up", 120, self.NOW - 30),
                  _trade("SELL", "Down", 80, self.NOW - 90)]
        assert flow.compute(trades, self.NOW) == flow.compute(trades, self.NOW)


# ---------------------------------------------------------------------------
# session features
# ---------------------------------------------------------------------------

class TestSessionFeatures:
    def test_cyclical_encodings_on_unit_circle(self):
        now = datetime.datetime(2026, 7, 22, 15, 30,
                                tzinfo=datetime.timezone.utc)
        out = session_features.compute(now)
        assert math.isclose(out["sess_tod_sin"] ** 2 + out["sess_tod_cos"] ** 2,
                            1.0, abs_tol=1e-9)
        assert math.isclose(out["sess_dow_sin"] ** 2 + out["sess_dow_cos"] ** 2,
                            1.0, abs_tol=1e-9)

    def test_midnight_continuity(self):
        before = session_features.compute(datetime.datetime(
            2026, 7, 22, 23, 59, 30, tzinfo=datetime.timezone.utc))
        after = session_features.compute(datetime.datetime(
            2026, 7, 23, 0, 0, 30, tzinfo=datetime.timezone.utc))
        assert abs(before["sess_tod_sin"] - after["sess_tod_sin"]) < 0.01

    def test_nyse_open_proximity_peaks(self):
        # 09:30 ET on a Wednesday == 13:30 UTC in July (EDT).
        at_open = datetime.datetime(2026, 7, 22, 13, 30,
                                    tzinfo=datetime.timezone.utc)
        off_hours = datetime.datetime(2026, 7, 22, 6, 0,
                                      tzinfo=datetime.timezone.utc)
        assert session_features.compute(at_open)["sess_nyse_prox"] > 0.95
        assert session_features.compute(off_hours)["sess_nyse_prox"] < 0.05

    def test_weekend_flag(self):
        sat = datetime.datetime(2026, 7, 25, 15, 0,
                                tzinfo=datetime.timezone.utc)
        out = session_features.compute(sat)
        assert out["sess_weekend"] == 1.0
        assert out["sess_nyse_prox"] == 0.0

    def test_session_labels(self):
        assert session_features.session_label(datetime.datetime(
            2026, 7, 22, 3, 0, tzinfo=datetime.timezone.utc)) == "asia"
        assert session_features.session_label(datetime.datetime(
            2026, 7, 22, 15, 0, tzinfo=datetime.timezone.utc)) == "us"


# ---------------------------------------------------------------------------
# regime
# ---------------------------------------------------------------------------

class TestRegime:
    def test_straight_trend_reads_trending(self):
        prices = [100.0 + i * 0.2 for i in range(40)]
        out = regime.compute(prices)
        assert out["regime_trend"] > 0.8
        assert out["regime_chop"] < 0.4

    def test_pure_chop_reads_choppy(self):
        prices = [100.0 + (0.5 if i % 2 else -0.5) for i in range(40)]
        out = regime.compute(prices)
        assert out["regime_trend"] < 0.2
        assert out["regime_chop"] > 0.6

    def test_short_input_zeros(self):
        out = regime.compute([100.0, 100.1])
        assert out["regime_trend"] == 0.0
        assert out["regime_chop"] == 0.0

    def test_bounds(self):
        prices = [100.0 + math.sin(i / 3.0) for i in range(60)]
        out = regime.compute(prices)
        for v in out.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# validation metrics
# ---------------------------------------------------------------------------

def _mk_sample(sig, yes_won, pm_yes=0.5, seq=0, extra=None):
    signals = {"x": sig}
    if extra:
        signals.update(extra)
    return Sample(market_id=f"m{seq}", time_remaining=120.0, btc_now=100.0,
                  strike=100.0, yes_won=yes_won, signals=signals,
                  pm_yes=pm_yes, market_seq=seq)


class TestValidationMetrics:
    def test_ic_positive_for_predictive_signal(self):
        samples = ([_mk_sample(0.8, True, seq=i) for i in range(30)]
                   + [_mk_sample(-0.8, False, seq=i) for i in range(30)])
        ic = information_coefficient(samples, "x")["ic"]
        assert ic is not None and ic > 0.9

    def test_ic_zero_for_noise(self):
        samples = ([_mk_sample(0.5, True, seq=i) for i in range(20)]
                   + [_mk_sample(0.5, False, seq=i) for i in range(20)])
        assert information_coefficient(samples, "x")["ic"] is None

    def test_slippage_reduces_ev(self):
        samples = [_mk_sample(0.5, True, pm_yes=0.5, seq=i) for i in range(20)]
        rule = lambda s: "yes"
        ev0 = net_edge(samples, rule, _flat_fee)["ev_per_share"]
        ev1 = net_edge(samples, rule, _flat_fee,
                       slippage=0.02)["ev_per_share"]
        assert ev1 == pytest.approx(ev0 - 0.02)

    def test_decay_analysis_buckets_by_recency(self):
        # Signal predictive in old markets (seq >= 20), noise in recent ones.
        samples = []
        for seq in range(30):
            predictive = seq >= 20
            for j in range(4):
                won = (j % 2 == 0)
                sig = (0.5 if won else -0.5) if predictive else 0.5
                samples.append(_mk_sample(sig, won, seq=seq))
        buckets = decay_analysis(samples, "x")
        assert buckets[0]["bucket"] == "recent"
        assert buckets[-1]["bucket"] == "oldest"
        assert buckets[-1]["follow_winrate"] > buckets[0]["follow_winrate"]

    def test_regime_split_isolates_where_signal_works(self):
        samples = []
        for i in range(90):
            trend = i / 90.0
            predictive = trend > 0.67          # signal only works when trending
            won = (i % 2 == 0)
            sig = (0.5 if won else -0.5) if predictive else 0.5
            samples.append(_mk_sample(
                sig, won, pm_yes=0.5, seq=i, extra={"trend": trend}))
        buckets = regime_split(samples, "x", "trend", _flat_fee)
        by_label = {b["regime"]: b for b in buckets}
        assert by_label["trend:high"]["follow_winrate"] > 0.9
        assert by_label["trend:low"]["follow_winrate"] == pytest.approx(
            0.5, abs=0.15)

    def test_rank_signals_orders_by_slip_adjusted_ev(self):
        samples = []
        for i in range(60):
            won = (i % 2 == 0)
            samples.append(_mk_sample(
                (0.5 if won else -0.5), won, pm_yes=0.5, seq=i % 6,
                extra={"noise": 0.3 if (i % 3) else -0.3}))
        rows = rank_signals(samples, ["noise", "x"], _flat_fee,
                            slippage=0.005)
        assert rows[0]["signal"] == "x"
        assert rows[0]["ev_slip"] > (rows[1]["ev_slip"] or -1.0)

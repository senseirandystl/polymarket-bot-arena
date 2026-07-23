"""Unit tests for the backtest package (synthetic data, no network)."""

import pytest

import config
import learning
from bots import base_bot
from bots.bot_momentum import MomentumBot

from backtest.books import synth_book
from backtest.broker import BacktestBroker, BacktestTrade
from backtest.data import HistoricalData, MarketRecord
from backtest.engine import run_backtest
from backtest.metrics import (max_drawdown, signal_contribution, summarize,
                              trade_stats)
from backtest.runtime import patched_runtime
from tools.lane_candidates import Series


def _trade(**kw):
    defaults = dict(bot_name="b", strategy_type="momentum", market_id="m",
                    side="yes", shares=10.0, cost=5.0, entry_price=0.5,
                    fee=0.05, confidence=0.5, entered_at=0.0,
                    time_remaining=120.0, context={})
    defaults.update(kw)
    return BacktestTrade(**defaults)


class TestBooks:
    def test_ladder_anchored_on_mid(self):
        book = synth_book(0.50)
        assert book["valid"]
        assert book["asks"][0][0] == pytest.approx(
            0.50 + config.BACKTEST_HALF_SPREAD)
        # Ascending prices, positive sizes.
        prices = [p for p, _ in book["asks"]]
        assert prices == sorted(prices)
        assert all(s > 0 for _, s in book["asks"])

    def test_invalid_mid_rejected(self):
        assert not synth_book(None)["valid"]
        assert not synth_book(0.0)["valid"]
        assert not synth_book(1.0)["valid"]


class TestBroker:
    def test_fill_resolve_win_and_loss(self):
        broker = BacktestBroker(bankroll=200.0)
        t = broker.place(bot=MomentumBot(), market_id="m1", side="yes",
                         side_mid=0.50, amount=20.0, expected_price=0.51,
                         confidence=0.5, entered_at=0.0, time_remaining=120.0,
                         context={})
        assert t is not None
        assert broker.available < 200.0
        settled = broker.resolve_market("m1", yes_won=True)
        assert settled[0].outcome == "win"
        assert settled[0].pnl == pytest.approx(
            settled[0].shares - settled[0].cost - settled[0].fee)
        assert not broker.open_trades
        # A NO trade on a YES-won market loses its full cost + fee.
        t2 = broker.place(bot=MomentumBot(), market_id="m2", side="no",
                          side_mid=0.40, amount=10.0, expected_price=0.41,
                          confidence=0.5, entered_at=0.0, time_remaining=60.0,
                          context={})
        loss = broker.resolve_market("m2", yes_won=True)[0]
        assert loss.outcome == "loss"
        assert loss.pnl == pytest.approx(-(loss.cost + loss.fee))

    def test_exposure_cap_enforced(self):
        broker = BacktestBroker(bankroll=200.0)
        cap = config.MARKET_SIDE_EXPOSURE_CAP * 200.0
        placed_cost = 0.0
        for _ in range(10):
            t = broker.place(bot=MomentumBot(), market_id="m1", side="yes",
                             side_mid=0.50, amount=15.0, expected_price=0.51,
                             confidence=0.5, entered_at=0.0,
                             time_remaining=120.0, context={})
            if t is None:
                break
            placed_cost += t.cost
        assert placed_cost <= cap + 1e-6
        assert broker.rejects.get("exposure_cap", 0) >= 1

    def test_bankroll_never_negative(self):
        broker = BacktestBroker(bankroll=5.0)
        for i in range(5):
            broker.place(bot=MomentumBot(), market_id=f"m{i}", side="yes",
                         side_mid=0.50, amount=100.0, expected_price=0.51,
                         confidence=0.5, entered_at=0.0, time_remaining=120.0,
                         context={})
        assert broker.available >= -1e-6

    def test_slippage_band_rejects_stale_expectation(self):
        broker = BacktestBroker(bankroll=200.0)
        t = broker.place(bot=MomentumBot(), market_id="m1", side="yes",
                         side_mid=0.50, amount=10.0,
                         expected_price=0.51 + config.MAX_FILL_SLIPPAGE + 0.02,
                         confidence=0.5, entered_at=0.0, time_remaining=120.0,
                         context={})
        assert t is None
        assert broker.rejects.get("slippage_band") == 1


class TestMetrics:
    def test_trade_stats_basics(self):
        trades = [
            _trade(outcome="win", pnl=4.0, entry_price=0.5),
            _trade(outcome="win", pnl=2.0, entry_price=0.6),
            _trade(outcome="loss", pnl=-3.0, entry_price=0.55),
        ]
        s = trade_stats(trades)
        assert s["n"] == 3
        assert s["win_rate"] == pytest.approx(2 / 3)
        assert s["total_pnl"] == pytest.approx(3.0)
        assert s["expectancy"] == pytest.approx(1.0)
        assert s["profit_factor"] == pytest.approx(2.0)
        assert s["breakeven_gap"] == pytest.approx(2 / 3 - 0.55)

    def test_trade_stats_empty(self):
        s = trade_stats([])
        assert s["n"] == 0 and s["win_rate"] is None

    def test_max_drawdown(self):
        curve = [(0, 100.0), (1, 120.0), (2, 90.0), (3, 130.0), (4, 100.0)]
        dd = max_drawdown(curve)
        assert dd["max_drawdown"] == pytest.approx(30.0)
        # 120 -> 90 is 30/120 = 25%, deeper in pct terms than 130 -> 100.
        assert dd["max_drawdown_pct"] == pytest.approx(30.0 / 120.0)

    def test_signal_contribution_follow_wr(self):
        samples = [
            {"drift": 0.5, "mom": 0.0, "pm_mom": 0.0, "yes_won": True},
            {"drift": 0.5, "mom": 0.0, "pm_mom": 0.0, "yes_won": True},
            {"drift": -0.5, "mom": 0.0, "pm_mom": 0.0, "yes_won": True},
        ]
        out = signal_contribution(samples, [])
        assert out["drift"]["sample_n"] == 3
        assert out["drift"]["follow_wr"] == pytest.approx(2 / 3)
        assert out["mom"]["sample_n"] == 0


class TestRuntimeIsolation:
    def test_patched_runtime_restores_hooks(self):
        broker = BacktestBroker(bankroll=123.0)
        orig = (base_bot._sizing_bankroll, base_bot._kelly_fraction,
                base_bot._lane_overrides, learning.get_learned_bias)
        with patched_runtime(broker, kelly_fraction=0.5):
            assert base_bot._sizing_bankroll("paper") == pytest.approx(123.0)
            assert base_bot._kelly_fraction() == pytest.approx(0.5)
            assert base_bot._lane_overrides() == {}
            assert learning.get_learned_bias("x", {}, 0.42) == 0.42
        assert (base_bot._sizing_bankroll, base_bot._kelly_fraction,
                base_bot._lane_overrides, learning.get_learned_bias) == orig


def _synthetic_data(n_markets=4, yes_won=True, rising=True):
    base = 1_700_000_000
    markets, opens, closes, pm = [], [], [], {}
    px = 100000.0
    step = 1.0005 if rising else 0.9995
    for i in range(n_markets):
        o = base + i * 300
        markets.append(MarketRecord(
            id=f"m{i}", question=f"mkt {i}", open_ts=o, close_ts=o + 300,
            yes_won=yes_won, up_token="tok"))
        sign = 1 if rising else -1
        pm[f"m{i}"] = [(o + s, 0.50 + sign * 0.0004 * s)
                       for s in range(0, 300, 30)]
    t = base - 3900
    while t < base + n_markets * 300:
        opens.append((t, px))
        closes.append((t + 60, px * step))
        px *= step
        t += 60
    return HistoricalData(markets=markets, btc_opens=Series(opens),
                          btc_closes=Series(closes), pm_prices=pm)


class TestEngine:
    def test_replay_trending_up_market(self):
        data = _synthetic_data(yes_won=True, rising=True)
        res = run_backtest([MomentumBot()], data, bankroll=200.0)
        assert res.markets_replayed == 4
        assert res.decisions > 0
        assert res.trades, "a strong uptrend should produce trades"
        assert all(t.side == "yes" for t in res.trades)
        assert all(t.outcome == "win" for t in res.trades)
        assert res.final_bankroll > res.initial_bankroll

    def test_one_trade_per_bot_per_market(self):
        data = _synthetic_data(yes_won=True, rising=True)
        res = run_backtest([MomentumBot()], data, bankroll=200.0)
        keys = [(t.bot_name, t.market_id) for t in res.trades]
        assert len(keys) == len(set(keys))

    def test_summary_shape(self):
        data = _synthetic_data()
        res = run_backtest([MomentumBot()], data, bankroll=200.0)
        s = summarize(res)
        for key in ("overall", "per_bot", "per_regime", "signal_contribution",
                    "max_drawdown", "skips", "config"):
            assert key in s

    def test_unsupported_bot_excluded(self):
        from bots.bot_arbitrage import ArbitrageBot
        data = _synthetic_data(n_markets=2)
        res = run_backtest([ArbitrageBot()], data, bankroll=200.0)
        assert res.trades == [] and res.decisions == 0

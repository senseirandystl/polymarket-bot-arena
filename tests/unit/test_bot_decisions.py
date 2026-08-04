"""Structured-decision contract + regime awareness + per-bot behavior.

Every bot's analyze()/make_decision now returns the strategy_decision shape
(action, side, edge, confidence, reasoning, signals, suggested_amount).
These tests pin that contract per bot, the regime conditioning added to the
directional strategies, the maker bots' inventory discipline, and the
arbitrage bot's fee-aware threshold attribution.
"""

from unittest import mock

import pytest

import polymarket_markets
from arena import market_data
from tests.conftest import make_market, make_signals

from bots.base_bot import DECISION_KEYS, strategy_decision
from bots.bot_arbitrage import ArbitrageBot
from bots.bot_btc_maker import BtcMakerBot
from bots.bot_fee_zone_maker import FeeZoneMakerBot
from bots.bot_hybrid import HybridBot
from bots.bot_late_window_maker import LateWindowMakerBot
from bots.bot_mean_rev import MeanRevBot
from bots.bot_momentum import MomentumBot
from bots.bot_phantom import PhantomBot
from bots.bot_sentiment import SentimentBot
from bots.bot_sniper import SniperBot


def _check_contract(sig):
    assert set(DECISION_KEYS) <= set(sig), (
        f"missing keys: {set(DECISION_KEYS) - set(sig)}")
    assert sig["action"] in ("buy", "hold", "skip")
    assert sig["side"] in ("yes", "no")
    assert isinstance(sig["edge"], float)
    assert 0.0 <= sig["confidence"] <= 1.0
    assert isinstance(sig["signals"], dict)
    assert isinstance(sig["reasoning"], str)


# ---------------------------------------------------------------------------
# strategy_decision builder
# ---------------------------------------------------------------------------

class TestStrategyDecisionBuilder:
    def test_fills_contract_defaults(self):
        d = strategy_decision("hold")
        _check_contract(d)
        assert d["suggested_amount"] == 0.0

    def test_clamps_confidence(self):
        assert strategy_decision("buy", confidence=1.7)["confidence"] == 1.0
        assert strategy_decision("buy", confidence=-0.2)["confidence"] == 0.0

    def test_extra_fields_pass_through(self):
        d = strategy_decision("buy", maker_bid=0.55, features={"f": 1})
        assert d["maker_bid"] == 0.55
        assert d["features"] == {"f": 1}


# ---------------------------------------------------------------------------
# Contract across the whole fleet
# ---------------------------------------------------------------------------

ALL_ANALYZE_BOTS = [
    (MomentumBot, "momentum"),
    (MeanRevBot, "meanrev"),
    (SentimentBot, "sentiment"),
    (PhantomBot, "phantom"),
    (HybridBot, "hybrid"),
    (SniperBot, "sniper"),
    (ArbitrageBot, "arbitrage"),
    (BtcMakerBot, "btc-maker"),
    (LateWindowMakerBot, "lwm"),
    (FeeZoneMakerBot, "fzm"),
]


@pytest.mark.parametrize("cls,name", ALL_ANALYZE_BOTS,
                         ids=[n for _, n in ALL_ANALYZE_BOTS])
def test_analyze_contract_neutral_tape(cls, name, arena_db):
    bot = cls(name=f"{name}-t", generation=0)
    _check_contract(bot.analyze(make_market(), make_signals()))


@pytest.mark.parametrize("cls,name", ALL_ANALYZE_BOTS,
                         ids=[n for _, n in ALL_ANALYZE_BOTS])
def test_analyze_contract_bullish_tape(cls, name, arena_db):
    bot = cls(name=f"{name}-t", generation=0)
    prices = [100_000.0 * (1.0004 ** i) for i in range(60)]
    sig = bot.analyze(
        make_market(yes_price=0.60),
        make_signals(prices=prices, latest=prices[-1] * 1.001, btc_drift=0.45,
                     vol_regime={"regime": "trending_up", "trend_score": 0.8}))
    _check_contract(sig)


# ---------------------------------------------------------------------------
# Regime conditioning
# ---------------------------------------------------------------------------

UPTREND = [100_000.0 * (1.001 ** i) for i in range(30)]


def _momo_conf(vol_regime):
    bot = MomentumBot(name="momo-t", generation=0)
    sig = bot.analyze(make_market(), make_signals(
        prices=UPTREND, latest=UPTREND[-1], vol_regime=vol_regime))
    assert sig["action"] == "buy" and sig["side"] == "yes"
    return sig["confidence"]


class TestMomentumRegime:
    def test_trending_beats_chop(self):
        trending = _momo_conf({"regime": "trending_up", "trend_score": 0.9})
        chop = _momo_conf({"regime": "choppy", "trend_score": 0.1})
        assert trending > chop

    def test_missing_regime_is_neutral(self):
        neutral = _momo_conf(None)
        # trend_score treated as 0.5 → factor exactly 1.0, same as omitting
        explicit_mid = _momo_conf({"regime": "normal", "trend_score": 0.5})
        assert neutral == pytest.approx(explicit_mid)

    def test_regime_factor_reported_in_signals(self):
        bot = MomentumBot(name="momo-t", generation=0)
        sig = bot.analyze(make_market(), make_signals(
            prices=UPTREND, latest=UPTREND[-1],
            vol_regime={"regime": "choppy", "trend_score": 0.1}))
        assert sig["signals"]["regime_factor"] < 1.0
        assert sig["signals"]["regime"] == "choppy"


def _meanrev_sig(vol_regime):
    bot = MeanRevBot(name="rev-t", generation=0)
    lookback = bot.strategy_params["lookback_candles"]
    # Sharp up-spike on flat tape → overextended UP → fade NO, backed by a
    # DOWN drift (BUG #28) and strike above the reversion mean (P0 PTB gate).
    prices = [100_000.0] * (lookback + 20) + [101_500.0]
    return bot.analyze(make_market(), make_signals(
        prices=prices, latest=prices[-1], btc_drift=-0.30,
        btc_strike=102_000.0,
        vol_regime=vol_regime))


class TestMeanRevRegime:
    def test_fires_full_confidence_when_ranging(self):
        ranging = _meanrev_sig({"regime": "quiet", "trend_score": 0.1})
        assert ranging["action"] == "buy" and ranging["side"] == "no"
        assert ranging["signals"]["regime_factor"] == pytest.approx(1.0)

    def test_damped_when_trending(self):
        ranging = _meanrev_sig({"regime": "quiet", "trend_score": 0.1})
        trending = _meanrev_sig({"regime": "trending_down", "trend_score": 0.9})
        assert trending["confidence"] < ranging["confidence"]
        damp = MeanRevBot(name="x").strategy_params["trending_conf_damp"]
        assert trending["signals"]["regime_factor"] == pytest.approx(1.0 - damp)

    def test_missing_regime_never_damps(self):
        sig = _meanrev_sig(None)
        assert sig["signals"]["regime_factor"] == pytest.approx(1.0)


class TestPhantomRegime:
    def _sig(self, vol_regime, *, drift=0.25):
        bot = PhantomBot(name="ph-t", generation=0)
        # Gentle rising tape (ATR in bounds) + latest above the recent high.
        # Drift required (2026-08 redesign: breakout only times PTB lean).
        prices = [100_000.0 * (1 + 0.0004 * i) for i in range(40)]
        return bot.analyze(
            make_market(yes_price=0.52),
            make_signals(
                prices=prices, latest=prices[-1] * 1.001,
                vol_regime=vol_regime, btc_drift=drift,
            ),
        )

    def test_breakout_long_boosted_by_trend(self):
        trending = self._sig({"regime": "trending_up", "trend_score": 0.9})
        # Chop is hard-blocked in redesign (false breakouts).
        chop = self._sig({"regime": "choppy", "trend_score": 0.1})
        assert trending["action"] == "buy" and trending["side"] == "yes"
        assert chop["action"] == "hold"
        ranging = self._sig({"regime": "quiet", "trend_score": 0.2})
        if ranging["action"] == "buy":
            assert trending["confidence"] >= ranging["confidence"] * 0.9


# ---------------------------------------------------------------------------
# Sniper
# ---------------------------------------------------------------------------

def _sniper_market():
    # YES 0.44 sits in the cheap zone [0.40, 0.48).
    return make_market(yes_price=0.44)


class TestSniper:
    def test_buys_cheap_zone_with_drift_backing(self, arena_db):
        bot = SniperBot(name="sniper-t", generation=0)
        sig = bot.make_decision(_sniper_market(), make_signals(btc_drift=0.20))
        _check_contract(sig)
        assert sig["action"] == "buy" and sig["side"] == "yes"
        assert sig["edge"] > 0
        assert sig["signals"]["drift"] == pytest.approx(0.20)

    def test_quiet_regime_raises_drift_bar(self, arena_db):
        bot = SniperBot(name="sniper-t", generation=0)
        base_min = bot.strategy_params["min_drift"]
        bump = bot.strategy_params["quiet_drift_bump"]
        drift = base_min + bump / 2.0  # clears base bar, not the quiet bar
        quiet = bot.make_decision(_sniper_market(), make_signals(
            btc_drift=drift, vol_regime={"regime": "quiet", "trend_score": 0.2}))
        normal = bot.make_decision(_sniper_market(), make_signals(
            btc_drift=drift, vol_regime={"regime": "normal", "trend_score": 0.5}))
        assert normal["action"] == "buy"
        assert quiet["action"] == "skip"

    def test_skip_carries_contract(self, arena_db):
        bot = SniperBot(name="sniper-t", generation=0)
        sig = bot.make_decision(_sniper_market(), make_signals(btc_drift=0.0))
        _check_contract(sig)
        assert sig["action"] == "skip"


# ---------------------------------------------------------------------------
# Maker bots — inventory discipline
# ---------------------------------------------------------------------------

def _lwm_inputs():
    market = make_market(yes_price=0.60, time_remaining=100)
    signals = make_signals(btc_drift=0.50)
    return market, signals


def _fzm_inputs():
    market = make_market(yes_price=0.60)
    signals = make_signals(btc_drift=0.40)
    return market, signals


class TestLateWindowMakerInventory:
    def test_buys_with_no_inventory(self, arena_db):
        bot = LateWindowMakerBot(name="lwm-t", generation=0)
        bot._inventory_usd = lambda m, s: 0.0
        market, signals = _lwm_inputs()
        sig = bot.analyze(market, signals)
        _check_contract(sig)
        assert sig["action"] == "buy" and sig["side"] == "yes"
        assert sig["edge"] > 0
        assert sig["signals"]["inventory_usd"] == 0.0

    def test_holds_at_inventory_cap(self):
        bot = LateWindowMakerBot(name="lwm-t", generation=0)
        cap = bot.strategy_params["max_inventory_usd"]
        bot._inventory_usd = lambda m, s: cap + 1.0
        market, signals = _lwm_inputs()
        sig = bot.analyze(market, signals)
        assert sig["action"] == "hold"
        assert "inventory cap" in sig["reasoning"]

    def test_clamps_size_to_headroom(self):
        bot = LateWindowMakerBot(name="lwm-t", generation=0)
        cap = bot.strategy_params["max_inventory_usd"]
        bot._inventory_usd = lambda m, s: cap - 3.0
        market, signals = _lwm_inputs()
        sig = bot.analyze(market, signals)
        assert sig["action"] == "buy"
        assert sig["suggested_amount"] <= 3.0 + 1e-9


class TestFeeZoneMakerInventory:
    def test_buys_with_no_inventory(self, arena_db):
        bot = FeeZoneMakerBot(name="fzm-t", generation=0)
        bot._inventory_usd = lambda m, s: 0.0
        market, signals = _fzm_inputs()
        sig = bot.analyze(market, signals)
        _check_contract(sig)
        assert sig["action"] == "buy" and sig["side"] == "yes"
        assert sig["edge"] > 0
        assert sig["signals"]["fee_bps"] > 0

    def test_holds_at_inventory_cap(self):
        bot = FeeZoneMakerBot(name="fzm-t", generation=0)
        cap = bot.strategy_params["max_inventory_usd"]
        bot._inventory_usd = lambda m, s: cap
        market, signals = _fzm_inputs()
        sig = bot.analyze(market, signals)
        assert sig["action"] == "hold"
        assert "inventory cap" in sig["reasoning"]

    def test_maker_quote_fields_preserved(self):
        """run_maker_section depends on the maker_* fields on every result."""
        bot = FeeZoneMakerBot(name="fzm-t", generation=0)
        bot._inventory_usd = lambda m, s: 0.0
        for signals in (make_signals(), _fzm_inputs()[1]):
            sig = bot.analyze(make_market(yes_price=0.60), signals)
            for key in ("maker_bid", "maker_ask", "maker_mid", "maker_side"):
                assert key in sig


class TestBtcMakerInventory:
    def _inputs(self):
        prices = [100_000.0] * 58 + [100_000.0, 100_300.0]
        return make_market(yes_price=0.55), make_signals(
            prices=prices, latest=prices[-1])

    def test_directional_quote_with_no_inventory(self, arena_db):
        bot = BtcMakerBot(name="mkr-t", generation=0)
        bot._inventory_usd = lambda m, s: 0.0
        market, signals = self._inputs()
        sig = bot.analyze(market, signals)
        _check_contract(sig)
        assert sig["action"] == "buy" and sig["side"] == "yes"

    def test_holds_at_inventory_cap(self):
        bot = BtcMakerBot(name="mkr-t", generation=0)
        cap = bot.strategy_params["max_inventory_usd"]
        bot._inventory_usd = lambda m, s: cap + 5.0
        market, signals = self._inputs()
        sig = bot.analyze(market, signals)
        assert sig["action"] == "hold"
        assert "inventory cap" in sig["reasoning"].lower()


# ---------------------------------------------------------------------------
# Arbitrage — fee-aware threshold + attribution
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_warm_store():
    market_data.store().prune(keep_market_id=None)
    yield
    market_data.store().prune(keep_market_id=None)


ARB_MARKET = {
    "id": "0xabc",
    "question": "BTC Up?",
    "polymarket_token_id": "YES",
    "polymarket_no_token_id": "NO",
}


def _book(price):
    return {"valid": True, "asks": [(price, 500.0)], "bids": [],
            "best_ask": price, "best_bid": None, "min_order_size": 0}


def _patched(yes_ask, no_ask):
    books = {"YES": _book(yes_ask), "NO": _book(no_ask)}
    return mock.patch.object(polymarket_markets, "get_order_book",
                             lambda tok: books[tok])


class TestArbitrageDecision:
    def test_buy_reports_fee_aware_edge(self):
        bot = ArbitrageBot()
        with _patched(0.42, 0.48):
            sig = bot.make_decision(ARB_MARKET, {})
        _check_contract(sig)
        assert sig["action"] == "buy"
        assert sig["edge"] > 0
        # Edge must be NET of both legs' fees: gross gap is 0.10/pair,
        # reported edge strictly less by the per-pair fee.
        assert sig["edge"] < 0.10
        assert sig["signals"]["fee_per_pair"] > 0
        assert sig["edge"] == pytest.approx(
            (1.0 - 0.42 - 0.48) - sig["signals"]["fee_per_pair"], abs=1e-6)

    def test_skip_when_fees_eat_the_gap(self):
        # Gross gap ~2.4c/pair but the two legs' taker fees exceed it.
        bot = ArbitrageBot()
        with _patched(0.49, 0.486):
            sig = bot.make_decision(ARB_MARKET, {})
        _check_contract(sig)
        assert sig["action"] == "skip"
        assert sig["edge"] < bot.strategy_params["min_margin"]
        assert "fees=" in sig["reasoning"]

    def test_legs_share_matched(self):
        bot = ArbitrageBot()
        with _patched(0.42, 0.48):
            sig = bot.make_decision(ARB_MARKET, {})
        legs = sig["legs"]
        assert {leg["side"] for leg in legs} == {"yes", "no"}
        assert legs[0]["shares"] == pytest.approx(legs[1]["shares"])


# ---------------------------------------------------------------------------
# Hybrid — ensemble attribution
# ---------------------------------------------------------------------------

class TestHybridAttribution:
    def test_signals_carry_weights_and_votes(self, arena_db):
        bot = HybridBot(name="hyb-t", generation=0)
        prices = [100_000.0 * (1.001 ** i) for i in range(40)]
        sig = bot.analyze(make_market(), make_signals(
            prices=prices, latest=prices[-1] * 1.001, btc_drift=0.3,
            vol_regime={"regime": "trending_up", "trend_score": 0.8}))
        _check_contract(sig)
        assert set(sig["signals"]["weights"]) == {
            "momentum", "mean_rev", "sentiment", "phantom"}
        assert sum(sig["signals"]["weights"].values()) == pytest.approx(1.0)
        if sig["action"] == "buy":
            assert sig["signals"]["votes"]  # at least one active sub

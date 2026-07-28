"""Phase 2: the four directional strategies must be genuinely distinct.

Before this change the per-strategy analyze() fired in only 6.6% of trades — all
bots traded the identical base stack, so evolution was selecting among clones.
Each strategy now expresses a distinct, frequently-firing, data-backed thesis:
  momentum      -> follow BTC trend
  mean_reversion-> fade BTC z-score extremes (opposite of momentum)
  sentiment     -> Polymarket in-market sentiment (PM price momentum + flow)
  hybrid        -> ensemble of the three
"""

from bots.bot_momentum import MomentumBot
from bots.bot_mean_rev import MeanRevBot
from bots.bot_sentiment import SentimentBot
from bots.bot_hybrid import HybridBot

RISING = [100.0 + i * 0.05 for i in range(12)]   # steady BTC uptrend


def _mkt(**over):
    base = {"current_price": 0.5, "no_price": 0.5, "time_remaining_seconds": 180}
    base.update(over)
    return base


def _sig(prices=RISING, **over):
    # Default strike above the RISING mean so a drift-backed NO fade can clear
    # the P0 mean-vs-PTB gate (mean ≤ strike) when tests inject down-drift.
    base = {"prices": prices, "latest": prices[-1], "volumes": [], "orderflow": {},
            "pm_momentum": 0.0, "obi": 0.0, "cvd": 0.0, "sentiment": {},
            "btc_strike": 100.5}
    base.update(over)
    return base


def test_momentum_fires_on_trend():
    d = MomentumBot(name="m").analyze(_mkt(), _sig())
    assert d["action"] == "buy" and d["side"] == "yes"


def test_mean_rev_fires_and_opposes_momentum():
    # Overextended up-move WITH a down-drift backing the fade -> NO
    # (opposite of momentum's YES). BUG #28: the ungated fade went 0/11 live
    # (-$55) — the fade now requires signed drift agreement, so the identity
    # is "fade the move the fundamentals don't back", not "fade everything".
    # P0: strike above mean so reversion target still supports DOWN.
    d = MeanRevBot(name="mr").analyze(_mkt(), _sig(btc_drift=-0.3, btc_strike=100.5))
    assert d["action"] == "buy" and d["side"] == "no"
    assert "strike=" in d["reasoning"] and "mean=" in d["reasoning"]
    assert "btc_now=" in d["reasoning"]


def test_mean_rev_holds_without_drift_backing():
    # Same overextension, drift flat -> no thesis (the 0/11 death class).
    d = MeanRevBot(name="mr").analyze(_mkt(), _sig())
    assert d["action"] == "hold"


def test_mean_rev_holds_when_mean_above_ptb_for_no():
    # User rule: mean reverting UP but mean still above PTB → do not bet DOWN.
    # Inject down-drift (would pass BUG #28 alone) but strike below mean.
    d = MeanRevBot(name="mr").analyze(
        _mkt(), _sig(btc_drift=-0.3, btc_strike=99.0))
    assert d["action"] == "hold"
    assert "mean above PTB" in d["reasoning"] or "PTB" in d["reasoning"]


def test_sentiment_fires_on_pm_flow():
    up = SentimentBot(name="s").analyze(_mkt(), _sig(pm_momentum=0.05, cvd=0.5))
    dn = SentimentBot(name="s").analyze(_mkt(), _sig(pm_momentum=-0.05, cvd=-0.5))
    assert up["action"] == "buy" and up["side"] == "yes"
    assert dn["action"] == "buy" and dn["side"] == "no"


def test_sentiment_holds_without_pm_or_flow():
    # No BTC-price dependence: with flat PM momentum + zero flow it stays neutral.
    d = SentimentBot(name="s").analyze(_mkt(), _sig(pm_momentum=0.0, cvd=0.0))
    assert d["action"] == "hold"


def test_momentum_and_meanrev_take_opposite_sides():
    # A BTC pop inside a DOWN window (drift negative): momentum follows the
    # pop (YES), meanrev fades it back toward the fundamentals (NO).
    m, s = _mkt(), _sig(btc_drift=-0.3, btc_strike=100.5)
    sides = {MomentumBot(name="m").analyze(m, s)["side"],
             MeanRevBot(name="mr").analyze(m, s)["side"]}
    assert sides == {"yes", "no"}   # genuinely distinct, not clones


def test_hybrid_fires_when_substrategies_lean():
    # PM flow bullish + BTC trend up -> hybrid should reach a buy, not hold.
    d = HybridBot(name="h").analyze(_mkt(), _sig(pm_momentum=0.05, cvd=0.5))
    assert d["action"] == "buy"

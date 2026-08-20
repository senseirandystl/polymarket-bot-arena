"""NO-side trading for sniper + maker bots (symmetric mirror of their YES logic).

Each bot evaluates its own strategy on BOTH the YES price and the NO price
(mirrored band + opposite momentum). Since yes+no ~= 1, at most one side's price
falls in the band, so there is never a double-quote and NO is not a mechanical
opposite of YES.
"""

from bots.bot_sniper import SniperBot
from bots.bot_late_window_maker import LateWindowMakerBot
from bots.bot_fee_zone_maker import FeeZoneMakerBot


def _market(yes, no=None, time_rem=180):
    return {
        "id": "mkt-1",
        "current_price": yes,
        "no_price": (round(1 - yes, 4)) if no is None else no,
        "polymarket_token_id": "yes-tok",
        "polymarket_no_token_id": "no-tok",
        "time_remaining_seconds": time_rem,
    }


def _sig(prices, drift=0.0):
    d_pct = 0.0
    if drift > 0:
        d_pct = 0.0008
    elif drift < 0:
        d_pct = -0.0008
    return {
        "prices": prices, "latest": prices[-1] if prices else 0.0,
        "orderflow": {}, "btc_drift": drift,
        "btc_drift_pct": d_pct,
        "btc_strike": 100000.0,
        "btc_now": 100000.0 * (1.0 + d_pct),
    }


# --- Sniper ---

def test_sniper_buys_no_on_cheap_no_zone_with_down_momentum():
    bot = SniperBot()
    m = _market(yes=0.55, no=0.45)
    s = _sig([100.5, 100.0], drift=-0.50)   # negative momentum + down drift
    d = bot.make_decision(m, s)
    assert d["action"] == "buy"
    assert d["side"] == "no"


def test_sniper_still_buys_yes_on_cheap_yes_zone_with_up_momentum():
    bot = SniperBot()
    m = _market(yes=0.45, no=0.55)
    s = _sig([100.0, 100.5], drift=0.50)    # positive momentum + up drift
    d = bot.make_decision(m, s)
    assert d["action"] == "buy"
    assert d["side"] == "yes"


def test_sniper_skips_no_when_momentum_contradicts():
    # NO would be in-zone, but BTC rising contradicts a NO bet -> skip
    bot = SniperBot()
    m = _market(yes=0.55, no=0.45)
    s = _sig([100.0, 100.5], drift=-0.50)   # positive momentum blocks NO
    d = bot.make_decision(m, s)
    assert d["action"] == "skip"


def test_sniper_skips_zone_without_drift_backing():
    # In-zone + momentum OK but drift flat: the cheap zone measured 37.5% WR
    # unbacked -> must skip.
    bot = SniperBot()
    m = _market(yes=0.45, no=0.55)
    s = _sig([100.0, 100.5], drift=0.0)
    d = bot.make_decision(m, s)
    assert d["action"] == "skip"


# --- Late-window maker ---

def test_late_window_maker_quotes_no_with_down_momentum():
    bot = LateWindowMakerBot()
    m = _market(yes=0.40, no=0.60, time_rem=100)   # no in [0.56,0.90] band
    s = _sig([101.0, 100.5, 100.0], drift=-0.5)    # down drift + down momentum
    d = bot.analyze(m, s)
    assert d["action"] == "buy"
    assert d["side"] == "no"
    assert d["maker_side"] == "no"


def test_late_window_maker_still_quotes_yes_with_up_momentum():
    bot = LateWindowMakerBot()
    m = _market(yes=0.60, no=0.40, time_rem=100)
    s = _sig([100.0, 100.5, 101.0], drift=0.5)     # up drift + up momentum
    d = bot.analyze(m, s)
    assert d["action"] == "buy"
    assert d["side"] == "yes"


# --- Fee-zone maker ---

def test_fee_zone_maker_quotes_no_when_no_price_in_zone():
    bot = FeeZoneMakerBot()
    m = _market(yes=0.40, no=0.60)                 # no in [0.56,0.86] zone
    s = _sig([100.0]*5, drift=-0.50)               # down drift backs NO
    d = bot.analyze(m, s)
    assert d["action"] == "buy"
    assert d["side"] == "no"
    assert d["maker_side"] == "no"


def test_fee_zone_maker_still_quotes_yes_when_yes_price_in_zone():
    bot = FeeZoneMakerBot()
    m = _market(yes=0.60, no=0.40)
    s = _sig([100.0]*5, drift=0.50)                # up drift backs YES
    d = bot.analyze(m, s)
    assert d["action"] == "buy"
    assert d["side"] == "yes"


def test_fee_zone_maker_holds_without_drift_backing():
    # In-zone favorite with flat drift measured barely break-even (+0.8c/sh)
    # offline -> hold until the fundamental backs the side.
    bot = FeeZoneMakerBot()
    m = _market(yes=0.60, no=0.40)
    d = bot.analyze(m, _sig([100.0]*5, drift=0.0))
    assert d["action"] == "hold"

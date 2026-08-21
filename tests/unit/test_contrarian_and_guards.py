"""BUG #28: ungated contrarian fade + guards-on-ask + one-sided slippage.

Live evidence (2026-07-18 afternoon, 40 resolved trades, 30% WR, -$58):
  * meanrev-v1 went 0/11 (-$55.30): its z-score fade fired with drift at
    0.00-0.08 on 10 of 11 trades — strat 0.30 x fade(±0.5-0.7) alone cleared
    the lean floor, making it a pure contrarian knife-catcher (the documented
    death class: "contrarian strategies lose money in 5-min markets").
  * The ask-pricing change moved the price GUARDS onto the ask: a bot bought
    YES at ask 0.41 while the mid was 0.26 — deep NO consensus the guard
    exists to block. Guards judge market INFORMATION (mid); cost judges the
    executable ask.
  * 9 fills landed >5c BELOW the decision ask (one at 0.06, seconds before
    expiry): the book had moved and the data was stale, but the slippage
    guard only rejected fills ABOVE expectation. That class ran 22% WR.

Fixes: (1) meanrev's fade only fires when signed btc_drift agrees
(min_drift) — "buy the dip in the WINNING direction" made structural, plus a
0.58 max side price (the harness's top rule: follow drift when the market
lags); (2) consensus/high-price guards key off the side's MID; (3) the
venue slippage guard is a symmetric band |fill - expected| <= MAX_FILL_SLIPPAGE.
"""

import pytest

import config
from bots.bot_mean_rev import MeanRevBot
from bots.bot_momentum import MomentumBot


def _mr():
    return MeanRevBot(name="mr-test", generation=0)


def _market(yes=0.52, no=None, tr=60, **extra):
    m = {
        "id": "m", "current_price": yes,
        "no_price": (round(1 - yes, 4)) if no is None else no,
        "polymarket_token_id": "y", "polymarket_no_token_id": "n",
        "time_remaining_seconds": tr,
    }
    m.update(extra)
    return m


def _sig(**over):
    # btc_drift_pct required by the TWAP dual drift gate (min 0.00030).
    # 0.001 = 0.1% moneyness at ~$100 BTC, well above the floor.
    base = {"prices": [100.0, 100.0], "latest": 100.0, "orderflow": {},
            "pm_momentum": 0.0, "obi": 0.0, "cvd": 0.0, "btc_drift": 0.0,
            "btc_strike": 100.0, "btc_drift_pct": 0.001}
    base.update(over)
    return base


def _dip_prices(direction="down", n=12):
    # A short-term overextension: steady tape then a sharp move.
    base = [100.0] * (n - 3)
    tail = [99.9, 99.75, 99.6] if direction == "down" else [100.1, 100.25, 100.4]
    return base + tail


# --- Fix 1: meanrev fade must be drift-backed ---

def test_fade_without_drift_holds():
    # Overextended down -> fade wants YES; drift ~0 -> NO trade thesis.
    bot = _mr()
    sig = _sig(prices=_dip_prices("down"), latest=99.6, btc_drift=0.0,
               btc_strike=99.5)
    out = bot.analyze(_market(), sig)
    assert out["action"] == "hold"


def test_fade_against_drift_holds():
    # Overextended down -> fade wants YES, but drift says DOWN: hold.
    bot = _mr()
    sig = _sig(prices=_dip_prices("down"), latest=99.6, btc_drift=-0.3,
               btc_strike=99.5)
    out = bot.analyze(_market(), sig)
    assert out["action"] == "hold"


def test_fade_with_drift_fires():
    # BTC above strike (drift +0.3) with a short-term dip: buy-the-dip YES.
    # P0: mean ≥ strike so reversion target still supports UP.
    bot = _mr()
    sig = _sig(prices=_dip_prices("down"), latest=99.6, btc_drift=0.3,
               btc_strike=99.5)
    out = bot.analyze(_market(), sig)
    assert out["action"] == "buy" and out["side"] == "yes"
    assert "strike=" in out["reasoning"] and "mean=" in out["reasoning"]


def test_fade_with_drift_fires_symmetric_no():
    # P0: mean ≤ strike so reversion target still supports DOWN.
    bot = _mr()
    sig = _sig(prices=_dip_prices("up"), latest=100.4, btc_drift=-0.3,
               btc_strike=100.5)
    out = bot.analyze(_market(), sig)
    assert out["action"] == "buy" and out["side"] == "no"


def test_fade_mean_above_ptb_blocks_no():
    # Drift would allow NO, but mean sits above Price-to-Beat → hold.
    bot = _mr()
    sig = _sig(prices=_dip_prices("up"), latest=100.4, btc_drift=-0.3,
               btc_strike=99.0)
    out = bot.analyze(_market(), sig)
    assert out["action"] == "hold"
    assert "PTB" in out["reasoning"]


def test_meanrev_max_side_price():
    # The harness's top rule is "follow drift when the side is <= 0.58c
    # (market lags)" — meanrev embodies it: strong drift but a 0.62 favorite
    # is priced-in, skip. Same signals at 0.52 trade.
    bot = _mr()
    sig = _sig(btc_drift=0.5)
    d52 = bot.make_decision(_market(yes=0.52, no=0.48), sig)
    assert d52["action"] in ("buy", "skip")
    d = bot.make_decision(_market(yes=0.62, no=0.38), sig)
    assert d["action"] == "skip"


# --- Fix 2: guards judge the MID, cost judges the ask ---

def test_consensus_guard_uses_mid_not_ask():
    # Mid 0.26 = deep NO consensus; a wide 0.41 ask must not sneak past.
    bot = _mr()
    m = _market(yes=0.26, no=0.74, yes_ask=0.41)
    d = bot.make_decision(m, _sig(btc_drift=0.5))
    assert d["action"] == "skip"
    why = (d.get("reasoning") or "").lower()
    assert (
        "onsensus" in why
        or d.get("skip_reason") in ("consensus", "no_thesis", "underdog")
    )


def test_high_price_guard_uses_mid():
    bot = MomentumBot(name="momentum-test", generation=0)
    m = _market(yes=0.75, no=0.25, yes_ask=0.70)  # mid over guard, ask under
    sig = _sig(btc_drift=0.6,
               prices=[100.0, 100.05, 100.12, 100.20, 100.30], latest=100.30)
    d = bot.make_decision(m, sig)
    assert d["action"] == "skip"


# --- Fix 3: symmetric slippage band in the paper venue ---

@pytest.fixture()
def db(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "slip_test.db")
    db_module.init_db()
    return db_module


def _mock_book(monkeypatch, asks):
    import polymarket_markets
    book = {"valid": True, "asks": list(asks), "bids": [],
            "best_ask": asks[0][0], "best_bid": None}
    monkeypatch.setattr(polymarket_markets, "get_order_book", lambda tok: book)


def _place(expected):
    from venues.paper import PaperEngine
    return PaperEngine().place(
        bot_name="momentum-v1", side="yes", amount=5.0,
        market={"id": "m1", "polymarket_token_id": "y",
                "polymarket_no_token_id": "n"},
        mode="paper", expected_price=expected)


def test_fill_far_below_expectation_rejected(db, monkeypatch):
    # Decision priced at 0.52 but the live book has collapsed to 0.06:
    # the market moved, the inputs are stale — reject (22% WR class live).
    _mock_book(monkeypatch, [(0.06, 1000)])
    res = _place(expected=0.52)
    assert not res.success
    assert "slippage" in (res.reason or "")


def test_fill_far_above_expectation_rejected(db, monkeypatch):
    _mock_book(monkeypatch, [(0.60, 1000)])
    res = _place(expected=0.52)
    assert not res.success


def test_fill_within_band_accepted(db, monkeypatch):
    _mock_book(monkeypatch, [(0.53, 1000)])
    res = _place(expected=0.52)
    assert res.success

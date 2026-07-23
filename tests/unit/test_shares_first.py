"""Tests for shares-first bet sizing in BaseBot.make_decision.

The invariant: the bot decides an exact SHARE count first, then derives the USD
spend as shares × price (never USD → shares). The returned ``suggested_amount``
must equal ``target_shares × market_price``, and ``target_shares`` must clear
Polymarket's 5-share minimum.
"""

import pytest

import config
from bots.base_bot import BaseBot


class _StubBot(BaseBot):
    def analyze(self, market, signals):
        return {"action": "buy", "side": "yes", "confidence": 1.0,
                "reasoning": "stub", "suggested_amount": 0}


@pytest.fixture()
def bot(monkeypatch):
    import learning
    import db
    monkeypatch.setattr(learning, "extract_features", lambda *a, **k: {})
    # Neutral learned bias = the strategy prior (no learning nudge).
    monkeypatch.setattr(learning, "get_learned_bias", lambda name, feats, prior: prior)
    monkeypatch.setattr(db, "get_bot_performance", lambda *a, **k: {"total_trades": 0})
    return _StubBot("t-momentum", "momentum", {})


def _decide(bot, price, signals=None):
    market = {"current_price": price, "time_remaining_seconds": 200,
              "polymarket_token_id": "up", "polymarket_no_token_id": "dn"}
    # btc_drift=0.2 (YES-favoring, matching the stub's YES thesis) keeps these
    # sizing-math tests clear of the dead-zone gate at mid-book prices — the
    # gate blocks flat-drift coin-flip trades, which is orthogonal to the
    # shares-first invariant under test here.
    sig = {"prices": [100.0, 101.0], "latest": 101.0, "obi": 1.0, "cvd": 1.0,
           "btc_drift": 0.2}
    if signals:
        sig.update(signals)
    return bot.make_decision(market, sig)


def test_amount_is_derived_from_shares(bot):
    sig = _decide(bot, 0.50)
    assert sig["action"] == "buy"
    assert "target_shares" in sig
    # USD spend is exactly shares × price (shares-first, not USD → shares).
    assert sig["suggested_amount"] == pytest.approx(
        sig["target_shares"] * 0.50, abs=1e-9
    )


def test_target_shares_clears_min(bot):
    sig = _decide(bot, 0.50)
    assert sig["target_shares"] >= config.POLYMARKET_MIN_SHARES  # >= 5


def test_amount_capped_at_max_position(bot):
    # Even with a strong edge the spend never exceeds max position.
    sig = _decide(bot, 0.50, signals={"obi": 1.0, "cvd": 1.0})
    assert sig["suggested_amount"] <= config.get_max_position() + 1e-9


def test_low_price_entry_still_clears_min_shares(bot):
    # At 40¢ (bottom of the allowed YES band), min-share floor must still hold.
    sig = _decide(bot, 0.40)
    if sig["action"] == "buy":
        assert sig["target_shares"] >= config.POLYMARKET_MIN_SHARES
        assert sig["suggested_amount"] == pytest.approx(
            sig["target_shares"] * 0.40, abs=1e-9
        )

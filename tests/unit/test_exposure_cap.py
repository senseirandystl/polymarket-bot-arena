"""Shared-pool per-(market, side) exposure cap (BUG #27, part 4).

Tandem clustering is structural: all directional bots read identical warm
lanes, so 3-6 bots pile the same side of the same market within seconds
(20 of 34 groups in the 2026-07-17 run). Per-bot Kelly doesn't know the pool
already holds correlated positions — hour 22's three 4-bot pile-ins were ~4x
effective leverage on single BTC candles.

Fix: cap the pool's total OPEN cost per (market, side) at
config.MARKET_SIDE_EXPOSURE_CAP x the gross paper pool. Later bots get the
remaining headroom (clamped) or skip. Arbitrage is exempt (its two legs are
hedged and it overrides execute()).
"""

import pytest

import config


@pytest.fixture()
def db(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "cap_test.db")
    db_module.init_db()
    return db_module


def _open_trade(db, bot, market="mkt-1", side="yes", amount=5.0):
    return db.log_trade(bot, market, side, amount, venue="polymarket",
                        mode="paper", fill_source="paper_sim")


def test_cap_config_exists():
    assert 0.0 < config.MARKET_SIDE_EXPOSURE_CAP <= 0.5


def test_open_exposure_sums_pending_only(db):
    _open_trade(db, "a", amount=4.0)
    _open_trade(db, "b", amount=6.0)
    _open_trade(db, "c", side="no", amount=9.0)           # other side
    tid = _open_trade(db, "d", amount=3.0)
    db.resolve_trade(tid, "win", 1.0)                     # resolved: excluded
    assert db.get_open_exposure("mkt-1", "yes", "paper") == pytest.approx(10.0)
    assert db.get_open_exposure("mkt-1", "no", "paper") == pytest.approx(9.0)
    assert db.get_open_exposure("mkt-2", "yes", "paper") == 0.0


def test_exposure_headroom_clamps_and_skips(db, monkeypatch):
    from bots.bot_momentum import MomentumBot
    monkeypatch.setattr(db, "get_paper_pool_gross", lambda: 100.0)
    bot = MomentumBot(name="momentum-test", generation=0)
    cap_usd = config.MARKET_SIDE_EXPOSURE_CAP * 100.0
    # Pool already holds cap - 2 on this side: a $5 request clamps to ~$2.
    _open_trade(db, "other-bot", amount=cap_usd - 2.0)
    allowed = bot._exposure_headroom("mkt-1", "yes", "paper")
    assert allowed == pytest.approx(2.0)
    # At/over the cap: no headroom.
    _open_trade(db, "third-bot", amount=2.0)
    assert bot._exposure_headroom("mkt-1", "yes", "paper") <= 0.0


def test_max_bots_per_side_blocks_new_bot(db, monkeypatch):
    """MARKET_SIDE_MAX_BOTS: fourth distinct bot gets zero headroom."""
    from bots.bot_momentum import MomentumBot
    monkeypatch.setattr(db, "get_paper_pool_gross", lambda: 1000.0)
    monkeypatch.setattr(config, "MARKET_SIDE_MAX_BOTS", 3, raising=False)
    monkeypatch.setattr(config, "EXPOSURE_CORR_AWARE", False, raising=False)
    for name in ("a", "b", "c"):
        _open_trade(db, name, amount=1.0)
    bot = MomentumBot(name="new-bot", generation=0)
    assert bot._exposure_headroom("mkt-1", "yes", "paper") == 0.0
    # Bot that already has a position can still add
    bot_a = MomentumBot(name="a", generation=0)
    assert bot_a._exposure_headroom("mkt-1", "yes", "paper") > 0


def test_corr_aware_weights_high_rho_peers(db, monkeypatch):
    """ρ≈1 peers nearly fully share the concentration budget."""
    from bots.bot_momentum import MomentumBot
    from arena import portfolio
    monkeypatch.setattr(db, "get_paper_pool_gross", lambda: 100.0)
    monkeypatch.setattr(config, "EXPOSURE_CORR_AWARE", True, raising=False)
    monkeypatch.setattr(config, "EXPOSURE_CORR_FLOOR", 0.35, raising=False)
    monkeypatch.setattr(config, "MARKET_SIDE_MAX_BOTS", 10, raising=False)
    # Peer open $5; with ρ=1 effective used ≈ $5
    _open_trade(db, "peer-bot", amount=5.0)
    bot = MomentumBot(name="momentum-test", generation=0)
    monkeypatch.setattr(
        portfolio, "load_state",
        lambda: {"correlations": {"momentum-test|peer-bot": 1.0}},
    )
    used, n = bot._effective_open_exposure("mkt-1", "yes", "paper")
    assert n == 1
    assert used == pytest.approx(5.0)
    # ρ=0.0 floors to EXPOSURE_CORR_FLOOR
    monkeypatch.setattr(
        portfolio, "load_state",
        lambda: {"correlations": {"momentum-test|peer-bot": 0.0}},
    )
    used0, _ = bot._effective_open_exposure("mkt-1", "yes", "paper")
    assert used0 == pytest.approx(5.0 * config.EXPOSURE_CORR_FLOOR)

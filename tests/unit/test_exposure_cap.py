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
    from bots.base_bot import invalidate_exposure_cache
    monkeypatch.setattr(db, "get_paper_pool_gross", lambda: 100.0)
    # Allow multi-bot for $ headroom clamp path (default max_bots=1 would
    # zero headroom as soon as any peer is open). Stub regime adapt so it
    # cannot tighten max_bots_side to 1 mid-test.
    monkeypatch.setattr(config, "MARKET_SIDE_MAX_BOTS", 5, raising=False)
    monkeypatch.setattr(config, "EXPOSURE_CORR_AWARE", False, raising=False)

    class _NoAdj:
        max_bots_side = None

    monkeypatch.setattr(
        "arena.regime_adapt.adjustments",
        lambda *a, **k: _NoAdj(),
        raising=False,
    )
    bot = MomentumBot(name="momentum-test", generation=0)
    cap_usd = config.MARKET_SIDE_EXPOSURE_CAP * 100.0
    # Pool already holds cap - 2 on this side: a $5 request clamps to ~$2.
    _open_trade(db, "other-bot", amount=cap_usd - 2.0)
    invalidate_exposure_cache()
    allowed = bot._exposure_headroom("mkt-1", "yes", "paper")
    assert allowed == pytest.approx(2.0)
    # At/over the cap: no headroom.
    _open_trade(db, "third-bot", amount=2.0)
    invalidate_exposure_cache()
    assert bot._exposure_headroom("mkt-1", "yes", "paper") <= 0.0


def test_max_bots_per_side_blocks_new_bot(db, monkeypatch):
    """MARKET_SIDE_MAX_BOTS: next distinct bot gets zero headroom at the cap."""
    from bots.bot_momentum import MomentumBot
    from bots.base_bot import invalidate_exposure_cache
    monkeypatch.setattr(db, "get_paper_pool_gross", lambda: 1000.0)
    monkeypatch.setattr(config, "MARKET_SIDE_MAX_BOTS", 1, raising=False)
    monkeypatch.setattr(config, "EXPOSURE_CORR_AWARE", False, raising=False)
    _open_trade(db, "a", amount=1.0)
    invalidate_exposure_cache()
    bot = MomentumBot(name="new-bot", generation=0)
    assert bot._exposure_headroom("mkt-1", "yes", "paper") == 0.0
    # Bot that already has a position can still add
    bot_a = MomentumBot(name="a", generation=0)
    assert bot_a._exposure_headroom("mkt-1", "yes", "paper") > 0


def test_pilein_ev_gate_blocks_second_bot(db, monkeypatch):
    """After a peer is open, a new bot with weak edge is blocked."""
    from bots.bot_momentum import MomentumBot
    from bots.base_bot import invalidate_exposure_cache
    monkeypatch.setattr(db, "get_paper_pool_gross", lambda: 1000.0)
    monkeypatch.setattr(config, "PILEIN_EV_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "PILEIN_EV_MIN_EDGE", 0.04, raising=False)
    monkeypatch.setattr(config, "PILEIN_EV_EDGE_STEP", 0.02, raising=False)
    monkeypatch.setattr(config, "PILEIN_EV_CONF_BYPASS", 0.85, raising=False)
    monkeypatch.setattr(config, "MARKET_SIDE_MAX_BOTS", 10, raising=False)
    _open_trade(db, "peer-bot", amount=5.0)
    invalidate_exposure_cache()
    bot = MomentumBot(name="momentum-test", generation=0)
    # Weak edge + mid conf → blocked
    msg = bot._pilein_ev_block(
        "mkt-1", "yes",
        {"edge": 0.02, "confidence": 0.60},
        "paper",
    )
    assert msg is not None
    assert "Pile-in EV gate" in msg
    # High conf bypass only at the configured bar (test sets 0.85)
    assert bot._pilein_ev_block(
        "mkt-1", "yes",
        {"edge": 0.02, "confidence": 0.90},
        "paper",
    ) is None
    # Strong edge clears
    assert bot._pilein_ev_block(
        "mkt-1", "yes",
        {"edge": 0.05, "confidence": 0.50},
        "paper",
    ) is None


def test_pilein_structure_conf_does_not_bypass_production_bar(db, monkeypatch):
    """quality_confidence routinely prints 0.82–0.91 on 55–62¢ lag trades.

    That is structure, not P(win). The soak's tandem doubles (unique +$24
    vs multi −$12) all cleared the old 0.85 bypass. Production bar sits
    above the 0.95 cap so ordinary structure cannot skip the extra edge.
    """
    from bots.bot_momentum import MomentumBot
    from bots.base_bot import invalidate_exposure_cache
    monkeypatch.setattr(db, "get_paper_pool_gross", lambda: 1000.0)
    monkeypatch.setattr(config, "PILEIN_EV_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "MARKET_SIDE_MAX_BOTS", 10, raising=False)
    _open_trade(db, "peer-bot", amount=5.0)
    invalidate_exposure_cache()
    bot = MomentumBot(name="momentum-test", generation=0)
    # Production bypass (≥0.96) — 0.90 structure must still need extra edge
    assert config.PILEIN_EV_CONF_BYPASS >= 0.96
    msg = bot._pilein_ev_block(
        "mkt-1", "yes",
        {"edge": 0.02, "confidence": 0.90},
        "paper",
    )
    assert msg is not None


def test_pilein_extra_peers_covers_same_tick_race(db, monkeypatch):
    """Same-tick second bot: DB may still show n_bots=0; extra_peers=1."""
    from bots.bot_momentum import MomentumBot
    monkeypatch.setattr(config, "PILEIN_EV_GATE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "PILEIN_EV_MIN_EDGE", 0.035, raising=False)
    monkeypatch.setattr(config, "PILEIN_EV_CONF_BYPASS", 0.96, raising=False)
    bot = MomentumBot(name="momentum-test", generation=0)
    # No DB peers, but trader already filled one this tick
    msg = bot._pilein_ev_block(
        "mkt-fresh", "yes",
        {"edge": 0.02, "confidence": 0.70},
        "paper",
        extra_peers=1,
    )
    assert msg is not None
    # extra_peers must not double-count a peer already in DB
    _open_trade(db, "peer-bot", market="mkt-fresh", amount=5.0)
    msg2 = bot._pilein_ev_block(
        "mkt-fresh", "yes",
        {"edge": 0.02, "confidence": 0.70},
        "paper",
        extra_peers=1,
    )
    assert msg2 is not None
    # Strong edge still clears with one in-tick peer
    assert bot._pilein_ev_block(
        "mkt-fresh", "yes",
        {"edge": 0.08, "confidence": 0.70},
        "paper",
        extra_peers=1,
    ) is None


def test_pilein_ev_gate_toggle_off(db, monkeypatch):
    from bots.bot_momentum import MomentumBot
    monkeypatch.setattr(config, "PILEIN_EV_GATE_ENABLED", False, raising=False)
    _open_trade(db, "peer-bot", amount=5.0)
    bot = MomentumBot(name="momentum-test", generation=0)
    monkeypatch.setattr(db, "get_pilein_ev_gate", lambda: False)
    assert bot._pilein_ev_block(
        "mkt-1", "yes", {"edge": 0.01, "confidence": 0.4}, "paper"
    ) is None


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


def test_execute_blocks_second_fill_same_market(db, monkeypatch):
    """Evolution reset wiped is_traded; DB open row must still block."""
    from bots.bot_momentum import MomentumBot
    monkeypatch.setattr(db, "get_bot_mode", lambda name: "paper")
    _open_trade(db, "momentum-test", market="mkt-1", amount=5.0)
    bot = MomentumBot(name="momentum-test", generation=0)
    bot.trading_mode = "paper"
    out = bot.execute(
        {"side": "yes", "suggested_amount": 3.0, "entry_price": 0.5},
        {"id": "mkt-1", "condition_id": "mkt-1"},
    )
    assert out.get("success") is False
    assert out.get("reason") == "already_in_market"

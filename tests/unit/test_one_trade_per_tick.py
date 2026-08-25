"""One directional trade per evaluation + window lock state."""

import config
from arena.state import SharedArenaState


def test_config_one_trade_defaults():
    # Paper-eval tandem pack: cluster open so every bot can fill.
    assert config.ONE_TRADE_PER_TICK is False
    assert config.MARKET_SIDE_MAX_BOTS == 0
    assert config.PILEIN_EV_GATE_ENABLED is False
    assert config.HYBRID_YIELD_ENABLED is False
    # Structure-confidence must not skip the extra-edge bar if pile-in is restored.
    assert config.PILEIN_EV_CONF_BYPASS >= 0.96
    assert config.DIRECTIONAL_WINDOW_LOCK is False
    assert "arbitrage" in config.ONE_TRADE_PER_TICK_EXEMPT
    assert "sweeper" in config.ONE_TRADE_PER_TICK_EXEMPT


def test_directional_window_lock_state():
    st = SharedArenaState()
    assert not st.is_directional_locked("mkt-a")
    st.mark_directional_lock("mkt-a")
    assert st.is_directional_locked("mkt-a")
    assert not st.is_directional_locked("mkt-b")
    st.reset()
    assert not st.is_directional_locked("mkt-a")


def test_reset_rehydrates_recent_fills(tmp_path, monkeypatch):
    """GA reset must not wipe (bot, market) keys for fills already on the book."""
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "rehyd.db")
    db_module.init_db()
    db_module.log_trade(
        "hybrid-v1", "mkt-live", "yes", 3.0,
        venue="polymarket", mode="paper", fill_source="paper_sim",
    )
    st = SharedArenaState()
    st.mark_traded(("other", "old"))
    st.reset()
    assert st.is_traded(("hybrid-v1", "mkt-live"))
    assert not st.is_traded(("other", "old"))


def test_discovery_interval_near_rollover():
    from arena.discovery import discovery_interval
    assert discovery_interval(tr=10, age=290) == 2.0
    assert discovery_interval(tr=200, age=5) == 2.0
    assert discovery_interval(tr=120, age=180) >= 15


def test_buy_score_is_dollar_ev_not_confidence():
    from arena.trader import directional_buy_score
    fat_conf = {"edge": 0.03, "confidence": 0.95, "entry_price": 0.50}
    thin_conf = {"edge": 0.08, "confidence": 0.10, "entry_price": 0.50}
    assert directional_buy_score(thin_conf) > directional_buy_score(fat_conf)


def test_window_lock_db_toggle(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "wl.db")
    db_module.init_db()
    # Default follows config (False)
    assert db_module.get_directional_window_lock() is False
    db_module.set_directional_window_lock(True)
    assert db_module.get_directional_window_lock() is True
    db_module.set_directional_window_lock(False)
    assert db_module.get_directional_window_lock() is False

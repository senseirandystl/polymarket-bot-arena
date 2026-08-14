"""One directional trade per evaluation + window lock state."""

import config
from arena.state import SharedArenaState


def test_config_one_trade_defaults():
    assert config.ONE_TRADE_PER_TICK is True
    assert config.MARKET_SIDE_MAX_BOTS == 1
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

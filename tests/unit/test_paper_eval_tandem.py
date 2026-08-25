"""Paper-eval tandem pack: open cluster, EV floor, no 1-bot coerce."""

import config
from arena.trader import directional_buy_score


def test_paper_eval_tandem_defaults():
    assert config.MARKET_SIDE_MAX_BOTS == 0
    assert config.MARKET_SIDE_MAX_BOTS_BAD_REGIME == 0
    assert config.MARKET_SIDE_MAX_BOTS_CHOP == 0
    assert config.ONE_TRADE_PER_TICK is False
    assert config.PILEIN_EV_GATE_ENABLED is False
    assert config.HYBRID_YIELD_ENABLED is False
    assert config.PORTFOLIO_NEG_EXP_MAX_WEIGHT == 0.10
    assert config.MARKET_SIDE_EXPOSURE_CAP == 0.30


def test_regime_adapt_zero_bad_regime_does_not_inject(monkeypatch):
    from arena.regime_adapt import adjustments
    monkeypatch.setattr(config, "MARKET_SIDE_MAX_BOTS_BAD_REGIME", 0, raising=False)
    monkeypatch.setattr(config, "MARKET_SIDE_MAX_BOTS_CHOP", 0, raising=False)
    from unittest import mock
    with mock.patch("arena.regime_adapt._refresh_cache"), \
         mock.patch("arena.regime_adapt._cache", (0.0, {}, {}, {}, {})), \
         mock.patch("arena.regime_stats.strategy_regime_cell",
                    return_value={"n": 20, "wins": 6, "wr": 0.30, "pnl": -10.0,
                                  "fast_n": 8, "fast_wins": 2, "fast_wr": 0.25,
                                  "fast_pnl": -4.0}), \
         mock.patch("arena.regime_stats.regime_cell",
                    return_value={"n": 20, "wins": 6, "wr": 0.30, "pnl": -10.0,
                                  "fast_n": 8, "fast_wins": 2, "fast_wr": 0.25,
                                  "fast_pnl": -4.0}), \
         mock.patch("arena.regime_stats.side_regime_cell",
                    return_value={"n": 0, "wins": 0, "pnl": 0, "wr": None,
                                  "fast_n": 0, "fast_wr": None, "fast_pnl": 0}), \
         mock.patch("arena.regime_stats.strategy_side_regime_cell",
                    return_value={"n": 0, "wins": 0, "pnl": 0, "wr": None,
                                  "fast_n": 0, "fast_wr": None, "fast_pnl": 0}), \
         mock.patch("arena.regime_stats.is_toxic_cell", return_value=True), \
         mock.patch("arena.regime_stats.effective_wr", return_value=0.30):
        a = adjustments("high_vol_trend", "momentum")
    assert a.max_bots_side is None


def test_db_tandem_getters_follow_config(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "tandem.db")
    db_module.init_db()
    assert db_module.get_one_trade_per_tick() is False
    assert db_module.get_hybrid_yield() is False
    assert db_module.get_market_side_max_bots() == 0
    db_module.set_one_trade_per_tick(True)
    db_module.set_hybrid_yield(True)
    db_module.set_market_side_max_bots(1)
    assert db_module.get_one_trade_per_tick() is True
    assert db_module.get_hybrid_yield() is True
    assert db_module.get_market_side_max_bots() == 1


def test_buy_score_still_dollar_ev():
    fat_conf = {"edge": 0.03, "confidence": 0.95, "entry_price": 0.50}
    thin_conf = {"edge": 0.08, "confidence": 0.10, "entry_price": 0.50}
    assert directional_buy_score(thin_conf) > directional_buy_score(fat_conf)

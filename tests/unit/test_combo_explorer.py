"""Signal Lab combo / foundational-rule explorer."""

import config
import db
import polymarket_fills
from arena import combo_explorer as ce
from arena.live_scorecard import unique_market_rows


def _insert(conn, **kw):
    cols = {
        "bot_name": "hybrid-v1",
        "strategy_type": "hybrid",
        "market_id": "m1",
        "action": "skip",
        "side": "yes",
        "skip_reason": "weak_lean",
        "edge": 0.04,
        "confidence": 0.2,
        "entry_price": 0.48,
        "drift": 0.02,
        "mom": 0.25,
        "strat": 0.05,
        "fut": 0.0,
        "tech": 0.22,
        "xasset": 0.20,
        "market_up": 1,
        "would_win": 1,
        "hyp_pnl": 0.16,
        "regime": "high_vol_chop",
    }
    cols.update(kw)
    conn.execute(
        """INSERT INTO decision_events (
               bot_name, strategy_type, market_id, action, side, skip_reason,
               edge, confidence, entry_price, drift, mom, strat, fut, tech,
               xasset, market_up, would_win, hyp_pnl, regime
           ) VALUES (
               :bot_name, :strategy_type, :market_id, :action, :side,
               :skip_reason, :edge, :confidence, :entry_price, :drift, :mom,
               :strat, :fut, :tech, :xasset, :market_up, :would_win, :hyp_pnl,
               :regime
           )""",
        cols,
    )


def test_official_crypto_fee_table_matches_docs():
    """Polymarket crypto table: fee = 0.07 · C · p · (1-p). Makers = 0."""
    assert config.POLYMARKET_TAKER_FEE_RATE == 0.07
    # Official 100-share crypto table (docs/trading/fees), 2-decimal display.
    table = {
        0.01: 0.07, 0.10: 0.63, 0.25: 1.31, 0.50: 1.75,
        0.75: 1.31, 0.90: 0.63, 0.99: 0.07,
    }
    for px, listed in table.items():
        raw = polymarket_fills.taker_fee(100.0, px)
        assert raw == 0.07 * 100.0 * px * (1.0 - px)
        assert round(raw, 2) == listed
    assert polymarket_fills.maker_fee(100.0, 0.50) == 0.0
    assert polymarket_fills.trading_fee(10.0, 0.50, is_maker=True) == 0.0


def test_combo_scores_agreement_on_unique_markets(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "combo.db")
    db.init_db()
    with db.get_conn() as conn:
        # 25 cheap windows: mom+tech agree and are correct (UP).
        for i in range(25):
            _insert(
                conn, market_id=f"up{i}", drift=0.02, mom=0.30, tech=0.28,
                xasset=0.01, entry_price=0.47, market_up=1, side="yes",
            )
        # 5 cheap windows: mom+tech agree and are WRONG.
        for i in range(5):
            _insert(
                conn, market_id=f"dn{i}", drift=0.01, mom=0.30, tech=0.28,
                xasset=0.00, entry_price=0.47, market_up=0, side="yes",
                would_win=0,
            )
        # Expensive 99c rows must not earn a combo (sweeper book).
        for i in range(12):
            _insert(
                conn, market_id=f"exp{i}", drift=0.90, mom=0.40, tech=0.40,
                xasset=0.40, entry_price=0.99, market_up=1, side="yes",
            )
        rows = unique_market_rows(conn)
        report = ce.build_combo_report(rows)
    mt = report["combos"]["mom+tech"]
    assert mt["cheap_markets"] == 30
    assert mt["cheap_accuracy"] == 25 / 30
    assert mt["verdict"] == "earned"
    assert mt["net_edge"] is not None and mt["net_edge"] > 0
    # 99c drift-heavy agreement must not be marked earned on cheap bar.
    dtx = report["combos"]["drift+tech+xasset"]
    assert dtx["cheap_markets"] == 0 or dtx["verdict"] != "earned"


def test_informed_rows_skip_bare_last_tick(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "inf.db")
    db.init_db()
    with db.get_conn() as conn:
        _insert(conn, market_id="m1", drift=0.20, mom=0.20, tech=0.20,
                skip_reason="weak_lean")
        _insert(conn, market_id="m1", drift=None, mom=None, tech=None,
                xasset=None, strat=None, fut=None,
                skip_reason="skip", entry_price=None)
        rows = ce.informed_market_rows(conn)
    assert len(rows) == 1
    assert rows[0]["skip_reason"] == "weak_lean"
    assert float(rows[0]["drift"]) == 0.20


def test_named_rule_drift_flat_confirm(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "flat.db")
    db.init_db()
    with db.get_conn() as conn:
        for i in range(22):
            _insert(
                conn, market_id=f"f{i}", drift=0.02, mom=0.25, tech=0.22,
                xasset=0.20, entry_price=0.40, market_up=1, side="yes",
            )
        report = ce.build_combo_report(unique_market_rows(conn))
    rule = report["rules"]["drift_flat_confirm"]
    assert rule["markets"] == 22
    assert rule["accuracy"] == 1.0
    assert rule["verdict"] == "earned"


def test_try_confirm_requires_earned_cheap_non_drift(monkeypatch):
    report = {
        "combos": {
            "mom+tech": {
                "lanes": ["mom", "tech"],
                "verdict": "earned",
                "accuracy": 0.70,
                "net_edge": 0.04,
                "cheap_markets": 30,
                "bypass_dual_gate": True,
            }
        },
        "rules": {},
        "earned": [{
            "name": "mom+tech",
            "lanes": ["mom", "tech"],
            "bypass_dual_gate": True,
            "accuracy": 0.70,
            "net_edge": 0.04,
        }],
    }
    monkeypatch.setattr(ce, "load_report", lambda: report)
    monkeypatch.setattr(config, "COMBO_CONFIRM_APPLY", True)
    ok = ce.try_confirm(
        {"mom": 0.30, "tech": 0.25, "xasset": 0.0, "drift": 0.02},
        yes_mid=0.40, no_mid=0.60, yes_ask=0.41, no_ask=0.61,
    )
    assert ok is not None
    assert ok["side"] == "yes"
    assert ok["bypass_dual_gate"] is True
    assert ok["p_model"] > 0.55

    # Expensive mid — refuse (do not mint 99c "edge").
    assert ce.try_confirm(
        {"mom": 0.30, "tech": 0.25, "drift": 0.02},
        yes_mid=0.99, no_mid=0.01,
    ) is None

    # Disagreement — refuse.
    assert ce.try_confirm(
        {"mom": 0.30, "tech": -0.25, "drift": 0.02},
        yes_mid=0.40, no_mid=0.60,
    ) is None


def test_try_confirm_does_not_fire_when_collecting(monkeypatch):
    monkeypatch.setattr(ce, "load_report", lambda: {"combos": {}, "earned": []})
    monkeypatch.setattr(config, "COMBO_CONFIRM_APPLY", True)
    assert ce.try_confirm(
        {"mom": 0.40, "tech": 0.40}, yes_mid=0.40, no_mid=0.60,
    ) is None


def _earned_mom_tech():
    return {
        "earned": [{
            "name": "mom+tech",
            "lanes": ["mom", "tech"],
            "bypass_dual_gate": True,
            "accuracy": 0.70,
            "net_edge": 0.04,
        }],
        "combos": {},
        "rules": {},
    }


def _stub_combo(_signals, **_kw):
    return {
        "name": "mom+tech",
        "lanes": ["mom", "tech"],
        "side": "yes",
        "strength": 0.30,
        "lean": 0.28,
        "p_model": 0.78,
        "bypass_dual_gate": True,
        "accuracy": 0.70,
        "net_edge": 0.04,
    }


def test_combo_confirm_can_buy_cheap_when_drift_flat(monkeypatch):
    """Earned mom+tech may trade a cheap lag without loosening dual-gate."""
    import db
    from bots.bot_momentum import MomentumBot
    monkeypatch.setattr(ce, "try_confirm", _stub_combo)
    monkeypatch.setattr(config, "COMBO_CONFIRM_APPLY", True)
    monkeypatch.setattr(db, "get_paper_available", lambda: 200.0)
    monkeypatch.setattr(db, "get_kelly_fraction", lambda: 0.25)
    bot = MomentumBot(name="mom-combo")
    bot._perf_cache = (9e12, 0)
    market = {
        "id": "m", "current_price": 0.59, "no_price": 0.41,
        "yes_ask": 0.60, "no_ask": 0.42, "time_remaining_seconds": 90,
    }
    signals = {
        "prices": [65000.0, 65080.0], "latest": 65080.0,
        "orderflow": {},
        "btc_drift": 0.04,
        "btc_drift_pct": 0.00008,
        "btc_strike": 65000.0, "btc_now": 65005.0,
        "tech_mtf": 0.35,
        "xasset": 0.05,
        "btc_momentum": 0.30,
    }
    d = bot.make_decision(market, signals)
    assert d["action"] == "buy"
    assert d["side"] == "yes"
    assert "combo(" in d["reasoning"]


def test_combo_confirm_does_not_bypass_dead_zone(monkeypatch):
    import db
    from bots.bot_momentum import MomentumBot
    monkeypatch.setattr(ce, "try_confirm", _stub_combo)
    monkeypatch.setattr(config, "COMBO_CONFIRM_APPLY", True)
    monkeypatch.setattr(db, "get_paper_available", lambda: 200.0)
    monkeypatch.setattr(db, "get_kelly_fraction", lambda: 0.25)
    bot = MomentumBot(name="mom-combo-dz")
    bot._perf_cache = (9e12, 0)
    market = {
        "id": "m", "current_price": 0.50, "no_price": 0.50,
        "yes_ask": 0.51, "no_ask": 0.51, "time_remaining_seconds": 90,
    }
    signals = {
        "prices": [65000.0, 65080.0], "latest": 65080.0,
        "orderflow": {},
        "btc_drift": 0.04,
        "btc_drift_pct": 0.00008,
        "btc_strike": 65000.0, "btc_now": 65005.0,
        "tech_mtf": 0.35,
        "btc_momentum": 0.30,
    }
    d = bot.make_decision(market, signals)
    assert d["action"] == "skip"
    assert "dead-zone" in d["reasoning"].lower()


def test_combo_confirm_does_not_bypass_underdog_band(monkeypatch):
    import db
    from bots.bot_momentum import MomentumBot
    monkeypatch.setattr(ce, "try_confirm", _stub_combo)
    monkeypatch.setattr(db, "get_paper_available", lambda: 200.0)
    monkeypatch.setattr(db, "get_kelly_fraction", lambda: 0.25)
    bot = MomentumBot(name="mom-combo-ud")
    bot._perf_cache = (9e12, 0)
    market = {
        "id": "m", "current_price": 0.38, "no_price": 0.62,
        "yes_ask": 0.39, "no_ask": 0.63, "time_remaining_seconds": 90,
    }
    signals = {
        "prices": [65000.0, 65080.0], "latest": 65080.0,
        "orderflow": {},
        "btc_drift": 0.04,
        "btc_drift_pct": 0.00008,
        "btc_strike": 65000.0, "btc_now": 65005.0,
        "tech_mtf": 0.35,
        "btc_momentum": 0.30,
    }
    d = bot.make_decision(market, signals)
    assert d["action"] == "skip"
    assert "underdog" in d["reasoning"].lower()

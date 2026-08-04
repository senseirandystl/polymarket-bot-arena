"""Data-driven skip/go/continuous rules, OOS, bandit, auto per-strategy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import config
from arena import learned_rules


@pytest.fixture()
def db(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "lr.db")
    db_module.init_db()
    monkeypatch.setattr(learned_rules, "db", db_module)
    # Reset caches
    learned_rules._eval_cache = (0.0, [])
    learned_rules._soften_cache = (0.0, {})
    learned_rules._mode_cache = (0.0, False)
    return db_module


def _ins(
    db, *, action, side, regime, entry, drift, would_win, hyp,
    strat="momentum", skip_reason=None, created_at=None,
):
    created = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with db.get_conn() as conn:
        conn.execute(
            """INSERT INTO decision_events
               (bot_name, strategy_type, market_id, action, side, skip_reason,
                entry_price, drift, regime, market_up, would_win, hyp_pnl, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f"{strat}-t", strat, f"m-{action}-{would_win}-{entry}-{created}",
                action, side, skip_reason,
                entry, drift, regime,
                1 if would_win else 0,
                1 if would_win else 0, hyp, created,
            ),
        )


def test_continuous_size_and_edge_mult():
    # Below min_n → neutral
    assert learned_rules.continuous_size_mult(0.40, 5) == 1.0
    # Low WR → size near min
    lo = learned_rules.continuous_size_mult(0.40, 40)
    hi = learned_rules.continuous_size_mult(0.65, 40)
    assert lo < 1.0 < hi
    # Low WR → edge tighter (>1)
    assert learned_rules.continuous_edge_mult(0.40, 40) > 1.0
    assert learned_rules.continuous_edge_mult(0.65, 40) < 1.0


def test_mine_promotes_skip_on_bad_buys(db, monkeypatch):
    monkeypatch.setattr(config, "LEARNED_RULES_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_MIN_N", 10, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_OOS_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY_AUTO", False, raising=False)
    for _ in range(12):
        _ins(db, action="buy", side="yes", regime="low_vol_range",
             entry=0.52, drift=0.15, would_win=False, hyp=-0.03)
    state = learned_rules.mine_and_update()
    skips = [r for r in state["rules"] if r["type"] == "skip"]
    assert skips, state
    ev = learned_rules.evaluate(
        regime="low_vol_range", side_price=0.52, drift=0.15,
        side="yes", strategy_type="momentum",
    )
    assert ev["action"] == "skip"


def test_mine_promotes_go_on_good_buys(db, monkeypatch):
    monkeypatch.setattr(config, "LEARNED_RULES_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_MIN_N", 10, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_OOS_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY_AUTO", False, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_GO_WR_MIN", 0.55, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_GO_HYP_MIN", 0.0, raising=False)
    for _ in range(12):
        _ins(db, action="buy", side="yes", regime="high_vol_trend",
             entry=0.45, drift=0.35, would_win=True, hyp=0.05)
    state = learned_rules.mine_and_update()
    goes = [r for r in state["rules"] if r["type"] == "go"]
    assert goes, state
    ev = learned_rules.evaluate(
        regime="high_vol_trend", side_price=0.45, drift=0.35,
        side="yes", strategy_type="momentum",
    )
    assert ev["action"] == "allow"
    assert ev["size_mult"] > 1.0 or ev["edge_mult"] < 1.0


def test_continuous_rule_mid_wr(db, monkeypatch):
    monkeypatch.setattr(config, "LEARNED_RULES_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_MIN_N", 10, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_OOS_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_CONTINUOUS", True, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY_AUTO", False, raising=False)
    # ~55% WR mid band → continuous soft rule, not hard skip/go
    for i in range(11):
        _ins(db, action="buy", side="yes", regime="normal",
             entry=0.50, drift=0.12, would_win=(i < 6), hyp=0.01 if i < 6 else -0.01)
    state = learned_rules.mine_and_update()
    cont = [r for r in state["rules"] if r["type"] == "continuous"]
    assert cont, state
    ev = learned_rules.evaluate(
        regime="normal", side_price=0.50, drift=0.12,
        side="yes", strategy_type="momentum",
    )
    assert ev["action"] == "allow"
    assert ev.get("type") == "continuous"


def test_oos_rejects_train_only_signal(db, monkeypatch):
    """Train toxic, test recovered → skip not promoted."""
    monkeypatch.setattr(config, "LEARNED_RULES_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_MIN_N", 8, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_OOS_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_OOS_MIN_EVENTS", 10, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_OOS_TRAIN_FRAC", 0.6, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY_AUTO", False, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_DEMOTE_SKIP_WR", 0.52, raising=False)
    base = datetime(2026, 7, 1, tzinfo=timezone.utc)
    # Early train: all losses
    for i in range(12):
        ts = (base + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        _ins(db, action="buy", side="yes", regime="low_vol_range",
             entry=0.52, drift=0.15, would_win=False, hyp=-0.04, created_at=ts)
    # Late test: all wins (signal reversed)
    for i in range(12):
        ts = (base + timedelta(days=2, hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        _ins(db, action="buy", side="yes", regime="low_vol_range",
             entry=0.52, drift=0.15, would_win=True, hyp=0.04, created_at=ts)
    state = learned_rules.mine_and_update()
    skips = [r for r in state["rules"] if r["type"] == "skip"]
    # Should reject OOS
    assert state.get("oos", {}).get("rejected_oos", 0) >= 1 or not skips


def test_skip_bandit_eases_dead_zone(db, monkeypatch):
    monkeypatch.setattr(config, "LEARNED_RULES_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_SKIP_BANDIT_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_SKIP_BANDIT_MIN_N", 10, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_SKIP_BANDIT_HIGH_CF", 0.55, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_OOS_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY_AUTO", False, raising=False)
    for _ in range(12):
        _ins(db, action="skip", side="yes", regime="low_vol_range",
             entry=0.50, drift=0.08, would_win=True, hyp=0.03,
             skip_reason="dead_zone")
    state = learned_rules.mine_and_update()
    soft = state.get("skip_soften") or {}
    assert "dead_zone" in soft
    assert soft["dead_zone"]["direction"] == "ease"
    assert soft["dead_zone"]["soften"] > 0
    out = learned_rules.skip_softening("dead_zone")
    assert out["factor"] < 1.0


def test_per_strategy_auto_off_until_mass(db, monkeypatch):
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY", False, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY_AUTO", True, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY_MIN_RESOLVED", 200, raising=False)
    with db.get_conn() as conn:
        on, reason = learned_rules.resolve_per_strategy(conn)
    assert on is False
    assert "resolved" in reason


def test_per_strategy_auto_on_with_mass(db, monkeypatch):
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY", False, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY_AUTO", True, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY_MIN_RESOLVED", 15, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY_MIN_CELLS", 2, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_MIN_N", 5, raising=False)
    # Two strategies × enough buys in distinct cells (16+ resolved)
    for strat, regime in [("momentum", "normal"), ("phantom", "high_vol_trend")]:
        for i in range(10):
            _ins(db, action="buy", side="yes", regime=regime,
                 entry=0.50, drift=0.25, would_win=True, hyp=0.02, strat=strat)
    with db.get_conn() as conn:
        on, reason = learned_rules.resolve_per_strategy(conn)
    assert on is True
    assert "auto_on" in reason


def test_evaluate_fail_open_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "LEARNED_RULES_ENABLED", False, raising=False)
    ev = learned_rules.evaluate(
        regime="x", side_price=0.5, drift=0.1, side="yes",
    )
    assert ev["action"] == "allow"
    assert ev["size_mult"] == 1.0


def test_snapshot_shape(db, monkeypatch):
    monkeypatch.setattr(config, "LEARNED_RULES_OOS_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "LEARNED_RULES_PER_STRATEGY_AUTO", False, raising=False)
    learned_rules.mine_and_update()
    snap = learned_rules.snapshot()
    assert "rules" in snap
    assert "skip_soften" in snap
    assert "per_strategy" in snap
    assert "config" in snap

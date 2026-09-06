"""Phase 3 audit fixes: paper gates ops, GA early cull, lab filter, hyp, skips."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
import db


# ---------------------------------------------------------------------------
# 1. Paper gate snapshot / ops
# ---------------------------------------------------------------------------

def test_paper_gate_snapshot_active_overrides():
    """Paper gates are off-by-default (profit-mode); activate via profile."""
    assert config.TRADING_MODE == "paper"
    snap = config.paper_gate_snapshot()
    # Default profile is off — profit-tight base gates.
    assert snap["active"] is False
    assert (snap.get("profile") or "").strip().lower() in ("off", "", "none")

def test_paper_gate_snapshot_data_gather_overrides(monkeypatch):
    monkeypatch.setattr(config, "PAPER_GATE_PROFILE", "data_gather_v1")
    snap = config.paper_gate_snapshot()
    assert snap["active"] is True
    assert snap["profile"] == "data_gather_v1"
    assert "z" in snap["overrides"]
    assert snap["overrides"]["z"]["overridden"] is True
    assert snap["overrides"]["z"]["effective"] == 0.28
    assert snap["overrides"]["lean"]["effective"] == 0.04


def test_paper_gate_snapshot_off(monkeypatch):
    monkeypatch.setattr(config, "PAPER_GATE_PROFILE", "off")
    snap = config.paper_gate_snapshot()
    assert snap["active"] is False
    assert snap["reason"] == "profile_off"


def test_ops_snapshot_includes_paper_gates(monkeypatch):
    from arena.ops_snapshot import ops_snapshot

    # Avoid heavy deps failing the whole snapshot — just check keys land.
    out = ops_snapshot()
    assert "paper_gates" in out
    assert "paper_gate_profile" in out
    assert "paper_gates_active" in out
    assert isinstance(out["paper_gates_active"], bool)
    assert out["paper_gate_profile"] == out["paper_gates"].get("profile")


# ---------------------------------------------------------------------------
# 2. GA early cull softened under paper profile
# ---------------------------------------------------------------------------

def test_early_cull_min_trades_raised_when_paper_active(monkeypatch):
    from evolution.ga import _early_cull_min_trades, _survives_legacy_bar

    monkeypatch.setattr(config, "PAPER_GATE_PROFILE", "data_gather_v1")
    monkeypatch.setattr(config, "TRADING_MODE", "paper")
    monkeypatch.setattr(config, "GA_EARLY_CULL_MIN_TRADES", 15)
    monkeypatch.setattr(config, "PAPER_GA_EARLY_CULL_MIN_TRADES", 30)
    assert _early_cull_min_trades() == 30
    # n=20 catastrophic would cull with base=15, but paper raises bar → immune
    assert _survives_legacy_bar({
        "trades": 20, "pnl": -20.0, "be_gap": -0.15, "generation": 1,
    })
    # n=35 still above paper min → early cull can fire
    assert not _survives_legacy_bar({
        "trades": 35, "pnl": -20.0, "be_gap": -0.15, "generation": 1,
    })


def test_early_cull_min_trades_base_when_paper_off(monkeypatch):
    from evolution.ga import _early_cull_min_trades

    monkeypatch.setattr(config, "PAPER_GATE_PROFILE", "off")
    monkeypatch.setattr(config, "GA_EARLY_CULL_MIN_TRADES", 15)
    assert _early_cull_min_trades() == 15


# ---------------------------------------------------------------------------
# 3. Lab empty-param filter
# ---------------------------------------------------------------------------

def test_lab_skips_empty_param_candidates(monkeypatch):
    from signals.strategy_pipeline import research

    class FakeStore:
        def list(self, limit=200):
            return []

        def insert(self, spec):
            self.inserted = getattr(self, "inserted", [])
            self.inserted.append(spec)
            return spec

        def recent_autopsies(self, limit=12):
            return []

        def open_by_stage(self, *stages):
            return []

    store = FakeStore()

    def fake_heuristic(context):
        # Use live-compat primitive (sniper) — PAPER_SLOTS==0 invent filter.
        return [
            {
                "primitive": "sniper",
                "name": "sniper-lab-empty",
                "thesis": "empty defaults",
                "params": {},
                "universe": ["polymarket:btc_5m"],
                "origin": "heuristic",
            },
            {
                "primitive": "sniper",
                "name": "sniper-strict",
                "thesis": "real genome",
                "params": {
                    "min_drift": 0.28,
                    "min_confidence": 0.25,
                    "quiet_drift_bump": 0.12,
                },
                "universe": ["polymarket:btc_5m"],
                "origin": "heuristic",
            },
        ]

    monkeypatch.setattr(research, "_heuristic_candidates", fake_heuristic)
    monkeypatch.setattr(research, "_gene_bank_mutations", lambda ctx: [])
    monkeypatch.setattr(research, "_llm_candidates", lambda ctx: [])
    monkeypatch.setattr(
        "signals.strategy_pipeline.fingerprint.active_peers", lambda: []
    )
    monkeypatch.setattr(
        "signals.strategy_pipeline.learning_spine.fingerprint_blocked",
        lambda *a, **k: False,
    )

    out = research.propose(store, max_new=5)
    names = [s.get("name") for s in out]
    assert "sniper-lab-empty" not in names
    assert any(n and n.startswith("sniper-strict") for n in names)


# ---------------------------------------------------------------------------
# 4. Entry-less skips: no would_win / hyp_pnl
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "dec.db")
    db.init_db()
    from arena import decision_log
    with decision_log._queue_lock:
        decision_log._queue.clear()
    with decision_log._throttle_lock:
        decision_log._throttle.clear()
    monkeypatch.setattr(config, "DECISION_LOG_ENABLED", True)
    monkeypatch.setattr(config, "DECISION_LOG_MIN_INTERVAL_SEC", 0.0)
    yield


def test_resolve_entryless_skip_leaves_would_win_null(tmp_db):
    from arena import decision_log

    sig = {
        "action": "skip",
        "side": "yes",
        "edge": 0.0,
        "confidence": 0.1,
        "entry_price": None,  # no actionable entry
        "reasoning": "no fade thesis",
        "signals": {"drift": 0.0, "regime": "low_vol_range"},
        "features": ["no_thesis"],
        "skip_reason": "no_thesis",
    }
    assert decision_log.enqueue(
        bot_name="meanrev-v1",
        strategy_type="mean_reversion",
        market_id="mkt-empty",
        signal=sig,
        force=True,
    )
    assert decision_log.flush() == 1
    n = decision_log.resolve_pending({"mkt-empty": True})
    assert n == 1
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT market_up, would_win, hyp_pnl, entry_price FROM decision_events"
        ).fetchone()
    assert row["market_up"] == 1
    assert row["would_win"] is None
    assert row["hyp_pnl"] is None
    assert row["entry_price"] is None


def test_resolve_with_entry_still_stamps_would_win(tmp_db):
    from arena import decision_log

    sig = {
        "action": "skip",
        "side": "yes",
        "edge": 0.05,
        "confidence": 0.2,
        "entry_price": 0.45,
        "reasoning": "edge but skipped",
        "signals": {"drift": 0.3},
        "features": ["ok"],
    }
    decision_log.enqueue(
        bot_name="b", strategy_type="momentum", market_id="m1",
        signal=sig, force=True,
    )
    decision_log.flush()
    decision_log.resolve_pending({"m1": True})
    with db.get_conn() as conn:
        row = conn.execute("SELECT would_win, hyp_pnl FROM decision_events").fetchone()
    assert row["would_win"] == 1
    assert row["hyp_pnl"] is not None and row["hyp_pnl"] > 0


# ---------------------------------------------------------------------------
# 5. /api/skips richer payload
# ---------------------------------------------------------------------------

def test_api_skips_windowed_shape(tmp_db, monkeypatch):
    from arena import decision_log
    from dashboard.server import get_skips

    for i in range(5):
        decision_log.enqueue(
            bot_name="mom-a" if i < 3 else "fade-b",
            strategy_type="momentum",
            market_id=f"s{i}",
            signal={
                "action": "skip",
                "side": "yes",
                "edge": 0.01,
                "confidence": 0.1,
                "entry_price": 0.5,
                "reasoning": "weak lean",
                "signals": {},
                "features": [],
                "skip_reason": "weak_lean" if i < 3 else "no_edge",
            },
            force=True,
        )
    decision_log.flush()

    resp = get_skips(hours=24.0, bot_name=None)
    # Starlette JSONResponse — body is in .body
    data = json.loads(resp.body.decode())
    assert "counts" in data
    assert "top_reasons" in data
    assert data["hours"] == 24.0
    assert data["skip_n"] >= 5
    assert data["decision_n"] >= 5
    assert data["counts"].get("weak_lean", 0) >= 3
    assert isinstance(data["top_reasons"], list)
    assert data["top_reasons"][0]["reason"] in ("weak_lean", "no_edge")

    resp2 = get_skips(hours=24.0, bot_name="fade-b")
    data2 = json.loads(resp2.body.decode())
    assert data2["bot_name"] == "fade-b"
    assert data2["counts"].get("no_edge", 0) >= 2
    assert "weak_lean" not in data2["counts"] or data2["counts"].get("weak_lean", 0) == 0

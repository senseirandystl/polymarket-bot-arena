"""Desk cycle: specs, store, heuristic research, promotion bars."""

from desk.compiler import sanitize_spec, new_spec_id, normalize_primitive
from desk.universe import phase_universe, tradable_slots
from desk.postmortem import write_autopsy
from desk.store import HypothesisStore
from desk.cycle import promotion_bars, DeskHost


def test_normalize_and_sanitize():
    spec = sanitize_spec({
        "primitive": "meanrev",
        "name": "fade desk!!",
        "lane_weights": {"drift": 3, "mom": 1, "strat": 0},
        "params": {"not_a_real_param": 1, "lookback_candles": 8},
        "universe": ["polymarket:btc_5m"],
        "thesis": "fade to strike",
    })
    assert spec["primitive"] == "mean_reversion"
    assert spec["name"] == "fade-desk"
    assert abs(sum(spec["lane_weights"].values()) - 1.0) < 1e-9
    assert spec["lane_weights"]["drift"] > spec["lane_weights"]["mom"]


def test_unknown_primitive_rejected():
    try:
        normalize_primitive("lstm-oracle")
        assert False, "should have raised"
    except ValueError:
        pass


def test_universe_phase_1_only_tradable_btc():
    slots = phase_universe(1)
    assert {s.slot_id for s in slots} == {"polymarket:btc_5m", "kalshi:btc_15m"}
    assert all(s.tradable for s in tradable_slots(1))
    phase2 = phase_universe(2)
    assert len(phase2) > len(slots)
    assert any(not s.tradable for s in phase2)


def test_store_and_autopsy(arena_db):
    store = HypothesisStore()
    spec = sanitize_spec({
        "spec_id": new_spec_id("sniper"),
        "primitive": "sniper",
        "name": "sniper-desk",
        "thesis": "lag hunt",
    })
    store.insert(spec)
    got = store.get(spec["spec_id"])
    assert got["stage"] == "coded" or got["name"] == "sniper-desk"
    write_autopsy(store, spec["spec_id"], stage="backtested",
                  reason="pnl_-1.2", evidence={"primitive": "sniper"})
    dead = store.get(spec["spec_id"])
    assert dead["status"] == "closed"
    assert dead["autopsy"]["reason"] == "pnl_-1.2"
    assert store.counts()["rejected"] >= 1


def test_heuristic_research_respects_max(arena_db, monkeypatch):
    monkeypatch.setattr("config.DESK_LLM_PROVIDER", "none")
    store = HypothesisStore()
    from desk.research import propose
    out = propose(store, max_new=2)
    assert 1 <= len(out) <= 2
    assert all(s["primitive"] in {
        "momentum", "mean_reversion", "sniper", "hybrid", "sweeper",
        "phantom", "lag_residual",
    } for s in out)


def test_promotion_bars_read_config(monkeypatch):
    monkeypatch.setattr("config.DESK_PROMOTE_MIN_TRADES", 100)
    monkeypatch.setattr("config.DESK_PROMOTE_MIN_DAYS", 7)
    monkeypatch.setattr("config.DESK_PROMOTE_TRADE_FLOOR", 30)
    assert promotion_bars() == (100, 7, 30)


def test_desk_tick_without_network(arena_db, monkeypatch):
    host = DeskHost()

    def fake_bt(hyp):
        return {
            "passed": False,
            "reason": "pnl_negative",
            "child_pnl": -1.0,
            "markets": 12,
            "primitive": (hyp.get("spec") or {}).get("primitive"),
        }

    monkeypatch.setattr(host, "_backtest", fake_bt)
    monkeypatch.setattr("config.DESK_LLM_PROVIDER", "none")
    monkeypatch.setattr("config.DESK_MAX_NEW_PER_TICK", 1)
    report = host.tick()
    assert report["proposed"] >= 1
    assert report["rejected"] >= 1
    snap = host.snapshot().as_dict()
    assert snap["pipeline_counts"]["rejected"] >= 1
    assert len(snap["roles"]) == 7

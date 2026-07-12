"""Tests for the paper (local-sim) and live (Polymarket) venue engines.

Covers the July-2026 rework that decoupled paper fills from Simmer's
50-buys/day cap: paper trades are now priced locally and always fill, live
trades go through the CLOB market-order path.
"""

import sys
import types

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "venues_test.db")
    db_module.init_db()
    return db_module


@pytest.fixture()
def paper_engine():
    from venues.paper import PaperEngine
    return PaperEngine()


def test_entry_price_for_sides():
    from venues.paper import entry_price_for
    m = {"current_price": 0.62}
    assert entry_price_for(m, "yes") == pytest.approx(0.62)
    assert entry_price_for(m, "no") == pytest.approx(0.38)


@pytest.mark.parametrize("price", [None, 0, 1, 1.5, -0.1, "x"])
def test_entry_price_for_rejects_bad_prices(price):
    from venues.paper import entry_price_for
    assert entry_price_for({"current_price": price}, "yes") is None


def test_paper_place_logs_local_sim_fill(db, paper_engine, monkeypatch):
    import config
    monkeypatch.setattr(config, "SIMMER_MIRROR_ENABLED", False, raising=False)
    market = {"id": "m1", "question": "BTC 5m", "current_price": 0.50}
    res = paper_engine.place(
        bot_name="momentum-v1", side="yes", amount=4.0, market=market,
        mode="paper", confidence=0.2, reasoning="r", features=None,
    )
    assert res.success and res.fill_source == "local_sim"
    assert res.shares == pytest.approx(8.0)          # 4.0 / 0.50
    assert res.entry_price == pytest.approx(0.50)
    with db.get_conn() as c:
        row = dict(c.execute(
            "SELECT shares_bought, entry_price, fill_source, trade_id, outcome "
            "FROM trades WHERE market_id='m1'").fetchone())
    assert row["fill_source"] == "local_sim"
    assert row["trade_id"] is None                   # no Simmer id for local sim
    assert row["outcome"] is None                    # pending until resolved
    assert row["shares_bought"] == pytest.approx(8.0)


def test_paper_place_skips_when_no_price(db, paper_engine, monkeypatch):
    import config
    monkeypatch.setattr(config, "SIMMER_MIRROR_ENABLED", False, raising=False)
    res = paper_engine.place(
        bot_name="b", side="yes", amount=2.0,
        market={"id": "m2", "current_price": None}, mode="paper",
    )
    assert not res.success and res.reason == "no_market_price"
    with db.get_conn() as c:
        assert c.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0


def test_paper_does_not_mirror_when_disabled(db, paper_engine, monkeypatch):
    """With the mirror off, no Simmer network call happens."""
    import config
    monkeypatch.setattr(config, "SIMMER_MIRROR_ENABLED", False, raising=False)
    called = {"n": 0}
    monkeypatch.setattr(paper_engine, "_mirror_to_simmer",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    paper_engine.place(bot_name="b", side="no", amount=1.0,
                       market={"id": "m3", "current_price": 0.4}, mode="paper")
    assert called["n"] == 0


def test_live_place_missing_token_id(db):
    from venues.live import LiveEngine
    res = LiveEngine().place(
        bot_name="b", side="yes", amount=1.0,
        market={"id": "m4", "question": "q"}, mode="live",
    )
    assert not res.success and res.reason == "missing_token_id"


def test_live_place_success(db, monkeypatch):
    from venues.live import LiveEngine
    # Stub the polymarket_client module the engine imports.
    stub = types.ModuleType("polymarket_client")
    stub.place_market_order = lambda **kw: {
        "success": True, "order_id": "0xabc", "price": 0.55, "size": 3.6,
    }
    monkeypatch.setitem(sys.modules, "polymarket_client", stub)
    market = {"id": "m5", "question": "q", "polymarket_token_id": "tok-yes"}
    res = LiveEngine().place(
        bot_name="b", side="yes", amount=2.0, market=market, mode="live",
    )
    assert res.success and res.fill_source == "polymarket"
    assert res.entry_price == pytest.approx(0.55)
    with db.get_conn() as c:
        row = dict(c.execute(
            "SELECT venue, fill_source, trade_id, entry_price "
            "FROM trades WHERE market_id='m5'").fetchone())
    assert row["venue"] == "polymarket" and row["fill_source"] == "polymarket"
    assert row["trade_id"] == "0xabc"

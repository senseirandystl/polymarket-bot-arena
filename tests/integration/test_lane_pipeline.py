"""Candidate-lane validation → proposal → approval pipeline.

Covers: tools/lane_candidates (pure parts), the db proposal lifecycle
(tmp DB), and base_bot's consumption of approved overrides.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import db
from tools import lane_candidates as lc
from tools.signal_validation import Sample
from polymarket_fills import taker_fee


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    yield


def _sample(fut=0.5, pm_yes=0.52, yes_won=True, tr=120.0):
    return Sample(
        market_id="m", time_remaining=tr, btc_now=100000.0, strike=99900.0,
        yes_won=yes_won, pm_yes=pm_yes,
        signals={"fut_taker": fut, "fut_funding": None, "fut_oi": None,
                 "tech_mtf": None, "tech_macd": None, "tech_bb": None,
                 "xasset": None})


# ---------------------------------------------------------------------------
# Series + attach
# ---------------------------------------------------------------------------

class TestSeries:
    def test_at_returns_last_at_or_before(self):
        s = lc.Series([(10, 1.0), (20, 2.0), (30, 3.0)])
        assert s.at(5) is None
        assert s.at(10) == 1.0
        assert s.at(25) == 2.0
        assert s.at(99) == 3.0

    def test_last_two_and_closes_until(self):
        s = lc.Series([(10, 1.0), (20, 2.0), (30, 3.0)])
        assert s.last_two(15) is None
        assert s.last_two(30) == (2.0, 3.0)
        assert s.closes_until(25, 5) == [1.0, 2.0]


class TestAttachCandidates:
    def test_attach_computes_lanes(self):
        open_ts = 1000.0
        samples = [_sample(tr=240.0)]  # decision at open_ts + 60
        n = 70
        btc = lc.Series([(open_ts - (n - i) * 60, 100000.0 * (1.001 ** i))
                         for i in range(n)])
        series = {
            "btc_close": btc,
            "eth_close": lc.Series([(open_ts - 60, 3000.0), (open_ts + 30, 3010.0)]),
            "sol_close": lc.Series([(open_ts - 60, 150.0), (open_ts + 30, 150.5)]),
            "taker": lc.Series([(open_ts + 30, 1.2)]),
            "funding": lc.Series([(open_ts - 3600, 0.0003)]),
            "oi": lc.Series([(open_ts - 300, 80000.0), (open_ts + 1, 80400.0)]),
        }
        lc.attach_candidates(samples, open_ts, series)
        sig = samples[0].signals
        assert sig["fut_taker"] > 0        # 1.2 buy/sell ratio -> positive
        assert sig["fut_funding"] > 0
        assert sig["fut_oi"] > 0           # OI rose 0.5%
        assert sig["tech_mtf"] > 0         # uptrending closes
        assert sig["xasset"] > 0           # both peers up

    def test_missing_series_reads_none(self):
        samples = [_sample(tr=240.0)]
        lc.attach_candidates(samples, 1000.0, {})
        sig = samples[0].signals
        # Keys that attach_candidates always writes (ms_mom_1m needs features).
        for k in ("fut_taker", "fut_funding", "fut_oi",
                  "tech_mtf", "tech_macd", "tech_bb", "xasset", "lag"):
            assert sig.get(k) is None


# ---------------------------------------------------------------------------
# Evaluation + proposal thresholds
# ---------------------------------------------------------------------------

class TestBuildProposals:
    def _metrics(self, n=300, wr=0.60, ev=0.02, key="fut_taker"):
        return {key: {"n": n, "follow_wr": wr, "net_n": n,
                      "net_wr": wr, "avg_price": 0.55,
                      "ev_per_share": ev}}

    def test_qualifying_lane_proposed_with_profile(self):
        props = lc.build_proposals(self._metrics())
        assert len(props) == 1
        assert props[0]["lane"] == "fut"
        assert props[0]["proposal"]["profile"] == lc.PROFILE_SUGGESTIONS["fut"]
        assert props[0]["metrics"]["signal_key"] == "fut_taker"

    def test_lag_and_ms_mom_are_promotable_lanes(self):
        """Expanded 2026-08 candidates must clear the same bar + have profiles."""
        assert "lag" in lc.LIVE_LANE_KEYS
        assert "ms_mom" in lc.LIVE_LANE_KEYS
        assert "lag" in lc.PROFILE_SUGGESTIONS
        assert "ms_mom" in lc.PROFILE_SUGGESTIONS
        assert "flow_decay" in lc.PROFILE_SUGGESTIONS  # live-shadow only
        m = {**self._metrics(key="lag"), **self._metrics(key="ms_mom_1m")}
        props = {p["lane"] for p in lc.build_proposals(m)}
        assert props == {"lag", "ms_mom"}

    def test_each_threshold_is_conjunctive(self):
        # This is the pm_mom lesson: high WR with negative EV must NOT pass.
        assert lc.build_proposals(self._metrics(ev=-0.008)) == []
        assert lc.build_proposals(self._metrics(wr=0.52)) == []
        assert lc.build_proposals(self._metrics(n=50)) == []

    def test_evaluate_candidates_end_to_end(self):
        # 40 samples where following fut_taker always wins at a cheap price.
        samples = ([_sample(fut=0.5, pm_yes=0.52, yes_won=True)] * 20
                   + [_sample(fut=-0.5, pm_yes=0.48, yes_won=False)] * 20)
        res = lc.evaluate_candidates(samples, taker_fee)
        m = res["fut_taker"]
        assert m["n"] == 40
        assert m["follow_wr"] == 1.0
        assert m["ev_per_share"] > 0.3     # ~48c gain minus fee


# ---------------------------------------------------------------------------
# DB proposal lifecycle
# ---------------------------------------------------------------------------

class TestProposalLifecycle:
    def test_run_recorded_and_latest_parsed(self, tmp_db):
        rid = db.record_lane_validation_run(300, 1200, {"fut_taker": {"n": 5}})
        run = db.get_latest_lane_run()
        assert run["id"] == rid
        assert run["results"] == {"fut_taker": {"n": 5}}

    def test_create_dedupes_pending(self, tmp_db):
        p1 = db.create_lane_proposal("fut", {"n": 1}, {"profile": {"momentum": 0.1}})
        p2 = db.create_lane_proposal("fut", {"n": 2}, {"profile": {"momentum": 0.2}})
        assert p1 == p2                     # refreshed, not duplicated
        pending = db.get_lane_proposals(status="pending")
        assert len(pending) == 1
        assert pending[0]["metrics"] == {"n": 2}   # evidence refreshed

    def test_approve_activates_override(self, tmp_db):
        pid = db.create_lane_proposal(
            "fut", {"n": 300}, {"profile": {"momentum": 0.1, "hybrid": 0.1}})
        assert db.decide_lane_proposal(pid, "approve") == "approved"
        ov = db.get_lane_overrides()
        assert ov["fut"]["enabled"] is True
        assert ov["fut"]["profile"] == {"momentum": 0.1, "hybrid": 0.1}
        # An approved lane blocks new proposals for itself.
        assert db.create_lane_proposal("fut", {}, {}) is None

    def test_deny_leaves_overrides_untouched(self, tmp_db):
        pid = db.create_lane_proposal("tech", {"n": 300}, {"profile": {}})
        assert db.decide_lane_proposal(pid, "deny") == "denied"
        assert db.get_lane_overrides() == {}
        # Denied is final for that proposal — deciding again raises.
        with pytest.raises(ValueError):
            db.decide_lane_proposal(pid, "approve")

    def test_bad_inputs_raise(self, tmp_db):
        with pytest.raises(ValueError):
            db.decide_lane_proposal(999, "approve")
        pid = db.create_lane_proposal("xasset", {}, {"profile": {}})
        with pytest.raises(ValueError):
            db.decide_lane_proposal(pid, "yolo")

    def test_disable_override(self, tmp_db):
        pid = db.create_lane_proposal("fut", {}, {"profile": {"momentum": 0.1}})
        db.decide_lane_proposal(pid, "approve")
        assert db.disable_lane_override("fut") is True
        assert db.get_lane_overrides()["fut"]["enabled"] is False
        assert db.disable_lane_override("nope") is False


# ---------------------------------------------------------------------------
# base_bot consumption of approved overrides
# ---------------------------------------------------------------------------

class TestBotConsumesOverrides:
    def _reset_cache(self):
        import bots.base_bot as bb
        bb._lane_override_cache = (0.0, {})

    def test_override_supplies_profile_weight(self, monkeypatch):
        from bots.bot_momentum import MomentumBot
        self._reset_cache()
        monkeypatch.setattr(db, "get_lane_overrides", lambda: {
            "fut": {"enabled": True, "profile": {"momentum": 0.2}}})
        bot = MomentumBot(name="t")
        # fut lane alone at +1.0: P = 0.5 + 0.5*0.2*1.0 = 0.60
        p = bot._model_prob_yes({"fut": 1.0})
        assert abs(p - 0.60) < 1e-9
        self._reset_cache()

    def test_unlisted_strategy_stays_zero(self, monkeypatch):
        from bots.bot_mean_rev import MeanRevBot
        self._reset_cache()
        monkeypatch.setattr(db, "get_lane_overrides", lambda: {
            "fut": {"enabled": True, "profile": {"momentum": 0.2}}})
        bot = MeanRevBot(name="t2")
        p = bot._model_prob_yes({"fut": 1.0})
        assert abs(p - 0.5) < 1e-9          # meanrev not in the profile
        self._reset_cache()

    def test_disabled_override_falls_back_to_profile(self, monkeypatch):
        from bots.bot_momentum import MomentumBot
        self._reset_cache()
        monkeypatch.setattr(db, "get_lane_overrides", lambda: {
            "fut": {"enabled": False, "profile": {"momentum": 0.2}}})
        bot = MomentumBot(name="t3")
        p = bot._model_prob_yes({"fut": 1.0})
        assert abs(p - 0.5) < 1e-9          # profile has fut at 0.00
        self._reset_cache()


# ---------------------------------------------------------------------------
# Dashboard: run-validation endpoints (subprocess mocked)
# ---------------------------------------------------------------------------

class TestRunValidationEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import dashboard.server as srv
        srv._validation_run.update({"proc": None, "started_at": None, "markets": None})
        yield TestClient(srv.app), srv
        srv._validation_run.update({"proc": None, "started_at": None, "markets": None})

    AUTH = ("admin", "Thor")

    class _FakeProc:
        def __init__(self):
            self._code = None

        def poll(self):
            return self._code

    def test_status_idle_initially(self, client):
        cl, _srv = client
        r = cl.get("/api/lane-validation/status", auth=self.AUTH)
        assert r.status_code == 200
        body = r.json()
        assert body["running"] is False
        assert body["returncode"] is None

    def test_run_starts_and_guards_concurrency(self, client, monkeypatch):
        import subprocess
        cl, srv = client
        fake = self._FakeProc()
        captured = {}

        def fake_popen(cmd, **kw):
            captured["cmd"] = cmd
            return fake

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        r = cl.post("/api/lane-validation/run", auth=self.AUTH,
                    json={"markets": 150})
        assert r.status_code == 200
        assert r.json()["markets"] == 150
        assert "--propose" in captured["cmd"]
        assert "150" in captured["cmd"]

        # Second click while running -> 409.
        r2 = cl.post("/api/lane-validation/run", auth=self.AUTH, json={})
        assert r2.status_code == 409

        # Process exits -> status reports done, run can start again.
        fake._code = 0
        st = cl.get("/api/lane-validation/status", auth=self.AUTH).json()
        assert st["running"] is False
        assert st["returncode"] == 0
        r3 = cl.post("/api/lane-validation/run", auth=self.AUTH, json={})
        assert r3.status_code == 200

    def test_markets_clamped(self, client, monkeypatch):
        import subprocess
        cl, _srv = client
        monkeypatch.setattr(subprocess, "Popen",
                            lambda cmd, **kw: self._FakeProc())
        r = cl.post("/api/lane-validation/run", auth=self.AUTH,
                    json={"markets": 5})
        assert r.json()["markets"] == 50   # clamped to the floor

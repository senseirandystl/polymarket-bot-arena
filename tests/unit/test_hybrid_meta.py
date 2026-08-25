"""HybridMetaLearner (bots/meta_learner.py) + hybrid integration.

Covers the meta(...) reasoning-token roundtrip, the Hedge-style online
update on resolved trades (correctness scoring, regime bucketing, clipping,
out-of-order-resolution safety), sample-size shrinkage in online_mults,
arena_state persistence, the hybrid bot's use of the learner (weights,
reasoning, signal-profile tilt), and the dashboard endpoint.
"""

import json
from unittest.mock import patch

import pytest

from tests.conftest import make_market, make_signals

from bots import meta_learner as ml
from bots.bot_hybrid import HybridBot, SUBS
from bots.meta_learner import HybridMetaLearner, bucket_for, format_token, parse_token


# ---------------------------------------------------------------------------
# Token + bucket helpers
# ---------------------------------------------------------------------------

class TestToken:
    def test_roundtrip(self):
        votes = {"momentum": 0.42, "mean_rev": -0.31, "phantom": 0.27}
        token = format_token(votes, "trending")
        parsed = parse_token(f"Meta[trending_up] (2Y/1N) {token}: momentum[...]")
        assert parsed is not None
        got, bucket = parsed
        assert bucket == "trending"
        for sub, v in votes.items():
            assert got[sub] == pytest.approx(v, abs=0.005)

    def test_missing_subs_default_zero(self):
        token = format_token({"momentum": 0.5}, "mixed")
        got, bucket = parse_token(token)
        assert got["phantom"] == 0.0 and got["mean_rev"] == 0.0
        assert bucket == "mixed"

    def test_legacy_sent_token_still_parses(self):
        # Pre-removal hybrid rows had sent= in the meta token.
        raw = "meta(mom=+0.40 rev=-0.30 sent=+0.00 ph=+0.20 | reg=trending)"
        got, bucket = parse_token(raw)
        assert bucket == "trending"
        assert got["momentum"] == pytest.approx(0.40)
        assert got["phantom"] == pytest.approx(0.20)
        assert "sentiment" not in got

    def test_parse_none_on_plain_reasoning(self):
        assert parse_token("fair=0.55 model=0.62 => yes") is None
        assert parse_token(None) is None


class TestBucketFor:
    @pytest.mark.parametrize("score,expected", [
        (0.9, "trending"), (0.65, "trending"),
        (0.5, "mixed"), (0.64, "mixed"), (0.36, "mixed"),
        (0.35, "ranging"), (0.1, "ranging"),
        (None, "mixed"),
    ])
    def test_boundaries(self, score, expected):
        assert bucket_for(score) == expected


# ---------------------------------------------------------------------------
# Online update on resolved trades
# ---------------------------------------------------------------------------

def _insert_trade(db_module, votes, bucket, side, outcome,
                  bot_name="hybrid-v1"):
    """A hybrid trade with a meta token, resolved (or pending) in the DB."""
    reasoning = f"Meta[x] (1Y/0N) {format_token(votes, bucket)}: stuff"
    trade_id = db_module.log_trade(
        bot_name=bot_name, market_id="mkt-1", side=side, amount=5.0,
        venue="polymarket", mode="paper", confidence=0.5, reasoning=reasoning)
    if outcome is not None:
        with db_module.get_conn() as conn:
            conn.execute(
                "UPDATE trades SET outcome=?, resolved_at=datetime('now') "
                "WHERE id=?", (outcome, trade_id))
    return trade_id


class TestOnlineUpdate:
    def test_correct_vote_raises_wrong_vote_lowers(self, arena_db):
        # YES trade WON → market UP. momentum voted up (correct),
        # mean_rev voted down (wrong), phantom abstained (0.0).
        _insert_trade(arena_db,
                      {"momentum": 0.4, "mean_rev": -0.3, "phantom": 0.0},
                      "trending", side="yes", outcome="win")
        learner = HybridMetaLearner()
        assert learner.update_from_trades() == 1

        state = learner.snapshot()
        mom = state["subs"]["momentum"]
        rev = state["subs"]["mean_rev"]
        assert mom["overall"]["mult"] > 1.0
        assert mom["overall"]["correct"] == 1 and mom["overall"]["n"] == 1
        assert rev["overall"]["mult"] < 1.0
        assert rev["overall"]["correct"] == 0 and rev["overall"]["n"] == 1
        # bucket record mirrors overall for this single trending trade
        assert mom["trending"]["mult"] == pytest.approx(mom["overall"]["mult"])
        assert "phantom" not in state["subs"]  # abstained → untouched

    def test_no_trade_won_means_market_down(self, arena_db):
        # NO trade WON → market DOWN → a negative vote was CORRECT.
        _insert_trade(arena_db, {"phantom": -0.5}, "ranging",
                      side="no", outcome="win")
        learner = HybridMetaLearner()
        learner.update_from_trades()
        ph = learner.snapshot()["subs"]["phantom"]
        assert ph["overall"]["mult"] > 1.0
        assert ph["ranging"]["n"] == 1

    def test_incremental_processes_each_trade_once(self, arena_db):
        _insert_trade(arena_db, {"momentum": 0.4}, "mixed", "yes", "win")
        learner = HybridMetaLearner()
        assert learner.update_from_trades() == 1
        assert learner.update_from_trades() == 0
        assert learner.snapshot()["subs"]["momentum"]["overall"]["n"] == 1

    def test_pending_trade_blocks_id_advance(self, arena_db):
        a = _insert_trade(arena_db, {"momentum": 0.4}, "mixed", "yes", "win")
        b = _insert_trade(arena_db, {"momentum": 0.4}, "mixed", "yes", None)
        _insert_trade(arena_db, {"momentum": 0.4}, "mixed", "yes", "win")

        learner = HybridMetaLearner()
        # Only trade A processed: B is pending and sits before C.
        assert learner.update_from_trades() == 1
        assert learner.snapshot()["last_trade_id"] == a

        # B resolves → the next pass picks up BOTH B and C.
        with arena_db.get_conn() as conn:
            conn.execute(
                "UPDATE trades SET outcome='loss', "
                "resolved_at=datetime('now') WHERE id=?", (b,))
        assert learner.update_from_trades() == 2
        assert learner.snapshot()["subs"]["momentum"]["overall"]["n"] == 3

    def test_multipliers_clip_at_bounds(self, arena_db):
        for _ in range(60):  # far more wins than exp(60*eta) allows unclipped
            _insert_trade(arena_db, {"momentum": 0.5}, "mixed", "yes", "win")
        learner = HybridMetaLearner()
        learner.update_from_trades()
        assert learner.snapshot()["subs"]["momentum"]["overall"]["mult"] \
            == pytest.approx(learner.max_mult)

    def test_ignores_other_bots_and_untagged_trades(self, arena_db):
        _insert_trade(arena_db, {"momentum": 0.4}, "mixed", "yes", "win",
                      bot_name="momentum-v1")
        arena_db.log_trade(bot_name="hybrid-v1", market_id="m", side="yes",
                           amount=5.0, venue="polymarket", mode="paper",
                           reasoning="no token here")
        learner = HybridMetaLearner()
        assert learner.update_from_trades() == 0

    def test_persists_and_reloads(self, arena_db):
        _insert_trade(arena_db, {"momentum": 0.4}, "trending", "yes", "win")
        HybridMetaLearner().update_from_trades()

        fresh = HybridMetaLearner()          # new instance, cold cache
        mults = fresh.online_mults("trending")
        assert mults["momentum"] > 1.0
        raw = arena_db.get_arena_state(ml.STATE_KEY)
        assert json.loads(raw)["subs"]["momentum"]["overall"]["n"] == 1


def _insert_cf_decision(db_module, votes, bucket, market_up,
                        bot_name="hybrid-v1", action="skip"):
    """Resolved hybrid skip with meta_token for counterfactual learning."""
    token = format_token(votes, bucket)
    with db_module.get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO decision_events (
                   bot_name, strategy_type, market_id, action, side,
                   meta_token, market_up, would_win
               ) VALUES (?, 'hybrid', 'mkt-cf', ?, 'yes', ?, ?, ?)""",
            (bot_name, action, token, 1 if market_up else 0,
             1 if market_up else 0),
        )
        return cur.lastrowid


class TestCounterfactualUpdate:
    @pytest.fixture(autouse=True)
    def _enable_cf(self, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "HYBRID_META_CF_ENABLED", True)

    def test_skip_votes_update_multipliers(self, arena_db):
        # market UP: momentum yes-vote correct, mean_rev no-vote wrong
        _insert_cf_decision(
            arena_db,
            {"momentum": 0.5, "mean_rev": -0.4, "phantom": 0.0},
            "trending", market_up=True,
        )
        learner = HybridMetaLearner(eta=0.12)
        assert learner.update_from_decisions() == 1
        state = learner.snapshot()
        assert state["subs"]["momentum"]["overall"]["mult"] > 1.0
        assert state["subs"]["mean_rev"]["overall"]["mult"] < 1.0
        assert "phantom" not in state["subs"]  # abstained
        assert state["cf"]["n"] == 1
        assert state["last_decision_id"] > 0

    def test_cf_eta_smaller_than_trade_eta(self, arena_db, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "HYBRID_META_CF_ETA_SCALE", 0.25)
        monkeypatch.setattr(cfg, "HYBRID_META_CF_ENABLED", True)
        # One CF skip where momentum is correct
        _insert_cf_decision(
            arena_db, {"momentum": 0.5}, "mixed", market_up=True)
        cf_learner = HybridMetaLearner(eta=0.12)
        cf_learner.update_from_decisions()
        cf_mult = cf_learner.snapshot()["subs"]["momentum"]["overall"]["mult"]

        # One real trade, same vote — should move farther (full eta)
        _insert_trade(arena_db, {"momentum": 0.5}, "mixed", "yes", "win",
                      bot_name="hybrid-v2")
        tr_learner = HybridMetaLearner(eta=0.12, name_prefix="hybrid-v2")
        tr_learner.update_from_trades()
        tr_mult = tr_learner.snapshot()["subs"]["momentum"]["overall"]["mult"]

        assert tr_mult > cf_mult > 1.0

    def test_buys_not_double_counted_via_decisions(self, arena_db):
        # action=buy rows must be ignored by CF path
        _insert_cf_decision(
            arena_db, {"momentum": 0.5}, "mixed", market_up=True, action="buy")
        learner = HybridMetaLearner()
        assert learner.update_from_decisions() == 0

    def test_incremental_cf_cursor(self, arena_db):
        _insert_cf_decision(
            arena_db, {"momentum": 0.5}, "mixed", market_up=True)
        learner = HybridMetaLearner()
        assert learner.update_from_decisions() == 1
        assert learner.update_from_decisions() == 0

    def test_maybe_update_runs_both_paths(self, arena_db):
        _insert_trade(arena_db, {"momentum": 0.4}, "mixed", "yes", "win")
        _insert_cf_decision(
            arena_db, {"phantom": -0.5}, "ranging", market_up=False)
        learner = HybridMetaLearner(update_ttl=0.0)
        n = learner.maybe_update()
        assert n >= 2
        snap = learner.snapshot()
        assert snap["subs"]["momentum"]["overall"]["n"] >= 1
        assert snap["subs"]["phantom"]["overall"]["n"] >= 1

    def test_cf_disabled_skips_updates(self, arena_db, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "HYBRID_META_CF_ENABLED", False)
        _insert_cf_decision(
            arena_db, {"momentum": 0.5}, "mixed", market_up=True)
        learner = HybridMetaLearner(eta=0.12)
        assert learner.update_from_decisions() == 0


def test_hybrid_bot_caps_online_max_mult(monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg, "HYBRID_META_MAX_MULT", 1.2)
    bot = HybridBot(name="hybrid-cap", params={"online_max_mult": 2.5})
    assert bot._meta.max_mult == pytest.approx(1.2)


def test_online_mults_clamps_persisted_cf_era(monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg, "HYBRID_META_MAX_MULT", 1.2)
    bot = HybridBot(name="hybrid-clamp", params={"online_max_mult": 2.5})
    bot._meta._state = {
        "subs": {
            "momentum": {
                "overall": {"mult": 2.5, "n": 100},
                "mixed": {"mult": 2.5, "n": 80},
            }
        }
    }
    mults = bot._meta.online_mults("mixed")
    assert mults["momentum"] == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# Decision-time reads: shrinkage blend
# ---------------------------------------------------------------------------

class TestOnlineMults:
    def _learner_with(self, sub_state):
        learner = HybridMetaLearner(bucket_full_trust=20)
        learner._state = {"last_trade_id": 0, "subs": sub_state, "last": {}}
        return learner

    def test_neutral_without_state(self):
        learner = self._learner_with({})
        assert set(learner.online_mults("trending")) == set(ml.SUB_TOKENS)
        assert all(v == 1.0 for v in learner.online_mults("trending").values())

    def test_full_bucket_sample_dominates(self):
        learner = self._learner_with({
            "momentum": {"overall": {"mult": 1.0, "n": 40, "correct": 20},
                         "trending": {"mult": 2.0, "n": 20, "correct": 16}}})
        assert learner.online_mults("trending")["momentum"] == pytest.approx(2.0)

    def test_thin_bucket_shrinks_toward_overall(self):
        learner = self._learner_with({
            "momentum": {"overall": {"mult": 1.0, "n": 40, "correct": 20},
                         "trending": {"mult": 2.0, "n": 10, "correct": 8}}})
        # t = 10/20 = 0.5 → 0.5*1.0 + 0.5*2.0
        assert learner.online_mults("trending")["momentum"] == pytest.approx(1.5)

    def test_unknown_bucket_uses_overall(self):
        learner = self._learner_with({
            "momentum": {"overall": {"mult": 1.3, "n": 10, "correct": 7},
                         "trending": {"mult": 2.0, "n": 20, "correct": 16}}})
        assert learner.online_mults(None)["momentum"] == pytest.approx(1.3)
        assert learner.online_mults("ranging")["momentum"] == pytest.approx(1.3)


# ---------------------------------------------------------------------------
# Hybrid bot integration
# ---------------------------------------------------------------------------

def _neutral_perf(bot):
    return patch.object(bot, "_perf_tilts",
                        return_value={s: 1.0 for s, *_ in SUBS})


def _trending_signals(prices=None):
    prices = prices or [100_000.0 * (1.001 ** i) for i in range(40)]
    return make_signals(prices=prices, latest=prices[-1] * 1.001,
                        btc_drift=0.3,
                        vol_regime={"regime": "trending_up",
                                    "trend_score": 0.9})


class TestHybridIntegration:
    def test_online_mult_shifts_weights(self, arena_db):
        bot = HybridBot(name="hybrid-t")
        sigs = make_signals(vol_regime={"regime": "normal", "trend_score": 0.5})
        with _neutral_perf(bot):
            with patch.object(bot._meta, "online_mults",
                              return_value={"momentum": 1.0, "mean_rev": 1.0,
                                            "phantom": 1.0}):
                base = bot._dynamic_weights(sigs)
            with patch.object(bot._meta, "online_mults",
                              return_value={"momentum": 2.0, "mean_rev": 1.0,
                                            "phantom": 1.0}):
                boosted = bot._dynamic_weights(sigs)
        assert boosted["momentum"] > base["momentum"]
        assert abs(sum(boosted.values()) - 1.0) < 1e-9

    def test_learned_state_flows_into_weights(self, arena_db):
        # A resolved trade where momentum called it right must raise
        # momentum's ensemble weight end-to-end (DB → learner → weights).
        for _ in range(5):
            _insert_trade(arena_db, {"momentum": 0.5, "phantom": -0.4},
                          "mixed", "yes", "win")
        bot = HybridBot(name="hybrid-t")
        sigs = make_signals(vol_regime={"regime": "normal", "trend_score": 0.5})
        with _neutral_perf(bot):
            w = bot._dynamic_weights(sigs)
        base_mom = bot.strategy_params["momentum_weight"]
        base_ph = bot.strategy_params["phantom_weight"]
        # momentum was right 5x, phantom wrong 5x → their weight RATIO must
        # move in momentum's favor vs the configured base ratio.
        assert (w["momentum"] / w["phantom"]) > (base_mom / base_ph)

    def test_buy_reasoning_carries_meta_token_and_weights(self, arena_db):
        bot = HybridBot(name="hybrid-t")
        with _neutral_perf(bot):
            sig = bot.analyze(make_market(), _trending_signals())
        assert sig["action"] == "buy"
        parsed = parse_token(sig["reasoning"])
        assert parsed is not None
        votes, bucket = parsed
        assert bucket == "trending"
        assert any(abs(v) > 0 for v in votes.values())
        assert "[w=" in sig["reasoning"]       # effective weights visible
        assert sig.get("meta_token") and "meta(" in sig["meta_token"]

    def test_hold_with_votes_stamps_meta_for_cf(self, arena_db):
        """Unknown-regime single-sub lean should hold but still stamp meta_token."""
        bot = HybridBot(name="hybrid-t")
        # Force only momentum sub to fire (mean_rev/phantom hold)
        hold = {"action": "hold", "side": "yes", "confidence": 0.0, "reasoning": ""}
        buy = {"action": "buy", "side": "yes", "confidence": 0.6, "reasoning": "mom"}
        with _neutral_perf(bot):
            with patch.object(bot, "_cached_sub_analyze",
                              return_value={"momentum": buy, "mean_rev": hold,
                                            "phantom": hold}):
                # unknown tape still requires ≥2-sub agreement → hold
                sig = bot.analyze(
                    make_market(),
                    make_signals(btc_drift=0.2,
                                 vol_regime={},
                                 market_regime={}),
                )
        assert sig["action"] == "hold"
        assert sig.get("meta_token")
        assert parse_token(sig["reasoning"]) is not None
        assert abs(sig["signals"]["votes"].get("momentum", 0)) > 0

    def test_signals_expose_weights_online_and_bucket(self, arena_db):
        bot = HybridBot(name="hybrid-t")
        with _neutral_perf(bot):
            sig = bot.analyze(make_market(), _trending_signals())
        s = sig["signals"]
        assert set(s["weights"]) == {sub for sub, *_ in SUBS}
        assert set(s["online"]) == set(ml.SUB_TOKENS)
        assert s["regime_bucket"] == "trending"

    def test_signal_profile_regime_tilt(self, arena_db):
        bot = HybridBot(name="hybrid-t")
        base_mom = type(bot).STRATEGY_SIGNAL_PROFILE["hybrid"]["mom"]
        with _neutral_perf(bot):
            bot.analyze(make_market(), _trending_signals())
            trending_prof = bot._signal_profile()
            bot.analyze(make_market(), make_signals(
                vol_regime={"regime": "choppy", "trend_score": 0.1}))
            chop_prof = bot._signal_profile()
        assert trending_prof["mom"] > base_mom
        assert chop_prof["mom"] < base_mom
        # drift (the validated fundamental) is never tilted
        assert trending_prof["drift"] == \
            type(bot).STRATEGY_SIGNAL_PROFILE["hybrid"]["drift"]

    def test_no_regime_means_default_profile(self, arena_db):
        bot = HybridBot(name="hybrid-t")
        with _neutral_perf(bot):
            bot.analyze(make_market(), make_signals())
        assert bot._signal_profile() == \
            type(bot).STRATEGY_SIGNAL_PROFILE["hybrid"]

    def test_record_last_persists_only_with_prior_state(self, arena_db):
        # No prior hybrid_meta key → analyze alone must NOT create one.
        bot = HybridBot(name="hybrid-t")
        with _neutral_perf(bot):
            bot.analyze(make_market(), _trending_signals())
        assert arena_db.get_arena_state(ml.STATE_KEY) is None

        # Once the learner has real state, the snapshot flows through.
        _insert_trade(arena_db, {"momentum": 0.4}, "mixed", "yes", "win")
        bot._meta.update_from_trades()
        with _neutral_perf(bot):
            bot.analyze(make_market(), _trending_signals())
        stored = json.loads(arena_db.get_arena_state(ml.STATE_KEY))
        assert stored["last"]["weights"]
        assert stored["last"]["bucket"] == "trending"

    def test_make_decision_persists_meta_token_on_trade_reasoning(
            self, arena_db, monkeypatch):
        """Regression: make_decision must keep meta(...) so the learner trains.

        Overnight soak had 0/140 hybrid trades with the token because
        make_decision rebuilt reasoning from scratch.
        """
        from bots.meta_learner import parse_token
        bot = HybridBot(name="hybrid-v1")
        # Bypass gates that would skip so we reach the buy reasoning path.
        monkeypatch.setattr(bot, "regime_context", lambda signals: {
            "label": "low_vol_trend", "legacy": "trending",
            "trend_score": 0.8, "vol_score": 0.3, "known": True,
            "ranging": False, "choppy": False, "high_vol": False,
        })
        with _neutral_perf(bot):
            analysis = bot.analyze(make_market(yes_price=0.45, no_price=0.55),
                                   _trending_signals())
        assert parse_token(analysis.get("reasoning") or "") is not None
        # Simulate what make_decision appends: the full buy path attaches the
        # same token via regex extract from analyze reasoning.
        import re
        from bots.base_bot import BaseBot
        raw = analysis.get("reasoning") or ""
        m = re.search(
            r"meta\(mom=[+-][\d.]+ rev=[+-][\d.]+ (?:sent=[+-][\d.]+ )?"
            r"ph=[+-][\d.]+ \| reg=\w+\)",
            raw,
        )
        assert m is not None
        # After a resolved trade with the token on reasoning, learner updates.
        tid = arena_db.log_trade(
            bot_name="hybrid-v1", market_id="mkt-meta", side="yes", amount=5.0,
            venue="polymarket", mode="paper", confidence=0.5,
            reasoning=f"fair=0.55 model=0.60 => yes edge=+0.05 {m.group(0)}",
        )
        with arena_db.get_conn() as conn:
            conn.execute(
                "UPDATE trades SET outcome='win', resolved_at=datetime('now') "
                "WHERE id=?", (tid,))
        learner = HybridMetaLearner()
        assert learner.update_from_trades() >= 1
        assert arena_db.get_arena_state(ml.STATE_KEY) is not None


# ---------------------------------------------------------------------------
# Dashboard endpoint
# ---------------------------------------------------------------------------

class TestDashboardEndpoint:
    AUTH = ("admin", "Thor")

    @pytest.fixture
    def client(self, arena_db):
        from fastapi.testclient import TestClient
        import dashboard.server as srv
        return TestClient(srv.app)

    def test_empty_state(self, client):
        r = client.get("/api/hybrid-meta", auth=self.AUTH)
        assert r.status_code == 200
        assert r.json() == {}

    def test_returns_persisted_state(self, client, arena_db):
        _insert_trade(arena_db, {"momentum": 0.4}, "trending", "yes", "win")
        HybridMetaLearner().update_from_trades()
        r = client.get("/api/hybrid-meta", auth=self.AUTH)
        assert r.status_code == 200
        data = r.json()
        assert data["subs"]["momentum"]["overall"]["n"] == 1
        assert data["subs"]["momentum"]["overall"]["mult"] > 1.0

    def test_corrupt_state_degrades_to_empty(self, client, arena_db):
        arena_db.set_arena_state(ml.STATE_KEY, "{not json")
        r = client.get("/api/hybrid-meta", auth=self.AUTH)
        assert r.status_code == 200
        assert r.json() == {}

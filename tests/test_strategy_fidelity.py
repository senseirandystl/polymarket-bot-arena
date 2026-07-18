"""Strategy fidelity (BUG #27, part 5): each bot trades ITS OWN thesis.

Live evidence: momentum/phantom/hybrid were near-clones (drift .40-.50 with
identical warm inputs -> P_model within +/-0.03 across bots in the same
second -> 4-bot tandem clusters). With the pm/obi/cvd lanes killed, honest
differentiation comes from which LIVE input dominates:

  momentum  — BTC short-term trend (mom lane + its analyze() trend read)
  phantom   — EMA-crossover/breakout swing (analyze()-dominant)
  meanrev   — fundamentals + fade: drift anchor + z-score reversion thesis
              ("buy the dip in the winning direction")
  hybrid    — balanced ensemble
  sentiment — in-market flow (pm/cvd analyze; lanes stay killed until
              validated)

The strat lane now carries a PER-STRATEGY profile weight (it was a flat
global 0.15 — too small to differentiate). The meanrev slate bot is renamed
meanrev-sl25-v1 -> meanrev-v1 (stop-loss long gone), strategy_type
mean_reversion_sl -> mean_reversion via idempotent DB migration.
"""

import pytest

from bots.base_bot import BaseBot
from bots.bot_mean_rev import MeanRevBot


PROF = BaseBot.STRATEGY_SIGNAL_PROFILE


def test_profiles_include_strat_weight():
    for stype in ("momentum", "phantom", "mean_reversion", "hybrid"):
        assert "strat" in PROF[stype], stype
        assert PROF[stype]["strat"] > 0.0, stype


def test_momentum_is_momentum_dominant():
    p = PROF["momentum"]
    assert p["mom"] > p["drift"]
    assert p["mom"] >= 0.40


def test_phantom_is_thesis_dominant():
    p = PROF["phantom"]
    assert p["strat"] >= p["mom"] > 0
    assert p["strat"] >= 0.40


def test_meanrev_is_drift_plus_fade():
    p = PROF["mean_reversion"]
    assert p["drift"] >= 0.60
    assert p["mom"] == 0.0
    assert p["strat"] > 0.0


def test_live_profiles_are_distinct():
    # Pairwise L1 distance over live lanes must be material — no near-clones.
    live = ("drift", "mom", "strat")
    types = ("momentum", "phantom", "mean_reversion", "hybrid")
    for i, a in enumerate(types):
        for b in types[i + 1:]:
            dist = sum(abs(PROF[a][k] - PROF[b][k]) for k in live)
            assert dist >= 0.25, (a, b, dist)


def test_strat_lane_uses_profile_weight():
    # A bot whose analyze() emits a strong thesis should get a materially
    # different model_prob than one whose profile ignores the strat lane.
    bot = MeanRevBot(name="mr-test", generation=0)
    lanes = {"drift": 0.0, "mom": 0.0, "pm": 0.0, "cvd": 0.0, "obi": 0.0,
             "strat": 0.8, "learn": 0.0}
    p = bot._model_prob_yes(lanes)
    w = PROF["mean_reversion"]["strat"]
    assert p == pytest.approx(0.5 + 0.5 * w * 0.8, abs=1e-6)


# --- meanrev rename ---

def test_meanrev_default_name():
    assert MeanRevBot().name == "meanrev-v1"


def test_default_slate_uses_meanrev_v1():
    from arena.startup import build_default_bots
    names = [b.name for b in build_default_bots()]
    assert "meanrev-v1" in names
    assert not any("sl25" in n for n in names)
    assert len(names) == 7


def test_db_migration_renames_meanrev(tmp_path, monkeypatch):
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "mig_test.db")
    db_module.init_db()
    with db_module.get_conn() as conn:
        conn.execute(
            "INSERT INTO bot_configs (bot_name, strategy_type, generation, lineage, params, active) "
            "VALUES ('meanrev-sl25-v1', 'mean_reversion_sl', 0, 'meanrev-sl25-v1', '{}', 1)")
    db_module.log_trade("meanrev-sl25-v1", "m1", "yes", 5.0,
                        venue="polymarket", mode="paper")
    db_module.init_db()  # migration is idempotent, runs on init
    with db_module.get_conn() as conn:
        cfg = conn.execute("SELECT bot_name, strategy_type FROM bot_configs").fetchall()
        assert [tuple(r) for r in cfg] == [("meanrev-v1", "mean_reversion")]
        tr = conn.execute("SELECT DISTINCT bot_name FROM trades").fetchall()
        assert [r[0] for r in tr] == ["meanrev-v1"]

"""Dual-window regime stats + strategy×side cells."""

from datetime import datetime, timedelta, timezone

import pytest

from arena import regime_stats as rs


def _row(bot, side, outcome, pnl, regime, hours_ago, created=None):
    now = datetime.now(timezone.utc)
    ts = created or (now - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "bot_name": bot,
        "side": side,
        "outcome": outcome,
        "pnl": pnl,
        "trade_features": f'["regime:{regime}"]',
        "created_at": ts,
    }


class FakeConn:
    def __init__(self, rows, smap=None):
        self.rows = rows
        self.smap = smap or {
            "momentum-v1": "momentum",
            "sniper-v1": "sniper",
        }
        self._result = []

    def execute(self, sql, params=None):
        if "bot_configs" in (sql or ""):
            self._result = [
                {"bot_name": k, "strategy_type": v}
                for k, v in self.smap.items()
            ]
        else:
            self._result = self.rows
        return self

    def fetchall(self):
        return list(self._result)

    def __iter__(self):
        return iter(self._result)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_dual_window_and_strategy_side(monkeypatch):
    rs.invalidate_cache()
    # 5h ago: good YES for momentum (long only)
    # 1h ago: bad YES for momentum (fast + long)
    rows = [
        _row("momentum-v1", "yes", "win", 3.0, "high_vol_chop", 5.0),
        _row("momentum-v1", "yes", "win", 3.0, "high_vol_chop", 4.0),
        _row("momentum-v1", "yes", "win", 3.0, "high_vol_chop", 3.5),
        _row("momentum-v1", "yes", "loss", -3.0, "high_vol_chop", 1.0),
        _row("momentum-v1", "yes", "loss", -3.0, "high_vol_chop", 0.8),
        _row("momentum-v1", "yes", "loss", -3.0, "high_vol_chop", 0.5),
        _row("momentum-v1", "yes", "loss", -3.0, "high_vol_chop", 0.3),
        _row("momentum-v1", "yes", "loss", -3.0, "high_vol_chop", 0.1),
        # NO still fine recently
        _row("momentum-v1", "no", "win", 2.0, "high_vol_chop", 0.5),
        _row("momentum-v1", "no", "win", 2.0, "high_vol_chop", 0.2),
        _row("sniper-v1", "yes", "loss", -2.0, "high_vol_chop", 0.4),
    ]
    monkeypatch.setattr(rs.db, "get_conn", lambda: FakeConn(rows))
    monkeypatch.setattr(rs.config, "REGIME_STATS_FAST_HOURS", 2.5, raising=False)
    monkeypatch.setattr(rs.config, "REGIME_STATS_LOOKBACK_HOURS", 72.0, raising=False)

    blob = rs.snapshot(force=True)
    cell = blob["by_strategy"]["high_vol_chop"]["momentum"]
    assert cell["n"] == 10  # 8 yes + 2 no
    assert cell["fast_n"] == 7  # 5 yes loss + 2 no win in last 2.5h
    assert cell["fast_wr"] is not None
    assert cell["fast_wr"] < cell["wr"]  # recent worse than full window

    yes = blob["by_strategy_side"]["high_vol_chop"]["momentum"]["yes"]
    no = blob["by_strategy_side"]["high_vol_chop"]["momentum"]["no"]
    assert yes["fast_n"] == 5
    assert yes["fast_wr"] == 0.0
    assert no["fast_n"] == 2
    assert no["fast_wr"] == 1.0

    # Helper
    sc = rs.strategy_side_regime_cell("high_vol_chop", "momentum", "yes")
    assert sc["fast_n"] == 5


def test_effective_wr_prefers_fast_blend(monkeypatch):
    cell = {
        "n": 40, "wins": 28, "wr": 0.70, "pnl": 20.0,
        "fast_n": 10, "fast_wins": 3, "fast_wr": 0.30, "fast_pnl": -15.0,
    }
    wr = rs.effective_wr(cell, min_n_fast=8, min_n_long=18, fast_blend=0.65)
    # 0.65*0.30 + 0.35*0.70 = 0.195+0.245 = 0.44
    assert wr == pytest.approx(0.44, abs=1e-6)


def test_is_toxic_fast_path():
    cell = {
        "n": 50, "wins": 30, "wr": 0.60, "pnl": 10.0,  # long healthy
        "fast_n": 12, "fast_wins": 3, "fast_wr": 0.25, "fast_pnl": -20.0,
    }
    assert rs.is_toxic_cell(cell, path="long") is False
    assert rs.is_toxic_cell(
        cell, path="fast", min_n=10, wr_bar=0.38, require_neg_pnl=True
    ) is True
    assert rs.is_toxic_cell(cell, path="either", min_n=10, wr_bar=0.38) is True

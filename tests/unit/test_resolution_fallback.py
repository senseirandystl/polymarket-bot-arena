"""Resolution helpers: extreme prices + direct Gamma fallback for stuck markets."""

from datetime import datetime, timezone, timedelta
from unittest import mock

import polymarket_markets as pm


class TestOutcomeFromPrices:
    def test_up_wins_extreme(self):
        assert pm.outcome_from_prices(["1", "0"]) is True
        assert pm.outcome_from_prices(["0.9995", "0.0005"]) is True
        assert pm.outcome_from_prices('["1","0"]') is True

    def test_down_wins_extreme(self):
        assert pm.outcome_from_prices(["0", "1"]) is False
        assert pm.outcome_from_prices(["0.0005", "0.9995"]) is False

    def test_not_decided(self):
        assert pm.outcome_from_prices(["0.55", "0.45"]) is None
        assert pm.outcome_from_prices(["0.90", "0.10"]) is None
        assert pm.outcome_from_prices(None) is None
        assert pm.outcome_from_prices([]) is None


class TestEndIsPast:
    def test_grace_window(self):
        now = datetime(2026, 8, 5, 15, 1, 0, tzinfo=timezone.utc)
        end = "2026-08-05T15:00:00Z"
        assert pm._end_is_past(end, now=now, grace_sec=120) is False
        later = datetime(2026, 8, 5, 15, 3, 0, tzinfo=timezone.utc)
        assert pm._end_is_past(end, now=later, grace_sec=120) is True


class TestFetchMarketOutcome:
    def _resp(self, payload, status=200):
        r = mock.Mock()
        r.status_code = status
        r.json.return_value = payload
        return r

    def test_closed_market_returns_outcome(self):
        payload = [{
            "conditionId": "0xabc",
            "closed": True,
            "endDate": "2026-08-05T15:00:00Z",
            "outcomePrices": '["0.9995","0.0005"]',
        }]
        with mock.patch.object(pm.http_client, "get",
                               return_value=self._resp(payload)):
            assert pm.fetch_market_outcome("0xabc") is True

    def test_open_live_market_not_resolved_early(self):
        # Extreme prices but end still in the future → do not settle.
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        payload = [{
            "conditionId": "0xlive",
            "closed": False,
            "endDate": future,
            "outcomePrices": '["0.999","0.001"]',
        }]
        with mock.patch.object(pm.http_client, "get",
                               return_value=self._resp(payload)):
            assert pm.fetch_market_outcome("0xlive") is None

    def test_defacto_past_end_not_closed(self):
        # The stuck-resolver class: end long past, prices extreme, still open.
        past = (datetime.now(timezone.utc) - timedelta(minutes=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        payload = [{
            "conditionId": "0xstuck",
            "closed": False,
            "endDate": past,
            "outcomePrices": '["0.0005","0.9995"]',
        }]
        with mock.patch.object(pm.http_client, "get",
                               return_value=self._resp(payload)):
            assert pm.fetch_market_outcome("0xstuck") is False

    def test_missing_market(self):
        with mock.patch.object(pm.http_client, "get",
                               return_value=self._resp([])):
            assert pm.fetch_market_outcome("0xnope") is None


class TestResolverFallback:
    def test_cycle_uses_direct_lookup_for_missing(self, tmp_path, monkeypatch):
        import db
        from arena.resolver import TradeResolver

        monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
        db.init_db()
        # Seed one pending paper trade.
        with db.get_conn() as conn:
            conn.execute(
                """INSERT INTO trades
                   (bot_name, market_id, market_question, side, amount,
                    confidence, reasoning, shares_bought, fee, fill_source,
                    entry_price, mode, venue)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("momentum-v1", "0xstuck", "BTC 5m", "yes", 2.0,
                 0.5, "test", 5.0, 0.0, "paper_sim", 0.4, "paper",
                 "polymarket"),
            )
            conn.commit()

        monkeypatch.setattr(pm, "recent_resolutions", lambda limit=100: {})
        monkeypatch.setattr(pm, "fetch_market_outcome", lambda mid: True)

        r = TradeResolver()
        r._do_resolution_cycle()

        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT outcome, pnl FROM trades WHERE market_id='0xstuck'"
            ).fetchone()
        assert row["outcome"] == "win"
        # win: shares - amount - fee = 5 - 2 - 0 = 3
        assert abs(float(row["pnl"]) - 3.0) < 1e-9


    def test_cycle_stamps_decision_events_without_pending_trades(
            self, tmp_path, monkeypatch):
        """Signal Lab must resolve skips even when no trade was placed."""
        import db
        from arena.resolver import TradeResolver

        monkeypatch.setattr(db, "DB_PATH", tmp_path / "dec.db")
        db.init_db()
        with db.get_conn() as conn:
            conn.execute(
                """INSERT INTO decision_events
                   (bot_name, strategy_type, market_id, action, side,
                    skip_reason, drift, created_at)
                   VALUES (?,?,?,?,?,?,?,datetime('now','-10 minutes'))""",
                ("momentum-v1", "momentum", "0xdone", "skip", "yes",
                 "weak_lean", 0.1),
            )
            conn.commit()

        monkeypatch.setattr(pm, "recent_resolutions",
                            lambda limit=100: {"0xdone": True})
        monkeypatch.setattr(pm, "fetch_market_outcome", lambda mid: None)

        r = TradeResolver()
        r._do_resolution_cycle()

        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT market_up, would_win FROM decision_events "
                "WHERE market_id='0xdone'"
            ).fetchone()
        assert int(row["market_up"]) == 1
        # Entry-less skips intentionally leave would_win NULL (phase3).
        assert row["would_win"] is None

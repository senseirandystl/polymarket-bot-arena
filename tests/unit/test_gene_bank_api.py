"""Dashboard /api/ga includes Elite Gene Bank payload."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_api_ga_includes_gene_bank():
    import dashboard.server as server

    # Storage order is oldest → newest (as load_bank returns).
    fake_entries = [
        {
            "name": "momentum-g1-200",
            "strategy_type": "momentum",
            "generation": 1,
            "cycle": 1,
            "fitness": 0.70,
            "pnl": 5.0,
            "win_rate": 0.55,
            "trades": 40,
            "params": {"lookback_candles": 9},
            "lineage": None,
            "source": "elite",
        },
        {
            "name": "meanrev-g2-100",
            "strategy_type": "mean_reversion",
            "generation": 2,
            "cycle": 2,
            "fitness": 0.88,
            "pnl": 18.0,
            "win_rate": 0.61,
            "trades": 35,
            "params": {"min_drift": 0.12, "bb_std_dev": 2.1},
            "lineage": "a+b -> meanrev-g2-100",
            "source": "elite",
        },
    ]

    with patch("db.get_ga_status", return_value={}), \
         patch("db.get_ga_history", return_value=[]), \
         patch("evolution.gene_bank.load_bank", return_value=list(fake_entries)), \
         patch("evolution.gene_bank._max_size", return_value=20):
        client = TestClient(server.app)
        # /api/ga may require auth — match other dashboard tests
        r = client.get("/api/ga", auth=("admin", "Thor"))
        if r.status_code == 401:
            r = client.get("/api/ga")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "gene_bank" in data
        gb = data["gene_bank"]
        assert gb["count"] == 2
        assert gb["max_size"] == 20
        # Newest first (reversed from storage order)
        assert gb["entries"][0]["name"] == "meanrev-g2-100"
        assert gb["entries"][1]["name"] == "momentum-g1-200"
        assert "min_drift" in gb["entries"][0]["params"]
        assert data["config"].get("gene_bank_size") == 20
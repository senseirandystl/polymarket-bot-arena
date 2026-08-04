"""Portfolio explore budget — cold bots share a capped total mass."""

from unittest import mock

from arena import portfolio


def test_cold_bots_share_explore_budget_not_full_mass():
    names = [
        "hybrid-v1",
        "momentum-g14-974",
        "late-window-maker-v1",
        "fee-zone-maker-v1",
        "arbitrage-v1",
    ]
    metrics = {
        "hybrid-v1": {
            "n": 50, "sharpe": 0.2, "expectancy": 0.1,
            "total_pnl": 10.0, "variance": 2.0, "ready": True,
        },
        "momentum-g14-974": {
            "n": 5, "sharpe": 0.0, "expectancy": 0.0,
            "total_pnl": 0.0, "variance": 1.0, "ready": False,
        },
        "late-window-maker-v1": {
            "n": 3, "sharpe": 0.0, "expectancy": 0.0,
            "total_pnl": 0.0, "variance": 1.0, "ready": False,
        },
        "fee-zone-maker-v1": {
            "n": 2, "sharpe": 0.0, "expectancy": 0.0,
            "total_pnl": 0.0, "variance": 1.0, "ready": False,
        },
        "arbitrage-v1": {
            "n": 20, "sharpe": 0.1, "expectancy": 0.3,
            "total_pnl": 8.0, "variance": 5.0, "ready": True,
        },
    }
    with mock.patch.object(portfolio, "compute_metrics", return_value=metrics), \
         mock.patch.object(portfolio, "_market_returns_by_bot", return_value={
             n: {} for n in names
         }), \
         mock.patch.object(portfolio, "_is_new_generation_bot", side_effect=
             lambda name, m=None: "g" in name and name.split("-g")[0] or
             ("-g" in name)):
        # Simpler: treat any gN name as explorer
        def _is_new(name, metrics_row=None):
            return "-g" in name or not (metrics.get(name) or {}).get("ready")

        with mock.patch.object(portfolio, "_is_new_generation_bot", _is_new):
            result = portfolio.allocate(names, method="kelly_portfolio")

    w = result["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-3
    # Cold explorers total mass ≤ explore budget (~0.12) + arb lock
    cold = w.get("momentum-g14-974", 0) + w.get("late-window-maker-v1", 0) + \
        w.get("fee-zone-maker-v1", 0)
    assert cold <= 0.20 + 1e-3, cold
    # Hybrid (ready winner) must get meaningful weight — not zeroed
    assert w.get("hybrid-v1", 0) >= 0.05, w

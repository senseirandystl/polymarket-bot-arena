from arena.regime_map import shrink, attribute


def test_shrink_thin_cell_pulls_to_prior():
    # 2 samples, k=40 -> estimate dominated by prior
    est = shrink(cell_mean=10.0, cell_n=2, prior_mean=0.0, k=40.0)
    assert abs(est - (2 * 10.0 + 40 * 0.0) / 42) < 1e-9
    assert est < 1.0  # strongly pulled toward prior


def test_shrink_rich_cell_trusts_itself():
    est = shrink(cell_mean=10.0, cell_n=400, prior_mean=0.0, k=40.0)
    assert est > 9.0  # mostly its own mean


def test_attribute_groups_by_cell_and_bot():
    trades = [
        {"bot_name": "a", "pnl": 2.0, "cell": ("r", 2, 3, "us", 0, 0)},
        {"bot_name": "a", "pnl": 4.0, "cell": ("r", 2, 3, "us", 0, 0)},
        {"bot_name": "b", "pnl": -1.0, "cell": ("r", 2, 3, "us", 0, 0)},
    ]
    out = attribute(trades, k=40.0)
    cell = ("r", 2, 3, "us", 0, 0)
    assert out[cell]["n"] == 3
    assert out[cell]["bots"]["a"]["n"] == 2
    assert "shrunk_pnl" in out[cell]["bots"]["a"]

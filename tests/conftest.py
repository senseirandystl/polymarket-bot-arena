"""Shared pytest fixtures for the Polymarket Bot Arena test suite.

Keeps the repo root importable regardless of how pytest is invoked, and
provides the canonical sample objects (markets, order books, signal dicts,
isolated DB) that unit and integration tests build on.
"""

import sys
from pathlib import Path

import pytest

# Repo root on sys.path so `import config`, `import db`, `from bots ...`
# work whether pytest is run bare or via `python -m pytest`, from any cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_collection_modifyitems(items):
    """Auto-mark tests by directory so `pytest -m unit` / `-m integration` work."""
    for item in items:
        parts = Path(str(item.fspath)).parts
        if "unit" in parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in parts:
            item.add_marker(pytest.mark.integration)


# ---------------------------------------------------------------------------
# Isolated database
# ---------------------------------------------------------------------------

@pytest.fixture()
def arena_db(tmp_path, monkeypatch):
    """A fresh SQLite DB in tmp_path — never touches the real bot_arena.db."""
    import db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test_arena.db")
    db_module.init_db()
    return db_module


# ---------------------------------------------------------------------------
# Sample market dicts (shape produced by polymarket_markets discovery +
# the market-data warmer laying warm book prices onto the dict)
# ---------------------------------------------------------------------------

def make_market(yes_price=0.55, time_remaining=150, market_id="mkt-test-1",
                **overrides):
    market = {
        "id": market_id,
        "condition_id": market_id,
        "question": "Bitcoin Up or Down - test window",
        "current_price": yes_price,
        "no_price": round(1.0 - yes_price, 4),
        "yes_ask": round(yes_price + 0.01, 4),
        "no_ask": round(1.0 - yes_price + 0.01, 4),
        "polymarket_token_id": "tok-yes",
        "polymarket_no_token_id": "tok-no",
        "time_remaining_seconds": time_remaining,
        "resolves_at": None,
    }
    market.update(overrides)
    return market


@pytest.fixture()
def sample_market():
    """A live-window market with YES mid at 0.55 and 150s remaining."""
    return make_market()


@pytest.fixture()
def cheap_market():
    """A market with YES deeply out of favor (consensus-guard territory)."""
    return make_market(yes_price=0.25)


# ---------------------------------------------------------------------------
# Sample order books (normalized shape from polymarket_markets.get_order_book:
# asks best-first as (price, size) tuples, `valid` flag, min_order_size)
# ---------------------------------------------------------------------------

def make_book(asks=None, bids=None, min_size=5.0):
    asks = asks if asks is not None else [(0.56, 50), (0.57, 100), (0.60, 200)]
    bids = bids if bids is not None else [(0.54, 50), (0.53, 100), (0.50, 200)]
    return {
        "valid": True,
        "asks": asks,
        "bids": bids,
        "best_ask": asks[0][0] if asks else None,
        "best_bid": bids[0][0] if bids else None,
        "min_order_size": min_size,
    }


@pytest.fixture()
def sample_order_book():
    """A healthy book: 50 shares at 56c, deeper liquidity behind."""
    return make_book()


@pytest.fixture()
def thin_order_book():
    """A nearly-empty book: 2 shares of depth total."""
    return make_book(asks=[(0.56, 2)], bids=[(0.54, 2)])


# ---------------------------------------------------------------------------
# Sample signal dicts (shape produced by arena/signals.build_combined_signals)
# ---------------------------------------------------------------------------

def make_signals(**overrides):
    # btc_drift_pct required by the TWAP dual drift gate (min 0.00030).
    # 0.001 = 0.1% moneyness at $100k BTC, well above the floor.
    base = {
        "prices": [100_000.0] * 60,
        "latest": 100_000.0,
        "volumes": [10.0] * 60,
        "orderflow": {},
        "pm_momentum": 0.0,
        "obi": 0.0,
        "cvd": 0.0,
        "btc_drift": 0.0,
        "btc_drift_pct": 0.001,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def neutral_signals():
    """Flat tape: no drift, no momentum, no flow."""
    return make_signals()


@pytest.fixture()
def bullish_signals():
    """Strong up-drift + rising BTC tape."""
    prices = [100_000.0 + i * 40 for i in range(60)]
    return make_signals(prices=prices, latest=prices[-1], btc_drift=0.45,
                       pm_momentum=0.002, cvd=0.4)


@pytest.fixture()
def bearish_signals():
    """Strong down-drift + falling BTC tape."""
    prices = [100_000.0 - i * 40 for i in range(60)]
    return make_signals(prices=prices, latest=prices[-1], btc_drift=-0.45,
                       pm_momentum=-0.002, cvd=-0.4)

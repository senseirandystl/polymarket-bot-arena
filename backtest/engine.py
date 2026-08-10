"""Backtest engine — replay resolved markets through the real decision path.

For each market window, the engine steps decision ticks (default one per
1-min candle, matching the Signal Lab harness's decision points), rebuilds
the ``market`` + ``signals`` dicts exactly as ``arena/signals.py`` shapes
them for the live trader, and calls each bot's REAL ``make_decision``. Buys
are filled by :mod:`backtest.broker` (depth-walked synthetic book, taker
fee, slippage band, shared-pool cap) and settled against the true outcome.

Supported bots: the directional strategies + sniper. The arbitrage and maker
bots override ``execute`` against the live warm-book store and are excluded
(their edge is microstructure the historical record does not carry).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import config
from signals import technicals, volatility_regime
from signals.macro_calendar import macro_caution
from signals.strike import drift_signal
from datetime import datetime, timezone

from backtest.broker import BacktestBroker
from backtest.data import HistoricalData
from backtest.runtime import patched_runtime, silence_perf_cache

logger = logging.getLogger("backtest.engine")

# Strategy types whose decision path is replayable offline.
SUPPORTED_STRATEGIES = {
    "momentum", "mean_reversion", "mean_reversion_sl", "mean_reversion_tp",
    "phantom", "hybrid", "sniper", "lag_residual", "regime_specialist", "no_lag",
    "sweeper",
}


@dataclass
class BacktestResult:
    """Everything a report needs from one run."""
    trades: list                       # resolved BacktestTrade rows
    skips: dict                        # reason-prefix -> count
    rejects: dict                      # broker reject reason -> count
    samples: list = field(default_factory=list)  # per-tick lane observations
    markets_replayed: int = 0
    decisions: int = 0
    initial_bankroll: float = 0.0
    final_bankroll: float = 0.0
    equity_curve: list = field(default_factory=list)  # [(close_ts, equity)]
    config_snapshot: dict = field(default_factory=dict)


def _skip_key(reasoning: str) -> str:
    """Collapse a skip reasoning string to a stable histogram key."""
    text = (reasoning or "unknown").split(":", 1)[0].strip()
    return text[:40] or "unknown"


def _build_signals(prices: list, btc_now: float, drift: float,
                   strike: float | None, pm_mom: float, tick_dt) -> dict:
    """The signals dict make_decision expects (arena/signals.py shape).

    Killed lanes (obi/cvd/pm carry weight 0 live) are fed neutral values;
    fut/xasset are stale-neutral. vol_regime + technicals run the production
    compute off the reconstructed candle stream, so the quiet-regime momentum
    damp and any tech override behave exactly as live.
    """
    return {
        "prices": prices,
        "latest": btc_now,
        "volumes": [],
        "orderflow": {},
        "obi": 0.0,
        "cvd": 0.0,
        "btc_drift": drift,
        "btc_strike": strike,
        "vol_regime": volatility_regime.compute(prices),
        "technicals": technicals.compute(prices),
        "xasset": 0.0,
        "futures": {"funding": 0.0, "oi_delta": 0.0, "taker_delta": 0.0,
                    "stale": True},
        "macro_caution": macro_caution(tick_dt),
        "pm_momentum": pm_mom,
        "pm_prices": [],
    }


def run_backtest(bots: list, data: HistoricalData,
                 bankroll: float | None = None,
                 kelly_fraction: float | None = None,
                 lane_overrides: dict | None = None,
                 tick_sec: int | None = None,
                 compound: bool = False) -> BacktestResult:
    """Replay ``data`` through ``bots``. One trade per (bot, market), as live.

    ``compound=False`` (default) sizes every Kelly bet off the FIXED initial
    bankroll so per-trade sizes stay comparable across the run; ``True``
    compounds off the live pool like the real arena.
    """
    tick = int(tick_sec or config.BACKTEST_TICK_SEC)
    supported = [b for b in bots if b.strategy_type in SUPPORTED_STRATEGIES]
    dropped = [b.name for b in bots if b.strategy_type not in SUPPORTED_STRATEGIES]
    if dropped:
        logger.warning(f"Excluding non-replayable bots: {', '.join(dropped)} "
                       f"(arbitrage/makers need live book depth)")
    broker = BacktestBroker(bankroll, compound=compound)
    silence_perf_cache(supported)
    result = BacktestResult(
        trades=[], skips={}, rejects=broker.rejects,
        initial_bankroll=broker.initial_bankroll,
        final_bankroll=broker.initial_bankroll,
        config_snapshot={
            "tick_sec": tick,
            "compound": compound,
            "half_spread": config.BACKTEST_HALF_SPREAD,
            "kelly_fraction": kelly_fraction or config.KELLY_FRACTION,
            "bots": [b.name for b in supported],
            "lane_overrides": sorted((lane_overrides or {}).keys()),
        })

    with patched_runtime(broker, kelly_fraction, lane_overrides):
        for mkt in data.markets:
            pm = data.pm_prices.get(mkt.id) or []
            if not pm:
                continue
            strike = data.btc_opens.at(mkt.open_ts)
            if not strike:
                continue
            traded: set = set()
            window = mkt.close_ts - mkt.open_ts
            elapsed = tick
            while elapsed < window:
                ts = mkt.open_ts + elapsed
                time_rem = window - elapsed
                btc_now = data.btc_opens.at(ts)
                past = [(t, p) for t, p in pm if t <= ts]
                if not btc_now or not past:
                    elapsed += tick
                    continue
                yes_mid = float(past[-1][1])
                if not (0.01 < yes_mid < 0.99):
                    elapsed += tick
                    continue
                pm_mom = 0.0
                if len(past) >= 2:
                    pm_mom = float(past[-1][1]) - float(past[-2][1])
                prices = data.btc_closes.closes_until(ts, 60)
                # Keep adaptive drift scale in step with replay tape (same as
                # live build_combined_signals).
                try:
                    from signals.drift_scale import get_drift_scale_estimator
                    get_drift_scale_estimator().update_from_prices(prices or [])
                except Exception:
                    pass
                drift = drift_signal(strike, btc_now, time_rem)
                tick_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                signals = _build_signals(prices, btc_now, drift, strike,
                                         pm_mom, tick_dt)
                no_mid = round(1.0 - yes_mid, 4)
                market = {
                    "id": mkt.id,
                    "market_id": mkt.id,
                    "question": mkt.question,
                    "current_price": yes_mid,
                    "no_price": no_mid,
                    "yes_ask": round(min(0.99, yes_mid + config.BACKTEST_HALF_SPREAD), 4),
                    "no_ask": round(min(0.99, no_mid + config.BACKTEST_HALF_SPREAD), 4),
                    "time_remaining_seconds": time_rem,
                    "polymarket_token_id": mkt.up_token or "bt-yes",
                    "polymarket_no_token_id": "bt-no",
                }
                # One lane-observation per tick (bot-independent inputs) for
                # the per-signal contribution report.
                mom_raw = 0.0
                if len(prices) >= 2 and prices[-2] > 0:
                    mom_raw = (prices[-1] - prices[-2]) / prices[-2]
                result.samples.append({
                    "market_id": mkt.id, "time_remaining": time_rem,
                    "yes_mid": yes_mid, "yes_won": mkt.yes_won,
                    "drift": drift, "mom": mom_raw, "pm_mom": pm_mom,
                    "regime": signals["vol_regime"].get("regime"),
                })

                for bot in supported:
                    if bot.name in traded:
                        continue
                    result.decisions += 1
                    try:
                        decision = bot.make_decision(market, signals)
                    except Exception as e:
                        logger.exception(f"[{bot.name}] decision failed on "
                                         f"{str(mkt.id)[:12]}…: {e}")
                        continue
                    if decision.get("action") != "buy":
                        key = _skip_key(decision.get("reasoning", ""))
                        result.skips[key] = result.skips.get(key, 0) + 1
                        continue
                    side = decision["side"]
                    side_mid = yes_mid if side == "yes" else no_mid
                    trade = broker.place(
                        bot=bot, market_id=mkt.id, side=side,
                        side_mid=side_mid,
                        amount=float(decision.get("suggested_amount") or 0.0),
                        expected_price=decision.get("entry_price"),
                        confidence=float(decision.get("confidence") or 0.0),
                        entered_at=ts, time_remaining=time_rem,
                        context={
                            "drift": drift, "mom": mom_raw, "pm_mom": pm_mom,
                            "regime": signals["vol_regime"].get("regime"),
                            "side_mid": side_mid,
                            "reasoning": decision.get("reasoning", "")[:300],
                        })
                    if trade is not None:
                        traded.add(bot.name)
                elapsed += tick

            broker.resolve_market(mkt.id, mkt.yes_won)
            result.markets_replayed += 1
            result.equity_curve.append(
                (mkt.close_ts, broker.initial_bankroll + broker.realized_pnl))

    result.trades = list(broker.resolved_trades)
    result.final_bankroll = broker.initial_bankroll + broker.realized_pnl
    return result

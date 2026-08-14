"""Strategy fitness scores by regime — capital / GA routing (not skip).

``score(strategy, regime)`` combines live regime_performance-style stats
(when available via bot attribution) with structural seed fit so portfolio
and type_alloc can tilt toward strategies that work *in this regime* without
hard-skipping others (explore floor still applies at the portfolio layer).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import config
import db

logger = logging.getLogger("arena.regime_router")

# Structural prior: which strategy types "fit" which regimes (0..1 seed fit).
_SEED_FIT: dict[str, dict[str, float]] = {
    "low_vol_range": {
        "mean_reversion": 0.90, "mean_reversion_tp": 0.90,
        "lag_residual": 0.85, "hybrid": 0.70, "sniper": 0.55,
        "momentum": 0.45, "phantom": 0.50, "regime_specialist": 0.65,
        "no_lag": 0.80, "arbitrage": 0.60, "late_window_maker": 0.55,
        "fee_zone_maker": 0.55, "true_maker": 0.50,
    },
    "low_vol_trend": {
        "momentum": 0.75, "phantom": 0.70, "hybrid": 0.75, "sniper": 0.80,
        "mean_reversion": 0.55, "lag_residual": 0.70, "regime_specialist": 0.70,
    },
    "high_vol_trend": {
        "momentum": 0.90, "phantom": 0.85, "hybrid": 0.80, "sniper": 0.75,
        "mean_reversion": 0.40, "lag_residual": 0.55,
    },
    "high_vol_chop": {
        "mean_reversion": 0.70, "hybrid": 0.65, "sniper": 0.50,
        "momentum": 0.40, "phantom": 0.45, "lag_residual": 0.60,
    },
    "normal": {
        "momentum": 0.70, "phantom": 0.70, "hybrid": 0.75,
        "mean_reversion": 0.70, "sniper": 0.70, "lag_residual": 0.70,
    },
}


def _seed_fit(strategy_type: str, regime: str) -> float:
    block = _SEED_FIT.get(regime) or _SEED_FIT.get("normal") or {}
    if strategy_type in block:
        return float(block[strategy_type])
    return 0.60


def _bot_stats_for_strategy(
    strategy_type: str,
    regime: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregate metrics for bots of this type — prefer regime-local stats."""
    # Prefer regime_stats.by_strategy[regime][strategy] when available
    if regime and regime not in ("unknown",):
        try:
            from arena.regime_stats import snapshot
            by = (snapshot().get("by_strategy") or {}).get(regime) or {}
            cell = by.get(strategy_type) or {}
            n = int(cell.get("n") or 0)
            if n > 0:
                wr = cell.get("wr")
                if wr is None and cell.get("wins") is not None and n:
                    wr = float(cell["wins"]) / n
                return {
                    "n": n,
                    "wr": float(wr) if wr is not None else None,
                    "pnl": float(cell.get("pnl") or 0.0),
                    "source": "regime_stats",
                }
        except Exception:
            pass

    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                """SELECT bot_name FROM bot_configs
                   WHERE strategy_type=? AND active=1""",
                (strategy_type,),
            ).fetchall()
        names = [r["bot_name"] for r in rows]
    except Exception:
        names = []
    if not names:
        return {"n": 0, "wr": None, "pnl": 0.0}

    # Prefer portfolio metrics in arena_state
    try:
        raw = db.get_arena_state("portfolio_allocation")
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        metrics = (data or {}).get("metrics") or {}
        n = 0
        pnl = 0.0
        for name in names:
            m = metrics.get(name) or {}
            ni = int(m.get("n") or 0)
            n += ni
            pnl += float(m.get("total_pnl") or 0.0)
        if n > 0:
            return {"n": n, "wr": None, "pnl": pnl, "source": "portfolio"}
    except Exception:
        pass

    try:
        raw = db.get_arena_state("risk_engine")
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
        bots = (data or {}).get("bots") or {}
        n = 0
        pnl = 0.0
        for name in names:
            b = bots.get(name) or {}
            ni = int(b.get("n_window") or 0)
            n += ni
            pnl += float(b.get("window_pnl") or b.get("daily_pnl") or 0.0)
        return {"n": n, "wr": None, "pnl": pnl, "source": "risk"}
    except Exception:
        return {"n": 0, "wr": None, "pnl": 0.0}


def score(
    strategy_type: str,
    regime: Optional[str],
    *,
    live: Optional[dict] = None,
) -> float:
    """Return fitness score in ~[0, 1.5] for routing (higher = prefer).

    Combines seed fit with **regime-local** live WR/pnl when sample mass is
    adequate (falls back to global portfolio/risk metrics).
    """
    regime = str(regime or "unknown")
    if regime == "unknown":
        return 0.65
    seed = _seed_fit(strategy_type, regime)
    live = (
        live if live is not None
        else _bot_stats_for_strategy(strategy_type, regime=regime)
    )
    n = int((live or {}).get("n") or 0)
    min_n = int(getattr(config, "REGIME_ROUTER_MIN_TRADES", 12))
    pnl = float((live or {}).get("pnl") or 0.0)
    wr = (live or {}).get("wr")

    # Live tilt: mild until n is solid
    live_tilt = 0.0
    if n >= min_n:
        if wr is not None:
            live_tilt = (float(wr) - 0.50) * 0.8
        else:
            # PnL proxy: +$10 → ~+0.15, −$10 → −0.15 (soft)
            live_tilt = max(-0.25, min(0.25, pnl / 60.0))
        trust = min(1.0, n / (2.0 * min_n))
        live_tilt *= trust

    return max(0.05, min(1.5, 0.55 * seed + 0.45 * (0.65 + live_tilt)))


def scores_for_regime(regime: Optional[str],
                      strategy_types: list[str] | None = None) -> dict[str, float]:
    """Map strategy_type → score for the given regime."""
    if strategy_types is None:
        strategy_types = list((_SEED_FIT.get(str(regime or "normal")) or {}).keys())
        if not strategy_types:
            strategy_types = list((_SEED_FIT["normal"]).keys())
    return {st: score(st, regime) for st in strategy_types}


def boost_type_alloc_scores(
    base_scores: dict[str, float],
    regime: Optional[str],
    *,
    blend: float | None = None,
) -> dict[str, float]:
    """Blend GA type fitness scores with regime router scores."""
    blend = float(
        blend if blend is not None
        else getattr(config, "REGIME_ROUTER_GA_BLEND", 0.35)
    )
    if blend <= 0 or not regime or regime == "unknown":
        return dict(base_scores)
    out = {}
    for st, base in base_scores.items():
        r = score(st, regime)
        # base may be unnormalized fitness; blend multiplicatively soft
        out[st] = float(base) * ((1.0 - blend) + blend * r)
    return out

"""Genes that must not be mutated by the GA.

Lane weights and signal-blend knobs are owned by the core-lane tuner / Signal
Lab — evolution manages the *roster and strategy params*, not the model blend.
Also freeze params live data has shown do not pay (or are obsolete with the
Chainlink BTC feed, e.g. volume-dependent knobs on a volume-less oracle).
"""

from __future__ import annotations

from typing import Any

import config

# Global freeze list (any strategy).
_DEFAULT_FROZEN = frozenset({
    # Signal blend — never evolve these if they leak into strategy_params
    "signal_weight_drift", "signal_weight_mom", "signal_weight_strat",
    "signal_weight_pm", "signal_weight_cvd", "signal_weight_obi",
    "signal_weight_fut", "signal_weight_tech", "signal_weight_xasset",
    # Volume is empty for Chainlink BTC; volume_weight only adds noise
    "volume_weight",
    # Kelly / pool sizing is dashboard-owned
    "kelly_fraction", "position_size_pct",
})

# Per-type allowlist: if non-empty for a type, ONLY these keys may mutate.
# Keys not listed are frozen for that type. Empty → all numeric except global freeze.
_DEFAULT_EVOLVABLE: dict[str, frozenset[str]] = {
    "momentum": frozenset({
        "lookback_candles", "momentum_threshold", "min_confidence",
        "trend_strength_weight", "regime_conf_weight",
    }),
    "mean_reversion": frozenset({
        "lookback_candles", "bb_std_dev", "rsi_period", "rsi_oversold",
        "rsi_overbought", "reversion_threshold", "min_drift",
        "min_confidence", "trending_conf_damp",
    }),
    "mean_reversion_tp": frozenset({
        "lookback_candles", "bb_std_dev", "rsi_period", "rsi_oversold",
        "rsi_overbought", "reversion_threshold", "min_drift",
        "min_confidence", "trending_conf_damp", "take_profit",
    }),
    "mean_reversion_sl": frozenset({
        "lookback_candles", "bb_std_dev", "rsi_period", "rsi_oversold",
        "rsi_overbought", "reversion_threshold", "min_drift",
        "min_confidence", "trending_conf_damp",
    }),
    "phantom": frozenset({
        "ema_fast", "ema_slow", "atr_period", "breakout_lookback",
        "min_atr_pct", "max_atr_pct", "min_confidence",
    }),
    "sniper": frozenset({
        "min_price_yes", "max_price_yes", "max_price_no",
        "skip_zone_low", "skip_zone_high", "min_drift",
        "min_confidence", "quiet_drift_bump",
    }),
    "hybrid": frozenset({
        "min_confidence", "w_momentum", "w_mean_reversion",
        "w_phantom", "w_sentiment",
    }),
    "sentiment": frozenset({
        "min_confidence", "lookback_candles",
    }),
}


def frozen_genes() -> frozenset[str]:
    extra = getattr(config, "GA_FROZEN_GENES", None) or ()
    return _DEFAULT_FROZEN | frozenset(extra)


def evolvable_keys(strategy_type: str, params: dict[str, Any]) -> set[str]:
    """Numeric keys in ``params`` that the GA may mutate for this type."""
    from evolution.bounds import is_numeric_gene

    frozen = frozen_genes()
    allow = _DEFAULT_EVOLVABLE.get(strategy_type)
    keys = set()
    for k, v in (params or {}).items():
        if k in frozen:
            continue
        if not is_numeric_gene(v):
            continue
        if allow is not None and k not in allow:
            continue
        keys.add(k)
    return keys


def filter_mutable(params: dict[str, Any], strategy_type: str) -> dict[str, Any]:
    """Return a copy of params with only evolvable keys (for mutation focus)."""
    allow = evolvable_keys(strategy_type, params)
    return {k: params[k] for k in allow if k in params}

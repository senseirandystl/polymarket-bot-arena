"""Abstract base class all arena bots inherit from."""

import random
import copy
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import db
import learning
import polymarket_fills
from signals.context import build_context
from signals.curves import smooth_ramp
from signals.lab import SignalView, get_lab

logger = logging.getLogger(__name__)

# Keys every analyze()/make_decision result carries — the structured decision
# contract. Plain dicts (not a class) so every existing caller, DB row and
# test fixture keeps working; strategy_decision() below guarantees the shape.
DECISION_KEYS = ("action", "side", "edge", "confidence", "reasoning",
                 "signals", "suggested_amount")


def strategy_decision(action: str, side: str = "yes", *, edge: float = 0.0,
                      confidence: float = 0.0, reasoning: str = "",
                      signals: dict | None = None,
                      suggested_amount: float = 0.0, **extra) -> dict:
    """Build a structured strategy decision.

    Every bot's analyze()/make_decision returns this shape: the action, the
    side, the strategy's own EDGE estimate (probability units, 0 when
    unknown/hold), a confidence in [0, 1], human-readable reasoning, and
    ``signals`` — the named signal readings that CONTRIBUTED to the decision
    (for attribution/debugging; the model-blend lanes have their own
    attribution via BlendResult). ``extra`` carries strategy-specific fields
    (maker_* quotes, arb legs, features, entry_price) through unchanged.
    """
    d = {
        "action": action,
        "side": side,
        "edge": float(edge),
        "confidence": max(0.0, min(1.0, float(confidence))),
        "reasoning": reasoning,
        "signals": dict(signals or {}),
        "suggested_amount": float(suggested_amount),
    }
    d.update(extra)
    return d


# Bankroll read for Kelly sizing, cached off the 1s hot path (the pool only
# changes on fills/resolutions). Shared across bots — the pool is shared too.
_bankroll_cache: tuple = (0.0, 0.0)  # (ts, value)
_kelly_cache: tuple = (0.0, 0.0)     # (ts, value)
_lane_override_cache: tuple = (0.0, {})  # (ts, overrides dict)
# Exposure headroom cache: (ts, {(market_id, side, mode): headroom})
_exposure_cache: tuple = (0.0, {})


def _lane_overrides() -> dict:
    """Approved candidate-lane overrides (dashboard Signal Lab).

    lane -> {enabled, profile: {strategy: weight}}. A lane the harness
    validated and a human APPROVED trades live through this DB override —
    no config edit, no restart. Cached off the 1s hot path.
    """
    global _lane_override_cache
    now = time.time()
    if (now - _lane_override_cache[0]) < getattr(config, "HOTPATH_CACHE_TTL_SEC", 30):
        return _lane_override_cache[1]
    try:
        value = db.get_lane_overrides()
    except Exception:
        value = _lane_override_cache[1]  # DB hiccup: keep last known
    _lane_override_cache = (now, value)
    return value


# The SignalLab reads overrides through this module-level function (late
# bound), so the backtest runtime's isolation patch on `_lane_overrides`
# reaches the lab too.
get_lab().overrides_provider = lambda: _lane_overrides()


def invalidate_exposure_cache() -> None:
    """Bust exposure headroom cache after a successful place (all bots)."""
    global _exposure_cache
    _exposure_cache = (0.0, {})


def _kelly_fraction() -> float:
    """Kelly fraction for sizing, live-editable in dashboard Settings.

    Read from the DB (the dashboard runs in a separate process, so a module
    constant would never see edits) with the same short hot-path cache as the
    bankroll.
    """
    global _kelly_cache
    now = time.time()
    if (now - _kelly_cache[0]) < getattr(config, "SIZING_BANKROLL_CACHE_SEC", 5.0):
        return _kelly_cache[1]
    value = db.get_kelly_fraction()
    _kelly_cache = (now, value)
    return value


def _sizing_bankroll(mode: str) -> float:
    """Current bankroll for bet sizing (cached, config.SIZING_BANKROLL_CACHE_SEC).

    Paper: the shared virtual pool's available cash. Live: a notional bankroll
    consistent with the per-trade cap (wallet reads are too slow for the 1s
    tick; LIVE_MAX_POSITION already hard-caps exposure).

    Portfolio allocation (when enabled) multiplies this by the bot's capital
    weight in ``make_decision`` / ``execute`` — this helper returns the *pool*.
    """
    if mode == "live":
        pct = max(config.MAX_POSITION_PCT_OF_BALANCE, 0.01)
        return config.LIVE_MAX_POSITION / pct
    global _bankroll_cache
    now = time.time()
    if (now - _bankroll_cache[0]) < getattr(config, "SIZING_BANKROLL_CACHE_SEC", 5.0):
        return _bankroll_cache[1]
    value = max(0.0, db.get_paper_available())
    _bankroll_cache = (now, value)
    return value


def _portfolio_weight(bot_name: str) -> float:
    """Fraction of the shared pool this bot may Kelly-size against.

    1.0 when portfolio allocation is off (legacy full-pool Kelly).
    """
    try:
        from arena.portfolio import get_weight
        return float(get_weight(bot_name))
    except Exception:
        return 1.0


def _portfolio_size_mult(bot_name: str) -> float:
    """Scale factor for zone/maker bots that size via max_position × pct.

    Equal weight → 1.0; winners size up, losers down. 1.0 when disabled.
    """
    try:
        from arena.portfolio import size_multiplier
        return float(size_multiplier(bot_name))
    except Exception:
        return 1.0


def _risk_size_mult(bot_name: str) -> float:
    """Risk-engine size taper (drawdown / portfolio stress). 0 = paused."""
    try:
        from arena.risk_engine import size_multiplier
        return float(size_multiplier(bot_name))
    except Exception:
        return 1.0


class BaseBot(ABC):
    name: str
    strategy_type: str
    strategy_params: dict
    generation: int
    lineage: str

    # Exit strategy: None = hold to resolution (default)
    # "stop_loss" = exit when position is down stop_loss_pct
    # "take_profit" = exit when position is up take_profit_pct
    exit_strategy: str | None = None
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0

    # Each strategy type gets different parameters for differentiation.
    # This creates real competition for evolution to select from.
    STRATEGY_PRIORS = {
        "momentum": 0.52,       # slight YES bias — momentum tends bullish
        "mean_reversion": 0.48, # slight NO bias — mean reversion bets against crowd
        "mean_reversion_sl": 0.48,
        "mean_reversion_tp": 0.48,
        "sniper": 0.50,         # neutral — sniper uses its own rules
        "phantom": 0.52,        # trend-following
        "hybrid": 0.50,         # neutral
    }
    # Per-strategy MODEL weight profile. Each lane arrives normalized to
    # [-1, 1] in YES-frame and the weighted sum maps to a model probability:
    #   P_model = 0.5 + 0.5 * sum(w_lane * lane)
    # Differentiation is by EMPHASIS, never by a hardcoded direction — every
    # weight is >= 0 and every lane is regime-agnostic, so no strategy carries
    # a baked-in YES/NO bias (see memory: regime-agnostic-signals).
    #   momentum / phantom  — trend followers: drift anchor + heavy flow/momentum
    #   mean_reversion*     — fundamentals-only: near-pure drift; by ignoring
    #                         momentum/flow it naturally fades price moves that
    #                         BTC's actual position doesn't back
    #   sentiment           — order-flow reader: CVD (executed aggression) heavy
    #   hybrid              — balanced blend of everything
    # Fidelity redesign (BUG #27): with the pm/obi/cvd lanes killed pending
    # offline validation, the LIVE inputs are drift, mom (BTC candle trend)
    # and strat (this strategy's own analyze() thesis — now carrying a
    # PER-STRATEGY weight here instead of the old flat global 0.15, which was
    # too small to differentiate anyone). Live weights sum to ~1.0 per
    # strategy; dead lanes stay listed at their revival weights' position (0)
    # so re-enabling a validated signal is a one-line profile edit.
    #   momentum — trades the BTC short-term trend (mom lane + trend analyze)
    #   phantom  — EMA-crossover/breakout swing: analyze()-dominant
    #   meanrev  — fundamentals + fade: drift anchor + z-score reversion
    #              thesis => "buy the dip in the winning direction" (the two
    #              agree only when price overextends AGAINST the drift side)
    #   sentiment— in-market flow reader (raw pm/cvd via analyze; its lanes
    #              stay killed until validated)
    #   hybrid   — balanced ensemble of the sub-strategies
    # (fut/tech/xasset are the 2026-07-18 candidate lanes — explicit 0.00 in
    # every profile so re-enabling a validated lane requires a DELIBERATE
    # per-strategy weight, never an accidental default-1.0 fallthrough.)
    _DEAD_LANES = {
        "pm": 0.00, "cvd": 0.00, "obi": 0.00,
        "fut": 0.00, "tech": 0.00, "xasset": 0.00,
        "lag": 0.00, "ms_mom": 0.00, "flow_decay": 0.00,
    }
    # Strat is confirmation-only (config.STRAT_LANE_MODE); mass toward drift/mom.
    # Sentiment strategy removed (2026-08 audit) — lanes stay kill-switched.
    STRATEGY_SIGNAL_PROFILE = {
        "momentum":          {"drift": 0.55, "mom": 0.30, "strat": 0.15, **_DEAD_LANES},
        "phantom":           {"drift": 0.50, "mom": 0.25, "strat": 0.25, **_DEAD_LANES},
        "mean_reversion":    {"drift": 0.75, "mom": 0.00, "strat": 0.25, **_DEAD_LANES},
        "mean_reversion_sl": {"drift": 0.75, "mom": 0.00, "strat": 0.25, **_DEAD_LANES},
        "mean_reversion_tp": {"drift": 0.75, "mom": 0.00, "strat": 0.25, **_DEAD_LANES},
        "hybrid":            {"drift": 0.55, "mom": 0.20, "strat": 0.25, **_DEAD_LANES},
        "sniper":            {"drift": 0.55, "mom": 0.10, "strat": 0.10, **_DEAD_LANES},
        # Menu-only strategies (not default slate)
        "lag_residual":      {"drift": 0.70, "mom": 0.10, "strat": 0.20, **_DEAD_LANES},
        "regime_specialist": {"drift": 0.60, "mom": 0.25, "strat": 0.15, **_DEAD_LANES},
        "no_lag":            {"drift": 0.80, "mom": 0.00, "strat": 0.20, **_DEAD_LANES},
        "true_maker":        {"drift": 0.60, "mom": 0.10, "strat": 0.10, **_DEAD_LANES},
    }
    DEFAULT_SIGNAL_PROFILE = {"drift": 0.55, "mom": 0.15, "strat": 0.15, **_DEAD_LANES}
    # How far fair value moves from the market mid toward the bot's own model.
    # fair = mid + trust * (P_model - mid): the bot only sees edge when its
    # model DISAGREES with the price — the honest replacement for the additive
    # tilt/alpha stack that manufactured edge by construction.
    STRATEGY_MODEL_TRUST = {
        "momentum": 0.50,
        "mean_reversion": 0.60,
        "mean_reversion_sl": 0.60,
        "mean_reversion_tp": 0.60,
        "sniper": 0.50,
        "phantom": 0.50,
        "hybrid": 0.50,
        "lag_residual": 0.55,
        "regime_specialist": 0.50,
        "no_lag": 0.55,
        "true_maker": 0.50,
    }
    # Minimum confidence to place a trade
    # (MIN_TRADE_CONFIDENCE removed 2026-07-17 — it was dead code: defined but
    # never read since the two-sided rewrite; the MIN_EDGE gate below is the
    # real trade filter.)
    # Minimum cost-adjusted edge (probability units) to place a trade. Two-sided
    # selection buys the side with the larger positive edge above this floor.
    # Per-strategy max side MID (judged alongside the global HIGH_PRICE_GUARD).
    # meanrev embodies the harness's top rule — "follow drift only when the
    # side is <= 58c (market lags)", +11.8c/share over 532 samples — so above
    # 0.58 the drift is priced-in and it stands down (BUG #28).
    # Hold-to-resolution: edge only exists when the market still LAGS the
    # directional thesis. Above these mids the side is largely priced-in —
    # high WR can still lose dollars (WR must beat entry + fees).
    STRATEGY_MAX_SIDE_PRICE = {
        "mean_reversion": 0.58,
        "mean_reversion_sl": 0.58,
        "mean_reversion_tp": 0.58,
        "momentum": 0.62,
        "phantom": 0.62,
        "lag_residual": 0.58,
        "no_lag": 0.58,
        "regime_specialist": 0.62,
    }
    # Re-tightened after data-gathering (2026-08 audit) — fee-aware floors.
    MIN_EDGE = {
        "momentum": 0.015,
        "mean_reversion": 0.020,
        "mean_reversion_sl": 0.020,
        "mean_reversion_tp": 0.020,
        "sniper": 0.020,
        "phantom": 0.015,
        "hybrid": 0.020,
        "lag_residual": 0.018,
        "regime_specialist": 0.018,
        "no_lag": 0.022,
        "true_maker": 0.015,
    }

    def __init__(self, name, strategy_type, params, generation=0, lineage=None):
        self.name = name
        self.strategy_type = strategy_type
        self.strategy_params = params
        self.generation = generation
        self.lineage = lineage or name
        self._paused = False
        self.trading_mode = "paper"
        # (ts, total_resolved) cache for the learning-weight ramp — refreshed
        # off the 1s hot path (see make_decision). Resolved count changes only
        # when a trade settles (~60s), so a short TTL is plenty.
        self._perf_cache = None

    @abstractmethod
    def analyze(self, market: dict, signals: dict) -> dict:
        """Analyze market + signals and return a structured decision.

        Build the return value with :func:`strategy_decision` so every bot
        exposes the same contract:
            {
                "action": "buy" | "hold",
                "side": "yes" | "no",
                "edge": float,          # strategy's own edge estimate (prob units)
                "confidence": 0.0-1.0,
                "reasoning": "why this trade",
                "signals": {...},       # named readings that contributed
                "suggested_amount": float,
            }
        Regime context is available via :meth:`regime_context` (the
        vol_regime block of ``signals``) — regime-sensitive strategies
        condition their confidence on it.
        """
        pass

    @staticmethod
    def regime_context(signals) -> dict:
        """Regime awareness input for strategies.

        Prefers the robust detector block (``market_regime`` / enriched
        ``vol_regime``): rich ids like ``high_vol_trend``, continuous
        feature scores, confidence, and legacy quiet/normal/trending/
        volatile labels for older conditionals.

        ``known`` is False when the regime feed hasn't produced a reading —
        strategies must treat that as NEUTRAL (no boost, no damp), never as
        chop. ``trend_score`` is 0..1 trendiness; ``trending``/``ranging``
        are convenience booleans at the 0.65/0.35 boundaries.
        """
        sv = SignalView.of(signals)
        mr = sv.market_regime
        vr = sv.vol_regime
        # Prefer detector snapshot; fall back to vol_regime enrichment.
        src = mr if mr.get("regime_id") or mr.get("label") else vr
        ts = src.get("trend_score", vr.get("trend_score"))
        vs = src.get("vol_score", vr.get("vol_score"))
        known = bool(src.get("known")) if "known" in src else (ts is not None)
        if ts is None and vs is None and not src.get("regime_id"):
            known = False
        trend_score = 0.5 if ts is None else max(0.0, min(1.0, float(ts)))
        vol_score = 0.5 if vs is None else max(0.0, min(1.0, float(vs)))
        rid = (
            src.get("regime_id")
            or src.get("label")
            or vr.get("regime_id")
            or vr.get("regime")
            or "unknown"
        )
        legacy = src.get("legacy") or vr.get("regime") or "unknown"
        return {
            "label": rid,                 # rich id preferred
            "legacy": legacy,             # quiet/normal/trending/volatile
            "known": known,
            "trend_score": trend_score,
            "vol_score": vol_score,
            "confidence": float(src.get("confidence") or 0.0),
            "features": dict(src.get("features") or vr.get("features") or {}),
            "meta_bucket": src.get("meta_bucket") or vr.get("meta_bucket") or "mixed",
            "trending": known and trend_score >= 0.65,
            "ranging": known and trend_score <= 0.35,
            "high_vol": known and vol_score >= 0.55,
            "chop": rid == "high_vol_chop" or (
                known and vol_score >= 0.55 and trend_score <= 0.35
            ),
        }

    def _inventory_usd(self, market: dict, side: str) -> float:
        """Open USD exposure already held on (market, side) across ALL bots.

        The maker bots use this for inventory management — quoting into a
        side the pool is already loaded on compounds one BTC candle's risk.
        Fails open to 0.0 (missing ids / DB hiccup): the shared-pool
        exposure cap in execute() still applies downstream.
        """
        market_id = (market or {}).get("condition_id") or (market or {}).get("id")
        if not market_id or side not in ("yes", "no"):
            return 0.0
        try:
            return db.get_open_exposure(market_id, side, self.trading_mode)
        except Exception:
            return 0.0

    @staticmethod
    def _normalize_analysis(raw: dict) -> dict:
        """Fill the structured-decision contract on a legacy analyze() dict.

        Subclasses (and evolution-spawned mutants of older code) may still
        return bare action/side/confidence dicts — normalize in one place so
        every downstream consumer can rely on DECISION_KEYS.
        """
        raw.setdefault("edge", 0.0)
        raw.setdefault("signals", {})
        raw.setdefault("suggested_amount", 0.0)
        raw.setdefault("reasoning", "")
        return raw

    def _signal_profile(self) -> dict:
        return self.STRATEGY_SIGNAL_PROFILE.get(
            self.strategy_type, self.DEFAULT_SIGNAL_PROFILE)

    def _model_prob_yes(self, lanes: dict) -> float:
        """Model probability of YES from normalized signal lanes.

        Thin wrapper over SignalLab.blend (per-strategy profile + approved
        lane overrides + live-validation gate). Kept as the bot-facing entry
        so strategy code and tests never touch the lab's internals.
        """
        return get_lab().blend(self.strategy_type, lanes,
                               self._signal_profile(),
                               overrides=_lane_overrides()).prob

    def _compute_fair_yes(self, yes_mid: float, model_prob: float,
                          trust: float) -> float:
        """Fair YES probability: market mid pulled toward the bot's model.

        fair = mid + trust * (P_model - mid). If the market already prices the
        model's view, fair == mid and there is NO edge — a signal the market
        has absorbed earns nothing (the flaw in the old additive stack, where
        edge equalled the bonus terms by construction).
        """
        fair = yes_mid + trust * (model_prob - yes_mid)
        return max(0.02, min(0.98, fair))

    def _assumed_maker(self) -> bool:
        """Whether edge math should use maker fee (0) under limit-first style."""
        if getattr(config, "ORDER_STYLE", "limit") != "limit":
            return False
        mode = getattr(config, "LIMIT_PRICE_MODE", "passive_mid")
        return mode in ("passive_mid", "join_bid")

    def _side_net_edges(self, model_prob: float, trust_eff: float,
                        yes_price: float, no_price: float) -> tuple:
        """Cost-adjusted edge per side, each anchored on its OWN book price.

        edge_side = trust_eff * (P_model_side - side_price) - fee. Fee is
        maker (0) when ORDER_STYLE=limit and LIMIT_PRICE_MODE is passive, else
        the crypto taker fee. Per-side anchoring makes edge purely
        model-vs-that-side's-price (BUG #27).
        """
        is_maker = self._assumed_maker()
        edge_yes = (trust_eff * (model_prob - yes_price)
                    - polymarket_fills.fee_per_share(yes_price, is_maker=is_maker))
        edge_no = (trust_eff * ((1.0 - model_prob) - no_price)
                   - polymarket_fills.fee_per_share(no_price, is_maker=is_maker))
        return edge_yes, edge_no

    def make_decision(self, market: dict, signals: dict) -> dict:
        """Make a trading decision using market price edge + strategy + learning.

        Signal hierarchy:
        1. Market price edge (strongest — when price is far from 50c, follow it)
        2. BTC momentum (if price is moving, lean that direction)
        3. Strategy analysis (adds differentiation between bots)
        4. Learned bias (accumulates over time, adjusts everything)

        Skips trades when confidence is too low (no edge = no bet).
        """
        # `or 0.5`, not a .get default: _normalize sets current_price=None
        # explicitly (key present), so a default arg wouldn't fire. A book that
        # is momentarily unavailable leaves it None — coalesce to neutral 0.5.
        market_price = market.get("current_price") or 0.5

        # --- Market-level lanes from the SignalLab ---
        # One cached computation per tick shared by EVERY bot: drift, mom
        # (incl. the quiet-regime damp), pm/cvd/obi and the fut/tech/xasset
        # candidates, with kill-switches and approved-lane overrides already
        # applied. ``raw`` carries the pre-kill-switch reads for feature
        # extraction and the cand(...) validation log. The lane math lives in
        # signals/lab.py — the calibration history stays documented there.
        lab = get_lab()
        overrides = _lane_overrides()
        market_lanes, raw = lab.compute_lanes(market, signals,
                                              overrides=overrides)
        price_momentum = raw["price_momentum"]
        momentum_signal = market_lanes["mom"]
        drift_signal_val = market_lanes["drift"]

        # --- Lane: strategy thesis from analyze() ---
        raw_signal = self._normalize_analysis(self.analyze(market, signals))
        strategy_signal = 0.0
        if raw_signal["action"] != "hold":
            strategy_yes = 1.0 if raw_signal["side"] == "yes" else -1.0
            strategy_signal = strategy_yes * raw_signal["confidence"]
        # Strat-lane confidence cap (BUG #30): live data showed WR falls as
        # the thesis gets MORE confident (>=0.6 magnitude ran 36.1% WR/-$60,
        # 0.3-0.6 ran 55.9%) — clamp so an overconfident read still blends at
        # the magnitude that actually performed instead of amplifying edge.
        strat_cap = getattr(config, "STRAT_LANE_CONF_CAP", 0.25)
        strategy_signal = max(-strat_cap, min(strat_cap, strategy_signal))

        # Strat is a *derived thesis*, not an independent market signal
        # (2026-08). Default mode "confirm": only contribute when the thesis
        # agrees with non-trivial drift; zero/damp when it fights. Prevents
        # double-counting candle patterns already reflected in drift/mom.
        _strat_mode = getattr(config, "STRAT_LANE_MODE", "confirm")
        if _strat_mode != "full" and abs(strategy_signal) > 1e-12:
            _d_agree = float(getattr(config, "STRAT_DRIFT_AGREE_MIN", 0.05))
            if abs(drift_signal_val) >= _d_agree:
                _agree = (strategy_signal * drift_signal_val) > 0
                if _agree:
                    strategy_signal *= float(
                        getattr(config, "STRAT_CONFIRM_SCALE", 0.55))
                else:
                    strategy_signal *= float(
                        getattr(config, "STRAT_FIGHT_SCALE", 0.0))
            elif _strat_mode == "confirm":
                # No drift to confirm against — keep only a residual whisper.
                strategy_signal *= float(
                    getattr(config, "STRAT_CONFIRM_SCALE", 0.55)) * 0.5

        # --- Signal 4: Learning bias ---
        sv = SignalView.of(signals)
        volume = sv.orderflow.get("volume_24h")
        time_rem = market.get("time_remaining_seconds")
        
        features = learning.extract_features(
            market_price, price_momentum, 
            volume=volume, time_rem=time_rem
        )
        # Stamp regime at decision time so evolution fitness / post-hoc
        # analysis can condition on the regime the trade was taken in.
        try:
            rctx = self.regime_context(signals)
            if rctx.get("label") and rctx["label"] != "unknown":
                features = list(features) + [f"regime:{rctx['label']}"]
                if rctx.get("legacy"):
                    features.append(f"regime_legacy:{rctx['legacy']}")
        except Exception:
            pass

        # Context vector is deferred until a buy is confirmed (CPU on skip path).
        ctx_vec = None
        # Hoist regime once for this decision (guards + reasoning reuse).
        _regime_label = self.regime_context(signals).get("label")
        prior = self.STRATEGY_PRIORS.get(self.strategy_type, 0.5)
        learned_yes_bias = learning.get_learned_bias(self.name, features, prior)
        # Convert from 0-1 to -0.5 to +0.5
        learning_signal = (learned_yes_bias - 0.5)

        # Dynamic learning weight: ramps up as bot accumulates data
        # Capped at 0.30 (was 0.60) — stale inherited data was making all bots identical
        # Cached off the hot path: the resolved-trade count changes only when a
        # trade settles, so re-querying it every 1s tick was pure waste.
        now_ts = time.time()
        ttl = getattr(config, "HOTPATH_CACHE_TTL_SEC", 30)
        if self._perf_cache is not None and (now_ts - self._perf_cache[0]) < ttl:
            total_resolved = self._perf_cache[1]
        else:
            total_resolved = db.get_bot_performance(
                self.name, hours=168
            ).get("total_trades", 0)
            self._perf_cache = (now_ts, total_resolved)
        # Live learning disabled (spec R5): the raw-YES-WR bias was
        # anti-predictive. Outcomes are still recorded for the redesign, but the
        # bias contributes 0 to live decisions until the edge-calibrated learner
        # replaces it.
        if config.LEARNING_ENABLED:
            learning_weight = min(0.30, 0.05 + total_resolved * 0.005)
        else:
            learning_weight = 0.0

        # --- Model probability via the SignalLab blend ---
        # Market lanes (computed above, shared) + this bot's own strat/learn
        # lanes, weighted by the per-strategy profile, approved-lane
        # overrides (the tuner/promoter closed loop) and the live-validation
        # gate. Edge appears ONLY where the model disagrees with the market
        # price ("follow drift only when the market lags" was the top rule in
        # the offline net-edge harness).
        # Regime-aware lane scales + hard stand-down when live regime is toxic.
        try:
            from arena.regime_adapt import adjustments as _regime_adj
            _radj = _regime_adj(
                _regime_label,
                strategy_type=self.strategy_type,
            )
        except Exception:
            from arena.regime_adapt import RegimeAdjust
            _radj = RegimeAdjust()
        if getattr(_radj, "block_directional", False):
            # Data-driven: regime_performance WR/P&L below hard-skip bar.
            # Clears automatically when WR recovers (hysteresis in regime_adapt).
            return strategy_decision(
                "skip",
                reasoning=(
                    f"Regime hard-skip: {_radj.label} toxic live "
                    f"({_radj.reason}) — directional stand-down"
                ),
                signals={
                    "drift": drift_signal_val, "mom": momentum_signal,
                    "strat": strategy_signal,
                    "regime": _radj.label,
                },
                features=features,
            )
        if abs(float(_radj.mom_lane_scale) - 1.0) > 1e-9:
            market_lanes = dict(market_lanes)
            market_lanes["mom"] = float(market_lanes.get("mom") or 0.0) * float(
                _radj.mom_lane_scale)
        if abs(float(_radj.strat_lane_scale) - 1.0) > 1e-9:
            strategy_signal = strategy_signal * float(_radj.strat_lane_scale)

        lanes = {
            **market_lanes,
            "strat": strategy_signal,
            "learn": learning_signal * 2.0 * learning_weight,
        }
        blend = lab.blend(self.strategy_type, lanes, self._signal_profile(),
                          overrides=overrides, signals=signals)
        model_prob = blend.prob

        # --- Macro-release caution: stand down around high-impact prints ---
        # Non-directional context (signals/macro_calendar.py): the smooth 0..1
        # caution peaks in the minutes around 08:30/14:00 ET weekday release
        # slots, where the window can gap violently against any model. Same
        # philosophy as the session filter — build the skip, default flat.
        # Structured skip: same contract as buys (edge 0, contributing
        # signals attached) so downstream consumers never branch on shape.
        def _skip(reason: str, side: str = "yes", confidence: float = 0.0,
                  edge: float = 0.0, entry_price: float | None = None):
            # Always attach lane reads so decision_events can score skips
            # counterfactually (same raw cand() values as buys).
            return strategy_decision(
                "skip", side, edge=edge, confidence=confidence, reasoning=reason,
                signals={
                    "drift": drift_signal_val, "mom": momentum_signal,
                    "strat": strategy_signal, "model_prob": model_prob,
                    "fut": raw.get("fut_taker"), "tech": raw.get("tech_mtf"),
                    "xasset": raw.get("xasset"),
                    "lag": raw.get("lag"), "ms_mom": raw.get("ms_mom"),
                    "flow_decay": raw.get("flow_decay"),
                    "regime": _regime_label,
                },
                features=features,
                entry_price=entry_price)

        macro = sv.macro_caution
        if macro >= getattr(config, "MACRO_CAUTION_SKIP", 0.75):
            return _skip(f"Macro-release caution {macro:.2f} — high-impact window")

        # --- Hard model-lean floor: no opinion, no trade (BUG #27) ---
        # Conviction-scaled trust damps a weak model but its residual edge
        # still scales with MARKET displacement, so near-ignorant models kept
        # clearing MIN_EDGE against displaced prices (lean < 0.10: 28.6% WR /
        # -$78.74 live; lean >= 0.10: 73% WR / +$96.12). Below the floor the
        # model has nothing tradable to say — skip outright.
        model_lean = abs(model_prob - 0.5)
        lean_min = float(config.MODEL_LEAN_MIN)
        try:
            from arena.learned_rules import skip_softening as _skip_soft
            _wl = _skip_soft("weak_lean")
            if _wl.get("soften"):
                lean_min = max(0.02, lean_min * float(_wl["factor"]))
        except Exception:
            pass
        if model_lean < lean_min:
            return _skip(
                f"Model lean too weak: |{model_prob:.3f}-0.5|="
                f"{model_lean:.3f} < {lean_min:.2f}")

        trust = self.STRATEGY_MODEL_TRUST.get(self.strategy_type, 0.5)
        # --- Conviction-scaled trust: the model's say is proportional to how
        # much it actually knows. edge = trust*(P_model - mid) takes its
        # magnitude from the MARKET's displacement, so a near-ignorant model
        # (P_model ~ 0.5) could book a phantom edge on any market move away
        # from 0.5 and systematically fade it (2026-07-17 chop run: underdog
        # buys 38.5% WR, YES side 10% WR). A decisive model (lean >= the
        # scale) keeps full trust — the validated market-lags-drift trade is
        # untouched.
        conviction = min(1.0, abs(model_prob - 0.5) / config.MODEL_CONVICTION_SCALE)
        trust_eff = trust * conviction
        fair_yes = self._compute_fair_yes(market_price, model_prob, trust_eff)

        # --- Per-side evaluation: each side scored on its OWN price + fee ---
        # Binary outcomes must sum to 1, so both sides share the one fair
        # probability (fair_yes / 1-fair_yes) — but each side is evaluated
        # INDEPENDENTLY: its own book price, its own taker fee, and therefore its
        # own net edge and its own confidence. The bot takes whichever side wins
        # its own evaluation. Fully symmetric / regime-agnostic — no side is
        # favored by a constant, only by the signals (the drift lane above being
        # the one that actually reads which way BTC is going).
        yes_price = market_price
        no_price = market.get("no_price")
        if no_price is None:
            no_price = round(1.0 - yes_price, 4)

        # --- Book-consistency gate (BUG #27) ---
        # YES and NO books disagreeing about the same event = suspect data
        # (stale/gapped book). A real gap is the arb bot's two-legged trade;
        # one-legged it is a coin flip minus fees — and Kelly max-sized
        # exactly those (sums 0.84-0.85 -> "13c edges" -> -$29.15 in 2 trades).
        book_sum = yes_price + no_price
        if abs(book_sum - 1.0) > config.BOOK_SUM_TOLERANCE:
            return _skip(
                f"Book inconsistency: yes={yes_price:.2f}+no={no_price:.2f}"
                f"={book_sum:.2f} outside 1±{config.BOOK_SUM_TOLERANCE:.2f}")

        # --- Executable prices: edge is measured against what a taker PAYS ---
        # Decisions used to price edge + entry off the MID while the fill
        # engines walk the ASKS: on wide books (3-8c spreads live) the fill
        # landed > MAX_FILL_SLIPPAGE above the decision price and the
        # slippage guard rejected most attempted trades (5 of 7 in an hour).
        # The trader lays the warm books' best asks onto the market dict;
        # until the warmer primes a market, fall back to the mid. The
        # book-sum gate above intentionally keeps judging the MIDS (asks sum
        # > 1 on any normal spread).
        yes_exec = market.get("yes_ask") or yes_price
        no_exec = market.get("no_ask") or no_price

        edge_yes, edge_no = self._side_net_edges(model_prob, trust_eff,
                                                 yes_exec, no_exec)

        # --- Model-lean eligibility: never fade the market on IGNORANCE ---
        # With no information the model sits at 0.5 and the blend would pull
        # fair toward 0.5, making the non-favorite side look "cheap" on every
        # market — a pure contrarian leak. A side is only tradable when the
        # model ACTIVELY leans toward it (its lanes point that way), so the
        # trade thesis is "market lags my information", never "market is more
        # confident than my nothing".
        if model_prob <= 0.5:
            edge_yes = float("-inf")
        if model_prob >= 0.5:
            edge_no = float("-inf")

        # --- Drift veto: never trade AGAINST a non-trivial drift reading ---
        # Drift is the validated fundamental (where BTC actually sits vs the
        # strike). Momentum/flow lanes may modulate conviction but must not
        # outvote it: live, trades contradicting a non-zero drift ran 26% WR
        # (vs 52% agreeing). Symmetric and regime-agnostic — the veto follows
        # whichever side BTC is on. Flow-only trades (|drift| below the floor)
        # remain allowed.
        veto = getattr(config, "DRIFT_VETO_MIN", 0.05)
        if drift_signal_val >= veto:
            edge_no = float("-inf")
        elif drift_signal_val <= -veto:
            edge_yes = float("-inf")

        # Side pick: still argmax of fee-net edge (YES-only or NO-only after
        # eligibility/veto). Confidence is NO LONGER edge × constant (that
        # inverted live: higher conf → worse WR). Quality conf is set after
        # side selection from structure (drift, lag mid, calibrated edge).
        if edge_yes >= edge_no:
            side, side_price, chosen_edge = "yes", yes_exec, edge_yes
        else:
            side, side_price, chosen_edge = "no", no_exec, edge_no
        confidence = 0.0  # filled below after mid is known

        # --- Dead-zone gate (2026-07-21): the single biggest live leak ---
        # A flat-drift opinion against a near-coin-flip market was 59 trades,
        # 39% WR, -$77.83 over the 290-trade run — the model manufacturing an
        # edge from noisy flow/strat lanes where the crowd is genuinely 50/50.
        # It fires BEFORE the edge gate: the coin-flip band with no drift
        # conviction is a "sit flat" region regardless of computed edge. The
        # SAME price band with |drift| >= DEAD_ZONE_DRIFT_MIN is the profitable
        # "market lags drift" trade (+$30.10, 65.7% WR) and passes through, so
        # the gate is drift-CONDITIONAL. Quiet/range regimes raise the floor
        # (DEAD_ZONE_QUIET_DRIFT_MIN) — weak-to-moderate drift in the mid band
        # was the 2026-07-29 soak's largest leak under low_vol_range.
        side_mid_dz = yes_price if side == "yes" else no_price
        dz_lo = getattr(config, "DEAD_ZONE_PRICE_LO", 0.42)
        dz_hi = getattr(config, "DEAD_ZONE_PRICE_HI", 0.58)
        dz_drift = float(getattr(config, "DEAD_ZONE_DRIFT_MIN", 0.10))
        try:
            rctx = self.regime_context(signals)
            quiet_regs = getattr(
                config, "DEAD_ZONE_QUIET_REGIMES",
                ("low_vol_range", "low_vol_trend", "quiet"),
            )
            label = (rctx.get("label") or rctx.get("legacy") or "")
            if label in quiet_regs:
                dz_drift = max(
                    dz_drift,
                    float(getattr(config, "DEAD_ZONE_QUIET_DRIFT_MIN", 0.20)),
                )
        except Exception:
            pass
        # Skip-reason bandit: if dead_zone skips often would have won, ease
        # the drift floor (factor < 1); if they correctly avoided losses, tighten.
        try:
            from arena.learned_rules import skip_softening as _skip_soft
            _dz_soft = _skip_soft("dead_zone")
            if _dz_soft.get("soften"):
                dz_drift = max(0.05, dz_drift * float(_dz_soft["factor"]))
        except Exception:
            pass
        # Regime prior can raise the dead-zone drift floor further.
        dz_drift = max(dz_drift, dz_drift + float(getattr(_radj, "extra_drift_floor", 0.0) or 0.0))
        if dz_lo <= side_mid_dz <= dz_hi and abs(drift_signal_val) < dz_drift:
            return _skip(
                f"Dead-zone gate: {side} mid={side_mid_dz:.2f} in "
                f"[{dz_lo:.2f},{dz_hi:.2f}] & |drift|={abs(drift_signal_val):.3f}"
                f"<{dz_drift:.2f} (coin-flip, no conviction)",
                side=side, confidence=confidence, entry_price=side_mid_dz)

        # --- NO-side intelligence (2026-08 soak: NO net −$15, YES +$245) ---
        # Prefer NO only as a true market-lag trade with real signed drift —
        # not a mirror of every YES rule. Momentum/meanrev keep mild extras;
        # sniper/hybrid/phantom get stricter strategy mults (config).
        if side == "no" and getattr(config, "NO_SIDE_ENABLED", True):
            signed_for_no = -float(drift_signal_val)
            no_min_d = float(getattr(config, "NO_SIDE_MIN_SIGNED_DRIFT", 0.12))
            no_min_d += float(getattr(_radj, "extra_drift_floor", 0.0) or 0.0)
            if signed_for_no < no_min_d:
                return _skip(
                    f"NO-side drift: signed_drift_for_NO={signed_for_no:+.3f} "
                    f"< {no_min_d:.2f} (need real down-conviction)",
                    side=side, confidence=confidence, entry_price=side_mid_dz)
            no_max_mid = float(getattr(config, "NO_SIDE_MAX_MID", 0.58))
            if side_mid_dz > no_max_mid:
                return _skip(
                    f"NO-side lag gate: mid={side_mid_dz:.2f} > {no_max_mid:.2f} "
                    f"(NO already priced in — not a lag trade)",
                    side=side, confidence=confidence, entry_price=side_mid_dz)

        # --- Cheap underdog band (0.35–0.42): require real drift ---
        ud_lo = float(getattr(config, "UNDERDOG_BAND_LO", 0.35))
        ud_hi = float(getattr(config, "UNDERDOG_BAND_HI", 0.42))
        if ud_lo <= side_mid_dz < ud_hi:
            ud_drift = float(getattr(config, "UNDERDOG_MIN_DRIFT", 0.18))
            ud_drift += float(getattr(_radj, "extra_drift_floor", 0.0) or 0.0)
            signed_side = (drift_signal_val if side == "yes"
                           else -drift_signal_val)
            if signed_side < ud_drift:
                return _skip(
                    f"Underdog band: {side} mid={side_mid_dz:.2f} needs "
                    f"signed_drift≥{ud_drift:.2f} (got {signed_side:+.3f})",
                    side=side, confidence=confidence, entry_price=side_mid_dz)

        # --- Coin-flip favorite band (0.50–0.58): need strong lag-worthy drift ---
        # 2026-08 soak: 229 trades @ 50.2% WR / −$37; low_vol_trend subset −$49.
        cf_lo = float(getattr(config, "MID_COINFLIP_LO", 0.50))
        cf_hi = float(getattr(config, "MID_COINFLIP_HI", 0.58))
        if cf_lo <= side_mid_dz <= cf_hi:
            cf_drift = float(getattr(config, "MID_COINFLIP_DRIFT_MIN", 0.28))
            # Regime prior / live depression can raise the floor further.
            if getattr(_radj, "mid_band_drift_min", None) is not None:
                cf_drift = max(cf_drift, float(_radj.mid_band_drift_min))
            cf_drift += float(getattr(_radj, "extra_drift_floor", 0.0) or 0.0)
            signed_side = (drift_signal_val if side == "yes"
                           else -drift_signal_val)
            if signed_side < cf_drift:
                return _skip(
                    f"Mid-band lag gate: {side} mid={side_mid_dz:.2f} in "
                    f"[{cf_lo:.2f},{cf_hi:.2f}] needs signed_drift≥{cf_drift:.2f} "
                    f"(got {signed_side:+.3f}; reg={getattr(_radj, 'label', '?')})",
                    side=side, confidence=confidence, entry_price=side_mid_dz)

        # --- Extreme-drift market-lag gate ---
        # |drift| ≥ DRIFT_EXTREME is only tradable when the market still LAGS
        # (side mid ≤ DRIFT_EXTREME_MAX_SIDE_MID). Extreme drift with price
        # already at 0.70+ is "priced in" — soak: |drift|≥0.50 → 41% WR.
        ext_abs = getattr(config, "DRIFT_EXTREME_ABS", 0.50)
        ext_max_mid = getattr(config, "DRIFT_EXTREME_MAX_SIDE_MID", 0.58)
        try:
            from arena.learned_rules import skip_softening as _skip_soft
            _ex_soft = _skip_soft("extreme_drift")
            if _ex_soft.get("soften"):
                # ease → allow higher mid (raise max); tighten → lower max
                sof = float(_ex_soft.get("soften") or 0.0)
                ext_max_mid = min(0.85, max(0.40, ext_max_mid + sof * 0.15))
        except Exception:
            pass
        # Hold-to-resolution: direction can be near-certain while the SIDE
        # PRICE already embeds that certainty. A 97% WR at 0.80 mid still
        # loses money. Stamp entry_price so skip counterfactuals use the
        # real mid (not the 0.50 default that inflated hyp_pnl overnight).
        if abs(drift_signal_val) >= ext_abs and side_mid_dz > ext_max_mid:
            return _skip(
                f"Extreme-drift lag gate: |drift|={abs(drift_signal_val):.3f}"
                f">={ext_abs:.2f} but {side} mid={side_mid_dz:.2f}"
                f">{ext_max_mid:.2f} (priced in — hold-to-resolution "
                f"needs BE gap, not just direction)",
                side=side, confidence=confidence, entry_price=side_mid_dz)

        # --- Learned skip/go rules (decision_events mining) ---
        # Data-driven context cells (regime × price_band × drift_band × side).
        # SKIP rules block historically toxic cells; GO rules ease min_edge and
        # boost size where the same context has been printing. Fail-open.
        _learned = {"action": "allow", "size_mult": 1.0, "edge_mult": 1.0,
                    "reason": ""}
        try:
            from arena.learned_rules import evaluate as _learned_eval
            _reg_lab = self.regime_context(signals).get("label")
            _learned = _learned_eval(
                regime=_reg_lab,
                side_price=side_mid_dz,
                drift=drift_signal_val,
                side=side,
                strategy_type=self.strategy_type,
            )
            if _learned.get("action") == "skip":
                return _skip(
                    _learned.get("reason")
                    or f"Learned skip rule for {side} mid={side_mid_dz:.2f}",
                    side=side, confidence=confidence)
        except Exception:
            pass

        # --- Minimum-edge gate (no edge = no bet) — SAME bar on both sides ---
        # Information-scaled: with drift flat the model's disagreement with the
        # market rests entirely on the noisy flow/momentum lanes, so a
        # flow-only claim must clear a HIGHER bar (overnight run: flow-only
        # cheap-side trades by the trend bots ran 29% WR in the 0.30-0.42
        # bucket; drift-backed trades in the same bucket were profitable).
        # Flow-only boundary raised 0.05 -> 0.10 after the 2026-07-19 24h run,
        # then made CONTINUOUS (BUG #30, 2026-07-20): the step function only
        # penalized |drift| < 0.10, but the next 24h run showed the 0.10-0.30
        # band it released to full trust was the biggest dollar loss (135
        # trades, 49.6% WR, -$76.32) while only |drift| >= 0.30 was genuinely
        # predictive (79.3% WR). The multiplier now tapers linearly from
        # FLOW_ONLY_EDGE_MULT_MAX at drift=0 to 1.0x at FLOW_ONLY_DRIFT_FULL_TRUST
        # instead of stepping to full trust at 0.10.
        # Learned GO rules multiply min_edge by edge_mult (<1 eases the bar
        # so historically good contexts can fire on thinner edges).
        min_edge = self.MIN_EDGE.get(self.strategy_type, config.MIN_EDGE_DEFAULT)
        mult_max = getattr(config, "FLOW_ONLY_EDGE_MULT_MAX", 2.0)
        full_trust = max(getattr(config, "FLOW_ONLY_DRIFT_FULL_TRUST", 0.25), 1e-6)
        # Regime can raise the |drift| needed for full flow trust (esp.
        # low_vol_trend — intermediate mom is noisy there).
        if getattr(_radj, "flow_full_trust", None) is not None:
            full_trust = max(full_trust, float(_radj.flow_full_trust))
        taper = max(0.0, 1.0 - abs(drift_signal_val) / full_trust)
        min_edge *= 1.0 + (mult_max - 1.0) * taper
        min_edge *= float(_learned.get("edge_mult") or 1.0)
        # Regime prior edge mult (structural + live WR boost).
        min_edge *= float(getattr(_radj, "edge_mult", 1.0) or 1.0)
        # NO-side and underdog edge taxes (intelligent, not blanket bans).
        if side == "no" and getattr(config, "NO_SIDE_ENABLED", True):
            min_edge *= float(getattr(config, "NO_SIDE_EDGE_MULT", 1.20))
            min_edge *= float(getattr(_radj, "no_edge_mult", 1.0) or 1.0)
            _st_mult = getattr(config, "NO_SIDE_STRATEGY_EDGE_MULT", {}) or {}
            min_edge *= float(_st_mult.get(self.strategy_type, 1.0))
            if ud_lo <= side_mid_dz < ud_hi:
                min_edge *= float(
                    getattr(config, "NO_SIDE_UNDERDOG_EDGE_MULT", 1.35))
        if ud_lo <= side_mid_dz < ud_hi:
            min_edge *= float(getattr(config, "UNDERDOG_EDGE_MULT", 1.40))
        # Wide-book size/skip tax: non-directional microstructure context.
        if getattr(config, "SPREAD_EDGE_MULT_ENABLED", True):
            try:
                spread = float(raw.get("micro_spread") or sv.micro_spread or 0.0)
                wide = float(getattr(config, "SPREAD_EDGE_WIDE", 0.04))
                smax = float(getattr(config, "SPREAD_EDGE_MULT_MAX", 1.35))
                if spread > wide and wide > 0:
                    # Linear ramp from 1.0 at wide to smax at 2× wide.
                    t = min(1.0, (spread - wide) / wide)
                    min_edge *= 1.0 + (smax - 1.0) * t
            except Exception:
                pass
        try:
            from arena.learned_rules import skip_softening as _skip_soft
            _ne = _skip_soft("no_edge")
            if _ne.get("soften"):
                min_edge *= float(_ne["factor"])  # ease → lower bar
        except Exception:
            pass
        if chosen_edge < min_edge:
            return _skip(
                f"No edge: {side} edge={chosen_edge:+.3f} < {min_edge:.3f} "
                f"| fair={fair_yes:.2f} yes={yes_price:.2f} no={no_price:.2f}",
                side=side, confidence=confidence)

        # --- Symmetric guards — keyed on the chosen side's MID (BUG #28) ---
        # The mid is the market's INFORMATION (what the crowd believes); the
        # ask is only the COST. Judging guards on the ask let a bot buy YES
        # at a wide 0.41 ask while the mid said 0.26 — the deep-consensus
        # fight the guard exists to block. Edge/sizing stay on the ask.
        side_mid = yes_price if side == "yes" else no_price
        max_price = min(config.HIGH_PRICE_GUARD,
                        self.STRATEGY_MAX_SIDE_PRICE.get(self.strategy_type, 1.0))
        consensus_floor = float(config.CONSENSUS_GUARD)
        try:
            from arena.learned_rules import skip_softening as _skip_soft
            _hp = _skip_soft("high_price")
            if _hp.get("soften"):
                # ease → allow higher mid (raise max_price)
                sof = float(_hp.get("soften") or 0.0)
                max_price = min(0.90, max_price + sof * 0.10)
            _cg = _skip_soft("consensus")
            if _cg.get("soften"):
                # ease → allow deeper underdog (lower floor)
                consensus_floor = max(0.20, consensus_floor * float(_cg["factor"]))
        except Exception:
            pass
        if side_mid > max_price:
            return _skip(
                f"High-price guard: {side} mid={side_mid:.2f} "
                f">{max_price:.2f}, priced-in / bad risk-reward",
                side=side, confidence=confidence)
        if side_mid < consensus_floor:
            return _skip(
                f"Consensus guard: {side} mid={side_mid:.2f} "
                f"<{consensus_floor:.2f}, fighting consensus",
                side=side, confidence=confidence)

        # --- Quality confidence (NOT edge × constant) ---
        # Live soak: conf = EDGE_TO_CONFIDENCE × edge inverted (high conf worst
        # WR). Confidence is now a structure score for logs / any min_conf
        # gates; sizing never multiplies by it.
        try:
            from bots.edge_calibration import quality_confidence as _qconf
            confidence = _qconf(
                edge=float(chosen_edge),
                abs_drift=abs(float(drift_signal_val)),
                side_mid=float(side_mid),
                side=side,
                regime_label=_regime_label,
            )
        except Exception:
            confidence = min(0.95, max(0.0, float(chosen_edge)) * 3.0)

        # Late-window: mild *size* boost via edge quality when drift is strong
        # and time is short — never inflate confidence (that re-inverts).
        time_rem = market.get("time_remaining_seconds")
        late_size_boost = 1.0
        if time_rem is not None and abs(drift_signal_val) >= 0.20:
            late = smooth_ramp(-float(time_rem), -90.0, -30.0)
            late_size_boost = 1.0 + 0.12 * late

        # --- Bet sizing: pure fractional Kelly, SHARES-FIRST ---
        # Binary-market Kelly: buying a side at price c with true probability p
        # grows fastest at bankroll fraction f* = (p - c)/(1 - c); with the
        # fee-adjusted edge already computed, f* = edge/(1 - price). We bet
        # the Kelly fraction (live-editable in dashboard Settings; full Kelly
        # over-bets estimation error) of the LIVE bankroll — no per-trade or
        # %-of-balance caps (removed 2026-07-17 to run pure Kelly sizing; the
        # venue's shared-pool gate still prevents spending cash the pool
        # lacks). Size scales with *calibrated* edge, odds, AND bankroll —
        # never with confidence (confidence inversion fix, 2026-08).
        price = max(side_price, 0.01)
        # Portfolio capital slice: when allocation is on, this bot sizes
        # against bankroll × weight (weights sum to 1 across the roster).
        # Risk engine may further taper (drawdown / stress) via size_mult.
        # Regime-adaptive size from adjustments() (live WR + priors).
        _reg_mult = float(getattr(_radj, "size_mult", 1.0) or 1.0)
        _learn_size = float(_learned.get("size_mult") or 1.0)
        bankroll = (_sizing_bankroll(self.trading_mode)
                    * _portfolio_weight(self.name)
                    * _risk_size_mult(self.name)
                    * _reg_mult
                    * _learn_size
                    * late_size_boost)
        # Concave edge calibration: modest edges get full credit; outsized
        # model–market disagreement gets diminishing Kelly input (and a hard
        # cap). Flat KELLY_EDGE_CAP alone still max-sized everything above cap.
        try:
            from bots.edge_calibration import calibrated_sizing_edge
            sizing_edge = calibrated_sizing_edge(float(chosen_edge))
        except Exception:
            sizing_edge = min(max(0.0, chosen_edge),
                              getattr(config, "KELLY_EDGE_CAP", 0.08))
        kelly_f = sizing_edge / max(1.0 - price, 0.05)
        kelly_usd = kelly_f * _kelly_fraction() * bankroll
        # SHARES-FIRST: derive the exact share count, then the USD from it.
        # Sizing USD-first and dividing by price rounds away PnL at low prices.
        # Floor to clear Polymarket's 5-share minimum (× buffer for slippage).
        target_shares = max(kelly_usd / price, config.POLYMARKET_MIN_SHARES * 1.15)
        target_shares = round(target_shares, 4)
        amount = target_shares * price

        # NOTE: the drift=/mom=/strat= and cand(...) tokens below are parsed
        # by arena/core_lane_tuner.py and arena/lane_monitor.py — keep their
        # exact format. blend.log_str() appends the per-lane CONTRIBUTIONS
        # (weight x value) so every decision's attribution is persisted.
        #
        # Hybrid meta(...) token: analyze() embeds ``meta(mom=… | reg=…)`` for
        # the online Hedge learner (bots/meta_learner.py). make_decision used
        # to rebuild reasoning from scratch and drop it — 0 hybrid trades had
        # the token, so hybrid_meta never persisted. Re-attach when present.
        reasoning = (
            f"fair={fair_yes:.2f} model={model_prob:.2f} "
            f"trust={trust:.2f}x{conviction:.2f}={trust_eff:.2f} "
            f"yes={yes_price:.2f} no={no_price:.2f} "
            f"ask={yes_exec:.2f}/{no_exec:.2f} "
            f"=> {side} edge={chosen_edge:+.3f} (eY={edge_yes:+.3f} eN={edge_no:+.3f}) "
            f"drift={drift_signal_val:+.3f} mom={momentum_signal:+.3f} "
            f"pm={lanes['pm']:+.3f} "
            f"of(obi={lanes['obi']:+.3f} cvd={lanes['cvd']:+.3f}) "
            # Raw candidate-lane reads (pre-kill-switch) — logged for the
            # offline validation dataset, they carry zero decision weight.
            f"cand(fut={raw.get('fut_taker', 0):+.2f} "
            f"tech={raw.get('tech_mtf', 0):+.2f} "
            f"xa={raw.get('xasset', 0):+.2f} "
            f"lag={raw.get('lag', 0):+.2f} "
            f"ms={raw.get('ms_mom', 0):+.2f} "
            f"fd={raw.get('flow_decay', 0):+.2f}) "
            f"strat={strategy_signal:+.3f} "
            f"{target_shares:.2f}sh conf={confidence:.2f} "
            f"reg={_regime_label or '?'} "
            f"{blend.log_str()}"
        )
        # Stamp context only on buys (deferred from skip path).
        try:
            ctx_vec = build_context(sv.prices, signals, datetime.now(tz=timezone.utc))
        except Exception:
            ctx_vec = None
        _meta_m = re.search(
            r"meta\(mom=[+-][\d.]+ rev=[+-][\d.]+ (?:sent=[+-][\d.]+ )?"
            r"ph=[+-][\d.]+ \| reg=\w+\)",
            raw_signal.get("reasoning") or "",
        )
        if _meta_m:
            reasoning = f"{reasoning} {_meta_m.group(0)}"

        return {
            "action": "buy",
            "side": side,
            "edge": chosen_edge,
            "confidence": confidence,
            "reasoning": reasoning,
            # Contributing signal readings (structured contract) — the model
            # blend's own lane attribution is in lane_contributions below.
            "signals": {
                "drift": drift_signal_val, "mom": momentum_signal,
                "strat": strategy_signal, "model_prob": model_prob,
                "trust_eff": trust_eff,
                # Raw candidate reads (pre kill-switch) for decision_events.
                "fut": raw.get("fut_taker"), "tech": raw.get("tech_mtf"),
                "xasset": raw.get("xasset"),
                "lag": raw.get("lag"), "ms_mom": raw.get("ms_mom"),
                "flow_decay": raw.get("flow_decay"),
                "regime": _regime_label,
            },
            "suggested_amount": amount,
            "target_shares": target_shares,
            # Price the decision expects to pay. execute() turns this into a
            # slippage limit so an adverse book move between decision and fill
            # rejects the trade instead of filling worse (config.MAX_FILL_SLIPPAGE).
            "entry_price": round(price, 4),
            "features": features,
            "context": ctx_vec,
            # Per-lane weight x value attribution for this decision (also in
            # the persisted reasoning via blend.log_str()).
            "lane_contributions": dict(blend.contributions),
        }

    def execute(self, signal: dict, market: dict) -> dict:
        """Place a trade via the venue engine (paper sim or live Polymarket)."""
        if self._paused:
            logger.info(f"[{self.name}] Paused, skipping trade")
            return {"success": False, "reason": "bot_paused"}

        # Per-bot mode: fresh read from DB so dashboard toggles take effect immediately
        self.trading_mode = db.get_bot_mode(self.name)
        mode = self.trading_mode

        # Pure Kelly sizing: paper amounts are uncapped (the shared-pool gate
        # in venues/paper.py still refuses to overspend the pool). LIVE keeps
        # the hard per-trade safety cap — real money.
        amount = signal.get("suggested_amount", 0.0)
        # Zone/maker bots size via max_position × pct and skip the Kelly path
        # above — scale their amount by portfolio weight so capital allocation
        # still applies. Kelly decisions already baked weight into bankroll;
        # size_multiplier is 1.0 when allocation is off, and for equal-weight
        # Kelly we'd double-count if we always multiplied — so only scale when
        # the signal was NOT produced by the model-blend Kelly path (no
        # target_shares from make_decision). Heuristic: apply multiplier when
        # target_shares is absent (sniper/makers/phantom analyze amounts).
        if signal.get("target_shares") is None:
            mult = _portfolio_size_mult(self.name)
            if mult != 1.0:
                amount = amount * mult

        # Centralized Risk Engine: kill switch, daily/DD limits, size taper.
        # Replaces the inline daily-loss checks (still available as legacy
        # fallback when the engine is disabled).
        try:
            from arena.risk_engine import pre_trade
            risk = pre_trade(self.name, mode=mode, amount=amount)
            if not risk.allow:
                if risk.action in ("pause", "kill"):
                    self._paused = True
                logger.warning(
                    f"[{self.name}] Risk block: {risk.reason} (action={risk.action})")
                return {"success": False, "reason": risk.reason}
            # Kelly path already applied risk mult via bankroll in
            # make_decision; zone/maker bots (no target_shares) need the
            # taper applied to their max_pos-based amount here.
            if risk.size_mult < 0.999 and signal.get("target_shares") is None:
                amount = amount * risk.size_mult
        except Exception as e:
            logger.warning(f"[{self.name}] Risk engine error (fail-open to legacy): {e}")
            daily_loss = db.get_bot_daily_loss(self.name, mode)
            max_daily = config.get_max_daily_loss_per_bot()
            if daily_loss >= max_daily:
                self._paused = True
                return {"success": False, "reason": "daily_loss_limit"}
            total_daily = db.get_total_daily_loss(mode)
            max_total = config.get_max_daily_loss_total()
            if total_daily >= max_total:
                return {"success": False, "reason": "arena_loss_limit"}

        if mode == "live":
            amount = min(amount, config.LIVE_MAX_POSITION)

        # Shared-pool concentration cap (BUG #27): per-bot Kelly can't see the
        # correlated positions the OTHER bots just opened on the same (market,
        # side) — clamp to the pool's remaining headroom, or stand down.
        # Arbitrage is exempt (overrides execute(); its legs are hedged).
        market_id = market.get("condition_id") or market.get("id")
        headroom = self._exposure_headroom(market_id, signal.get("side"), mode)
        if headroom is not None and amount > headroom:
            min_viable = config.POLYMARKET_MIN_SHARES * max(
                signal.get("entry_price") or 0.5, 0.05)
            if headroom < min_viable:
                logger.info(
                    f"[{self.name}] Exposure cap: (market, {signal.get('side')}) "
                    f"pool headroom ${headroom:.2f} < min viable ${min_viable:.2f}, skipping")
                return {"success": False, "reason": "exposure_cap"}
            logger.info(
                f"[{self.name}] Exposure cap: clamping ${amount:.2f} -> ${headroom:.2f}")
            amount = headroom

        try:
            return self._place_via_engine(signal, market, amount, mode)
        except Exception as e:
            logger.error(f"[{self.name}] Trade exception: {e}")
            return {"success": False, "reason": str(e)}

    def _peer_corr(self, other_bot: str) -> float | None:
        """Pairwise correlation with another bot, or None if unknown.

        Unknown peers count at full weight in exposure (conservative). Measured
        ρ is floored at EXPOSURE_CORR_FLOOR so weakly related bots still share
        some budget.
        """
        if other_bot == self.name:
            return 1.0
        try:
            from arena.portfolio import load_state
            st = load_state() or {}
            pairs = st.get("correlations") or {}
            a, b = sorted([self.name, other_bot])
            key = f"{a}|{b}"
            if key in pairs:
                return max(0.0, min(1.0, float(pairs[key])))
        except Exception:
            pass
        return None

    def _effective_open_exposure(self, market_id, side, mode) -> tuple[float, int]:
        """Correlation-weighted open cost on (market, side) + bot count.

        Peers with ρ≈1 (momentum/phantom/hybrid) nearly fully share the
        concentration budget so tandem fills cannot 4× one candle. Returns
        (effective_usd, n_bots_open).
        """
        with db.get_conn() as conn:
            rows = conn.execute(
                """SELECT bot_name, SUM(amount) cost FROM trades
                   WHERE market_id=? AND side=? AND mode=? AND outcome IS NULL
                   GROUP BY bot_name""",
                (market_id, side, mode),
            ).fetchall()
        if not rows:
            # Fall back to unweighted total if query shape differs
            return float(db.get_open_exposure(market_id, side, mode) or 0.0), 0
        floor = float(getattr(config, "EXPOSURE_CORR_FLOOR", 0.35))
        aware = bool(getattr(config, "EXPOSURE_CORR_AWARE", True))
        eff = 0.0
        n_bots = 0
        for r in rows:
            name = r["bot_name"]
            cost = float(r["cost"] or 0.0)
            if cost <= 0:
                continue
            n_bots += 1
            if not aware or name == self.name:
                eff += cost
            else:
                rho = self._peer_corr(name)
                # Unknown corr → full weight (safe). Known → max(floor, ρ).
                weight = 1.0 if rho is None else max(floor, rho)
                eff += cost * weight
        return eff, n_bots

    def _exposure_headroom(self, market_id, side, mode) -> float | None:
        """Remaining shared-pool budget for this (market, side), or None when
        it can't be computed (missing ids — fail open, other guards still
        apply). Cap base: gross paper pool in paper mode; a fixed
        2x LIVE_MAX_POSITION per market-side in live mode.

        Long-term concentration control: correlation-weighted open exposure
        + hard max bots per (market, side) (MARKET_SIDE_MAX_BOTS).

        Short TTL cache (EXPOSURE_CACHE_TTL_SEC) shared across bots on the
        same tick; invalidated after every successful place.
        """
        if not market_id or side not in ("yes", "no"):
            return None
        global _exposure_cache
        now = time.time()
        ttl = float(getattr(config, "EXPOSURE_CACHE_TTL_SEC", 1.5))
        key = (str(market_id), side, mode, self.name)
        cache_ts, cache_map = _exposure_cache
        if (now - cache_ts) < ttl and key in cache_map:
            return cache_map[key]
        if (now - cache_ts) >= ttl:
            cache_map = {}
            cache_ts = now

        if mode == "live":
            cap_usd = 2.0 * config.LIVE_MAX_POSITION
        else:
            cap_usd = config.MARKET_SIDE_EXPOSURE_CAP * db.get_paper_pool_gross()
        try:
            used, n_bots = self._effective_open_exposure(market_id, side, mode)
        except Exception:
            used = float(db.get_open_exposure(market_id, side, mode) or 0.0)
            n_bots = 0
        max_bots = int(getattr(config, "MARKET_SIDE_MAX_BOTS", 3) or 0)
        headroom = cap_usd - used
        if max_bots > 0 and n_bots >= max_bots:
            # Already at thesis-cluster limit — no headroom for another bot
            # unless this bot already has a position (adding size).
            try:
                mine = 0.0
                with db.get_conn() as conn:
                    r = conn.execute(
                        """SELECT SUM(amount) c FROM trades
                           WHERE market_id=? AND side=? AND mode=? AND bot_name=?
                             AND outcome IS NULL""",
                        (market_id, side, mode, self.name),
                    ).fetchone()
                    mine = float((r["c"] if r else 0) or 0)
                if mine <= 0:
                    headroom = 0.0
            except Exception:
                headroom = 0.0
        cache_map[key] = headroom
        _exposure_cache = (cache_ts, cache_map)
        return headroom

    def _place_via_engine(self, signal, market, amount, mode) -> dict:
        """Delegate order placement to the paper or live venue engine.

        Paper → local simulated fill (``venues.paper``); live → Polymarket CLOB
        (``venues.live``). Adapts the engine's ``TradeResult`` to the legacy
        dict shape callers (trader.py, maker bots) expect.

        When the market carries a warm side book (``yes_book`` / ``no_book``
        from the market-data warmer), that snapshot is passed to the engine
        so the fill walks the SAME book the decision priced — not a fresh
        CLOB fetch that can move several cents in under a second.
        """
        from arena.market_data import side_book
        from venues import get_engine

        # Slippage band: reject a fill that deviates more than
        # MAX_FILL_SLIPPAGE from the decision's expected price in EITHER
        # direction (BUG #28: a fill far below expectation is stale data, not
        # a bargain). Only applied when the signal carries an expected
        # ``entry_price`` (all buy signals now do).
        expected = signal.get("entry_price")
        book = side_book(market, signal.get("side"))
        # Limit-first: price the resting/marketable buy from the book mode,
        # falling back to the decision entry when book is missing.
        lim = signal.get("limit_price")
        if lim is None and getattr(config, "ORDER_STYLE", "limit") == "limit":
            try:
                import polymarket_fills
                mid = (market.get("current_price") if signal.get("side") == "yes"
                       else market.get("no_price"))
                if book:
                    lim = polymarket_fills.limit_buy_price(book, mid=mid)
                if lim is None:
                    lim = expected
            except Exception:
                lim = expected

        # Note: do NOT pass Kelly ``target_shares`` here — that flag means
        # share-matched arb legs in the venue engines. Directional bots size
        # via USD amount (derived shares-first above) and use the limit path.
        res = get_engine(mode).place(
            bot_name=self.name,
            side=signal["side"],
            amount=amount,
            market=market,
            mode=mode,
            confidence=signal.get("confidence"),
            reasoning=signal.get("reasoning"),
            features=signal.get("features"),
            expected_price=expected,
            limit_price=lim,
            book=book,
            context=signal.get("context"),
        )
        if res.success:
            invalidate_exposure_cache()
        return {
            "success": res.success,
            "trade_id": res.trade_id,
            "reason": res.reason,
            "fill_source": res.fill_source,
        }

    def get_performance(self, hours=12) -> dict:
        """Get bot performance stats."""
        perf = db.get_bot_performance(self.name, hours)
        perf["name"] = self.name
        perf["strategy_type"] = self.strategy_type
        perf["generation"] = self.generation
        perf["paused"] = self._paused
        return perf

    def export_params(self) -> dict:
        return {
            "name": self.name,
            "strategy_type": self.strategy_type,
            "generation": self.generation,
            "lineage": self.lineage,
            "params": copy.deepcopy(self.strategy_params),
        }

    def mutate(self, winning_params: dict, mutation_rate: float | None = None) -> dict:
        """Create mutated params via Gaussian noise inside sensible bounds.

        Delegates to ``evolution.operators.mutate`` (the GA operator). The
        optional ``mutation_rate`` is the per-gene flip probability.
        """
        from evolution.operators import mutate as ga_mutate
        rate = mutation_rate
        if rate is None:
            rate = getattr(config, "GA_MUTATION_RATE", None) or config.MUTATION_RATE
        return ga_mutate(winning_params, rate=rate)

    def reset_daily(self):
        """Reset daily pause state."""
        self._paused = False

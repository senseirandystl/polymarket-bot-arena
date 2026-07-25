"""Abstract base class all arena bots inherit from."""

import random
import copy
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import db
import learning
import polymarket_fills
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
        "sentiment": 0.50,      # neutral
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
    _DEAD_LANES = {"pm": 0.00, "cvd": 0.00, "obi": 0.00,
                   "fut": 0.00, "tech": 0.00, "xasset": 0.00}
    # momentum/hybrid REBALANCED (BUG #30, 2026-07-20): the 24h/279-trade run
    # made momentum-v1 the worst-performing bot (-$31.85, 40.9% WR) and the
    # hybrid family collectively negative across all 4 live generations
    # (-$43 total) — both profiles lean heavily on the two lanes shown live to
    # be currently harmful (mid-magnitude mom noise pushing model_prob into
    # the toxic 0.10-0.30 drift band; high-confidence strat reads, see
    # STRAT_LANE_CONF_CAP above). Shifted weight toward drift, the one lane
    # that measured genuinely predictive at high magnitude (79.3% WR) and
    # whose own conviction scaling (MODEL_CONVICTION_SCALE) already keeps it
    # honest when uninformative — a smaller, conservative nudge, not a full
    # re-derivation, pending a longer run to confirm the direction.
    STRATEGY_SIGNAL_PROFILE = {
        "momentum":          {"drift": 0.35, "mom": 0.40, "strat": 0.25, **_DEAD_LANES},
        "phantom":           {"drift": 0.20, "mom": 0.30, "strat": 0.50, **_DEAD_LANES},
        "mean_reversion":    {"drift": 0.70, "mom": 0.00, "strat": 0.30, **_DEAD_LANES},
        "mean_reversion_sl": {"drift": 0.70, "mom": 0.00, "strat": 0.30, **_DEAD_LANES},
        "mean_reversion_tp": {"drift": 0.70, "mom": 0.00, "strat": 0.30, **_DEAD_LANES},
        "sentiment":         {"drift": 0.30, "mom": 0.00, "strat": 0.70, **_DEAD_LANES},
        "hybrid":            {"drift": 0.50, "mom": 0.20, "strat": 0.30, **_DEAD_LANES},
        "sniper":            {"drift": 0.50, "mom": 0.10, "strat": 0.15, **_DEAD_LANES},
    }
    DEFAULT_SIGNAL_PROFILE = {"drift": 0.50, "mom": 0.10, "strat": 0.15, **_DEAD_LANES}
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
        "sentiment": 0.50,
        "hybrid": 0.50,
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
    STRATEGY_MAX_SIDE_PRICE = {
        "mean_reversion": 0.58,
        "mean_reversion_sl": 0.58,
        "mean_reversion_tp": 0.58,
    }
    # 2026-07-21 (data-gathering): floors lowered (0.015->0.010, 0.02->0.012) to
    # un-starve the evaluation dataset — the fee-net bar + flow tax + conviction
    # scaling stacked into a ~6.5pt model-vs-ask requirement, yielding ~63k
    # no_edge skips per ~12 trades. Safety guards (drift-veto, dead-zone,
    # consensus, book-sum) are unchanged. Restore 0.015/0.02 after the window.
    MIN_EDGE = {
        "momentum": 0.010,
        "mean_reversion": 0.012,
        "mean_reversion_sl": 0.012,
        "mean_reversion_tp": 0.012,
        "sniper": 0.012,
        "phantom": 0.010,
        "sentiment": 0.012,
        "hybrid": 0.012,
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

    def _side_net_edges(self, model_prob: float, trust_eff: float,
                        yes_price: float, no_price: float) -> tuple:
        """Cost-adjusted edge per side, each anchored on its OWN book price.

        edge_side = trust_eff * (P_model_side - side_price) - taker_fee. The
        old form anchored fair on the YES mid but paid the NO book, so any
        cross-book gap (stale/inconsistent books, yes+no != 1) landed in the
        NO edge as phantom directional signal with zero model input — and
        Kelly max-sized exactly those trades (BUG #27). Per-side anchoring
        makes edge purely model-vs-that-side's-price; real cross-book gaps
        belong to the arbitrage bot's two-legged trade.
        """
        edge_yes = (trust_eff * (model_prob - yes_price)
                    - polymarket_fills.taker_fee(1.0, yes_price))
        edge_no = (trust_eff * ((1.0 - model_prob) - no_price)
                   - polymarket_fills.taker_fee(1.0, no_price))
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
        strat_cap = getattr(config, "STRAT_LANE_CONF_CAP", 0.60)
        strategy_signal = max(-strat_cap, min(strat_cap, strategy_signal))

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
        def _skip(reason: str, side: str = "yes", confidence: float = 0.0):
            return strategy_decision(
                "skip", side, confidence=confidence, reasoning=reason,
                signals={"drift": drift_signal_val, "mom": momentum_signal,
                         "strat": strategy_signal},
                features=features)

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
        if model_lean < config.MODEL_LEAN_MIN:
            return _skip(
                f"Model lean too weak: |{model_prob:.3f}-0.5|="
                f"{model_lean:.3f} < {config.MODEL_LEAN_MIN:.2f}")

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

        conf_yes = min(0.95, max(0.0, edge_yes) * config.EDGE_TO_CONFIDENCE)
        conf_no = min(0.95, max(0.0, edge_no) * config.EDGE_TO_CONFIDENCE)
        if edge_yes >= edge_no:
            side, side_price, chosen_edge, confidence = "yes", yes_exec, edge_yes, conf_yes
        else:
            side, side_price, chosen_edge, confidence = "no", no_exec, edge_no, conf_no

        # --- Dead-zone gate (2026-07-21): the single biggest live leak ---
        # A flat-drift opinion against a near-coin-flip market was 59 trades,
        # 39% WR, -$77.83 over the 290-trade run — the model manufacturing an
        # edge from noisy flow/strat lanes where the crowd is genuinely 50/50.
        # It fires BEFORE the edge gate: the coin-flip band with no drift
        # conviction is a "sit flat" region regardless of computed edge. The
        # SAME price band with |drift| >= DEAD_ZONE_DRIFT_MIN is the profitable
        # "market lags drift" trade (+$30.10, 65.7% WR) and passes through, so
        # the gate is drift-CONDITIONAL and regime-agnostic (keys off |drift|).
        side_mid_dz = yes_price if side == "yes" else no_price
        dz_lo = getattr(config, "DEAD_ZONE_PRICE_LO", 0.42)
        dz_hi = getattr(config, "DEAD_ZONE_PRICE_HI", 0.58)
        dz_drift = getattr(config, "DEAD_ZONE_DRIFT_MIN", 0.10)
        if dz_lo <= side_mid_dz <= dz_hi and abs(drift_signal_val) < dz_drift:
            return _skip(
                f"Dead-zone gate: {side} mid={side_mid_dz:.2f} in "
                f"[{dz_lo:.2f},{dz_hi:.2f}] & |drift|={abs(drift_signal_val):.3f}"
                f"<{dz_drift:.2f} (coin-flip, no conviction)",
                side=side, confidence=confidence)

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
        min_edge = self.MIN_EDGE.get(self.strategy_type, config.MIN_EDGE_DEFAULT)
        mult_max = getattr(config, "FLOW_ONLY_EDGE_MULT_MAX", 2.0)
        full_trust = max(getattr(config, "FLOW_ONLY_DRIFT_FULL_TRUST", 0.30), 1e-6)
        taper = max(0.0, 1.0 - abs(drift_signal_val) / full_trust)
        min_edge *= 1.0 + (mult_max - 1.0) * taper
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
        if side_mid > max_price:
            return _skip(
                f"High-price guard: {side} mid={side_mid:.2f} "
                f">{max_price:.2f}, priced-in / bad risk-reward",
                side=side, confidence=confidence)
        if side_mid < config.CONSENSUS_GUARD:
            return _skip(
                f"Consensus guard: {side} mid={side_mid:.2f} "
                f"<{config.CONSENSUS_GUARD:.2f}, fighting consensus",
                side=side, confidence=confidence)

        # --- Late-window conviction boost (smooth) ---
        # BTC direction increasingly locked in toward market close. The boost
        # ramps smoothly from x1.0 at 90s remaining to x1.25 inside 30s — the
        # old hard step at exactly 60s made 61s and 59s decisions discontinuous.
        time_rem = market.get("time_remaining_seconds")
        if time_rem is not None:
            late = smooth_ramp(-float(time_rem), -90.0, -30.0)
            confidence = min(0.95, confidence * (1.0 + 0.25 * late))

        # --- Bet sizing: pure fractional Kelly, SHARES-FIRST ---
        # Binary-market Kelly: buying a side at price c with true probability p
        # grows fastest at bankroll fraction f* = (p - c)/(1 - c); with the
        # fee-adjusted edge already computed, f* = edge/(1 - price). We bet
        # the Kelly fraction (live-editable in dashboard Settings; full Kelly
        # over-bets estimation error) of the LIVE bankroll — no per-trade or
        # %-of-balance caps (removed 2026-07-17 to run pure Kelly sizing; the
        # venue's shared-pool gate still prevents spending cash the pool
        # lacks). Size scales with edge, odds, AND bankroll — the old formula
        # (flat % of max_pos by confidence) sized wins and losses almost
        # identically ($3.83 vs $3.76).
        price = max(side_price, 0.01)
        # Portfolio capital slice: when allocation is on, this bot sizes
        # against bankroll × weight (weights sum to 1 across the roster).
        # Risk engine may further taper (drawdown / stress) via size_mult.
        bankroll = (_sizing_bankroll(self.trading_mode)
                    * _portfolio_weight(self.name)
                    * _risk_size_mult(self.name))
        # Edge is CLAMPED for sizing only (the trade/skip gate above used the
        # raw edge): outsized edges mean maximal model-vs-market disagreement,
        # which live correlates with stale inputs, not extra information (the
        # 15 biggest bets of the 24h run went 8/15 for -$34).
        sizing_edge = min(max(0.0, chosen_edge),
                          getattr(config, "KELLY_EDGE_CAP", 0.10))
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
            f"cand(fut={raw['fut_taker']:+.2f} "
            f"tech={raw['tech_mtf']:+.2f} "
            f"xa={raw['xasset']:+.2f}) "
            f"strat={strategy_signal:+.3f} "
            f"{target_shares:.2f}sh conf={confidence:.2f} "
            f"reg={self.regime_context(signals).get('label', '?')} "
            f"{blend.log_str()}"
        )

        return {
            "action": "buy",
            "side": side,
            "edge": chosen_edge,
            "confidence": confidence,
            "reasoning": reasoning,
            # Contributing signal readings (structured contract) — the model
            # blend's own lane attribution is in lane_contributions below.
            "signals": {"drift": drift_signal_val, "mom": momentum_signal,
                        "strat": strategy_signal, "model_prob": model_prob,
                        "trust_eff": trust_eff},
            "suggested_amount": amount,
            "target_shares": target_shares,
            # Price the decision expects to pay. execute() turns this into a
            # slippage limit so an adverse book move between decision and fill
            # rejects the trade instead of filling worse (config.MAX_FILL_SLIPPAGE).
            "entry_price": round(price, 4),
            "features": features,
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

    def _exposure_headroom(self, market_id, side, mode) -> float | None:
        """Remaining shared-pool budget for this (market, side), or None when
        it can't be computed (missing ids — fail open, other guards still
        apply). Cap base: gross paper pool in paper mode; a fixed
        2x LIVE_MAX_POSITION per market-side in live mode."""
        if not market_id or side not in ("yes", "no"):
            return None
        if mode == "live":
            cap_usd = 2.0 * config.LIVE_MAX_POSITION
        else:
            cap_usd = config.MARKET_SIDE_EXPOSURE_CAP * db.get_paper_pool_gross()
        return cap_usd - db.get_open_exposure(market_id, side, mode)

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
            book=book,
        )
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

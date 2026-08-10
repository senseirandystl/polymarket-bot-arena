"""Hybrid bot: regime-switching, online-learning meta-learner.

Upgraded from a static-weight ensemble (2026-07-18; online layer 2026-07-23).
Sub-strategy weights are dynamic, combining three smooth multipliers onto the
evolvable base weights:

1. **Volatility/trend regime** (``signals["vol_regime"]``, local compute from
   the BTC candle stream): trending tape up-weights the trend followers
   (momentum, phantom) and down-weights the fade book (mean reversion) — and
   vice versa in chop. The tilt is continuous in ``trend_score`` (0..1), never
   a hard regime switch.
2. **Recent live performance** of each sub-strategy's real arena bot (same
   shared DB, cached off the 1s hot path): a smooth logistic tilt around 50%
   WR, only trusted in proportion to sample size. A sub-strategy that is
   actually losing this session gets a quieter voice.
3. **Online meta-learning** (``bots/meta_learner.py``): every hybrid buy logs
   its sub-votes in the trade reasoning (``meta(...)`` token); at resolution
   each vote is scored against the actual market direction and the sub's
   multiplier updates Hedge-style (exp(±eta), clipped), kept PER REGIME
   BUCKET with sample-size shrinkage. State persists in arena_state
   ``hybrid_meta`` — shared across hybrid generations, visible in the
   dashboard (/api/hybrid-meta).

Sentiment sub-analyzer removed (2026-08 audit) — it injected kill-switched
pm/cvd flow into the strat lane. Sub-set is momentum / mean_rev / phantom.

The regime also tilts the hybrid's own SIGNAL profile (``_signal_profile``):
the mom lane's weight scales continuously with trend_score, bounded, while
drift — the validated fundamental — is never reduced below its class default.

The ensemble score still feeds the ``strat`` lane of the shared model blend —
all of BaseBot's validated guards (lean floor, drift veto, book-sum gate,
Kelly sizing) apply downstream unchanged.
"""

import logging
import time

import db
import config
from bots.base_bot import BaseBot, strategy_decision
from bots.bot_momentum import MomentumBot
from bots.bot_mean_rev import MeanRevBot
from bots.bot_phantom import PhantomBot
from bots.meta_learner import HybridMetaLearner, bucket_for, format_token
from signals.lab import SignalLab, SignalView

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "momentum_weight": 0.35,
    "mean_rev_weight": 0.35,
    "phantom_weight": 0.30,
    # 0.55->0.15: momentum and mean-reversion are opposite theses that partly
    # cancel in the weighted score, so a 0.55 gate meant the ensemble almost
    # never fired. 0.15 lets a net lean through (still scaled by the strat
    # lane weight in the shared model blend).
    "confidence_threshold": 0.15,
    "agreement_bonus": 0.15,   # bonus when multiple strategies agree
    # How far the regime tilt may swing a trend-sensitive weight (+/-).
    "regime_tilt": 0.5,
    # How far recent live WR may swing any weight (+/-).
    "perf_tilt": 0.4,
    # Online meta-learner (bots/meta_learner.py): Hedge step size and the
    # clip band for the per-sub multipliers learned from the hybrid's OWN
    # resolved trades. Evolution may tune these like any numeric param.
    "online_eta": 0.12,
    "online_min_mult": 0.4,
    "online_max_mult": 2.5,
    # How far the regime may tilt the hybrid's OWN mom-lane profile weight
    # (signal-level dynamic weighting; drift is never tilted down).
    "signal_regime_tilt": 0.4,
    "position_size_pct": 0.06,
    "min_confidence": 0.5,
}

# Sub-strategy -> (weight param, live bot-name prefix, trend sensitivity).
# Sensitivity +1 = thrives in trends, -1 = thrives in chop, 0 = agnostic.
SUBS = [
    ("momentum",  "momentum_weight",  "momentum",  +1.0),
    ("mean_rev",  "mean_rev_weight",  "meanrev",   -1.0),
    ("phantom",   "phantom_weight",   "phantom",   +1.0),
]
PERF_LOOKBACK_HOURS = 12
PERF_MIN_TRADES = 8      # below this, the WR tilt fades toward neutral


class HybridBot(BaseBot):
    def __init__(self, name="hybrid-v1", params=None, generation=0, lineage=None):
        merged = {**DEFAULT_PARAMS, **(params or {})}
        # Drop legacy sentiment_weight if present on old DB rows.
        merged.pop("sentiment_weight", None)
        super().__init__(
            name=name,
            strategy_type="hybrid",
            params=merged,
            generation=generation,
            lineage=lineage,
        )
        # Internal sub-analyzers (not full bots, just use their analyze logic)
        self._analyzers = {
            "momentum": MomentumBot(name="_internal_mom"),
            "mean_rev": MeanRevBot(name="_internal_mr"),
            "phantom": PhantomBot(name="_internal_ph"),
        }
        self._perf_tilt_cache: tuple = (0.0, {})  # (ts, {sub: tilt})
        # Online meta-learner: state is shared via arena_state, so every
        # hybrid generation (and each mutant) reads/extends one record.
        self._meta = HybridMetaLearner(
            eta=self.strategy_params.get("online_eta", 0.12),
            min_mult=self.strategy_params.get("online_min_mult", 0.4),
            max_mult=self.strategy_params.get("online_max_mult", 2.5),
        )
        # Regime context stashed by analyze() for _signal_profile(), which
        # BaseBot.make_decision calls AFTER analyze() on the same tick.
        self._last_regime: dict | None = None
        self._last_weight_detail: dict = {}
        # Tick-level cache for sub-analyze (shared within the same second).
        self._sub_analyze_cache: tuple = (0.0, None, {})

    # ------------------------------------------------------------------
    # Dynamic weighting
    # ------------------------------------------------------------------

    def _perf_tilts(self) -> dict:
        """Per-sub multiplicative tilt from recent LIVE arena performance.

        The scoring (logistic around 50% WR, sample-size damped) lives in
        SignalLab.score_perf_tilts; the fetch stays here (this module's
        ``db``) with the per-instance hot-path cache.
        """
        now = time.time()
        ttl = getattr(config, "HOTPATH_CACHE_TTL_SEC", 30)
        if (now - self._perf_tilt_cache[0]) < ttl:
            return self._perf_tilt_cache[1]

        subs = {sub: prefix for sub, _param, prefix, _sens in SUBS}
        try:
            perf = db.get_all_bots_performance(hours=PERF_LOOKBACK_HOURS)
            tilts = SignalLab.score_perf_tilts(
                perf, subs, min_trades=PERF_MIN_TRADES,
                max_tilt=self.strategy_params.get("perf_tilt", 0.4))
        except Exception as e:
            logger.debug(f"perf tilt unavailable: {e}")
            tilts = {sub: 1.0 for sub in subs}

        self._perf_tilt_cache = (now, tilts)
        return tilts

    def _dynamic_weights(self, signals: dict) -> dict:
        """Base weights x regime tilt x performance tilt x online multiplier.

        The full attribution (per-layer multipliers) is stashed on
        ``self._last_weight_detail`` for the reasoning/signals/dashboard.
        """
        sv = SignalView.of(signals)
        regime = sv.vol_regime
        # trend_score 0..1; recentre so 0.5-ish tape is neutral. NOTE: 0.0 is
        # a legitimate reading (pure chop) — only a MISSING key means neutral,
        # so no `or 0.5` coalescing here.
        raw_trend = regime.get("trend_score")
        if raw_trend is None:
            raw_trend = sv.market_regime.get("trend_score")
        trend = 2.0 * (0.5 if raw_trend is None else float(raw_trend)) - 1.0
        regime_tilt = self.strategy_params.get("regime_tilt", 0.5)
        perf_tilts = self._perf_tilts()

        # Online layer: lazily fold any newly-resolved hybrid trades into the
        # multipliers (TTL-throttled), then read the current-bucket blend.
        # Prefer robust detector id so high_vol_chop gets its own book.
        rid = (sv.regime_label
               or regime.get("regime_id")
               or (self._last_regime or {}).get("label"))
        bucket = bucket_for(
            raw_trend if raw_trend is None else float(raw_trend),
            regime_id=rid,
        )
        try:
            self._meta.maybe_update()
            online = self._meta.online_mults(bucket)
        except Exception as e:  # learner must never stall a tick
            logger.debug(f"meta learner unavailable: {e}")
            online = {sub: 1.0 for sub, *_ in SUBS}

        weights = {}
        for sub, param, _prefix, sens in SUBS:
            base = self.strategy_params.get(param, 0.25)
            w = (base * (1.0 + regime_tilt * sens * trend)
                 * perf_tilts.get(sub, 1.0) * online.get(sub, 1.0))
            weights[sub] = max(0.0, w)

        total = sum(weights.values())
        if total <= 0:
            weights = {sub: 1.0 / len(SUBS) for sub, *_ in SUBS}
        else:
            weights = {sub: w / total for sub, w in weights.items()}

        self._last_weight_detail = {
            "bucket": bucket, "trend": trend,
            "perf": dict(perf_tilts), "online": dict(online),
        }
        return weights

    def _signal_profile(self) -> dict:
        """Signal-level dynamic weighting: regime-tilted mom-lane weight.

        The mom lane (BTC 1-candle trend) earns more say on trending tape
        and less in chop, continuously and bounded by ``signal_regime_tilt``.
        Drift — the validated fundamental — keeps its class-default weight
        untouched. Uses the regime stashed by analyze() (make_decision calls
        analyze() first); no stash → class default profile.
        """
        prof = dict(super()._signal_profile())
        ctx = self._last_regime
        if ctx and ctx.get("known"):
            tilt = self.strategy_params.get("signal_regime_tilt", 0.4)
            t = 2.0 * ctx["trend_score"] - 1.0
            prof["mom"] = max(0.0, prof.get("mom", 0.0) * (1.0 + tilt * t))
        return prof

    def _cached_sub_analyze(self, market: dict, signals: dict) -> dict:
        """Cache sub-analyze results for this tick (market_id + second)."""
        mkt = (market or {}).get("id") or (market or {}).get("market_id")
        now = time.time()
        key = (mkt, int(now))
        ts, cached_key, results = self._sub_analyze_cache
        if cached_key == key and (now - ts) < 1.5:
            return results
        out = {}
        for sub, _param, _prefix, _sens in SUBS:
            out[sub] = self._analyzers[sub].analyze(market, signals)
        self._sub_analyze_cache = (now, key, out)
        return out

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def analyze(self, market: dict, signals: dict) -> dict:
        """Regime-, performance- and online-weighted vote over the subs."""
        # Stash regime for _signal_profile() (called later on this tick).
        self._last_regime = self.regime_context(signals)
        weights = self._dynamic_weights(signals)
        detail = self._last_weight_detail

        # Pre-gate: if drift is essentially flat and no lean, skip heavy work.
        sv = SignalView.of(signals)
        drift = float(sv.btc_drift or 0.0)

        weighted_score = 0.0
        active = []
        reasons = []
        votes = {}
        sub_results = self._cached_sub_analyze(market, signals)
        for sub, _param, _prefix, _sens in SUBS:
            sig = sub_results.get(sub) or {"action": "hold"}
            if sig["action"] == "hold":
                continue
            direction = 1 if sig["side"] == "yes" else -1
            weighted_score += direction * sig["confidence"] * weights[sub]
            active.append((sub, direction))
            votes[sub] = direction * sig["confidence"]
            reasons.append(f"{sub}[w={weights[sub]:.2f}]:{sig.get('reasoning', '')[:40]}")

        contributing = {"weights": dict(weights), "votes": votes,
                        "weighted_score": weighted_score,
                        "online": detail.get("online", {}),
                        "perf": detail.get("perf", {}),
                        "regime_bucket": detail.get("bucket", "mixed"),
                        "drift": drift}

        # Keep the dashboard's view of the effective weights fresh
        # (arena_state 'hybrid_meta' -> /api/hybrid-meta; persist throttled).
        try:
            self._meta.record_last(
                weights, detail.get("online", {}),
                self._last_regime["label"], detail.get("bucket", "mixed"))
        except Exception as e:
            logger.debug(f"meta record_last failed: {e}")

        if not active:
            return strategy_decision("hold", signals=contributing,
                                     reasoning="All sub-strategies say hold")

        yes_votes = sum(1 for _, d in active if d > 0)
        no_votes = sum(1 for _, d in active if d < 0)
        agreement = max(yes_votes, no_votes) >= 2

        confidence = abs(weighted_score)
        if agreement:
            # Scale the agreement bonus by STRAT_LANE_CONF_CAP so the hybrid's
            # internal confidence does not exceed what the shared model blend
            # will actually use (strat lane is capped at 0.25 downstream).
            strat_cap = float(getattr(config, "STRAT_LANE_CONF_CAP", 0.25))
            bonus = self.strategy_params.get("agreement_bonus", 0.15)
            # Full bonus only when weighted_score already in the cap's useful
            # range; scale down when near or above cap to avoid over-confidence.
            scale = max(0.3, 1.0 - abs(weighted_score) / max(strat_cap * 2.0, 1e-6))
            confidence = min(1.0, confidence + bonus * scale)

        if confidence < self.strategy_params.get("confidence_threshold", 0.15):
            return strategy_decision(
                "hold", signals=contributing,
                reasoning=f"Hybrid lean too weak: conf={confidence:.3f}")

        side = "yes" if weighted_score > 0 else "no"
        meta_tok = format_token(votes, detail.get("bucket", "mixed"))
        return strategy_decision(
            "buy", side,
            confidence=confidence,
            edge=abs(weighted_score),
            reasoning=(
                f"Hybrid {side} ({confidence:.2f}) {meta_tok} | "
                + "; ".join(reasons)[:180]
            ),
            signals=contributing,
        )

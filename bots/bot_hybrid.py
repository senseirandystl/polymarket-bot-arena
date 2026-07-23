"""Hybrid bot: regime-switching meta-learner over the sub-strategies.

Upgraded from a static-weight ensemble (2026-07-18). Sub-strategy weights are
now dynamic, combining two smooth multipliers onto the evolvable base weights:

1. **Volatility/trend regime** (``signals["vol_regime"]``, local compute from
   the BTC candle stream): trending tape up-weights the trend followers
   (momentum, phantom) and down-weights the fade book (mean reversion) — and
   vice versa in chop. The tilt is continuous in ``trend_score`` (0..1), never
   a hard regime switch.
2. **Recent live performance** of each sub-strategy's real arena bot (same
   shared DB, cached off the 1s hot path): a smooth logistic tilt around 50%
   WR, only trusted in proportion to sample size. A sub-strategy that is
   actually losing this session gets a quieter voice.

The ensemble score still feeds the ``strat`` lane of the shared model blend —
all of BaseBot's validated guards (lean floor, drift veto, book-sum gate,
Kelly sizing) apply downstream unchanged.
"""

import logging
import time

import db
import config
from bots.base_bot import BaseBot
from bots.bot_momentum import MomentumBot
from bots.bot_mean_rev import MeanRevBot
from bots.bot_sentiment import SentimentBot
from bots.bot_phantom import PhantomBot
from signals.curves import sigmoid

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "momentum_weight": 0.30,
    "mean_rev_weight": 0.30,
    "sentiment_weight": 0.15,
    "phantom_weight": 0.25,
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
    "position_size_pct": 0.06,
    "min_confidence": 0.5,
}

# Sub-strategy -> (weight param, live bot-name prefix, trend sensitivity).
# Sensitivity +1 = thrives in trends, -1 = thrives in chop, 0 = agnostic.
SUBS = [
    ("momentum",  "momentum_weight",  "momentum",  +1.0),
    ("mean_rev",  "mean_rev_weight",  "meanrev",   -1.0),
    ("sentiment", "sentiment_weight", "sentiment",  0.0),
    ("phantom",   "phantom_weight",   "phantom",   +1.0),
]
PERF_LOOKBACK_HOURS = 12
PERF_MIN_TRADES = 8      # below this, the WR tilt fades toward neutral


class HybridBot(BaseBot):
    def __init__(self, name="hybrid-v1", params=None, generation=0, lineage=None):
        merged = {**DEFAULT_PARAMS, **(params or {})}
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
            "sentiment": SentimentBot(name="_internal_sent"),
            "phantom": PhantomBot(name="_internal_ph"),
        }
        self._perf_tilt_cache: tuple = (0.0, {})  # (ts, {sub: tilt})

    # ------------------------------------------------------------------
    # Dynamic weighting
    # ------------------------------------------------------------------

    def _perf_tilts(self) -> dict:
        """Per-sub multiplicative tilt from recent LIVE arena performance.

        Logistic around 50% WR, damped by sample size (a 3-trade streak
        should barely move the needle). Cached off the 1s hot path.
        """
        now = time.time()
        ttl = getattr(config, "HOTPATH_CACHE_TTL_SEC", 30)
        if (now - self._perf_tilt_cache[0]) < ttl:
            return self._perf_tilt_cache[1]

        tilts = {}
        try:
            perf = db.get_all_bots_performance(hours=PERF_LOOKBACK_HOURS)
            max_tilt = self.strategy_params.get("perf_tilt", 0.4)
            for sub, _param, prefix, _sens in SUBS:
                rows = [p for name, p in perf.items()
                        if name.startswith(prefix)]
                trades = sum(p["total_trades"] for p in rows)
                wins = sum(p["wins"] for p in rows)
                if trades == 0:
                    tilts[sub] = 1.0
                    continue
                wr = wins / trades
                trust = min(1.0, trades / (2.0 * PERF_MIN_TRADES))
                # sigmoid(wr; 0.5, 12): 40% WR -> ~0.23, 60% WR -> ~0.77
                lean = 2.0 * sigmoid(wr, center=0.5, steepness=12.0) - 1.0
                tilts[sub] = 1.0 + max_tilt * lean * trust
        except Exception as e:
            logger.debug(f"perf tilt unavailable: {e}")
            tilts = {sub: 1.0 for sub, *_ in SUBS}

        self._perf_tilt_cache = (now, tilts)
        return tilts

    def _dynamic_weights(self, signals: dict) -> dict:
        """Base weights x smooth regime tilt x smooth performance tilt."""
        regime = signals.get("vol_regime", {}) or {}
        # trend_score 0..1; recentre so 0.5-ish tape is neutral. NOTE: 0.0 is
        # a legitimate reading (pure chop) — only a MISSING key means neutral,
        # so no `or 0.5` coalescing here.
        raw_trend = regime.get("trend_score")
        trend = 2.0 * (0.5 if raw_trend is None else float(raw_trend)) - 1.0
        regime_tilt = self.strategy_params.get("regime_tilt", 0.5)
        perf_tilts = self._perf_tilts()

        weights = {}
        for sub, param, _prefix, sens in SUBS:
            base = self.strategy_params.get(param, 0.25)
            w = base * (1.0 + regime_tilt * sens * trend) * perf_tilts.get(sub, 1.0)
            weights[sub] = max(0.0, w)

        total = sum(weights.values())
        if total <= 0:
            return {sub: 0.25 for sub, *_ in SUBS}
        return {sub: w / total for sub, w in weights.items()}

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def analyze(self, market: dict, signals: dict) -> dict:
        """Regime- and performance-weighted vote over the sub-strategies."""
        weights = self._dynamic_weights(signals)

        weighted_score = 0.0
        active = []
        reasons = []
        for sub, _param, _prefix, _sens in SUBS:
            sig = self._analyzers[sub].analyze(market, signals)
            if sig["action"] == "hold":
                continue
            direction = 1 if sig["side"] == "yes" else -1
            weighted_score += direction * sig["confidence"] * weights[sub]
            active.append((sub, direction))
            reasons.append(f"{sub}[w={weights[sub]:.2f}]:{sig.get('reasoning', '')[:40]}")

        if not active:
            return {"action": "hold", "side": "yes", "confidence": 0,
                    "reasoning": "All sub-strategies say hold"}

        yes_votes = sum(1 for _, d in active if d > 0)
        no_votes = sum(1 for _, d in active if d < 0)
        agreement = max(yes_votes, no_votes) >= 2

        confidence = abs(weighted_score)
        if agreement:
            confidence += self.strategy_params["agreement_bonus"]
        confidence = min(0.95, confidence)

        threshold = self.strategy_params["confidence_threshold"]
        if confidence < threshold:
            return {"action": "hold", "side": "yes", "confidence": confidence,
                    "reasoning": f"Ensemble confidence {confidence:.2f} below threshold {threshold}"}

        side = "yes" if weighted_score > 0 else "no"
        amount = config.get_max_position() * self.strategy_params["position_size_pct"]
        regime_label = (signals.get("vol_regime", {}) or {}).get("regime", "?")

        return {
            "action": "buy",
            "side": side,
            "confidence": confidence,
            "reasoning": (f"Meta[{regime_label}] ({yes_votes}Y/{no_votes}N, "
                          f"agree={agreement}): " + " | ".join(reasons)),
            "suggested_amount": amount,
        }

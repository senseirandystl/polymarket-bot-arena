"""SignalLab — the one place signals are read, weighted, gated and logged.

Every bot decision used to assemble its signal lanes ad-hoc inside
``BaseBot.make_decision`` (and each strategy's ``analyze()`` grabbed raw dict
keys). The lab centralizes that:

- **Consistent fetch + cache**: :meth:`SignalLab.compute_lanes` turns one
  combined-signals dict (arena/signals.build_combined_signals) into the
  normalized lane values every bot consumes — computed ONCE per tick per
  market and shared by all bots (the 8-bot slate no longer recomputes the
  same tanh eight times a second).
- **Dynamic weighting**: :meth:`blend` merges the per-strategy profile with
  the DB lane overrides (the closed-loop tuner/promoter output — the
  performance-based half) and a regime-conditional damp map (the
  regime-based half, currently the validated quiet-regime momentum damp).
  A ``set_model_hook`` seam is left for a light ML model later.
- **Validation gating**: lanes whose LIVE monitor report
  (arena/lane_monitor.py, arena_state ``lane_monitor``) shows failing
  accuracy are zero-weighted here as defense-in-depth, even before the
  override flip propagates.
- **Clean API**: bots read signals through :class:`SignalView` (typed
  accessors, dict-compatible) and probabilities through :meth:`blend`.
- **Contribution logging**: every blend returns per-lane contributions and
  logs them at debug; ``make_decision`` embeds them in trade reasoning.

The lab holds NO decision policy: guards, gates, sizing and side selection
stay in ``BaseBot.make_decision`` — the lab only answers "what do the
signals say and how much does each one count for this strategy".
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import config
from signals.curves import sigmoid, soft_saturate

logger = logging.getLogger(__name__)

# Lanes computed from MARKET-level signals (per-bot strat/learn lanes are
# appended by the caller). Order matters only for logging.
MARKET_LANES = (
    "drift", "mom", "pm", "cvd", "obi", "fut", "tech", "xasset",
    "lag", "ms_mom", "flow_decay",
)

# Regime-conditional lane damps — the regime-based half of dynamic weighting.
# Keys accept BOTH legacy labels (quiet/normal/trending/volatile) and the
# robust detector ids (high_vol_trend / low_vol_range / high_vol_chop /
# low_vol_trend). Quiet / low-vol damps the mom lane (validated 2026-07-19:
# momentum-driven trades in chop ran 47.9% WR / -$74). High-vol chop also
# damps mom (whipsaw) and slightly damps strat (overconfident theses).
# Add entries only with live or harness evidence.
def _quiet_mom():
    return getattr(config, "MOM_QUIET_REGIME_DAMP", 0.5)


def _chop_mom():
    return getattr(config, "MOM_CHOP_REGIME_DAMP", 0.45)


def _chop_strat():
    return getattr(config, "STRAT_CHOP_REGIME_DAMP", 0.70)


REGIME_LANE_DAMP: dict = {
    # Legacy
    "quiet": {"mom": _quiet_mom},
    "volatile": {"mom": _chop_mom},
    # Robust detector ids
    "low_vol_range": {"mom": _quiet_mom},
    "low_vol_trend": {"mom": _quiet_mom},  # quiet tape: 1-candle mom is noise
    "high_vol_chop": {"mom": _chop_mom, "strat": _chop_strat},
    "high_vol_trend": {},  # trend followers keep full mom weight
    "normal": {},
}

_LANE_CACHE_TTL = 1.0        # one warmer tick — all bots in a tick share
_MONITOR_CACHE_TTL = 30.0    # lane_monitor report refresh


class SignalView(Mapping):
    """Typed, read-only accessor over the combined-signals dict.

    Bots use the named properties instead of raw ``signals.get(...)`` so a
    renamed key breaks loudly in one place. It is still a Mapping, so code
    (and tests) that pass or read plain dicts keep working unchanged.
    """

    __slots__ = ("_d",)

    def __init__(self, signals: Optional[dict]):
        self._d = signals or {}

    @classmethod
    def of(cls, signals) -> "SignalView":
        return signals if isinstance(signals, cls) else cls(signals)

    # Mapping protocol — full backward compatibility with dict access.
    def __getitem__(self, k):
        return self._d[k]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    # BTC candle stream ---------------------------------------------------
    @property
    def prices(self) -> list:
        return self._d.get("prices", []) or []

    @property
    def volumes(self) -> list:
        return self._d.get("volumes", []) or []

    @property
    def latest(self) -> float:
        return float(self._d.get("latest", 0.0) or 0.0)

    # Fundamentals / flow --------------------------------------------------
    @property
    def btc_drift(self) -> float:
        return float(self._d.get("btc_drift", 0.0) or 0.0)

    @property
    def btc_strike(self) -> Optional[float]:
        """Window Price-to-Beat (Binance open @ eventStartTime), or None."""
        v = self._d.get("btc_strike")
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f > 0 else None

    @property
    def pm_momentum(self) -> float:
        return float(self._d.get("pm_momentum", 0.0) or 0.0)

    @property
    def cvd(self) -> float:
        return float(self._d.get("cvd", 0.0) or 0.0)

    @property
    def obi(self) -> float:
        return float(self._d.get("obi", 0.0) or 0.0)

    @property
    def orderflow(self) -> dict:
        return self._d.get("orderflow", {}) or {}

    # Context / candidates -------------------------------------------------
    @property
    def vol_regime(self) -> dict:
        return self._d.get("vol_regime", {}) or {}

    @property
    def market_regime(self) -> dict:
        """Robust detector snapshot (regime_id, features, confidence)."""
        return self._d.get("market_regime", {}) or {}

    @property
    def regime_label(self) -> Optional[str]:
        """Preferred rich regime id; falls back to legacy vol_regime label."""
        mr = self.market_regime
        if mr.get("regime_id") and mr["regime_id"] != "unknown":
            return mr["regime_id"]
        if mr.get("label") and mr["label"] != "unknown":
            return mr["label"]
        vr = self.vol_regime
        return vr.get("regime_id") or vr.get("regime")

    @property
    def technicals(self) -> dict:
        return self._d.get("technicals", {}) or {}

    @property
    def futures(self) -> dict:
        return self._d.get("futures", {}) or {}

    @property
    def xasset(self) -> float:
        return float(self._d.get("xasset", 0.0) or 0.0)

    @property
    def lag_residual(self) -> float:
        return float(self._d.get("lag_residual", 0.0) or 0.0)

    @property
    def ms_mom_1m(self) -> float:
        return float(self._d.get("ms_mom_1m", 0.0) or 0.0)

    @property
    def flow_cvd_decay(self) -> float:
        return float(self._d.get("flow_cvd_decay", 0.0) or 0.0)

    @property
    def micro_spread(self) -> float:
        return float(self._d.get("micro_spread", 0.0) or 0.0)

    @property
    def micro_spread_score(self) -> float:
        return float(self._d.get("micro_spread_score", 0.5) or 0.5)

    @property
    def macro_caution(self) -> float:
        return float(self._d.get("macro_caution", 0.0) or 0.0)

    @property
    def sentiment(self) -> dict:
        return self._d.get("sentiment", {}) or {}


@dataclass(frozen=True)
class BlendResult:
    """Model probability + full attribution for one decision."""
    prob: float
    weights: dict                  # lane -> weight actually applied
    contributions: dict            # lane -> weight * value (signed)
    gated: tuple = ()              # lanes zeroed by the validation gate

    def log_str(self) -> str:
        """Compact per-lane contribution string for reasoning/log lines."""
        parts = [f"{k}={v:+.3f}" for k, v in self.contributions.items()
                 if abs(v) > 1e-9]
        gate = f" gated={','.join(self.gated)}" if self.gated else ""
        return "P=" + f"{self.prob:.3f}[" + " ".join(parts) + "]" + gate


class SignalLab:
    """Central signal service — see module docstring.

    ``overrides_provider`` returns the approved-lane overrides dict
    (db.get_lane_overrides shape); the default reader lives in
    ``bots.base_bot._lane_overrides`` and is injected so the backtest
    runtime's isolation patch keeps working. ``monitor_provider`` returns
    the live lane-monitor report (arena_state ``lane_monitor``).
    """

    def __init__(self,
                 overrides_provider: Optional[Callable[[], dict]] = None,
                 monitor_provider: Optional[Callable[[], dict]] = None):
        self.overrides_provider = overrides_provider or (lambda: {})
        self.monitor_provider = monitor_provider or self._monitor_from_db
        self._model_hook: Optional[Callable] = None
        self._lock = threading.Lock()
        self._lane_cache: dict = {}          # key -> (ts, lanes, raw)
        self._monitor_cache: tuple = (0.0, {})
        self._perf_cache: dict = {}          # cache_key -> (ts, tilts)

    # ------------------------------------------------------------------
    # Providers / caches
    # ------------------------------------------------------------------

    @staticmethod
    def _monitor_from_db() -> dict:
        import db
        try:
            raw = db.get_arena_state("lane_monitor")
            return json.loads(raw) if raw else {}
        except Exception:
            return {}

    def _monitor_report(self) -> dict:
        now = time.time()
        with self._lock:
            ts, report = self._monitor_cache
            if (now - ts) < _MONITOR_CACHE_TTL:
                return report
        try:
            report = self.monitor_provider() or {}
        except Exception:
            report = {}
        with self._lock:
            self._monitor_cache = (now, report)
        return report

    def gated_lanes(self) -> frozenset:
        """Lanes currently failing LIVE validation (lane_monitor report).

        A lane is gated when its monitor verdict is ``disabled`` OR its
        accuracy sits below the demotion bar with a full sample — belt and
        braces on top of db.disable_lane_override, so a failing lane stops
        counting within one cache TTL even if the override flip lags.
        """
        report = self._monitor_report()
        gated = set()
        for lane, r in report.items():
            try:
                if r.get("verdict") == "disabled":
                    gated.add(lane)
                    continue
                n = int(r.get("n") or 0)
                acc = r.get("accuracy")
                min_n = int(r.get("min_trades")
                            or getattr(config, "LANE_MONITOR_MIN_TRADES", 50))
                min_acc = float(r.get("min_accuracy")
                                or getattr(config, "LANE_MONITOR_MIN_ACCURACY",
                                           0.53))
                if acc is not None and n >= min_n and float(acc) < min_acc:
                    gated.add(lane)
            except (TypeError, ValueError):
                continue
        return frozenset(gated)

    # ------------------------------------------------------------------
    # Lane computation (market-level, shared by every bot)
    # ------------------------------------------------------------------

    def compute_lanes(self, market: Optional[dict], signals,
                      overrides: Optional[dict] = None) -> tuple:
        """Normalized market-level lanes + raw reads, cached per tick.

        Returns ``(lanes, raw)``:
          * ``lanes`` — lane -> value in [-1, 1], kill-switches/overrides and
            the regime damp already applied (the exact values the model
            blend consumes),
          * ``raw`` — pre-kill-switch reads used for feature extraction and
            the ``cand(...)`` reasoning log (the offline-validation dataset).

        Cached by a VALUE key (market id + the numeric inputs + the override
        set) for one warmer tick, so all bots deciding on the same tick share
        one computation and see IDENTICAL values. A value key, not object
        identity: a freed signals dict's id() can be recycled by the next
        tick's dict inside the TTL, which would silently serve stale lanes.
        """
        sv = SignalView.of(signals)
        mkt_id = (market or {}).get("id") or (market or {}).get("market_id")
        overrides = overrides if overrides is not None else self.overrides_provider()

        prices = sv.prices
        fut_raw_k = float(sv.futures.get("taker_delta", 0.0) or 0.0)
        tech_raw_k = float(sv.technicals.get("mtf_score", 0.0) or 0.0)
        key = (mkt_id, sv.latest, sv.btc_drift, sv.pm_momentum, sv.cvd,
               sv.obi, fut_raw_k, tech_raw_k, sv.xasset, sv.regime_label,
               len(prices),
               prices[-1] if prices else 0.0,
               prices[-2] if len(prices) >= 2 else 0.0,
               tuple(sorted((k, bool((v or {}).get("enabled")))
                            for k, v in overrides.items())))
        now = time.time()
        with self._lock:
            hit = self._lane_cache.get(key)
            if hit and (now - hit[0]) < _LANE_CACHE_TTL:
                return hit[1], hit[2]

        def _mult(lane: str, config_switch: float) -> float:
            if (overrides.get(lane, {}) or {}).get("enabled"):
                return 1.0
            return config_switch

        # --- mom: BTC 1-candle momentum, tanh at 0.2% (~p97; BUG #25) ---
        prices = sv.prices
        btc_latest = sv.latest
        price_momentum = 0.0
        if len(prices) >= 2 and prices[-1] > 0:
            price_momentum = (prices[-1] - prices[-2]) / prices[-2]
        elif btc_latest > 0 and len(prices) >= 1 and prices[-1] > 0:
            price_momentum = (btc_latest - prices[-1]) / prices[-1]
        mom = soft_saturate(price_momentum, 0.002)

        # Regime-conditional damp (see REGIME_LANE_DAMP). Prefer rich id,
        # then legacy label so both detector and older vol_regime work.
        rid = sv.regime_label or ""
        legacy = (sv.market_regime.get("legacy")
                  or sv.vol_regime.get("regime") or "")
        damps = dict(REGIME_LANE_DAMP.get(rid, {}) or {})
        if not damps and legacy:
            damps = dict(REGIME_LANE_DAMP.get(legacy, {}) or {})
        if "mom" in damps:
            mom *= damps["mom"]()

        # --- pm: PM in-market momentum (0.15 move saturates; kill-switched) ---
        pm = max(-1.0, min(1.0, sv.pm_momentum / 0.15))
        pm *= _mult("pm", config.SIGNAL_WEIGHT_PM)

        # --- flow: OBI (killed) + CVD (killed pending re-validation) ---
        obi = max(-1.0, min(1.0, sv.obi)) * _mult(
            "obi", config.SIGNAL_WEIGHT_OBI)
        cvd = max(-1.0, min(1.0, sv.cvd)) * _mult(
            "cvd", config.SIGNAL_WEIGHT_CVD)

        # --- candidate lanes: fut / tech / xasset (killed until approved) ---
        fut_raw = float(sv.futures.get("taker_delta", 0.0) or 0.0)
        fut = max(-1.0, min(1.0, fut_raw)) * _mult(
            "fut", getattr(config, "SIGNAL_WEIGHT_FUT", 0.0))
        tech_raw = float(sv.technicals.get("mtf_score", 0.0) or 0.0)
        tech = max(-1.0, min(1.0, tech_raw)) * _mult(
            "tech", getattr(config, "SIGNAL_WEIGHT_TECH", 0.0))
        xa_raw = sv.xasset
        xasset = max(-1.0, min(1.0, xa_raw)) * _mult(
            "xasset", getattr(config, "SIGNAL_WEIGHT_XASSET", 0.0))

        # --- Expanded candidates (2026-08): lag residual, multiscale mom, flow ---
        lag_raw = float(sv.lag_residual or 0.0)
        lag = max(-1.0, min(1.0, lag_raw)) * _mult(
            "lag", getattr(config, "SIGNAL_WEIGHT_LAG", 0.0))
        ms_raw = float(sv.ms_mom_1m or 0.0)
        ms_mom = max(-1.0, min(1.0, ms_raw)) * _mult(
            "ms_mom", getattr(config, "SIGNAL_WEIGHT_MS_MOM", 0.0))
        fd_raw = float(sv.flow_cvd_decay or 0.0)
        flow_decay = max(-1.0, min(1.0, fd_raw)) * _mult(
            "flow_decay", getattr(config, "SIGNAL_WEIGHT_FLOW_DECAY", 0.0))

        # --- drift: the validated fundamental (already bounded/time-scaled) ---
        drift = max(-1.0, min(1.0, sv.btc_drift))

        lanes = {
            "drift": drift, "mom": mom, "pm": pm, "cvd": cvd,
            "obi": obi, "fut": fut, "tech": tech, "xasset": xasset,
            "lag": lag, "ms_mom": ms_mom, "flow_decay": flow_decay,
        }
        raw = {
            "price_momentum": price_momentum, "fut_taker": fut_raw,
            "tech_mtf": tech_raw, "xasset": xa_raw,
            "lag": lag_raw, "ms_mom": ms_raw, "flow_decay": fd_raw,
            "micro_spread": float(sv.micro_spread or 0.0),
        }

        with self._lock:
            if len(self._lane_cache) > 16:
                cutoff = now - 5 * _LANE_CACHE_TTL
                self._lane_cache = {k: v for k, v in self._lane_cache.items()
                                    if v[0] >= cutoff}
            self._lane_cache[key] = (now, lanes, raw)
        return lanes, raw

    # ------------------------------------------------------------------
    # Weighting + blend
    # ------------------------------------------------------------------

    def weights_for(self, strategy_type: str, lanes: dict, profile: dict,
                    overrides: Optional[dict] = None) -> tuple:
        """Effective per-lane weights for one strategy: (weights, gated).

        Priority per lane: an ENABLED override's per-strategy profile weight
        (the tuner/promoter closed loop) beats the static profile; lanes the
        profile doesn't name keep weight 1.0 (their value carries the weight
        — the strat/learn convention). Lanes failing live validation are
        zeroed last.
        """
        overrides = overrides if overrides is not None else self.overrides_provider()
        gated = self.gated_lanes()
        weights = {}
        hit_gate = []
        for k in lanes:
            ov = overrides.get(k)
            if ov and ov.get("enabled"):
                w = float(ov.get("profile", {}).get(strategy_type, 0.0))
            else:
                w = profile.get(k, 1.0)
            if k in gated and w != 0.0:
                w = 0.0
                hit_gate.append(k)
            weights[k] = w
        return weights, tuple(hit_gate)

    @staticmethod
    def regime_damps_for(signals) -> dict:
        """Active regime-conditional lane multipliers (lane -> float in 0..1)."""
        sv = SignalView.of(signals)
        rid = sv.regime_label or ""
        legacy = (sv.market_regime.get("legacy")
                  or sv.vol_regime.get("regime") or "")
        raw = dict(REGIME_LANE_DAMP.get(rid, {}) or {})
        if not raw and legacy:
            raw = dict(REGIME_LANE_DAMP.get(legacy, {}) or {})
        return {k: (fn() if callable(fn) else float(fn)) for k, fn in raw.items()}

    def blend(self, strategy_type: str, lanes: dict, profile: dict,
              overrides: Optional[dict] = None,
              signals=None) -> BlendResult:
        """Model probability of YES from weighted lanes, with attribution.

        P_model = clamp(0.5 + 0.5 * sum(w_lane * lane)). If a model hook is
        installed (future light-ML), it may replace the probability; the
        linear contributions are still logged so attribution never goes dark.

        When ``signals`` is provided, regime damps for non-mom lanes (e.g.
        strat under high_vol_chop) are applied here — mom is already damped
        inside :meth:`compute_lanes`.
        """
        adj_lanes = dict(lanes)
        if signals is not None:
            for k, mult in self.regime_damps_for(signals).items():
                if k == "mom":
                    continue  # already applied in compute_lanes
                if k in adj_lanes:
                    adj_lanes[k] = adj_lanes[k] * float(mult)

        weights, gated = self.weights_for(strategy_type, adj_lanes, profile,
                                          overrides)
        contributions = {k: weights[k] * v for k, v in adj_lanes.items()}
        s = sum(contributions.values())
        prob = max(config.MODEL_PROB_MIN,
                   min(config.MODEL_PROB_MAX, 0.5 + 0.5 * s))

        if self._model_hook is not None:
            try:
                hooked = self._model_hook(strategy_type, dict(adj_lanes),
                                          dict(weights))
                if hooked is not None and 0.0 <= float(hooked) <= 1.0:
                    prob = max(config.MODEL_PROB_MIN,
                               min(config.MODEL_PROB_MAX, float(hooked)))
            except Exception as e:  # a broken model must never stall a tick
                logger.warning(f"signal-lab model hook failed: {e}")

        result = BlendResult(prob=prob, weights=weights,
                             contributions=contributions, gated=gated)
        logger.debug(f"[{strategy_type}] {result.log_str()}")
        return result

    def set_model_hook(self, fn: Optional[Callable]) -> None:
        """Install a probability model: fn(strategy_type, lanes, weights) ->
        prob in [0,1] or None (= keep the linear blend). The seam for a
        light ML model later; None clears it."""
        self._model_hook = fn

    # ------------------------------------------------------------------
    # Strategy-performance tilts (hybrid's meta-learner input)
    # ------------------------------------------------------------------

    @staticmethod
    def score_perf_tilts(perf: dict, subs: dict,
                         min_trades: int = 8, max_tilt: float = 0.4) -> dict:
        """Pure scoring: per-sub multiplicative tilt from live performance.

        ``perf`` is db.get_all_bots_performance's shape (bot name -> stats);
        ``subs`` maps sub-name -> live bot-name prefix. Logistic around 50%
        WR, damped by sample size (a 3-trade streak barely moves the
        needle). No I/O — callers own fetching/caching so their tests can
        stub the data source.
        """
        tilts = {sub: 1.0 for sub in subs}
        for sub, prefix in subs.items():
            rows = [p for name, p in perf.items() if name.startswith(prefix)]
            trades = sum(p["total_trades"] for p in rows)
            wins = sum(p["wins"] for p in rows)
            if trades == 0:
                continue
            wr = wins / trades
            trust = min(1.0, trades / (2.0 * min_trades))
            lean = 2.0 * sigmoid(wr, center=0.5, steepness=12.0) - 1.0
            tilts[sub] = 1.0 + max_tilt * lean * trust
        return tilts

    def perf_tilts(self, subs: dict, lookback_hours: int = 12,
                   min_trades: int = 8, max_tilt: float = 0.4) -> dict:
        """Cached, DB-backed :meth:`score_perf_tilts` — one query per TTL
        shared by every caller with the same parameters."""
        import db
        cache_key = (tuple(sorted(subs.items())), lookback_hours,
                     min_trades, round(max_tilt, 4))
        now = time.time()
        ttl = getattr(config, "HOTPATH_CACHE_TTL_SEC", 30)
        with self._lock:
            hit = self._perf_cache.get(cache_key)
            if hit and (now - hit[0]) < ttl:
                return hit[1]

        try:
            perf = db.get_all_bots_performance(hours=lookback_hours)
            tilts = self.score_perf_tilts(perf, subs, min_trades, max_tilt)
        except Exception as e:
            logger.debug(f"perf tilts unavailable: {e}")
            tilts = {sub: 1.0 for sub in subs}

        with self._lock:
            self._perf_cache[cache_key] = (now, tilts)
        return tilts


_lab: Optional[SignalLab] = None


def get_lab() -> SignalLab:
    """Process-wide SignalLab singleton. ``bots.base_bot`` injects the
    overrides provider on import so backtest isolation keeps working."""
    global _lab
    if _lab is None:
        _lab = SignalLab()
    return _lab

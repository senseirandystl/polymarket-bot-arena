"""Bot 2: Mean Reversion strategy."""

import math
from typing import Optional, Sequence

import config
from bots.base_bot import BaseBot, strategy_decision
from signals.curves import smooth_ramp
from signals.lab import SignalView

DEFAULT_PARAMS = {
    # Continuous-tape fallback lookback when the live window has too few
    # closed 1m candles for a stable z-score (see _resolve_lookback).
    "lookback_candles": 10,
    # Prefer a window-local series once this many closed 1m bars exist
    # since event open (P1). 5-min markets yield at most 5 closed candles.
    "min_window_candles": 3,
    "bb_std_dev": 2.0,         # Bollinger Band width
    "rsi_period": 14,
    "rsi_oversold": 40,
    "rsi_overbought": 60,
    # Retrace of TWAP toward this window's strike (not a 4-bar z-score).
    "pullback_min": 0.15,  # Pass B (was 0.20); keep min_drift=0.20
    # Drift-agreement gate (BUG #28): the fade may only fire toward the side
    # a signed btc_drift of at least this magnitude already favors. Ungated,
    # the z-fade was a pure contrarian knife-catcher — 10 of 11 live trades
    # fired with drift 0.00-0.08 and ALL lost (-$55.30; the documented
    # "contrarian loses in 5-min markets" death class). Gated, the identity
    # becomes "buy the dip in the WINNING direction": drift picks the side,
    # TWAP-vs-strike pullback times the entry.
    # Raised 0.12→0.20 (2026-08-11): weak drift + high profile weight made
    # meanrev a mom clone at 55–58¢.
    "min_drift": 0.20,
    # PTB mean gate (P0): reversion TARGET (the z-score mean) must sit on the
    # same side of the Price-to-Beat as the bet. Fading UP → NO only when
    # mean ≤ strike (reversion still finishes ≤ PTB); fading DOWN → YES only
    # when mean ≥ strike. Drift alone (current vs strike) is not enough —
    # if the mean is still above PTB, reverting to it does not win a DOWN bet.
    # Regime conditioning: pure mean-reversion is a RANGING-market thesis —
    # "contrarian loses in 5-min markets" is the documented death class, and
    # fading a genuine trend is exactly how. Confidence is damped by up to
    # this fraction as trend_score rises past ~0.35 (full damp by ~0.75);
    # clearly-ranging tape (or no regime reading) fades nothing.
    "trending_conf_damp": 0.60,
    "position_size_pct": 0.05,
    "min_confidence": 0.55,
}


def _window_age_seconds(market: Optional[dict]) -> float:
    """Seconds since window open (0 if unknown)."""
    if not market:
        return 0.0
    window = float(getattr(config, "MARKET_WINDOW_SEC", 300) or 300)
    tr = market.get("time_remaining_seconds")
    if tr is not None:
        try:
            return max(0.0, window - float(tr))
        except (TypeError, ValueError):
            pass
    age = market.get("window_age_seconds")
    if age is not None:
        try:
            return max(0.0, float(age))
        except (TypeError, ValueError):
            pass
    return 0.0


def resolve_lookback(
    market: Optional[dict],
    n_prices: int,
    *,
    max_lookback: int = 10,
    min_window_candles: int = 3,
) -> tuple[int, str]:
    """Pick z-score lookback: window-local when possible, else continuous.

    Closed 1m candles since open ≈ floor(window_age / 60), capped at the
    5-min window (5 bars). Prefer that series once ≥ ``min_window_candles``
    closed bars exist and the feed has them; otherwise fall back to the
    continuous ``max_lookback`` (PTB mean gate still applies).

    Returns ``(lookback, source)`` where source is ``"window"``,
    ``"continuous"``, or ``"none"``.
    """
    max_lookback = max(1, int(max_lookback))
    min_window = max(1, int(min_window_candles))
    window_sec = float(getattr(config, "MARKET_WINDOW_SEC", 300) or 300)
    max_window_bars = max(1, int(window_sec // 60))

    age = _window_age_seconds(market)
    closed_in_window = min(int(age // 60), max_window_bars, max_lookback)

    if closed_in_window >= min_window and n_prices >= closed_in_window:
        return closed_in_window, "window"
    # Cross-window tape is the wrong object vs this strike — do not fade.
    return 0, "none"


def strike_pullback(
    btc_now: float,
    strike: float,
    window_prices: Sequence[float],
) -> tuple[float, int, float]:
    """Retrace of current TWAP toward this window's strike.

    Returns ``(pullback_frac, side_sign, extreme)``:
    * side_sign +1 if ``btc_now`` is above strike (YES winning), −1 below
    * pullback_frac 0 = sitting on the window extreme, 1 = back at strike
    * extreme is the window high (YES) or low (NO)
    """
    if strike is None or strike <= 0 or btc_now is None or btc_now <= 0:
        return 0.0, 0, 0.0
    path = [float(p) for p in (window_prices or []) if p and float(p) > 0]
    if not path:
        return 0.0, 0, 0.0
    now = float(btc_now)
    k = float(strike)
    if now >= k:
        extreme = max(path)
        if extreme <= k:
            return 0.0, 1, extreme
        width = extreme - k
        if width <= 0:
            return 0.0, 1, extreme
        frac = max(0.0, min(1.0, (extreme - now) / width))
        return frac, 1, extreme
    extreme = min(path)
    if extreme >= k:
        return 0.0, -1, extreme
    width = k - extreme
    if width <= 0:
        return 0.0, -1, extreme
    frac = max(0.0, min(1.0, (now - extreme) / width))
    return frac, -1, extreme


class MeanRevBot(BaseBot):
    def __init__(self, name="meanrev-v1", params=None, generation=0, lineage=None):
        super().__init__(
            name=name,
            strategy_type="mean_reversion",
            params=params or DEFAULT_PARAMS.copy(),
            generation=generation,
            lineage=lineage,
        )

    def _calc_rsi(self, prices, period):
        if len(prices) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, len(prices)):
            delta = prices[i] - prices[i-1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))

        gains = gains[-period:]
        losses = losses[-period:]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_zscore_and_mean(
        self, prices: Sequence[float], lookback: int
    ) -> tuple[float, float]:
        """Return (z-score, window mean) over the last ``lookback`` prices."""
        if lookback <= 0 or len(prices) < lookback:
            return 0.0, 0.0
        window = list(prices[-lookback:])
        mean = sum(window) / len(window)
        variance = sum((p - mean) ** 2 for p in window) / len(window)
        std = math.sqrt(variance) if variance > 0 else 1.0
        z = (window[-1] - mean) / std if std > 0 else 0.0
        return z, mean

    def _twap_path(self, signals: dict, min_n: int, market: dict | None = None) -> list[float]:
        """60s settlement candles — TWAP on Polymarket, BRTI on Kalshi."""
        from signals.tape import candle_prices, is_kalshi_market
        raw = signals if isinstance(signals, dict) else {}
        if not is_kalshi_market(market) and not raw.get("btc_twap_ticks"):
            # No TWAP series → fixture/spot path (mix guard can fire).
            return []
        tw = candle_prices(market, raw, sample_sec=60.0)
        if len(tw) < min_n:
            return []
        return [float(x) for x in tw if x and float(x) > 0]

    def analyze(self, market: dict, signals: dict) -> dict:
        """Buy the TWAP dip toward this window's strike, gated by drift."""
        sv = SignalView.of(signals)
        p = self.strategy_params
        max_lb = int(p.get("lookback_candles", 10))
        min_win = int(p.get("min_window_candles", 3))
        sig_dict = signals if isinstance(signals, dict) else {}

        tw_prices = self._twap_path(sig_dict, min_win, market)
        if tw_prices:
            prices = tw_prices
            from signals.tape import is_kalshi_market
            path_kind = "brti" if is_kalshi_market(market) else "twap"
        else:
            prices = list(sv.prices)
            path_kind = "fixture"
            last = float(prices[-1]) if prices else 0.0
            now_hint = float(sv.btc_now or 0.0)
            # Live TWAP now vs spot candles is BUG #23-class mix — sit out.
            if now_hint > 0 and last > 0 and abs(now_hint - last) / now_hint > 0.0005:
                return strategy_decision(
                    "hold",
                    reasoning=(
                        "meanrev: TWAP now vs spot path mix "
                        f"(now={now_hint:.2f} last={last:.2f})"
                    ),
                )

        lookback, lb_source = resolve_lookback(
            market, len(prices),
            max_lookback=max_lb,
            min_window_candles=min_win,
        )
        if lookback <= 0:
            return strategy_decision("hold", reasoning="insufficient data")

        window = list(prices[-lookback:])
        zscore, mean = self._calc_zscore_and_mean(prices, lookback)
        rsi = self._calc_rsi(prices, p["rsi_period"])
        amount = config.get_max_position() * p["position_size_pct"]
        pb_min = float(p.get(
            "pullback_min",
            getattr(config, "MEANREV_PULLBACK_MIN", 0.20),
        ))

        strike = sv.btc_strike
        # Stay on the resolution object: TWAP now, else last of the chosen path.
        btc_now = float(sv.btc_now or 0.0) or (
            float(window[-1]) if window else 0.0
        )
        if strike is None or btc_now <= 0:
            return strategy_decision("hold", reasoning="insufficient strike/TWAP")

        drift = sv.btc_drift
        min_drift = p.get("min_drift", 0.10)
        fade_no_ok = drift <= -min_drift
        fade_yes_ok = drift >= min_drift

        _min_fade_floor = float(getattr(config, "MEANREV_MIN_FADE_DRIFT", 0.40))
        if abs(drift) >= _min_fade_floor:
            return strategy_decision(
                "hold",
                signals={"drift": drift, "btc_now": btc_now, "strike": strike},
                reasoning=(
                    f"Meanrev identity guard: |drift|={abs(drift):.3f}"
                    f">={_min_fade_floor:.2f} — strong trend, not fading;"
                    f" stand down to avoid duplicate trend-following"
                ),
            )

        pb, side_sign, extreme = strike_pullback(btc_now, strike, window)

        regime = self.regime_context(signals)
        damp = p.get("trending_conf_damp", 0.60)
        regime_factor = 1.0
        if regime["known"] and not regime["ranging"]:
            regime_factor = 1.0 - damp * smooth_ramp(
                regime["trend_score"], 0.35, 0.75)

        strike_s = f"{strike:.2f}" if strike is not None else "na"
        soak = (
            f"strike={strike_s} btc_now={btc_now:.2f} extreme={extreme:.2f} "
            f"pb={pb:.2f} lb={lookback}/{lb_source}/{path_kind}"
        )
        contributing = {
            "zscore": zscore,
            "rsi": rsi,
            "drift": drift,
            "mean": mean,
            "pullback": pb,
            "extreme": extreme,
            "strike": strike,
            "btc_now": btc_now,
            "lookback": lookback,
            "lookback_source": lb_source,
            "regime": regime["label"],
            "regime_factor": regime_factor,
        }

        if pb < pb_min or side_sign == 0:
            return strategy_decision(
                "hold", signals=contributing,
                reasoning=(
                    f"No TWAP pullback vs strike: pb={pb:.2f}<{pb_min:.2f} "
                    f"| {soak}"))

        if side_sign < 0:
            # Still below strike: bounce toward PTB in a DOWN window → NO.
            if not fade_no_ok:
                return strategy_decision(
                    "hold", signals=contributing,
                    reasoning=(
                        f"Fade NO not drift-backed: pb={pb:.2f}, "
                        f"drift={drift:+.3f} | {soak}"))
            rsi_boost = (
                max(0.0, rsi - p["rsi_overbought"]) * 0.005
                if rsi is not None else 0.0
            )
            confidence = min(0.95, (0.35 + pb * 0.40 + rsi_boost)
                             * regime_factor)
            return strategy_decision(
                "buy", "no",
                edge=min(0.10, pb * 0.08 * regime_factor),
                confidence=confidence,
                reasoning=(
                    f"Mean reversion SHORT: pullback={pb:.2f} vs strike, "
                    f"RSI={rsi if rsi is not None else 'na'} "
                    f"(bounce in DOWN window, "
                    f"regime={regime['label']}x{regime_factor:.2f}) "
                    f"| {soak}"),
                signals=contributing,
                suggested_amount=amount,
            )

        # Still above strike: dip toward PTB in an UP window → YES.
        if not fade_yes_ok:
            return strategy_decision(
                "hold", signals=contributing,
                reasoning=(
                    f"Fade YES not drift-backed: pb={pb:.2f}, "
                    f"drift={drift:+.3f} | {soak}"))
        rsi_boost = (
            max(0.0, p["rsi_oversold"] - rsi) * 0.005
            if rsi is not None else 0.0
        )
        confidence = min(0.95, (0.35 + pb * 0.40 + rsi_boost)
                         * regime_factor)
        return strategy_decision(
            "buy", "yes",
            edge=min(0.10, pb * 0.08 * regime_factor),
            confidence=confidence,
            reasoning=(
                f"Mean reversion LONG: pullback={pb:.2f} vs strike, "
                f"RSI={rsi if rsi is not None else 'na'} "
                f"(dip in UP window, "
                f"regime={regime['label']}x{regime_factor:.2f}) "
                f"| {soak}"),
            signals=contributing,
            suggested_amount=amount,
        )

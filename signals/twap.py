"""Chainlink / Polymarket TWAP helpers for BTC 5-min resolution.

Effective 2026-08-07 00:00 UTC, Polymarket crypto up/down markets resolve on
Chainlink-computed **time-weighted average prices**, not single snapshots.

Lookback windows (config-driven; override via ``TWAP_WINDOW_SEC`` env):

  * 5-minute markets  → **60-second TWAP** (was 30s at initial cutover)
  * 15-minute / 4h    → 60-second TWAP

Both the **opening Price to Beat** and the **final settlement price** come
from the applicable TWAP feed (PolymarketDevs clarification).

This module is pure math + policy helpers:

  * ``window_seconds_for_market`` — pick lookback from market duration
  * ``rtds_topic`` — RTDS topic for the active lookback (thirty vs sixty)
  * ``compute_twap`` — local time-weighted average over a tick series
  * ``settlement_nowcast`` — partial average inside the settlement window
  * ``resolution_price`` — pick RTDS TWAP vs nowcast for drift ``btc_now``

Do **not** try to independently reproduce Chainlink's signed report without
their sampling specification — prefer the RTDS/Data Streams value. Local
TWAP is for nowcast / diagnostics / offline ranking only.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional, Sequence

import config

logger = logging.getLogger(__name__)

Tick = tuple[float, float]  # (epoch_sec, price)

# Rate-limit "coverage outage" warnings (settlement can tick every ~1s).
_LAST_COVERAGE_OUTAGE_LOG_MONO: float = 0.0
_COVERAGE_OUTAGE_LOG_INTERVAL_SEC: float = 30.0


def window_seconds_for_market(
    market_window_sec: Optional[float] = None,
) -> int:
    """TWAP lookback for a market duration (seconds).

    Defaults to the arena's 5-min series (``TWAP_WINDOW_SEC``, 60s). Longer
    windows use ``TWAP_WINDOW_SEC_15M`` (also 60s per Polymarket's table).
    """
    mw = float(
        market_window_sec
        if market_window_sec is not None
        else getattr(config, "MARKET_WINDOW_SEC", 300) or 300
    )
    if mw <= 300 + 1e-6:
        return int(getattr(config, "TWAP_WINDOW_SEC", 60) or 60)
    return int(getattr(config, "TWAP_WINDOW_SEC_15M", 60) or 60)


def rtds_topic(window_sec: Optional[int] = None) -> str:
    """Polymarket RTDS topic for the given TWAP lookback.

    60s → ``crypto_prices_twap_sixty``; 30s → ``crypto_prices_twap_thirty``.
    """
    w = int(
        window_sec
        if window_sec is not None
        else getattr(config, "TWAP_WINDOW_SEC", 60) or 60
    )
    if w >= 60:
        return str(
            getattr(config, "TWAP_RTDS_TOPIC_60", "crypto_prices_twap_sixty")
            or "crypto_prices_twap_sixty"
        )
    return str(
        getattr(config, "TWAP_RTDS_TOPIC_30", "crypto_prices_twap_thirty")
        or "crypto_prices_twap_thirty"
    )


def settlement_entry_horizon_sec() -> int:
    """Seconds before expiry covering pre_settle lead + full TWAP window.

    Used by sweeper / late-window strategies as the default entry gate.
    """
    w = int(getattr(config, "TWAP_WINDOW_SEC", 60) or 60)
    lead = int(getattr(config, "TWAP_PRE_SETTLE_LEAD_SEC", 20) or 20)
    return max(1, w + lead)


def in_settlement_window(
    time_remaining_sec: Optional[float],
    *,
    twap_window_sec: Optional[int] = None,
) -> bool:
    """True when wall-clock is inside the final TWAP averaging window."""
    if time_remaining_sec is None:
        return False
    w = int(
        twap_window_sec
        if twap_window_sec is not None
        else getattr(config, "TWAP_WINDOW_SEC", 60) or 60
    )
    try:
        tr = float(time_remaining_sec)
    except (TypeError, ValueError):
        return False
    return 0.0 <= tr <= float(w)


def compute_twap(
    ticks: Sequence[Tick],
    window_start: float,
    window_end: float,
) -> tuple[Optional[float], int, float]:
    """Time-weighted average of ``ticks`` over ``[window_start, window_end]``.

    Each tick holds until the next tick (or ``window_end``). Returns
    ``(twap, n_ticks, coverage)`` where coverage is fraction of the window
    with observed tape (0–1). Empty / zero-duration → ``(None, 0, 0.0)``.
    """
    if window_end <= window_start:
        return None, 0, 0.0
    if not ticks:
        return None, 0, 0.0

    # Filter + sort; also keep the last tick *before* the window so we can
    # carry its price into the window start (standard last-tick carry).
    ordered = sorted(
        ((float(ts), float(px)) for ts, px in ticks if px and px > 0),
        key=lambda t: t[0],
    )
    if not ordered:
        return None, 0, 0.0

    carry: Optional[float] = None
    in_window: list[Tick] = []
    for ts, px in ordered:
        if ts < window_start:
            carry = px
        elif ts <= window_end:
            in_window.append((ts, px))
        else:
            break

    points: list[Tick] = []
    if carry is not None:
        points.append((window_start, carry))
    # Drop any in-window tick that lands exactly on start if we already
    # carried (avoid double-weighting the same instant).
    for ts, px in in_window:
        if points and abs(ts - points[-1][0]) < 1e-9:
            points[-1] = (ts, px)
        else:
            points.append((ts, px))

    if not points:
        return None, 0, 0.0

    weighted = 0.0
    covered = 0.0
    for i, (ts, px) in enumerate(points):
        next_ts = points[i + 1][0] if i + 1 < len(points) else window_end
        # Clamp segment to the window
        t0 = max(ts, window_start)
        t1 = min(next_ts, window_end)
        dur = t1 - t0
        if dur <= 0:
            continue
        weighted += px * dur
        covered += dur

    total = window_end - window_start
    if covered <= 0 or total <= 0:
        return None, len(points), 0.0
    return weighted / covered, len(points), min(1.0, covered / total)


def ensure_nowcast_ticks(
    ticks: Sequence[Tick],
    *,
    now_epoch: float,
    price: Optional[float],
    expiry_epoch: Optional[float] = None,
    twap_window_sec: Optional[int] = None,
) -> list[Tick]:
    """Append the live print only when the ring has no earlier tick.

    Does **not** back-date the current price to window open (that would
    invent a full-window TWAP and fake coverage≈1). Empty tape stays
    empty so settlement policy treats it as an outage and ``btc_now``
    falls back to the official rolling RTDS TWAP.
    """
    out = list(ticks or [])
    if not price or price <= 0 or now_epoch is None:
        return out
    now = float(now_epoch)
    if any(float(ts) < now - 1e-9 for ts, _px in out):
        return out
    out.append((now - 0.05, float(price)))
    return out


def settlement_nowcast(
    ticks: Sequence[Tick],
    *,
    now_epoch: float,
    expiry_epoch: float,
    twap_window_sec: Optional[int] = None,
    fill_price: Optional[float] = None,
) -> dict:
    """Estimate final settlement TWAP given ticks seen so far.

    Settlement lookback is ``[expiry − W, expiry]``. While ``now < expiry``:

      * observed = TWAP of ticks on ``[expiry−W, now]`` (clamped)
      * remaining interval ``[now, expiry]`` filled with ``fill_price``
        (last tick / current spot / current rolling TWAP)

    Returns a dict with ``nowcast``, ``observed_twap``, ``coverage``,
    ``frac_elapsed``, ``in_window``, ``window_start``, ``window_end``.
    """
    w = float(
        twap_window_sec
        if twap_window_sec is not None
        else getattr(config, "TWAP_WINDOW_SEC", 60) or 60
    )
    window_end = float(expiry_epoch)
    window_start = window_end - w
    now = float(now_epoch)

    out: dict = {
        "nowcast": None,
        "observed_twap": None,
        "coverage": 0.0,
        "frac_elapsed": 0.0,
        "in_window": False,
        "window_start": window_start,
        "window_end": window_end,
        "twap_window_sec": int(w),
    }

    if now < window_start:
        # Settlement window not open yet — no partial observation.
        return out

    out["in_window"] = True
    # How far through the settlement window we are (capped at 1).
    elapsed = min(w, max(0.0, now - window_start))
    out["frac_elapsed"] = elapsed / w if w > 0 else 0.0

    obs_end = min(now, window_end)
    obs_twap, n_ticks, coverage = compute_twap(ticks, window_start, obs_end)
    out["observed_twap"] = obs_twap
    out["coverage"] = float(coverage)
    out["n_ticks"] = n_ticks

    if obs_twap is None or obs_twap <= 0:
        return out

    remaining = max(0.0, window_end - now)
    if remaining <= 1e-9 or now >= window_end:
        out["nowcast"] = float(obs_twap)
        return out

    fill = fill_price
    if fill is None or fill <= 0:
        # Last observed tick price as forward fill.
        ordered = [t for t in ticks if t[0] <= now and t[1] and t[1] > 0]
        fill = float(ordered[-1][1]) if ordered else float(obs_twap)
    observed_dur = max(1e-9, obs_end - window_start)
    # Weight by wall time inside the settlement window.
    total = w
    nowcast = (obs_twap * observed_dur + float(fill) * remaining) / total
    out["nowcast"] = float(nowcast)
    out["fill_price"] = float(fill)
    return out


def resolution_btc_now(
    *,
    rtds_twap: Optional[float],
    spot: Optional[float],
    time_remaining_sec: Optional[float],
    ticks: Sequence[Tick] = (),
    now_epoch: Optional[float] = None,
    expiry_epoch: Optional[float] = None,
    twap_window_sec: Optional[int] = None,
    prefer_remaining_expiry: bool = True,
) -> dict:
    """Choose the BTC level used for drift moneyness under TWAP resolution.

    Preference order when ``TWAP_USE_FOR_DRIFT``:

      1. Settlement nowcast (inside final W seconds, coverage OK)
      2. Official rolling RTDS TWAP
      3. Spot Chainlink (if ``TWAP_FALLBACK_TO_SPOT``)
      4. 0 / unavailable

    Outside the settlement window the rolling RTDS TWAP is the right object:
    both open and close are TWAP prints, so mid-window moneyness is
    TWAP_now − TWAP_open.
    """
    use_twap = bool(getattr(config, "TWAP_USE_FOR_DRIFT", True))
    use_nowcast = bool(getattr(config, "TWAP_NOWCAST_ENABLED", True))
    min_cov = float(getattr(config, "TWAP_NOWCAST_MIN_COVERAGE", 0.40) or 0.40)
    fallback_spot = bool(getattr(config, "TWAP_FALLBACK_TO_SPOT", True))
    w = int(
        twap_window_sec
        if twap_window_sec is not None
        else getattr(config, "TWAP_WINDOW_SEC", 60) or 60
    )

    result = {
        "btc_now": 0.0,
        "source": "none",
        "rtds_twap": float(rtds_twap) if rtds_twap and rtds_twap > 0 else None,
        "spot": float(spot) if spot and spot > 0 else None,
        "nowcast": None,
        "in_settlement_window": in_settlement_window(
            time_remaining_sec, twap_window_sec=w
        ),
        "nowcast_coverage": 0.0,
        "nowcast_frac_elapsed": 0.0,
    }

    if not use_twap:
        px = result["spot"] or result["rtds_twap"]
        if px:
            result["btc_now"] = float(px)
            result["source"] = "spot" if result["spot"] else "rtds_twap"
        return result

    # Settlement nowcast when we can build it.
    # Expiry from remaining-time is the same clock the phase uses. A drifted
    # Gamma endDate used to put [expiry−W, now] in the future → coverage=0
    # every window even though the feed was live.
    if (
        prefer_remaining_expiry
        and now_epoch is not None
        and time_remaining_sec is not None
    ):
        try:
            rem_exp = float(now_epoch) + float(time_remaining_sec)
            if expiry_epoch is None or abs(float(expiry_epoch) - rem_exp) > 5.0:
                expiry_epoch = rem_exp
        except (TypeError, ValueError):
            pass

    if (
        use_nowcast
        and result["in_settlement_window"]
        and now_epoch is not None
        and expiry_epoch is not None
    ):
        fill = result["rtds_twap"] or result["spot"]
        ticks = ensure_nowcast_ticks(
            ticks,
            now_epoch=float(now_epoch),
            price=fill,
            expiry_epoch=float(expiry_epoch),
            twap_window_sec=w,
        )
        nc = settlement_nowcast(
            ticks,
            now_epoch=float(now_epoch),
            expiry_epoch=float(expiry_epoch),
            twap_window_sec=w,
            fill_price=fill,
        )
        result["nowcast"] = nc.get("nowcast")
        result["nowcast_coverage"] = float(nc.get("coverage") or 0.0)
        result["nowcast_frac_elapsed"] = float(nc.get("frac_elapsed") or 0.0)
        if (
            nc.get("nowcast")
            and float(nc["nowcast"]) > 0
            and float(nc.get("coverage") or 0.0) >= min_cov
        ):
            result["btc_now"] = float(nc["nowcast"])
            result["source"] = "settlement_nowcast"
            return result

    if result["rtds_twap"]:
        result["btc_now"] = float(result["rtds_twap"])
        result["source"] = "rtds_twap"
        return result

    if fallback_spot and result["spot"]:
        result["btc_now"] = float(result["spot"])
        result["source"] = "spot_fallback"
        return result

    return result


def twap_certainty(
    frac_elapsed: float,
    coverage: float,
    abs_drift: float,
    *,
    min_drift: float = 0.15,
) -> float:
    """0–1 score: how locked the settlement outcome looks under TWAP.

    Used by late-window / sweeper strategies so single-tick spikes don't
    look like free money inside the averaging window.
    """
    fe = max(0.0, min(1.0, float(frac_elapsed)))
    cov = max(0.0, min(1.0, float(coverage)))
    ad = max(0.0, float(abs_drift))
    md = max(1e-6, float(min_drift))
    # More of the window observed + stronger moneyness → higher certainty.
    drift_term = min(1.0, ad / (md * 2.0))
    return max(0.0, min(1.0, 0.45 * fe + 0.25 * cov + 0.30 * drift_term))


def soft_dampen_vol_scale(scale: float) -> float:
    """Apply TWAP smoothness mult to a full-window vol scale.

    Default mult is 1.0 when σ is already TWAP-calibrated (2026-08-07).
    """
    mult = float(getattr(config, "TWAP_DRIFT_VOL_MULT", 1.0) or 1.0)
    if not math.isfinite(scale) or scale <= 0:
        return scale
    return float(scale) * mult


def market_phase(
    time_remaining_sec: Optional[float],
    *,
    twap_window_sec: Optional[int] = None,
    market_window_sec: Optional[float] = None,
) -> str:
    """Coarse phase label for feature stamps / policy.

    * ``open`` — first ~TWAP window of the market (open TWAP just printed)
    * ``mid`` — bulk of the window
    * ``pre_settle`` — lead-in before settlement averaging
    * ``settlement`` — final TWAP averaging window
    * ``unknown`` — missing clock
    """
    if time_remaining_sec is None:
        return "unknown"
    try:
        tr = float(time_remaining_sec)
    except (TypeError, ValueError):
        return "unknown"
    w = int(
        twap_window_sec
        if twap_window_sec is not None
        else getattr(config, "TWAP_WINDOW_SEC", 60) or 60
    )
    mw = float(
        market_window_sec
        if market_window_sec is not None
        else getattr(config, "MARKET_WINDOW_SEC", 300) or 300
    )
    lead = float(getattr(config, "TWAP_PRE_SETTLE_LEAD_SEC", 15) or 15)
    if tr < 0:
        return "unknown"
    if tr <= w:
        return "settlement"
    if tr <= w + lead:
        return "pre_settle"
    if tr >= mw - w:
        return "open"
    return "mid"


def settlement_adjustments(
    *,
    time_remaining_sec: Optional[float] = None,
    twap_certainty_val: float = 0.0,
    nowcast_frac_elapsed: float = 0.0,
    nowcast_coverage: float = 0.0,
    abs_drift: float = 0.0,
    in_settlement: Optional[bool] = None,
) -> dict:
    """Trading policy mults for TWAP settlement / pre-settlement phases.

    Returns a dict consumed by ``base_bot.make_decision``, SignalLab, and
    late-window strategies. Neutral (all 1.0 / 0 boost) when policy is off
    or phase is mid-window.
    """
    phase = market_phase(time_remaining_sec)
    if in_settlement is None:
        in_settlement = phase == "settlement"

    out = {
        "phase": phase,
        "in_settlement_window": bool(in_settlement),
        "pre_settle": phase == "pre_settle",
        "certainty": max(0.0, min(1.0, float(twap_certainty_val or 0.0))),
        "frac_elapsed": max(0.0, min(1.0, float(nowcast_frac_elapsed or 0.0))),
        "coverage": max(0.0, min(1.0, float(nowcast_coverage or 0.0))),
        "edge_mult": 1.0,
        "size_mult": 1.0,
        "conf_boost": 0.0,
        "mom_damp": 1.0,
        "block_fade": False,
        "policy_active": False,
    }

    if not bool(getattr(config, "TWAP_SETTLEMENT_POLICY", True)):
        return out
    if not bool(getattr(config, "TWAP_RESOLUTION_ENABLED", True)):
        return out

    cert = out["certainty"]
    # If certainty not stamped yet, derive a light estimate from frac/drift.
    if cert <= 0.0 and (out["frac_elapsed"] > 0 or abs_drift):
        cert = twap_certainty(
            out["frac_elapsed"], out["coverage"], abs(float(abs_drift)),
        )
        out["certainty"] = cert

    hi = float(getattr(config, "TWAP_SETTLE_CERT_HIGH", 0.55) or 0.55)
    lo = float(getattr(config, "TWAP_SETTLE_CERT_LOW", 0.25) or 0.25)

    if phase == "settlement":
        out["policy_active"] = True
        out["mom_damp"] = float(getattr(config, "TWAP_SETTLE_MOM_DAMP", 0.40) or 0.40)
        if cert >= hi:
            out["edge_mult"] = float(
                getattr(config, "TWAP_SETTLE_EDGE_MULT_HIGH", 0.92) or 0.92
            )
            out["size_mult"] = float(
                getattr(config, "TWAP_SETTLE_SIZE_MULT_HIGH", 1.12) or 1.12
            )
            out["conf_boost"] = float(
                getattr(config, "TWAP_SETTLE_CONF_BOOST", 0.08) or 0.08
            ) * min(1.0, cert)
        elif cert <= lo:
            out["edge_mult"] = float(
                getattr(config, "TWAP_SETTLE_EDGE_MULT_LOW", 1.40) or 1.40
            )
            out["size_mult"] = float(
                getattr(config, "TWAP_SETTLE_SIZE_MULT_LOW", 0.80) or 0.80
            )
            out["conf_boost"] = 0.0
        else:
            # Interpolate mid band between low and high certainty.
            t = (cert - lo) / max(1e-6, hi - lo)
            e_lo = float(getattr(config, "TWAP_SETTLE_EDGE_MULT_LOW", 1.40) or 1.40)
            e_hi = float(getattr(config, "TWAP_SETTLE_EDGE_MULT_HIGH", 0.92) or 0.92)
            s_lo = float(getattr(config, "TWAP_SETTLE_SIZE_MULT_LOW", 0.80) or 0.80)
            s_hi = float(getattr(config, "TWAP_SETTLE_SIZE_MULT_HIGH", 1.12) or 1.12)
            mid_e = float(getattr(config, "TWAP_SETTLE_EDGE_MULT_MID", 1.12) or 1.12)
            # Blend toward mid at t=0.5, endpoints at 0/1
            out["edge_mult"] = e_lo + (e_hi - e_lo) * t
            # Soft pull toward mid_e near the middle of the band
            out["edge_mult"] = 0.7 * out["edge_mult"] + 0.3 * mid_e
            out["size_mult"] = s_lo + (s_hi - s_lo) * t
            out["conf_boost"] = float(
                getattr(config, "TWAP_SETTLE_CONF_BOOST", 0.08) or 0.08
            ) * t * 0.5

        if bool(getattr(config, "TWAP_SETTLE_BLOCK_FADE", True)):
            need_c = float(getattr(config, "TWAP_SETTLE_BLOCK_FADE_CERT", 0.50) or 0.50)
            need_d = float(getattr(config, "TWAP_SETTLE_BLOCK_FADE_DRIFT", 0.20) or 0.20)
            if cert >= need_c and abs(float(abs_drift)) >= need_d:
                out["block_fade"] = True

    elif phase == "pre_settle":
        out["policy_active"] = True
        out["mom_damp"] = float(
            getattr(config, "TWAP_PRE_SETTLE_MOM_DAMP", 0.70) or 0.70
        )
        # Mild edge tax: not yet observing settlement but close
        out["edge_mult"] = 1.08
        out["size_mult"] = 0.95

    elif phase == "open":
        # Open TWAP just printed — early tape still noisy; slight mom damp
        out["policy_active"] = True
        out["mom_damp"] = 0.85
        out["edge_mult"] = 1.05

    # TWAP coverage outage guard — settlement only.
    # nowcast_coverage is only computed inside the final TWAP window; open and
    # pre_settle always report coverage≈0 by design. Treating that as an outage
    # spammed logs every tick and wiped intentional open/pre_settle damps.
    # When the RTDS/spot tick ring is truly empty during settlement, skip the
    # low-cert 1.40× edge tax (missing data ≠ noisy TWAP).
    if (
        phase == "settlement"
        and cert <= lo + 1e-9
        and out.get("coverage", 1.0) < 0.05
    ):
        global _LAST_COVERAGE_OUTAGE_LOG_MONO
        now_m = time.monotonic()
        if (
            now_m - _LAST_COVERAGE_OUTAGE_LOG_MONO
            >= _COVERAGE_OUTAGE_LOG_INTERVAL_SEC
        ):
            _LAST_COVERAGE_OUTAGE_LOG_MONO = now_m
            logger.warning(
                "TWAP coverage outage (settlement): cert=%.2f coverage=%.2f — "
                "falling back to spot-only (no settlement penalty)",
                cert, out.get("coverage", 0.0),
            )
        out["edge_mult"] = 1.0
        out["size_mult"] = 1.0
        out["mom_damp"] = 1.0
        out["conf_boost"] = 0.0
        out["coverage_outage"] = True

    return out

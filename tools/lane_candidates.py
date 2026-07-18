"""Candidate-lane backfill + evaluation for the signal-validation harness.

Stage 1-2 of the lane-promotion pipeline (see CLAUDE.md "Signal-stack
expansion"): reconstruct what each kill-switched candidate lane (fut / tech /
xasset) WOULD have read at every decision point of the harness's resolved
markets, then measure follow-WR and — the number that decides everything —
NET edge per share after paying the actual Polymarket price + taker fee.

Lane values are computed with the SAME production code paths
(``signals.technicals.compute``, ``signals.curves.soft_saturate`` with the
``signals.futures_meta`` / ``signals.cross_asset`` calibration constants) so
the harness validates exactly what would ship.

Data sources (public, no auth):
  * Binance spot 1m klines (BTC history for technicals; ETH/SOL for xasset)
  * Binance USD-M futures: funding history, open-interest hist (5m),
    taker long/short ratio (5m). NOTE: the /futures/data/* endpoints only
    retain ~30 days — older markets simply get no fut readings (skipped).

Pure functions (attach/evaluate/build_proposals) do no I/O and are
unit-tested; fetchers live at the bottom.
"""

from bisect import bisect_right
from typing import Optional

from signals.technicals import compute as tech_compute
from signals.curves import soft_saturate
from signals.futures_meta import (
    FUNDING_SCALE, OI_DELTA_SCALE, TAKER_RATIO_SCALE,
)
from signals.cross_asset import XASSET_SCALE

# Live lane -> the sample signal key that matches its production definition.
LIVE_LANE_KEYS = {
    "fut": "fut_taker",      # live fut lane consumes taker_delta
    "tech": "tech_mtf",      # live tech lane consumes mtf_score
    "xasset": "xasset",
}
# All candidate keys reported (the extra ones are informational context that
# may motivate a redefinition of a live lane later).
CANDIDATE_KEYS = ["fut_taker", "fut_funding", "fut_oi",
                  "tech_mtf", "tech_macd", "tech_bb", "xasset"]

# Promotion thresholds: a lane must clear ALL of them on its LIVE definition.
# Net edge is the bar that killed pm_mom (predictive, -0.80c/share); the WR
# and sample floors keep one lucky regime from qualifying.
MIN_SAMPLES = 200
MIN_FOLLOW_WR = 0.55
MIN_NET_EDGE = 0.005          # +0.5c/share after price + fee
DEADBAND = 0.05               # |lane| below this = no directional reading

# Suggested per-strategy profile weights on promotion — conservative 0.10
# starters on the strategies whose character the lane fits (trend/flow lanes
# don't belong in the drift-pure meanrev book). Evolution tunes from there.
PROFILE_SUGGESTIONS = {
    "fut":    {"momentum": 0.10, "phantom": 0.10, "hybrid": 0.10},
    "tech":   {"momentum": 0.10, "phantom": 0.10, "hybrid": 0.10},
    "xasset": {"momentum": 0.10, "hybrid": 0.10},
}


class Series:
    """Sorted (ts_sec, value) series with last-at-or-before lookup."""

    def __init__(self, points):
        pts = sorted((float(t), v) for t, v in (points or []))
        self._ts = [t for t, _ in pts]
        self._vals = [v for _, v in pts]

    def __len__(self):
        return len(self._ts)

    def at(self, ts: float) -> Optional[float]:
        i = bisect_right(self._ts, ts)
        return self._vals[i - 1] if i else None

    def last_two(self, ts: float):
        """The two most recent values at/before ``ts`` (or None)."""
        i = bisect_right(self._ts, ts)
        if i < 2:
            return None
        return self._vals[i - 2], self._vals[i - 1]

    def closes_until(self, ts: float, n: int) -> list:
        """Up to ``n`` most recent values at/before ``ts`` (oldest first)."""
        i = bisect_right(self._ts, ts)
        return self._vals[max(0, i - n):i]


def attach_candidates(samples: list, open_ts: float, series: dict) -> None:
    """Compute candidate lane values into each sample's ``signals`` dict.

    ``samples`` are one market's Samples (tools/signal_validation.Sample,
    window-relative times); ``open_ts`` is that market's open as epoch
    seconds; ``series`` maps name -> Series with epoch-second keys:
    ``btc_close``, ``eth_close``, ``sol_close`` (1m closes, keyed by candle
    CLOSE time so a candle is only visible once finished), ``funding``
    (rate), ``oi`` (open interest), ``taker`` (buy/sell ratio).
    """
    for s in samples:
        elapsed = (300 - s.time_remaining)
        ts = open_ts + elapsed

        taker = series["taker"].at(ts) if "taker" in series else None
        s.signals["fut_taker"] = (
            None if taker is None else soft_saturate(taker - 1.0, TAKER_RATIO_SCALE))

        funding = series["funding"].at(ts) if "funding" in series else None
        s.signals["fut_funding"] = (
            None if funding is None else soft_saturate(funding, FUNDING_SCALE))

        oi2 = series["oi"].last_two(ts) if "oi" in series else None
        if oi2 is None or not oi2[0]:
            s.signals["fut_oi"] = None
        else:
            s.signals["fut_oi"] = soft_saturate(
                (oi2[1] - oi2[0]) / oi2[0], OI_DELTA_SCALE)

        closes = (series["btc_close"].closes_until(ts, 60)
                  if "btc_close" in series else [])
        if len(closes) >= 6:
            tech = tech_compute(closes)
            s.signals["tech_mtf"] = tech["mtf_score"]
            s.signals["tech_macd"] = tech["macd_score"]
            s.signals["tech_bb"] = tech["bb_score"]
        else:
            s.signals["tech_mtf"] = None
            s.signals["tech_macd"] = None
            s.signals["tech_bb"] = None

        peers = []
        for key in ("eth_close", "sol_close"):
            two = series[key].last_two(ts) if key in series else None
            if two and two[0] and two[0] > 0:
                peers.append(soft_saturate((two[1] - two[0]) / two[0],
                                           XASSET_SCALE))
        s.signals["xasset"] = (sum(peers) / len(peers)) if peers else None


def _lane_rule(key: str, deadband: float = DEADBAND):
    """Follow-the-lane side rule for the net-edge metric."""
    def rule(s):
        v = s.signals.get(key)
        if v is None or abs(v) < deadband:
            return None
        return "yes" if v > 0 else "no"
    return rule


def evaluate_candidates(all_samples: list, taker_fee) -> dict:
    """Per-candidate metrics: follow-WR (predictiveness) + net edge vs price."""
    from tools.signal_validation import predictiveness, net_edge

    results = {}
    for key in CANDIDATE_KEYS:
        pred = predictiveness(all_samples, key, deadband=DEADBAND)
        ne = net_edge(all_samples, _lane_rule(key), taker_fee)
        results[key] = {
            "n": pred["n"],
            "follow_wr": pred["follow_winrate"],
            "net_n": ne["n"],
            "net_wr": ne.get("winrate"),
            "avg_price": ne.get("avg_price"),
            "ev_per_share": ne.get("ev_per_share"),
        }
    return results


def build_proposals(results: dict,
                    min_samples: int = MIN_SAMPLES,
                    min_follow_wr: float = MIN_FOLLOW_WR,
                    min_net_edge: float = MIN_NET_EDGE) -> list:
    """Lanes whose LIVE definition clears every promotion threshold.

    Returns [{lane, metrics, proposal}] ready for db.create_lane_proposal.
    The thresholds are conjunctive on purpose: pm_mom would have passed a
    WR-only bar and lost money (BUG #26).
    """
    proposals = []
    for lane, key in LIVE_LANE_KEYS.items():
        m = results.get(key)
        if not m:
            continue
        n = m.get("net_n") or 0
        wr = m.get("follow_wr")
        ev = m.get("ev_per_share")
        if (n >= min_samples and wr is not None and wr >= min_follow_wr
                and ev is not None and ev >= min_net_edge):
            proposals.append({
                "lane": lane,
                "metrics": {**m, "signal_key": key},
                "proposal": {"profile": dict(PROFILE_SUGGESTIONS[lane])},
            })
    return proposals


# ---------------------------------------------------------------------------
# Fetchers (network — harness only, never the trading hot path)
# ---------------------------------------------------------------------------

SPOT_KLINES = "https://api.binance.com/api/v3/klines"
FAPI = "https://fapi.binance.com"


def fetch_klines_series(symbol: str, start_ms: int, end_ms: int) -> Series:
    """1m closes keyed by candle CLOSE time (epoch sec), paginated."""
    import requests

    points, cursor = [], start_ms
    while cursor < end_ms:
        r = requests.get(SPOT_KLINES, params={
            "symbol": symbol, "interval": "1m",
            "startTime": cursor, "endTime": end_ms, "limit": 1000,
        }, timeout=20)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        for c in rows:
            # c[6] = close time ms, c[4] = close price
            points.append(((c[6] + 1) / 1000.0, float(c[4])))
        cursor = rows[-1][6] + 1
        if len(rows) < 1000:
            break
    return Series(points)


def _fetch_fapi(path: str, params: dict, start_ms: int, end_ms: int,
                limit: int, ts_key: str, val_key: str, val_cast=float) -> Series:
    import requests

    points, cursor = [], start_ms
    while cursor < end_ms:
        r = requests.get(f"{FAPI}{path}", params={
            **params, "startTime": cursor, "endTime": end_ms, "limit": limit,
        }, timeout=20)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        for row in rows:
            points.append((float(row[ts_key]) / 1000.0, val_cast(row[val_key])))
        last = int(float(rows[-1][ts_key]))
        if last + 1 <= cursor or len(rows) < limit:
            break
        cursor = last + 1
    return Series(points)


def fetch_futures_series(start_ms: int, end_ms: int) -> dict:
    """funding / oi / taker Series for BTCUSDT over [start, end].

    The /futures/data/* endpoints retain ~30 days; on empty responses the
    Series is simply empty and affected samples read None (excluded).
    """
    return {
        "funding": _fetch_fapi(
            "/fapi/v1/fundingRate", {"symbol": "BTCUSDT"},
            start_ms, end_ms, 1000, "fundingTime", "fundingRate"),
        "oi": _fetch_fapi(
            "/futures/data/openInterestHist",
            {"symbol": "BTCUSDT", "period": "5m"},
            start_ms, end_ms, 500, "timestamp", "sumOpenInterest"),
        "taker": _fetch_fapi(
            "/futures/data/takerlongshortRatio",
            {"symbol": "BTCUSDT", "period": "5m"},
            start_ms, end_ms, 500, "timestamp", "buySellRatio"),
    }

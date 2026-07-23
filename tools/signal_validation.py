"""Core logic for the offline signal-validation harness (pure + testable).

Given a resolved BTC 5-min market (its open-time strike, intra-window BTC price
trajectory, and the true Up/Down outcome), build decision-time SAMPLES and
measure whether a candidate signal actually predicts the outcome — the
confirms-side vs contradicts-side win-rate split that MUST be checked before any
signal earns a live weight (BUG_HISTORY #23).

Fetching lives in ``tools/validate_signals.py``; this module does no network or
disk I/O so it can be unit-tested with synthetic data.
"""

from dataclasses import dataclass
from typing import Callable, Optional

# Reuse the PRODUCTION drift formula so we validate exactly what ships.
from signals.strike import drift_signal


@dataclass(frozen=True)
class Sample:
    """One decision-time observation within a market window."""
    market_id: str
    time_remaining: float      # seconds to window close at this decision point
    btc_now: float
    strike: float
    yes_won: bool              # ground truth: did Up (YES) win?
    signals: dict              # signal_name -> value (YES-frame: >0 leans Up)
    pm_yes: Optional[float] = None   # Polymarket YES mid at this decision point
    market_seq: int = 0        # market recency rank (0 = most recent market)


def build_samples(market_id: str, strike: float, trajectory: list,
                  yes_won: bool, window_sec: int = 300,
                  pm_prices: Optional[list] = None,
                  market_seq: int = 0) -> list:
    """Build decision-time samples from a BTC price trajectory.

    ``trajectory`` is a list of ``(seconds_from_open, btc_price)`` points (e.g.
    each 1-min candle open). The open point (drift == 0) is skipped. Each sample
    carries several candidate signals in YES/Up-frame (>0 = leans Up):

      * ``drift_raw``   — signed fractional distance from strike
      * ``drift_prod``  — the production ``drift_signal`` (tanh, time-scaled)
      * ``mom2``        — BTC change over the last two trajectory points
      * ``pm_mom``      — per-minute PM YES-price change from the last two PM
        points at/before decision time (``None`` without >=2 points). This is
        the shape of the live ``pm_momentum`` lane, un-clamped — the live lane
        saturates at a 0.19c/step move (SCALE=80 then /0.15), i.e. it degrades
        to sign(last tick); validate the RAW quantity here before any weight.

    ``pm_prices`` (optional) is a list of ``(seconds_from_open, yes_mid)``
    points from Polymarket's price history; each sample gets the most recent
    PM YES mid at-or-before its decision time (``pm_yes``), enabling net-edge
    (vs-the-price) metrics, not just raw predictiveness.
    """
    samples = []
    prev = None
    for elapsed, btc in trajectory:
        tr = window_sec - elapsed
        if elapsed <= 0 or tr <= 0 or not strike:
            prev = btc
            continue
        drift_raw = (btc - strike) / strike
        mom2 = 0.0 if prev is None or prev <= 0 else (btc - prev) / prev
        pm_yes = None
        pm_mom = None
        if pm_prices:
            past = [(t, p) for t, p in pm_prices if t <= elapsed]
            if past:
                pm_yes = float(past[-1][1])
            if len(past) >= 2:
                (t0, p0), (t1, p1) = past[-2], past[-1]
                dt_min = max((float(t1) - float(t0)) / 60.0, 1e-9)
                pm_mom = (float(p1) - float(p0)) / dt_min   # per-minute delta
        samples.append(Sample(
            market_id=market_id, time_remaining=float(tr), btc_now=float(btc),
            strike=float(strike), yes_won=bool(yes_won),
            signals={
                "drift_raw": drift_raw,
                "drift_prod": drift_signal(strike, btc, tr),
                "mom2": mom2,
                "pm_mom": pm_mom,
            },
            pm_yes=pm_yes,
            market_seq=market_seq,
        ))
        prev = btc
    return samples


def predictiveness(samples: list, signal_key: str,
                   deadband: float = 1e-6,
                   filt: Optional[Callable] = None) -> dict:
    """Confirms-side vs contradicts-side win rate for one signal.

    For each sample the signal points to a side (>0 → Up/YES, <0 → Down/NO).
    "confirms" = the sample where the signal's side actually won. We report the
    win rate of *following the signal* on the up-leaning and down-leaning
    subsets, so a genuine signal shows ``up_winrate`` high and ``down_winrate``
    low (i.e. following it wins). ``edge_pp`` summarises: how much better is
    following the signal than fading it.
    """
    up_n = up_win = dn_n = dn_win = 0
    for s in samples:
        if filt is not None and not filt(s):
            continue
        v = s.signals.get(signal_key)
        if v is None or abs(v) <= deadband:
            continue
        if v > 0:                      # signal says Up/YES
            up_n += 1
            up_win += 1 if s.yes_won else 0
        else:                          # signal says Down/NO
            dn_n += 1
            dn_win += 1 if (not s.yes_won) else 0   # following it = betting Down
    up_wr = (up_win / up_n) if up_n else None
    dn_wr = (dn_win / dn_n) if dn_n else None
    # Follow-the-signal win rate across both directions (this is what a bot using
    # the signal directionally would realise).
    follow_n = up_n + dn_n
    follow_win = up_win + dn_win
    follow_wr = (follow_win / follow_n) if follow_n else None
    return {
        "signal": signal_key,
        "n": follow_n,
        "follow_winrate": follow_wr,     # >0.5 = predictive, <0.5 = INVERTED
        "up_n": up_n, "up_winrate": up_wr,
        "down_n": dn_n, "down_winrate": dn_wr,
    }


def magnitude_distribution(samples: list, signal_key: str,
                           percentiles=(50, 75, 90, 97)) -> dict:
    """Percentiles of ``|signal|`` across samples (None values skipped).

    Use to pick an HONEST saturation point for a lane normalization: a lane
    that saturates below the median of its own input distribution degrades to
    a sign() detector (the BUG #25 momentum mistake, replayed live by pm).
    """
    vals = sorted(abs(s.signals[signal_key]) for s in samples
                  if s.signals.get(signal_key) is not None)
    out = {"signal": signal_key, "n": len(vals)}
    for p in percentiles:
        if not vals:
            out[f"p{p}"] = None
        else:
            idx = min(len(vals) - 1, max(0, round(p / 100.0 * (len(vals) - 1))))
            out[f"p{p}"] = vals[idx]
    return out


def net_edge(samples: list, side_of: Callable, fee: Callable,
             filt: Optional[Callable] = None,
             slippage: float = 0.0) -> dict:
    """Realised per-share net EV of a decision rule against the ACTUAL PM price.

    ``side_of(sample) -> "yes" | "no" | None`` picks the side to buy (None =
    skip). The buy price is the PM mid for that side at decision time and
    ``fee(shares, price)`` is the canonical taker fee. This is the metric that
    matters: a signal can be predictive yet have NO edge once the market has
    already priced it in (BUG: raw follow-WR ≠ profit).

    ``slippage`` is a flat per-share execution penalty added onto the mid
    (fraction of $1, e.g. 0.005 = half a cent). PM history mids are stale and
    optimistic; the live fill walks the asks. A signal that only survives at
    zero slippage has no real edge.
    """
    n = wins = 0
    ev_sum = price_sum = 0.0
    for s in samples:
        if s.pm_yes is None or (filt is not None and not filt(s)):
            continue
        side = side_of(s)
        if side is None:
            continue
        price = s.pm_yes if side == "yes" else (1.0 - s.pm_yes)
        if price <= 0.01 or price >= 0.99:
            continue
        paid = min(0.99, price + slippage)
        won = s.yes_won if side == "yes" else (not s.yes_won)
        ev = (1.0 - paid if won else -paid) - fee(1.0, paid)
        n += 1
        wins += 1 if won else 0
        ev_sum += ev
        price_sum += paid
    return {
        "n": n,
        "winrate": (wins / n) if n else None,
        "avg_price": (price_sum / n) if n else None,
        "ev_per_share": (ev_sum / n) if n else None,
    }


def information_coefficient(samples: list, signal_key: str,
                            filt: Optional[Callable] = None) -> dict:
    """Point-biserial correlation between signal value and the Up outcome.

    IC in [-1, 1]: positive = higher signal values associate with Up wins
    (correctly signed), negative = inverted, ~0 = no information. Unlike
    follow-WR this uses the signal's MAGNITUDE, so a lane whose size means
    something scores higher than a same-sign coin-flip.
    """
    xs, ys = [], []
    for s in samples:
        if filt is not None and not filt(s):
            continue
        v = s.signals.get(signal_key)
        if v is None:
            continue
        xs.append(float(v))
        ys.append(1.0 if s.yes_won else 0.0)
    n = len(xs)
    if n < 3:
        return {"signal": signal_key, "n": n, "ic": None}
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
    if sx <= 0 or sy <= 0:
        return {"signal": signal_key, "n": n, "ic": None}
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return {"signal": signal_key, "n": n, "ic": cov / (sx * sy)}


def decay_analysis(samples: list, signal_key: str, buckets: int = 3) -> list:
    """Predictiveness + IC split by market recency (performance decay check).

    Markets are fetched newest-first, so ``market_seq`` 0 is the most recent.
    Splitting the seq range into ``buckets`` chronological slices shows
    whether a signal's edge is decaying: strong in the OLD slice but flat in
    the RECENT slice = the market has adapted (or the regime moved on) — do
    not weight it off the pooled number.
    """
    seqs = sorted({s.market_seq for s in samples})
    if not seqs:
        return []
    per = max(1, (len(seqs) + buckets - 1) // buckets)
    out = []
    for b in range(buckets):
        chunk = set(seqs[b * per:(b + 1) * per])
        if not chunk:
            continue
        subset = [s for s in samples if s.market_seq in chunk]
        res = predictiveness(subset, signal_key)
        res["ic"] = information_coefficient(subset, signal_key)["ic"]
        res["bucket"] = ("recent" if b == 0 else
                         ("oldest" if b == buckets - 1 else f"mid-{b}"))
        out.append(res)
    return out


def regime_split(samples: list, signal_key: str, regime_key: str,
                 fee: Callable, slippage: float = 0.0,
                 deadband: float = 1e-6) -> list:
    """Follow-WR / IC / net edge of a signal per tercile of a regime feature.

    ``regime_key`` is a non-directional context feature attached to the same
    samples (e.g. ``regime_trend``, ``ms_rvol_5m``). Answers "where does this
    signal actually earn?" — a lane that only pays in trending tape should be
    gated by that context, not run flat.
    """
    vals = sorted(s.signals[regime_key] for s in samples
                  if s.signals.get(regime_key) is not None)
    if len(vals) < 9:
        return []
    lo_cut = vals[len(vals) // 3]
    hi_cut = vals[(2 * len(vals)) // 3]

    def rule(s):
        v = s.signals.get(signal_key)
        if v is None or abs(v) <= deadband:
            return None
        return "yes" if v > 0 else "no"

    out = []
    bounds = [("low", lambda v: v <= lo_cut),
              ("mid", lambda v: lo_cut < v <= hi_cut),
              ("high", lambda v: v > hi_cut)]
    for label, member in bounds:
        def filt(s, member=member):
            rv = s.signals.get(regime_key)
            return rv is not None and member(rv)
        subset = [s for s in samples if filt(s)]
        pred = predictiveness(subset, signal_key, deadband=deadband)
        ne = net_edge(subset, rule, fee, slippage=slippage)
        out.append({
            "regime": f"{regime_key}:{label}",
            "n": pred["n"],
            "follow_winrate": pred["follow_winrate"],
            "ic": information_coefficient(subset, signal_key)["ic"],
            "net_n": ne["n"],
            "ev_per_share": ne["ev_per_share"],
        })
    return out


def rank_signals(samples: list, keys: list, fee: Callable,
                 slippage: float = 0.005,
                 deadband: float = 1e-6) -> list:
    """Ranked scorecard for a set of DIRECTIONAL signal keys.

    Per key: n, IC, follow-WR, net EV at mid, net EV after ``slippage``, and
    the recent-third follow-WR (decay check). Sorted by slippage-adjusted EV
    (the number a live taker actually keeps), IC as tiebreaker. A signal
    earns/keeps weight only when slip-adjusted EV is positive AND the recent
    slice hasn't collapsed — the pooled number alone is exactly how pm_mom
    almost got weighted (BUG #26).
    """
    rows = []
    for key in keys:
        def rule(s, key=key):
            v = s.signals.get(key)
            if v is None or abs(v) <= deadband:
                return None
            return "yes" if v > 0 else "no"
        pred = predictiveness(samples, key, deadband=deadband)
        ic = information_coefficient(samples, key)["ic"]
        ne_mid = net_edge(samples, rule, fee, slippage=0.0)
        ne_slip = net_edge(samples, rule, fee, slippage=slippage)
        decay = decay_analysis(samples, key)
        recent_wr = decay[0]["follow_winrate"] if decay else None
        rows.append({
            "signal": key,
            "n": pred["n"],
            "ic": ic,
            "follow_wr": pred["follow_winrate"],
            "ev_mid": ne_mid["ev_per_share"],
            "ev_slip": ne_slip["ev_per_share"],
            "net_n": ne_slip["n"],
            "recent_wr": recent_wr,
        })
    return sorted(rows, key=lambda r: ((r["ev_slip"] is None),
                                       -(r["ev_slip"] or 0.0),
                                       -(r["ic"] or 0.0)))


def net_edge_time_buckets(samples: list, side_of: Callable, fee: Callable,
                          edges=(60, 120, 180, 300)) -> list:
    """``net_edge`` split by time-remaining bucket."""
    out = []
    lo = 0
    for hi in edges:
        res = net_edge(samples, side_of, fee,
                       filt=lambda s, lo=lo, hi=hi: lo < s.time_remaining <= hi)
        res["bucket"] = f"{lo}-{hi}s"
        out.append(res)
        lo = hi
    return out


def time_buckets(samples: list, signal_key: str, edges=(60, 120, 180, 300)) -> list:
    """Predictiveness of a signal split by time-remaining bucket.

    Answers "is the signal predictive near expiry but noise early?" — the crux
    of whether drift can be salvaged as a late-window-only signal.
    """
    out = []
    lo = 0
    for hi in edges:
        subset = [s for s in samples if lo < s.time_remaining <= hi]
        res = predictiveness(subset, signal_key)
        res["bucket"] = f"{lo}-{hi}s"
        out.append(res)
        lo = hi
    return out

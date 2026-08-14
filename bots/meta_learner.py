"""HybridMetaLearner — online meta-learning for the hybrid ensemble.

The hybrid's sub-strategy weights combine three layers (see bot_hybrid):

1. **Regime tilt** — continuous in trend_score (trend followers up in
   trends, the fade book up in chop).
2. **Cross-bot performance tilt** — each sub-strategy's REAL arena bot's
   recent live WR (SignalLab.perf_tilts).
3. **THIS module** — an online multiplicative-weights (Hedge-style)
   learner trained on:
     (a) the hybrid's OWN resolved **trades** (full eta), and
     (b) resolved **decision_events** skips with a ``meta(...)`` token
         (counterfactuals — scaled eta so one skip ≠ one fill).

   Every hybrid decision that has sub-votes logs::

       meta(mom=+0.42 rev=+0.00 ph=+0.30 | reg=trending)

   At resolution each sub that voted (|vote| >= deadband) is scored
   against the actual market direction. A correct vote multiplies the
   sub's weight by ``exp(+eta)``, a wrong one by ``exp(-eta)``, clipped
   to [min_mult, max_mult]. Multipliers are kept PER REGIME BUCKET
   (trending / ranging / mixed / chop) *and* overall, blended by bucket
   sample size at decision time.

State persists in arena_state key ``hybrid_meta`` (JSON): it survives
restarts, is SHARED by every hybrid generation (evolution mutants inherit
the lineage's learning instead of starting cold), and the dashboard reads
it directly (``/api/hybrid-meta``). All DB access is exception-safe — a
missing table or hiccup degrades to neutral multipliers, never a stalled
tick.
"""

import json
import logging
import math
import re
import threading
import time
from typing import Optional

import config
import db

logger = logging.getLogger("bots.meta_learner")

STATE_KEY = "hybrid_meta"
# "chop" is the high-vol non-directional bucket from the robust regime
# detector; older rows without it still score under "mixed".
BUCKETS = ("trending", "ranging", "mixed", "chop")

# sub-strategy name -> compact reasoning-token key (kept short: the token
# rides in every hybrid trade's persisted reasoning).
# Sentiment sub removed 2026-08; parser still accepts optional sent= for
# historical hybrid trades so meta learning does not break on old rows.
SUB_TOKENS = {"momentum": "mom", "mean_rev": "rev", "phantom": "ph"}

# New: meta(mom=+0.42 rev=+0.00 ph=+0.30 | reg=trending)
# Legacy: meta(mom=… rev=… sent=… ph=… | reg=…)
_META_RE = re.compile(
    r"meta\(mom=([+-][\d.]+) rev=([+-][\d.]+) "
    r"(?:sent=([+-][\d.]+) )?ph=([+-][\d.]+) \| reg=(\w+)\)")
_TOKEN_GROUP = {"momentum": 1, "mean_rev": 2, "phantom": 4}


def bucket_for(trend_score: Optional[float] = None,
               regime_id: Optional[str] = None) -> str:
    """Regime bucket for online-weight bookkeeping.

    Prefers the robust detector id (high_vol_trend / low_vol_range /
    high_vol_chop / …) via ``regime_id``; falls back to trend_score
    boundaries matching BaseBot.regime_context. None → "mixed".
    """
    if regime_id:
        try:
            from signals.regime_detector import meta_bucket
            b = meta_bucket(regime_id, trend_score)
            if b in BUCKETS:
                return b
        except Exception:
            pass
    if trend_score is None:
        return "mixed"
    if trend_score >= 0.65:
        return "trending"
    if trend_score <= 0.35:
        return "ranging"
    return "mixed"


def format_token(votes: dict, bucket: str) -> str:
    """Serialize sub-votes for a trade's reasoning (parsed at resolution)."""
    parts = " ".join(
        f"{tok}={votes.get(sub, 0.0):+.2f}" for sub, tok in SUB_TOKENS.items())
    return f"meta({parts} | reg={bucket})"


def parse_token(reasoning: Optional[str]):
    """Inverse of format_token: (votes dict, bucket) or None."""
    m = _META_RE.search(reasoning or "")
    if not m:
        return None
    votes = {sub: float(m.group(g)) for sub, g in _TOKEN_GROUP.items()}
    # Group 5 is always the regime bucket (sent is optional group 3).
    return votes, m.group(5)


class HybridMetaLearner:
    """Online per-sub multipliers from the hybrid's own resolved trades."""

    def __init__(self, eta: float = 0.12, min_mult: float = 0.4,
                 max_mult: float = 2.5, deadband: float = 0.05,
                 bucket_full_trust: int = 20, update_ttl: float = 60.0,
                 name_prefix: str = "hybrid"):
        self.eta = eta
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.deadband = deadband
        # Bucket-specific multiplier earns full say at this many bucket
        # samples; below it the overall multiplier dominates (shrinkage).
        self.bucket_full_trust = bucket_full_trust
        self.update_ttl = update_ttl
        self.name_prefix = name_prefix
        self._lock = threading.Lock()
        self._state: Optional[dict] = None
        self._last_update_check = 0.0
        self._last_persist = 0.0
        # True once the arena_state key exists (loaded or written) —
        # record_last only persists then, so a decision alone never writes
        # meta state to a DB the learner has no record in (keeps unrelated
        # unit tests and read-only inspection from creating state).
        self._key_exists = False

    # ------------------------------------------------------------------
    # State load / persist (arena_state JSON)
    # ------------------------------------------------------------------

    @staticmethod
    def _fresh_state() -> dict:
        return {
            "last_trade_id": 0,
            "last_decision_id": 0,
            "subs": {},
            "cf": {"n": 0, "correct": 0},  # counterfactual counters
            "updated_at": None,
            "last": {},
        }

    def _load(self) -> dict:
        if self._state is not None:
            return self._state
        try:
            raw = db.get_arena_state(STATE_KEY)
            self._key_exists = raw is not None
            self._state = json.loads(raw) if raw else self._fresh_state()
            # Backfill keys for older arena_state rows.
            self._state.setdefault("last_decision_id", 0)
            self._state.setdefault("cf", {"n": 0, "correct": 0})
        except Exception:
            self._state = self._fresh_state()
        return self._state

    def _persist(self) -> None:
        try:
            db.set_arena_state(STATE_KEY, json.dumps(self._state))
            self._key_exists = True
            self._last_persist = time.time()
        except Exception as e:  # never let bookkeeping stall a tick
            logger.debug(f"hybrid_meta persist failed: {e}")

    def _sub_rec(self, sub: str, bucket: str) -> dict:
        subs = self._load().setdefault("subs", {})
        rec = subs.setdefault(sub, {})
        return rec.setdefault(bucket, {"mult": 1.0, "n": 0, "correct": 0})

    def _apply_votes(self, votes: dict, bucket: str, market_up: bool,
                       eta: float) -> int:
        """Score one set of sub-votes; returns number of subs updated."""
        if bucket not in BUCKETS:
            bucket = "mixed"
        n_subs = 0
        for sub, vote in votes.items():
            try:
                v = float(vote)
            except (TypeError, ValueError):
                continue
            if abs(v) < self.deadband:
                continue  # this sub abstained
            correct = (v > 0) == bool(market_up)
            for b in (bucket, "overall"):
                rec = self._sub_rec(sub, b)
                step = eta if correct else -eta
                rec["mult"] = max(self.min_mult,
                                  min(self.max_mult,
                                      rec["mult"] * math.exp(step)))
                rec["n"] += 1
                rec["correct"] += int(correct)
            n_subs += 1
        return n_subs

    # ------------------------------------------------------------------
    # Online update (multiplicative weights on resolved trades + CF skips)
    # ------------------------------------------------------------------

    def update_from_trades(self) -> int:
        """Score newly-resolved hybrid trades and update the multipliers.

        Incremental by trade row id, but ``last_trade_id`` only advances
        past ids with no UNRESOLVED hybrid meta-trade before them — trades
        resolve out of placement order, and skipping a still-pending row
        would silently drop its lesson when it settles.
        Returns the number of trades processed (0 = no persist, so calling
        this against a DB with no meta-tagged trades writes nothing).
        """
        state = self._load()
        last_id = int(state.get("last_trade_id") or 0)
        try:
            with db.get_conn() as conn:
                pending = conn.execute(
                    """SELECT MIN(id) AS mid FROM trades
                       WHERE (outcome IS NULL OR outcome='pending')
                         AND bot_name LIKE ? AND reasoning LIKE '%meta(%'""",
                    (self.name_prefix + "%",)).fetchone()
                pending_min = pending["mid"] if pending else None
                # Hold-to-resolution + optional TP/SL exits: any resolved
                # directional outcome teaches the learner. exit_tp/exit_sl
                # count as win/loss of the held side.
                q = """SELECT id, side, outcome, reasoning FROM trades
                       WHERE id > ? AND outcome IN (
                           'win', 'loss', 'exit_tp', 'exit_sl')
                         AND bot_name LIKE ? AND reasoning LIKE '%meta(%'"""
                args = [last_id, self.name_prefix + "%"]
                if pending_min is not None:
                    q += " AND id < ?"
                    args.append(pending_min)
                rows = conn.execute(q + " ORDER BY id", args).fetchall()
        except Exception as e:
            logger.debug(f"hybrid_meta trade scan failed: {e}")
            return 0

        processed = 0
        with self._lock:
            for r in rows:
                parsed = parse_token(r["reasoning"])
                if parsed is None:
                    continue
                votes, bucket = parsed
                # Market went UP iff a YES trade won (or TP'd) or a NO trade lost.
                side = (r["side"] or "").lower()
                out = (r["outcome"] or "").lower()
                won = out in ("win", "exit_tp")
                market_up = (side == "yes") == won
                if self._apply_votes(votes, bucket, market_up, self.eta) > 0:
                    processed += 1
            if rows:
                # Advance past the last *processed* id that had a parseable
                # meta token; still advance to max scanned so unparseable
                # rows don't permanently block the cursor.
                state["last_trade_id"] = max(r["id"] for r in rows)
            if processed:
                state["updated_at"] = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.gmtime())
                self._persist()
        return processed

    def update_from_decisions(self) -> int:
        """Counterfactual learning from resolved hybrid *skips*.

        Uses ``decision_events`` rows with a stored ``meta_token`` and known
        ``market_up``. Buys are excluded (handled by ``update_from_trades``)
        so a filled trade is never double-counted. CF Hedge step is
        ``eta * HYBRID_META_CF_ETA_SCALE`` so volume of skips cannot dominate
        real fills.
        """
        if not getattr(config, "HYBRID_META_CF_ENABLED", True):
            return 0
        if not getattr(config, "DECISION_LEARN_FROM_ALL", True):
            return 0

        state = self._load()
        last_id = int(state.get("last_decision_id") or 0)
        eta_scale = float(getattr(config, "HYBRID_META_CF_ETA_SCALE", 0.25))
        eta_cf = max(1e-6, self.eta * eta_scale)
        max_n = int(getattr(config, "HYBRID_META_CF_MAX_PER_CYCLE", 200))
        try:
            with db.get_conn() as conn:
                # Prefer meta_token column; fall back to features/legacy
                # rows that may only have the token in features JSON.
                rows = conn.execute(
                    """SELECT id, market_up, meta_token, features, regime, action
                       FROM decision_events
                       WHERE id > ?
                         AND strategy_type = 'hybrid'
                         AND market_up IS NOT NULL
                         AND action = 'skip'
                         AND (meta_token IS NOT NULL OR features LIKE '%meta(%')
                       ORDER BY id
                       LIMIT ?""",
                    (last_id, max(1, max_n)),
                ).fetchall()
        except Exception as e:
            # Column missing on unmigrated DB — soft no-op.
            logger.debug(f"hybrid_meta decision scan failed: {e}")
            return 0

        processed = 0
        cf_correct = 0
        with self._lock:
            max_seen = last_id
            for r in rows:
                max_seen = max(max_seen, int(r["id"]))
                token = r["meta_token"]
                if not token and r["features"]:
                    m = re.search(r"meta\([^)]+\)", r["features"] or "")
                    token = m.group(0) if m else None
                if not token:
                    continue
                parsed = parse_token(token)
                if parsed is None:
                    continue
                votes, bucket = parsed
                # Prefer token bucket; fall back to detector regime → meta bucket
                if bucket not in BUCKETS:
                    try:
                        from signals.regime_detector import meta_bucket
                        bucket = meta_bucket(r["regime"], None) or "mixed"
                    except Exception:
                        bucket = "mixed"
                market_up = bool(int(r["market_up"]))
                n_subs = self._apply_votes(votes, bucket, market_up, eta_cf)
                if n_subs <= 0:
                    continue
                processed += 1
                # Rough CF accuracy: fraction of non-abstain subs correct
                for sub, vote in votes.items():
                    try:
                        v = float(vote)
                    except (TypeError, ValueError):
                        continue
                    if abs(v) < self.deadband:
                        continue
                    if (v > 0) == market_up:
                        cf_correct += 1
            if max_seen > last_id:
                state["last_decision_id"] = max_seen
            if processed:
                cf = state.setdefault("cf", {"n": 0, "correct": 0})
                cf["n"] = int(cf.get("n") or 0) + processed
                cf["correct"] = int(cf.get("correct") or 0) + cf_correct
                state["updated_at"] = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.gmtime())
                self._persist()
        return processed

    def maybe_update(self) -> int:
        """TTL-throttled trade + CF decision update — safe on the 1s path."""
        now = time.time()
        if (now - self._last_update_check) < self.update_ttl:
            return 0
        self._last_update_check = now
        n = self.update_from_trades()
        try:
            n += self.update_from_decisions()
        except Exception as e:
            logger.debug(f"hybrid_meta CF update failed: {e}")
        return n

    # ------------------------------------------------------------------
    # Decision-time reads
    # ------------------------------------------------------------------

    def online_mults(self, bucket: Optional[str] = None) -> dict:
        """Per-sub multiplier: overall blended toward the bucket's own.

        blend = (1-t)*overall + t*bucket with t = min(1, n_bucket /
        bucket_full_trust) — a thin bucket leans on the overall record, a
        fat one speaks for itself. No state → neutral 1.0 everywhere.
        """
        state = self._load()
        out = {}
        for sub in SUB_TOKENS:
            rec = state.get("subs", {}).get(sub, {})
            overall = rec.get("overall", {"mult": 1.0, "n": 0})
            mult = float(overall.get("mult", 1.0))
            if bucket in BUCKETS:
                brec = rec.get(bucket)
                if brec:
                    t = min(1.0, float(brec.get("n", 0)) / self.bucket_full_trust)
                    mult = (1.0 - t) * mult + t * float(brec.get("mult", 1.0))
            out[sub] = mult
        return out

    def record_last(self, weights: dict, online: dict, regime_label: str,
                    bucket: str, min_interval: float = 30.0) -> None:
        """Persist the latest effective weights for the dashboard (throttled)."""
        with self._lock:
            state = self._load()
            state["last"] = {
                "weights": {k: round(v, 4) for k, v in weights.items()},
                "online": {k: round(v, 4) for k, v in online.items()},
                "regime": regime_label,
                "bucket": bucket,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            }
            # Persist only once the learner has a real record in this DB
            # (first created by update_from_trades processing a resolved
            # trade) — a decision alone never creates state.
            if ((self._key_exists or state.get("subs"))
                    and (time.time() - self._last_persist) >= min_interval):
                self._persist()

    def snapshot(self) -> dict:
        """Full state + computed per-bucket multipliers (dashboard shape)."""
        state = self._load()
        return {
            **state,
            "mults": {b: self.online_mults(b) for b in BUCKETS},
            "params": {"eta": self.eta, "min_mult": self.min_mult,
                       "max_mult": self.max_mult, "deadband": self.deadband,
                       "bucket_full_trust": self.bucket_full_trust},
        }

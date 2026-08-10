"""Live strategy×regime and side×regime stats from resolved trades.

Data-driven inputs for:
  * core-lane by_regime earn bars (P&L gate)
  * strategy-level style-skip (toxic strategy in a regime, not whole market)
  * strategy×regime×side continuous tax / side-skip
  * NO-side edge tax / drift floor when NO is bleeding in a regime
  * tandem max-bots reduction in toxic regimes

Dual window (2026-08-07):
  * **long** — full lookback (default 72h) for stability / clear hysteresis
  * **fast** — recent hours (default 2.5h) so 5m markets can move knobs
    without waiting for n=20 on a diluted overnight pool

Hot-path cached; fail-open to empty dicts.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import config
import db

logger = logging.getLogger("arena.regime_stats")

_cache: tuple[float, dict[str, Any]] = (0.0, {})


def _ttl() -> float:
    return float(getattr(config, "REGIME_STATS_CACHE_SEC", 15.0))


def parse_regime_tag(trade_features: Any) -> Optional[str]:
    """Extract ``regime:<id>`` from trade_features list/JSON."""
    if not trade_features:
        return None
    try:
        arr = (
            json.loads(trade_features)
            if isinstance(trade_features, str)
            else trade_features
        )
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arr, list):
        return None
    for t in arr:
        if isinstance(t, str) and t.startswith("regime:") and not t.startswith(
            "regime_legacy"
        ):
            rid = t.split(":", 1)[1].strip()
            if rid and rid != "unknown":
                return rid
    return None


def _empty_cell() -> dict[str, Any]:
    return {
        "n": 0, "wins": 0, "pnl": 0.0, "wr": None,
        "fast_n": 0, "fast_wins": 0, "fast_pnl": 0.0, "fast_wr": None,
    }


def _bump(cell: dict[str, Any], *, win: bool, pnl: float, in_fast: bool) -> None:
    cell["n"] = int(cell.get("n") or 0) + 1
    cell["wins"] = int(cell.get("wins") or 0) + int(win)
    cell["pnl"] = float(cell.get("pnl") or 0.0) + float(pnl)
    if in_fast:
        cell["fast_n"] = int(cell.get("fast_n") or 0) + 1
        cell["fast_wins"] = int(cell.get("fast_wins") or 0) + int(win)
        cell["fast_pnl"] = float(cell.get("fast_pnl") or 0.0) + float(pnl)


def _finalize(cell: dict[str, Any]) -> dict[str, Any]:
    n = int(cell.get("n") or 0)
    wins = int(cell.get("wins") or 0)
    cell["n"] = n
    cell["wins"] = wins
    cell["pnl"] = float(cell.get("pnl") or 0.0)
    cell["wr"] = (wins / n) if n else None
    fn = int(cell.get("fast_n") or 0)
    fw = int(cell.get("fast_wins") or 0)
    cell["fast_n"] = fn
    cell["fast_wins"] = fw
    cell["fast_pnl"] = float(cell.get("fast_pnl") or 0.0)
    cell["fast_wr"] = (fw / fn) if fn else None
    return cell


def _parse_created_at(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    s = str(raw).strip()
    if not s:
        return None
    # SQLite timestamps: "YYYY-MM-DD HH:MM:SS" (sometimes with fractional)
    try:
        if "T" in s:
            s2 = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s2)
        else:
            # take first 19 chars for base format
            base = s[:19]
            dt = datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _rebuild() -> dict[str, Any]:
    """Scan recent resolved trades → nested stats maps (long + fast)."""
    hours = float(getattr(config, "REGIME_STATS_LOOKBACK_HOURS", 72.0))
    fast_hours = float(getattr(config, "REGIME_STATS_FAST_HOURS", 2.5))
    limit = int(getattr(config, "REGIME_STATS_MAX_TRADES", 4000))
    by_strat: dict[str, dict[str, dict[str, Any]]] = {}
    by_side: dict[str, dict[str, dict[str, Any]]] = {}
    by_reg: dict[str, dict[str, Any]] = {}
    by_strat_side: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    now = datetime.now(timezone.utc)
    fast_cut = now - timedelta(hours=fast_hours)

    try:
        with db.get_conn() as conn:
            smap = {
                r["bot_name"]: r["strategy_type"]
                for r in conn.execute(
                    "SELECT bot_name, strategy_type FROM bot_configs"
                )
            }
            rows = conn.execute(
                """SELECT bot_name, side, outcome, pnl, trade_features, created_at
                   FROM trades
                   WHERE outcome IN ('win', 'loss', 'exit_tp', 'exit_sl')
                     AND created_at >= datetime('now', ?)
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (f"-{int(hours)} hours", limit),
            ).fetchall()
    except Exception as e:
        logger.debug("regime_stats rebuild failed: %s", e)
        return {
            "by_strategy": {},
            "by_side": {},
            "by_regime": {},
            "by_strategy_side": {},
            "fast_hours": fast_hours,
            "lookback_hours": hours,
            "updated_at": time.time(),
        }

    for r in rows:
        regime = parse_regime_tag(r["trade_features"])
        if not regime:
            continue
        strat = smap.get(r["bot_name"])
        side = (r["side"] or "").lower()
        if side not in ("yes", "no"):
            continue
        outcome = r["outcome"] or ""
        win = outcome in ("win", "exit_tp")
        pnl = float(r["pnl"] or 0.0)
        created = _parse_created_at(r["created_at"] if "created_at" in r.keys()
                                    else None)
        in_fast = bool(created and created >= fast_cut)

        rc = by_reg.setdefault(regime, _empty_cell())
        _bump(rc, win=win, pnl=pnl, in_fast=in_fast)

        if strat:
            sc = by_strat.setdefault(regime, {}).setdefault(strat, _empty_cell())
            _bump(sc, win=win, pnl=pnl, in_fast=in_fast)
            ssc = (by_strat_side.setdefault(regime, {})
                   .setdefault(strat, {})
                   .setdefault(side, _empty_cell()))
            _bump(ssc, win=win, pnl=pnl, in_fast=in_fast)

        scell = by_side.setdefault(regime, {}).setdefault(side, _empty_cell())
        _bump(scell, win=win, pnl=pnl, in_fast=in_fast)

    for reg, cell in by_reg.items():
        by_reg[reg] = _finalize(cell)
    for reg, strats in by_strat.items():
        for st, cell in strats.items():
            strats[st] = _finalize(cell)
    for reg, sides in by_side.items():
        for sd, cell in sides.items():
            sides[sd] = _finalize(cell)
    for reg, strats in by_strat_side.items():
        for st, sides in strats.items():
            for sd, cell in sides.items():
                sides[sd] = _finalize(cell)

    return {
        "by_strategy": by_strat,
        "by_side": by_side,
        "by_regime": by_reg,
        "by_strategy_side": by_strat_side,
        "fast_hours": fast_hours,
        "lookback_hours": hours,
        "updated_at": time.time(),
    }


def snapshot(force: bool = False) -> dict[str, Any]:
    """Cached full stats blob."""
    global _cache
    now = time.time()
    if not force and (now - _cache[0]) < _ttl() and _cache[1]:
        return _cache[1]
    data = _rebuild()
    _cache = (now, data)
    return data


def strategy_regime_cell(
    regime: Optional[str], strategy_type: Optional[str]
) -> dict[str, Any]:
    if not regime or not strategy_type:
        return _empty_cell()
    blob = snapshot()
    return dict(
        (blob.get("by_strategy") or {}).get(regime, {}).get(strategy_type)
        or _empty_cell()
    )


def strategy_side_regime_cell(
    regime: Optional[str],
    strategy_type: Optional[str],
    side: str,
) -> dict[str, Any]:
    """strategy × regime × side cell (long + fast fields)."""
    if not regime or not strategy_type or side not in ("yes", "no"):
        return _empty_cell()
    blob = snapshot()
    return dict(
        (blob.get("by_strategy_side") or {})
        .get(regime, {})
        .get(strategy_type, {})
        .get(side)
        or _empty_cell()
    )


def side_regime_cell(regime: Optional[str], side: str) -> dict[str, Any]:
    if not regime or side not in ("yes", "no"):
        return _empty_cell()
    blob = snapshot()
    return dict(
        (blob.get("by_side") or {}).get(regime, {}).get(side) or _empty_cell()
    )


def regime_cell(regime: Optional[str]) -> dict[str, Any]:
    if not regime:
        return _empty_cell()
    blob = snapshot()
    return dict((blob.get("by_regime") or {}).get(regime) or _empty_cell())


def is_toxic_cell(
    cell: dict[str, Any],
    *,
    min_n: Optional[int] = None,
    wr_bar: Optional[float] = None,
    require_neg_pnl: bool = True,
    path: str = "long",
) -> bool:
    """True when cell has enough samples and is live-toxic.

    ``path``:
      * ``long`` (default) — use n/wr/pnl (backward-compatible)
      * ``fast`` — use fast_n/fast_wr/fast_pnl
      * ``either`` — toxic if long OR fast path is toxic
    """
    if path == "either":
        return (
            is_toxic_cell(cell, min_n=min_n, wr_bar=wr_bar,
                          require_neg_pnl=require_neg_pnl, path="long")
            or is_toxic_cell(cell, min_n=min_n, wr_bar=wr_bar,
                             require_neg_pnl=require_neg_pnl, path="fast")
        )
    if path == "fast":
        min_n = int(
            min_n
            if min_n is not None
            else getattr(config, "REGIME_STYLE_SKIP_FAST_MIN_N", 10)
        )
        wr_bar = float(
            wr_bar
            if wr_bar is not None
            else getattr(config, "REGIME_STYLE_SKIP_FAST_WR", 0.38)
        )
        n = int(cell.get("fast_n") or 0)
        wr = cell.get("fast_wr")
        if wr is None and n:
            wins = int(cell.get("fast_wins") or 0)
            wr = wins / n
        pnl = float(cell.get("fast_pnl") or 0.0)
    else:
        min_n = int(
            min_n
            if min_n is not None
            else getattr(config, "REGIME_STYLE_SKIP_MIN_TRADES", 18)
        )
        wr_bar = float(
            wr_bar
            if wr_bar is not None
            else getattr(config, "REGIME_STYLE_SKIP_WR", 0.42)
        )
        n = int(cell.get("n") or 0)
        wr = cell.get("wr")
        if wr is None and n:
            wins = int(cell.get("wins") or 0)
            wr = wins / n
        pnl = float(cell.get("pnl") or 0.0)
    if n < min_n:
        return False
    if wr is None:
        wr = 0.5
    if wr > wr_bar:
        return False
    if require_neg_pnl and pnl >= 0:
        return False
    return True


def is_healthy_cell(
    cell: dict[str, Any],
    *,
    min_n: Optional[int] = None,
    wr_clear: Optional[float] = None,
    path: str = "long",
) -> bool:
    if path == "fast":
        min_n = int(
            min_n
            if min_n is not None
            else getattr(config, "REGIME_STYLE_SKIP_FAST_MIN_N", 10)
        )
        n = int(cell.get("fast_n") or 0)
        wr = cell.get("fast_wr")
        if wr is None and n:
            wins = int(cell.get("fast_wins") or 0)
            wr = wins / n
        pnl = float(cell.get("fast_pnl") or 0.0)
    else:
        min_n = int(
            min_n
            if min_n is not None
            else getattr(config, "REGIME_STYLE_SKIP_MIN_TRADES", 18)
        )
        n = int(cell.get("n") or 0)
        wr = cell.get("wr")
        if wr is None and n:
            wins = int(cell.get("wins") or 0)
            wr = wins / n
        pnl = float(cell.get("pnl") or 0.0)
    wr_clear = float(
        wr_clear
        if wr_clear is not None
        else getattr(config, "REGIME_STYLE_SKIP_CLEAR_WR", 0.48)
    )
    if n < min_n:
        return False
    if wr is None:
        wr = 0.5
    return wr >= wr_clear and pnl >= 0


def effective_wr(
    cell: dict[str, Any],
    *,
    min_n_fast: Optional[int] = None,
    min_n_long: Optional[int] = None,
    fast_blend: Optional[float] = None,
) -> Optional[float]:
    """Blended WR for continuous adaptation.

    When fast has enough samples: ``fast_blend * fast_wr + (1-blend) * long_wr``
    (long falls back to fast if long thin). When only long is thick: long WR.
    Thin everywhere → None (caller treats as no opinion).
    """
    min_n_fast = int(
        min_n_fast
        if min_n_fast is not None
        else getattr(config, "REGIME_ADAPT_CONT_MIN_N", 8)
    )
    min_n_long = int(
        min_n_long
        if min_n_long is not None
        else getattr(config, "REGIME_STYLE_SKIP_MIN_TRADES", 18)
    )
    blend = float(
        fast_blend
        if fast_blend is not None
        else getattr(config, "REGIME_ADAPT_FAST_BLEND", 0.65)
    )
    blend = max(0.0, min(1.0, blend))
    fn = int(cell.get("fast_n") or 0)
    ln = int(cell.get("n") or 0)
    fwr = cell.get("fast_wr")
    lwr = cell.get("wr")
    if fwr is None and fn:
        fwr = int(cell.get("fast_wins") or 0) / fn
    if lwr is None and ln:
        lwr = int(cell.get("wins") or 0) / ln
    if fn >= min_n_fast and fwr is not None:
        if ln >= min_n_long and lwr is not None:
            return blend * float(fwr) + (1.0 - blend) * float(lwr)
        return float(fwr)
    if ln >= min_n_long and lwr is not None:
        return float(lwr)
    if fn >= min_n_fast and fwr is not None:
        return float(fwr)
    return None


def invalidate_cache() -> None:
    global _cache
    _cache = (0.0, {})

"""Pure helpers for market filtering and ET time arithmetic.

No threads, no I/O, no side effects. Importable from anywhere inside the
arena runtime package without dragging in the signal feeds, the bot classes,
or the credentials store.

Extracted from the old monolithic arena.py so each threaded worker can pull
in only the helpers it actually needs.  ``compute_time_remaining_seconds``
was inlined into the old main_loop and is now reused by both
``MarketDiscovery`` (who decorates its snapshot) and ``Trader`` (who
freshens the time_remaining on every 1s tick so the staleness guard sees
live numbers even between discovery cycles).
"""

import re
from datetime import datetime, timedelta, timezone


# Eastern Time zone (handles EST/EDT transitions automatically when the
# ``zoneinfo`` package is available — Python 3.9+ on macOS/Linux).
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    _ET = _ZoneInfo("America/New_York")

    def to_et(dt_utc: datetime) -> datetime:
        return dt_utc.astimezone(_ET)
except Exception:  # pragma: no cover -- Windows without tzdata
    def to_et(dt_utc: datetime) -> datetime:
        year = dt_utc.year
        # DST start: 2nd Sunday of March at 07:00 UTC (= 2:00 AM EST)
        mar1 = datetime(year, 3, 1, tzinfo=timezone.utc)
        dst_start = (
            mar1
            + timedelta(days=(6 - mar1.weekday()) % 7)
            + timedelta(weeks=1, hours=7)
        )
        # DST end: 1st Sunday of November at 06:00 UTC (= 2:00 AM EDT)
        nov1 = datetime(year, 11, 1, tzinfo=timezone.utc)
        dst_end = (
            nov1
            + timedelta(days=(6 - nov1.weekday()) % 7)
            + timedelta(hours=6)
        )
        offset = timedelta(hours=-4 if dst_start <= dt_utc < dst_end else -5)
        return dt_utc + offset


# "9:00PM-9:05PM" or "9:00 PM – 9:05 PM" -- Simmer phrases windows in ET.
_5MIN_RANGE_RE = re.compile(
    r'(\d{1,2}):(\d{2})\s*(am|pm)\s*[-–]\s*(\d{1,2}):(\d{2})\s*(am|pm)'
)


def _parse_range_minutes(question: str):
    """Parse a "9:00PM-9:05PM"-style window out of a market question.

    Returns ``(start_min, end_min)`` as minutes-of-day in 24h ET, or
    ``None`` if the question doesn't carry an ET time range.
    """
    m = _5MIN_RANGE_RE.search((question or "").lower())
    if not m:
        return None
    h1, m1, ap1 = int(m.group(1)), int(m.group(2)), m.group(3)
    h2, m2, ap2 = int(m.group(4)), int(m.group(5)), m.group(6)
    if ap1 == 'pm' and h1 != 12: h1 += 12
    if ap1 == 'am' and h1 == 12: h1 = 0
    if ap2 == 'pm' and h2 != 12: h2 += 12
    if ap2 == 'am' and h2 == 12: h2 = 0
    return h1 * 60 + m1, h2 * 60 + m2


def is_btc_updown(m: dict) -> bool:
    """True if this market looks like a BTC up/down window.

    Matches either via the Simmer ``fast-5m`` tag plus a BTC keyword, or as
    a fallback via any BTC keyword + an up/down phrasing in the question.
    """
    q = (m.get("question", "") or "").lower()
    tags = m.get("tags") or []
    if "fast-5m" in tags or "fast" in tags:
        if "bitcoin" in q or "btc" in q:
            return True
    is_btc = "bitcoin" in q or "btc" in q
    is_updown = (
        "up or down" in q or "up/down" in q
        or "higher or lower" in q or "above or below" in q
    )
    return is_btc and is_updown


def is_5min_market(question: str) -> bool:
    """True if the market question represents a *5-minute* window."""
    parsed = _parse_range_minutes(question)
    if not parsed:
        return False
    start_min, end_min = parsed
    diff = end_min - start_min
    if diff < 0:
        diff += 24 * 60
    return diff == 5


def select_current_market(markets: list, now_utc: datetime) -> dict | None:
    """Pick the live 5-minute BTC window from *decorated* markets.

    ``markets`` must already carry ``time_remaining_seconds`` (see
    ``compute_time_remaining_seconds``). A market qualifies as *current* only
    when BOTH hold:

      * it is a genuine 5-minute window (``is_5min_market`` on its question), and
      * its real ``resolves_at`` timestamp puts it inside its window right now,
        i.e. ``0 < time_remaining_seconds <= 300``.

    Selection is by the actual timestamp, never by ET time-of-day, so a
    future-dated market whose clock window happens to straddle "now" (e.g. a
    *next-day* 8:15-8:30 window at 8:29 today) is never chosen. Likewise a
    15-minute window is rejected outright. Returns the soonest-resolving
    qualifying market, or ``None`` when no 5-minute window is live.
    """
    live = [
        m for m in markets
        if is_5min_market(m.get("question", "") or "")
        and 0 < m.get("time_remaining_seconds", 0) <= 300
    ]
    live.sort(key=lambda m: m.get("time_remaining_seconds", 999))
    return live[0] if live else None


def compute_time_remaining_seconds(market: dict, now_utc: datetime) -> int:
    """Seconds until ``market`` resolves; falls back to 300 (5-min default).

    Tolerates both ISO-8601 ``resolves_at`` strings (with/without trailing
    ``Z``) and the older ``end_time`` field.  When neither parses we assume
    the standard 5-minute window so downstream staleness math doesn't blow
    up on bad data.
    """
    resolves_at_str = market.get("resolves_at") or market.get("end_time")
    time_remaining = 300
    if resolves_at_str:
        try:
            s = resolves_at_str.replace("Z", "+00:00").replace(" ", "T")
            resolves_at = datetime.fromisoformat(s)
            if resolves_at.tzinfo is None:
                resolves_at = resolves_at.replace(tzinfo=timezone.utc)
            time_remaining = int((resolves_at - now_utc).total_seconds())
        except Exception:
            pass
    return time_remaining

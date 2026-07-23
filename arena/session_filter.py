"""Session-timing skip filter — 'build the skip, default state is flat'.

The profitable-bot research (0xSurferX / 0x_Punisher, Jun 2026) is emphatic that
the best bots skip far more than they trade, and that *specific clock windows*
are structurally hostile to 5-minute directional bets:

  * **Session handovers** — one region logging off as another logs on. Conviction
    dies and windows chop. NYSE open (~09:30 ET) and NYSE close (~16:00 ET) log
    the most direction flips per 5-min window (up to 16), with recorded single-slot
    losses. These are the concrete high-flip windows to sit out.
  * **Weekend regime** — casual money, fast spikes that snap back, nothing trends;
    the same signal that hits 63% WR Monday can drop to 44% Saturday.

This module is the *mechanism*, defaulted to the research's known-bad windows.
Once the arena has its own logs, tighten the windows to the personal flip-heavy
slots (that is the intended follow-up, per the research). All windows are
expressed in America/New_York because the markets trade on ET.

Pure and side-effect free so it is trivially testable; the Trader calls
:func:`session_skip` once per tick and sits flat when it returns a reason.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import config

logger = logging.getLogger(__name__)

_ET_ZONE = "America/New_York"


def _to_et(now_utc: datetime) -> datetime:
    """Convert an aware/naive UTC datetime to ET (DST-correct when tzdata is present)."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        return now_utc.astimezone(ZoneInfo(_ET_ZONE))
    except Exception:
        # Fallback: fixed −4 (EDT). Coarse but only used for a session gate.
        return now_utc.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_window(spec: str) -> Optional[tuple]:
    """Parse ``'HH:MM-HH:MM'`` into ``(start_min, end_min)`` minutes-of-day."""
    try:
        start_s, end_s = spec.split("-")
        sh, sm = (int(x) for x in start_s.split(":"))
        eh, em = (int(x) for x in end_s.split(":"))
        return (sh * 60 + sm, eh * 60 + em)
    except (ValueError, AttributeError):
        return None


def session_skip(now_utc: datetime) -> Optional[str]:
    """Return a reason string if directional trading should be skipped now, else ``None``.

    Honors three config knobs (all safe defaults):
      * ``SESSION_SKIP_ENABLED`` — master switch (default True).
      * ``SESSION_SKIP_WEEKENDS`` — skip Sat/Sun entirely (default False; crypto
        trades weekends and v2 has no weekend data yet).
      * ``SESSION_SKIP_WINDOWS_ET`` — list of ``'HH:MM-HH:MM'`` ET windows to sit out.
    """
    if not getattr(config, "SESSION_SKIP_ENABLED", True):
        return None

    et = _to_et(now_utc)

    if getattr(config, "SESSION_SKIP_WEEKENDS", False) and et.weekday() >= 5:
        return f"weekend skip ({et.strftime('%a')} ET)"

    minute_of_day = et.hour * 60 + et.minute
    for spec in getattr(config, "SESSION_SKIP_WINDOWS_ET", []) or []:
        win = _parse_window(spec)
        if win and win[0] <= minute_of_day < win[1]:
            return f"session skip ({spec} ET handover)"
    return None

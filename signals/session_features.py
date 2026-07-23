"""Time-of-day and session features (pure, deterministic).

``compute(now)`` takes an explicit timezone-aware datetime — no wall-clock
reads inside — and returns cyclical time encodings plus session context. All
outputs are CONTEXT (non-directional): they never pick a side; the harness
uses them to condition other signals ("does drift work better in the US
session?") and future selectivity work can scale conviction by them.

Outputs:
- ``sess_tod_sin`` / ``sess_tod_cos``: UTC time-of-day on the unit circle —
  smooth and continuous across midnight (an hour-number feature jumps 23→0).
- ``sess_dow_sin`` / ``sess_dow_cos``: day-of-week on the unit circle.
- ``sess_label``: "asia" / "europe" / "us" / "us_late" by UTC hour — for
  logs/report bucketing only; numeric consumers use the cyclical encodings.
- ``sess_nyse_prox``: smooth 0..1 proximity to NYSE open (09:30 ET) or close
  (16:00 ET) — the two high-flip windows the session filter already skips
  (Gaussian shoulder, same shape as macro_calendar).
- ``sess_weekend``: 1.0 on Sat/Sun ET, else 0.0.
"""

import datetime
import math
import zoneinfo

from signals.curves import gaussian_zone

ET = zoneinfo.ZoneInfo("America/New_York")
NYSE_EVENTS_ET = [(9, 30), (16, 0)]     # open / close
NYSE_PROX_WIDTH_MIN = 15.0              # Gaussian shoulder width in minutes

# UTC-hour session boundaries (approximate, stable): Asia 00-07, Europe 07-13,
# US 13-21, late/overnight 21-24.
_SESSIONS = ((0, 7, "asia"), (7, 13, "europe"), (13, 21, "us"),
             (21, 24, "us_late"))


def session_label(now_utc: datetime.datetime) -> str:
    h = now_utc.hour
    for lo, hi, label in _SESSIONS:
        if lo <= h < hi:
            return label
    return "us_late"


def nyse_proximity(now: datetime.datetime) -> float:
    """Smooth 0..1 proximity to NYSE open/close (ET); 0.0 on weekends."""
    et = now.astimezone(ET)
    if et.weekday() >= 5:
        return 0.0
    minutes_now = et.hour * 60 + et.minute + et.second / 60.0
    prox = 0.0
    for h, m in NYSE_EVENTS_ET:
        prox = max(prox, gaussian_zone(minutes_now, h * 60 + m,
                                       NYSE_PROX_WIDTH_MIN))
    return prox


def compute(now: datetime.datetime) -> dict:
    """All session features for an explicit timezone-aware datetime."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.timezone.utc)
    utc = now.astimezone(datetime.timezone.utc)

    day_frac = (utc.hour * 3600 + utc.minute * 60 + utc.second) / 86400.0
    week_frac = (utc.weekday() + day_frac) / 7.0
    et = now.astimezone(ET)

    return {
        "sess_tod_sin": math.sin(2.0 * math.pi * day_frac),
        "sess_tod_cos": math.cos(2.0 * math.pi * day_frac),
        "sess_dow_sin": math.sin(2.0 * math.pi * week_frac),
        "sess_dow_cos": math.cos(2.0 * math.pi * week_frac),
        "sess_label": session_label(utc),
        "sess_nyse_prox": nyse_proximity(now),
        "sess_weekend": 1.0 if et.weekday() >= 5 else 0.0,
    }

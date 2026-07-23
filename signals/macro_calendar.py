"""High-impact macro release awareness (time-based, no API dependency).

US macro prints that move BTC hardest cluster at two ET slots on weekdays:
08:30 ET (CPI / PPI / NFP / jobless claims) and 14:00 ET (FOMC statement
days). Around those slots the 5-min market's first seconds can gap violently
— the same class of risk the session filter already handles for NYSE
open/close ("build the skip, default flat").

``macro_caution(now)`` returns a smooth 0..1 caution score: 1.0 inside the
core minutes around a slot, decaying smoothly over the shoulder (Gaussian in
minutes-from-slot). Consumers may scale selectivity by it; it is NOT a
directional signal and carries no lane weight.

This is intentionally time-based only — an economic-calendar API would add a
network dependency and an auth key for marginal gain; the slots themselves
are stable. If a calendar source is added later, keep this fallback.
"""

import datetime
import zoneinfo

from signals.curves import gaussian_zone

ET = zoneinfo.ZoneInfo("America/New_York")
# (hour, minute) ET slots; width in minutes of the Gaussian shoulder.
SLOTS = [(8, 30), (14, 0)]
CORE_WIDTH_MIN = 6.0


def macro_caution(now: datetime.datetime | None = None) -> float:
    """Smooth 0..1 caution around high-impact macro release slots (ET)."""
    if now is None:
        now = datetime.datetime.now(tz=ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=ET)
    else:
        now = now.astimezone(ET)

    if now.weekday() >= 5:  # weekend: no US macro prints
        return 0.0

    minutes_now = now.hour * 60 + now.minute + now.second / 60.0
    caution = 0.0
    for h, m in SLOTS:
        slot = h * 60 + m
        caution = max(caution,
                      gaussian_zone(minutes_now, slot, CORE_WIDTH_MIN))
    return caution

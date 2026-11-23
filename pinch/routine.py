"""The daily routine (decision 82): what time of day it is for Pinch, and what that changes.

  * night (22:00-06:00): sleepiness builds three times faster, so the evening ends in Rest on its own and
    Pinch sleeps through the night unless something wakes it; by day the rate is the usual one
  * quiet hours, set by the owner ("quiet hours from 10 pm to 7 am", or the HTTP API): no sounds and only
    quiet activities inside them, as in a meeting; "no quiet hours" clears them
  * the local hour: from a fixed UTC offset when one is set (the twin sets India's +5.5, so a run replays
    exactly on any machine), else the robot's own clock and time zone

Settings live in the memory file (memory.data["settings"]), so they survive restarts.
"""

import re
import time

NIGHT = (22.0, 6.0)               # local hours
NIGHT_SLEEP_DRIVE = 3.0           # sleepiness builds this much faster at night
PARTS = ((6.0, "morning"), (12.0, "afternoon"), (18.0, "evening"), (22.0, "night"))
DEFAULT_SETTINGS = dict(utc_offset_h=None, quiet_hours=None, face_recognition=True,
                        ask_before_remembering=True)      # facts about its person wait for their yes (ARC-014)

_WORDS = {w: i + 1 for i, w in enumerate("one two three four five six seven eight nine ten eleven twelve".split())}
_HOUR = r"(\d{1,2}|" + "|".join(_WORDS) + r")(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?"


def local_hour(clock_s, utc_offset_h=None):
    """Hour of the day (0-24, fractional) at time clock_s (seconds since the epoch)."""
    if utc_offset_h is None:
        lt = time.localtime(clock_s)
        return lt.tm_hour + lt.tm_min / 60 + lt.tm_sec / 3600
    return (clock_s / 3600.0 + utc_offset_h) % 24.0


def in_window(hour, start, end):
    """Is hour inside [start, end)? Windows may run over midnight (22 -> 7)."""
    return start <= hour < end if start <= end else (hour >= start or hour < end)


def _to_24h(num, minutes, ampm):
    h = _WORDS.get(num) or int(num)
    h = h % 12 + (12 if ampm and ampm.startswith("p") else 0) if ampm else h
    return (h % 24) + (int(minutes) / 60 if minutes else 0.0)


def parse_quiet_hours(text):
    """"quiet hours from 10 pm to 7 am" -> (22.0, 7.0); "no quiet hours" -> "clear"; else None. Works on the
    raw text (speech recognisers write numbers as digits)."""
    t = text.lower()
    if re.search(r"\b(no|cancel|clear|stop|remove|turn off)\b.*\bquiet hours\b|\bquiet hours\b.*\b(off|cancel)", t):
        return "clear"
    m = re.search(r"\bquiet hours?\b.*?\b(?:from\s+)?" + _HOUR + r"\s*(?:to|until|till|-)\s*" + _HOUR, t)
    if not m:
        return None
    a, b = _to_24h(*m.group(1, 2, 3)), _to_24h(*m.group(4, 5, 6))
    return None if a == b else (a, b)


class Routine:
    def __init__(self, settings):
        self.settings = settings                  # memory.data["settings"] (shared, saved with the memory)
        for k, v in DEFAULT_SETTINGS.items():
            settings.setdefault(k, v)

    def hour(self, clock_s):
        return local_hour(clock_s, self.settings.get("utc_offset_h"))

    def part_of_day(self, clock_s):
        h = self.hour(clock_s)
        name = "night"
        for start, part in PARTS:
            if h >= start:
                name = part
        return name

    def is_night(self, clock_s):
        return in_window(self.hour(clock_s), *NIGHT)

    def sleep_drive(self, clock_s):
        """How much faster than usual sleepiness builds right now."""
        return NIGHT_SLEEP_DRIVE if self.is_night(clock_s) else 1.0

    def quiet_hours(self, clock_s):
        qh = self.settings.get("quiet_hours")
        return bool(qh) and in_window(self.hour(clock_s), *qh)

    def set_quiet_hours(self, start, end):
        if not (0 <= start < 24 and 0 <= end < 24) or start == end:
            raise ValueError(f"quiet hours must be two different hours in 0-24, got {start}, {end}")
        self.settings["quiet_hours"] = [float(start), float(end)]

    def clear_quiet_hours(self):
        self.settings["quiet_hours"] = None

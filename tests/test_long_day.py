"""The 24-hour full-autonomy run, in the suite (sim/benches/longrun.py; the bench runs the full version with five
10-minute physics windows). A whole simulated day in fast-forward - the real mind on a kinematic desk - checked for
behaviour quality, quiet hours, meetings, sleep and variety; a short day with two physics windows warmed from the
fast-forward, checked for safety; and exact replay.

Checks listed in KNOWN_ISSUES are failures of the mind these runs found: they must still fail here (a strict
xfail), so fixing one turns its test red until the entry is deleted."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.benches.longrun import KNOWN_ISSUES, day_run  # noqa: E402


def assert_checks(r, only=None):
    checks = {k: v for k, v in r["checks"].items() if only is None or k in only}
    bad = {k: (c["value"], c["want"]) for k, c in checks.items() if not c["ok"] and k not in KNOWN_ISSUES}
    assert not bad, f"failed: {bad}"
    if only is None:
        fixed = [k for k, c in checks.items() if c["ok"] and k in KNOWN_ISSUES]
        assert not fixed, f"{fixed} pass now: delete them from KNOWN_ISSUES in sim/benches/longrun.py"


def test_24h_fast_forward():
    r = day_run(0, physics=False)
    assert_checks(r)
    assert {"sleeps_at_night", "quiet_hours_silent", "variety", "observer_no_lock_or_oscillation"} <= set(r["checks"])
    assert r["decisions"] > 1000 and r["wall_s"]["ff_speedup"] > 300, (r["decisions"], r["wall_s"])


def test_physics_windows_are_safe():
    r = day_run(1, hours=3.6, windows=(("morning", 7.1), ("standup", None)), window_s=60.0, physics=True)
    assert len(r["windows"]) == 2, r["windows"]
    assert_checks(r, only={"physics_safe", "meetings_silent", "quiet_hours_silent", "quiet_activities_only",
                           "observer_no_lock_or_oscillation"})
    standup = r["windows"][1]
    assert standup["activities"].get("go_quiet"), f"no go_quiet in the stand-up window: {standup['activities']}"


def test_replays_exactly():
    a, b, c = day_run(2, hours=1.0, physics=False), day_run(2, hours=1.0, physics=False), day_run(3, hours=1.0,
                                                                                               physics=False)
    assert a["digest"] == b["digest"] and a["activities"] == b["activities"], "same seed, different day"
    assert a["digest"] != c["digest"], "two seeds gave the same day"

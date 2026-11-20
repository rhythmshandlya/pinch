"""The 7-day memory run, in the suite: seven simulated days with the robot switched on 09:30-11:00 each day (the
bench, sim/benches/longrun.py week, runs every hour of every day with a reboot each night). Each switch-on loads
the memory file the last switch-off wrote. Checked with numbers: the memory survives every reboot, the owner and a
colleague grow familiar, a guest seen once fades by the 90-day half-life and their fear by the 3-day one, a learned
danger spot fades with awake time, the daily stand-up becomes a known routine, and likes don't move without praise.

Checks in KNOWN_ISSUES must still fail (strict xfail): see tests/test_long_day.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.benches.longrun import week_run  # noqa: E402
from tests.test_long_day import assert_checks  # noqa: E402


def test_week_of_memory():
    r = week_run(0, awake=(9.5, 11.0))
    assert_checks(r)
    assert len(r["restarts"]) == 6 and len({d["day"] for d in r["daily"]}) == 7
    c = r["checks"]
    assert c["standup_routine_learned"]["value"]["learned_on_day"] <= 3
    sam = c["owner_familiarity_grows"]["value"]
    assert sam[-1] - sam[0] > 0.1, sam

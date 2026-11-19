"""What every activity shares: the Activity base class, the quiet-mode list, and small helpers."""

import math

import numpy as np

QUIET_OK = {"go_quiet", "rest", "react_touch", "comfort", "idle", "startle", "tidy", "guard_edge"}

GRASP_ENABLED = True       # the redesigned claw picks up sticks and blocks (decision 24); pushing is the fallback

# Top-level behaviour states (decision 63). Every activity belongs to one; the mind reports the current one
# and logs every change. "task" (desk work you asked for) is Pinch's addition to the usual animal set.
STATES = ("idle", "explore", "social", "play", "alert", "defensive", "rest", "recover", "task")


class Activity:
    """One thing Pinch can do. utility() says how much it wants to now (0 = not at all); run() is a generator
    of steps (motions, faces, sounds, waits, and ("phase", name) to say which part of the activity it is in).
    Contract: state is one of STATES; max_s bounds how long it may run (then it ends with reason "timeout");
    interruptible activities give way to clearly better or urgent ones, and may be resumed afterwards."""
    name = "activity"
    state = "idle"
    interruptible = True
    min_s = 2.0
    max_s = 300.0
    quiet_ok = False

    def utility(self, m) -> float:
        return 0.0

    def run(self, m):
        yield ("wait", 1.0)

def _owner_near(m):
    return m.world.desk.owner_xy[0], m.world.desk.owner_xy[1] + 0.32       # a spot on the desk near you

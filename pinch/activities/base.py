"""What every activity shares: the Activity base class, the quiet-mode list, and small helpers."""

import math

import numpy as np

QUIET_OK = {"go_quiet", "rest", "react_touch", "comfort", "idle", "startle", "tidy", "guard_edge"}

GRASP_ENABLED = True       # the redesigned claw picks up sticks and blocks (decision 24); pushing is the fallback

# Top-level behaviour states (decision 63). Every activity belongs to one; the mind reports the current one
# and logs every change. "task" (desk work you asked for) is Pinch's addition to the usual animal set.
STATES = ("idle", "explore", "social", "play", "alert", "defensive", "rest", "sleep", "recover", "task")   # sleep: rest in its asleep phase


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
    own_jitter = False         # True: its tie-break jitter comes from its own random stream (Selection._own_jitter)

    def prepare(self, m):
        """Book-keeping before this choice (resetting a counter, drawing a due time): runs for every activity before
        any is scored. utility() itself must not change anything, so a choice can be scored again, compared or
        learned from (docs/research/v2_1: a learned scorer needs pure scores)."""

    def utility(self, m) -> float:
        """How much it wants this now (0 = not at all). Reads the mind; changes nothing."""
        return 0.0

    def run(self, m):
        yield ("wait", 1.0)

def _owner_near(m):
    """A spot on the desk near its person, nearer the more it likes and knows them, further if it fears them
    (proxemics by relationship: Mumm & Mutlu 2011, docs/research/v2_1/gaps/notes.md)."""
    ox, oy = m.world.desk.owner_xy
    pid = m.feedback_person() if hasattr(m, "feedback_person") else None
    p = m.people.people.get(pid, {}) if pid else {}
    w, f, r = p.get("W", 0.5), p.get("F", 0.5), p.get("R", 0.0)
    d = 0.32 - 0.06 * max(0.0, w) * f + 0.08 * max(r, 1.0 - f) * (0.0 if p.get("owner") else 1.0)
    return ox, oy + float(min(0.42, max(0.24, d)))

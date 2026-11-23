"""Startle and caution: flinching, keeping its distance from someone it fears, hiding."""

import numpy as np

from pinch.activities.base import (  # noqa: F401
    Activity,
)
from pinch.affect import EMO
from pinch.attention import bearing
from pinch.world import BODY_SCALE

FEAR = EMO["fear_threshold"]                 # configs/mind/expression.toml: one line for fear


class Startle(Activity):
    state, max_s = "alert", 30.0
    name, interruptible, min_s, quiet_ok = "startle", False, 2.0, True

    def utility(self, m):
        return 2.2 * (m.affect.needs.safety > FEAR)    # the same line as the scared feeling

    def run(self, m):
        yield ("face", "scared")
        if not m.affect.quiet:
            yield ("sound", "startled_squeak")
        f = m.attention.focus                                 # where the fright came from, if it knows
        d = bearing(f, m.world.pose) if f is not None and m.t - f.t < 3.0 else None
        d = 0.0 if d is None else d
        b = 0.04 * BODY_SCALE                                 # (a bigger crab backs off further)
        yield ("walk", -b * np.cos(d), -b * np.sin(d), 0.0, 1.0)   # first a flinch away from it (straight back if
        yield ("emote", "scared")                             # unknown), then it cowers: the startle reflex's order


class Orient(Activity):
    """Something off to one side called its attention (pinch/attention.py): turn toward it, look a moment,
    then let attention move on - whatever it was doing comes back (the state layer resumes it). Weak stimuli
    don't win over a job in hand: the utility only beats idling."""
    state, max_s = "alert", 15.0
    name, min_s, quiet_ok = "orient", 1.0, True

    def utility(self, m):
        f, sal = m.attention.update(m.t)
        d = bearing(f, m.world.pose)
        if d is None or abs(d) < 0.4 or m.t - f.t > 2.0:
            return 0.0
        return 0.15 + 0.9 * sal

    def run(self, m):
        f, _ = m.attention.update(m.t)
        d = bearing(f, m.world.pose)
        if d is None:
            return
        x, y, h = m.world.pose
        target = tuple(f.xy) if f.xy is not None else (x + 0.3 * np.cos(h + d), y + 0.3 * np.sin(h + d))
        yield ("phase", f"orient_{f.kind}")
        yield ("face", "surprised" if f.kind == "loud" else "curious")
        yield ("look_at", target)
        yield ("turn_to", target)
        yield ("wait", 1.5)
        m.attention.attended(f, m.t)


class Shy(Activity):
    """Someone it doesn't know yet: stay near home, peek at them, warm up slowly."""
    state, max_s = "defensive", 120.0
    name, min_s = "shy", 4.0
    quiet_ok = True

    def utility(self, m):
        now = m.world.clock()
        return 0.9 if m.people.strangers_here(now) and not m.people.feared_here(now) else 0.0

    def run(self, m):
        yield ("face", "shy")
        yield ("goto", m.memory.home_of("pinch") or m.world.desk.home_xy, 0.05)
        peeks = 0
        while m.people.strangers_here(m.world.clock()) and peeks < 12:    # one shy spell, not a restart loop
            yield ("look_at", m.world.desk.owner_xy)
            yield ("wait", 4.0 + 2.0 * m.rng.random())
            yield ("face", "curious" if peeks % 2 else "shy")
            if peeks == 0 and not m.affect.quiet:
                yield ("sound", "curious_chirp")
            peeks += 1


class Hide(Activity):
    """Someone it's afraid of is here: back off to its home spot, freeze, keep an eye on them."""
    state, max_s = "defensive", 600.0
    name, min_s = "hide", 4.0
    quiet_ok = True

    def utility(self, m):
        return 3.0 if m.people.feared_here(m.world.clock()) else 0.0

    def run(self, m):
        yield ("face", "scared")
        if not m.affect.quiet:
            yield ("sound", "whimper")
        yield ("goto", m.memory.home_of("pinch") or m.world.desk.home_xy, 0.04)
        yield ("pose", "guard")                              # crouched, claws up as a shield
        while m.people.feared_here(m.world.clock()):
            yield ("look_at", m.world.desk.owner_xy)          # keeps an eye on them (states.toml: gaze on the threat)
            yield ("wait", 3.0)

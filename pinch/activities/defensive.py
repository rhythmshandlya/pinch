"""Startle and caution: flinching, keeping its distance from someone it fears, hiding."""

from pinch.activities.base import (  # noqa: F401
    Activity,
)


class Startle(Activity):
    state, max_s = "alert", 30.0
    name, interruptible, min_s, quiet_ok = "startle", False, 2.0, True

    def utility(self, m):
        return 2.2 * (m.affect.needs.safety > 0.6)

    def run(self, m):
        yield ("face", "scared")
        if not m.affect.quiet:
            yield ("sound", "startled_squeak")
        yield ("emote", "scared")
        yield ("walk", -0.04, 0.0, 0.0, 1.0)


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
        yield ("look_at", m.world.desk.owner_xy)
        yield ("wait", 4.0)
        yield ("face", "curious")
        if not m.affect.quiet:
            yield ("sound", "curious_chirp")


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
        yield ("pose", "low")
        while m.people.feared_here(m.world.clock()):
            yield ("wait", 3.0)
        yield ("face", "neutral")

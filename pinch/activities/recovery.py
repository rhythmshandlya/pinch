"""Looking after itself: rolling back over, getting unstuck, heading home when the battery is low, saving power."""

from pinch.activities.base import (  # noqa: F401
    Activity,
)


class RightSelf(Activity):
    """On its back (it fell, or someone put it down that way): a startled squeak, then roll back over with the
    righting swing (crab/righting.py); if two tries fail, wave its legs and plead for help."""
    state, max_s = "recover", 60.0
    name, min_s, quiet_ok, interruptible = "right_self", 1.0, True, False

    def utility(self, m):
        return 6.0 if m.flipped else 0.0

    def run(self, m):
        yield ("face", "scared")
        if not m.affect.quiet:
            yield ("sound", "startled_squeak")
        for _ in range(2):
            yield ("phase", "roll")
            yield ("right_self",)
            if m.last_step_ok:
                m.affect.feel("unstuck")                     # relief first: the fear drops, then a pleased look
                yield ("face", "happy")
                if not m.affect.quiet:
                    yield ("sound", "happy_trill")
                m.memory.remember("fell", "fell on my back and got up again")
                return
            yield ("wait", 1.0)
        yield ("face", "sad")
        if not m.affect.quiet:
            yield ("sound", "whimper")
        m.memory.remember("help", "fell on my back and couldn't get up")
        yield ("wait", 15.0)


class Unstick(Activity):
    """Stuck (a foot over the edge, caught on something, or walking without getting anywhere): grumble, try
    twice to work free, then ask for help - with long gaps, so it pleads rather than nags."""
    state, max_s = "recover", 180.0
    name, min_s = "unstick", 1.0
    quiet_ok = True

    def utility(self, m):
        return 4.0 if getattr(m, "stuck", False) else 0.0

    def run(self, m):
        yield ("face", "grumpy")
        if not m.affect.quiet:
            yield ("sound", "angry_buzz")
        for _ in range(2):
            yield ("phase", "work_free")
            yield ("recover",)
            yield ("wait", 2.5)                               # (StuckSense needs a moment to see it cleared)
            if not m.stuck:
                break
        asks = 0
        while m.stuck and asks < 4:
            yield ("phase", "ask_help")
            asks += 1
            yield ("face", "sad")
            if not m.affect.quiet:
                yield ("sound", "uh_oh" if asks == 1 else "whimper")
            yield ("emote", "beg")
            if asks == 1:
                m.memory.remember("help", "got stuck and asked for help")
            yield ("wait", 20.0)
        if not m.stuck:
            yield ("face", "happy")
            if not m.affect.quiet:
                yield ("sound", "happy_trill")


class HeadHome(Activity):
    """Battery getting low: go home and ask to be charged (a rumbly tummy at 40 %, walking home and pleading
    with long gaps below 25 %), then wait there, sleepy. Relief when put on charge (mind._self_preservation)."""
    state, max_s = "recover", 300.0
    name, min_s = "head_home", 5.0
    quiet_ok = True

    def utility(self, m):
        if m.charging:
            return 0.0
        return 3.0 * max(0.0, m.hunger() - 0.3)                 # 0.79 at 25 %, 1.53 at 10 %

    def run(self, m):
        home = m.memory.home_of("pinch") or m.world.desk.home_xy
        yield ("face", "sleepy")
        if not m.affect.quiet:
            yield ("sound", "sleepy_hum")
        yield ("goto", home, 0.03)
        asks = 0
        while not m.charging and m.world.battery > 0.10 and asks < 6:
            asks += 1
            yield ("turn_to", m.world.desk.owner_xy)
            yield ("face", "sad")
            if not m.affect.quiet and m.world.owner_here(120):
                yield ("sound", "attention_chirp")
                yield ("emote", "beg")
            yield ("wait", 90.0)


class LowPower(Activity):
    """Almost empty (10 %): stop walking, save memory, say goodnight, and doze until charged."""
    state, max_s = "rest", 7200.0
    name, min_s = "low_power", 5.0
    quiet_ok = True
    interruptible = False

    def utility(self, m):
        return 5.0 if (m.world.battery <= 0.10 and not m.charging) else 0.0

    def run(self, m):
        m.memory.data["danger"] = m.danger_map.to_list()
        m.memory.save()
        yield ("face", "sleepy")
        if not m.affect.quiet:
            yield ("sound", "yawn")
        yield ("emote", "sleep")
        yield ("face", "asleep")
        while m.world.battery <= 0.12 and not m.charging:
            yield ("wait", 30.0)

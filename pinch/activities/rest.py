"""Being still: idling, resting, napping."""

from pinch.activities.base import (  # noqa: F401
    Activity,
    np,
)


class Idle(Activity):
    state, max_s = "idle", 120.0
    name, min_s, quiet_ok = "idle", 1.0, True

    def utility(self, m):
        return 0.25

    def run(self, m):
        rng = m.rng
        while True:
            yield ("face", m.affect.feeling())
            r = rng.random()
            if m.affect.quiet:
                yield ("emote", "fidget") if r < 0.25 else ("wait", 2.0)
            elif r < 0.15:
                yield ("emote", "look_around")
            elif r < 0.25:
                yield ("emote", rng.choice(["stretch", "yawn", "fidget", "sneeze"],
                                           p=[0.3, 0.2 + 0.3 * m.affect.needs.sleep, 0.4, 0.1] / np.sum(
                                               [0.3, 0.2 + 0.3 * m.affect.needs.sleep, 0.4, 0.1])))
            else:
                yield ("wait", rng.uniform(1.5, 4.0))


class Rest(Activity):
    # a whole night is one rest (decision 94): the watchdog limit covers 22:00-06:00 plus margin
    state, max_s = "rest", 9 * 3600.0
    name, min_s, quiet_ok = "rest", 20.0, True

    def utility(self, m):
        n = m.affect.needs
        asked = m.t < m.sleeping_until
        return 1.5 * asked + 1.2 * n.sleep ** 2 + 1.4 * max(0.0, n.energy - 0.7) + 0.5 * m.situation_strength("owner_away") * (n.sleep > 0.4)

    def run(self, m):
        yield ("face", "sleepy")
        yield ("goto", m.world.desk.home_xy, 0.05)
        yield ("emote", "yawn")
        while True:
            yield ("phase", "asleep")                        # the sleep state (configs/mind/states.toml)
            yield ("face", "asleep")
            yield ("pose", "nap", 10.0)
            night = m.routine.is_night(m.world.clock())
            # by day a nap pays off sleepiness and ends; at night it sleeps on until morning (or something wakes it:
            # touch, voice and the owner preempt Rest through the usual selection), paying it off 4x more slowly
            m.affect.needs.sleep = max(0.0, m.affect.needs.sleep - (0.02 if night else 0.08))
            if m.affect.needs.sleep < 0.05 and m.t > m.sleeping_until and not night:
                break
        yield ("phase", "wake")
        yield ("face", "sleepy")
        yield ("emote", "stretch")

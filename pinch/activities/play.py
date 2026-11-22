"""Play and curiosity: its own games, showing off tricks you liked, exploring the desk."""

from pinch.activities.base import (  # noqa: F401
    Activity,
    math,
    np,
)


class Play(Activity):
    """Bored and playful: pick a game from the PlayBook (pinch/play.py) - push a toy somewhere, carry it,
    pick it up and wave it about - play it, and learn from what happened which games are fun."""
    state, max_s = "play", 300.0
    name, min_s = "play", 6.0

    def utility(self, m):
        if m.affect.quiet or m.world.owner.busy == "meeting":
            return 0.0
        n = m.affect.needs
        asked = 1.0 if m.t - getattr(m, "play_request_t", -1e9) < self.ASKED_S else 0.0   # "pinch, play!"
        return (0.8 * m.affect.personality.playfulness * max(0.0, n.play - 0.35) + asked) * (1 - n.sleep) * (1 - m.hunger())

    ASKED_S = 60.0                                   # an invitation to play stands this long

    def run(self, m):
        m.play_request_t = -1e9                      # (the invitation is taken up)
        now = m.world.clock()
        game, score = m.playbook.choose(m.world, m.danger, m.memory.data["likes"], m.rng, now)
        yield ("face", "excited")
        if not m.affect.quiet:
            yield ("sound", "giggle")
        if game is None:                                      # nothing safe to play with: a happy wiggle
            yield ("emote", m.rng.choice(["dance", "excited", "applause"]))
            m.affect.feel("played_with", 0.4)
            return
        verb, toy, target = game
        yield ("phase", verb)
        ob = m.world.objects[toy]
        start = tuple(ob.xy)
        goal = m.playbook.target_xy(target, m.world) if target != "here" else start
        m._say("play", f"let's play {verb} {toy} -> {target} (score {score:.2f}, expect it to move "
                       f"{m.playbook.predict(f'{verb}:{toy}:{target}') * 100:.0f} cm)")
        yield ("look_at", ob.xy)
        won = False
        if verb == "push_to":
            yield ("push", toy, goal)
            won = m.last_step_ok
        elif verb == "carry_to":
            yield ("pick", toy)
            if m.last_step_ok:
                yield ("drop", goal)
                won = m.last_step_ok
        else:                                                 # show: lift it up proudly, hold it, put it down
            yield ("pick", toy)                               # (no arm-waving emote while gripping: in the twin
            if m.last_step_ok:                                #  that squeezed a pen out and flung it off the desk)
                yield ("face", "proud")
                yield ("wait", 1.5)
                yield ("release",)
                won = True
        ob = m.world.objects.get(toy)
        end = tuple(ob.xy) if ob else start
        moved = math.hypot(end[0] - start[0], end[1] - start[1])
        gid, err, fun = m.playbook.record(verb, toy, target, moved, won, m.world.desk.edge_distance(end), now)
        m.memory.data["play"] = m.playbook.to_dict()
        m.last_trick = ("game:" + gid, m.t)                  # your praise or scolding now teaches this game
        if err > 1.5:                                         # that went differently than it expected!
            yield ("face", "surprised")
            if not m.affect.quiet:
                yield ("sound", "curious_chirp")
            m.affect.feel("novelty", 0.6)
        yield ("face", "happy" if won else "curious")
        m.affect.feel("played_with", 0.4 + 0.4 * fun)


class ShowOff(Activity):
    """Happy and watched: does a trick it has learned you like, hoping for praise."""
    state, max_s = "play", 120.0
    name, min_s = "show_off", 4.0

    def utility(self, m):
        if m.affect.quiet or not m.world.owner_attending(30) or m.t - m.showed_off_t < 90:
            return 0.0
        return 0.5 * max(0.0, m.affect.valence)

    def run(self, m):
        m.showed_off_t = m.t
        likes = m.memory.data["likes"]
        tricks = ["wave", "dance", "beg", "handshake", "applause", "play_dead", "excited"]
        w = np.array([1.0 + 2.0 * max(-0.45, likes.get(t, 0.0)) for t in tricks])
        trick = m.rng.choice(tricks, p=w / w.sum())
        yield ("turn_to", m.world.desk.owner_xy)
        yield ("face", "excited")
        yield ("emote", trick)
        m.last_trick = (trick, m.t)
        yield ("praise_wait", 6.0)


class Explore(Activity):
    state, max_s = "explore", 300.0
    name, min_s = "explore", 5.0

    def utility(self, m):
        if m.affect.quiet:
            return 0.0
        return 0.65 * max(0.0, m.affect.needs.curiosity - 0.35) * (1 - m.affect.needs.sleep) * (1 - m.hunger())

    def run(self, m):
        d = m.world.desk
        for _ in range(8):                                    # somewhere it isn't afraid of
            target = d.clamp((m.rng.uniform(-d.half_x, d.half_x), m.rng.uniform(-d.half_y, d.half_y)), 0.1)
            if m.danger(target) < 0.15:
                break
        else:                                                 # everywhere it drew is scary: look near home instead
            target = m.memory.home_of("pinch") or d.home_xy
        yield ("face", "curious")
        yield ("phase", "go_look")
        yield ("goto", target, 0.05)
        yield ("emote", "look_around")
        m.affect.feel("novelty", 0.6)

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
        """A bout, not a single hop: one to three spots by how curious it is, a pause at each to take it in (a look
        here and there, now and then a look-around), ending early once the curiosity is spent (animal
        exploration comes in bouts of several stops; the rhythm bench measured one-hop explores at ~8 s)."""
        d = m.world.desk
        stops = 1 + min(2, int(2.5 * max(0.0, m.affect.needs.curiosity - 0.35) / 0.65 + 0.5))
        last = m.world.pose[:2]
        yield ("face", "curious")
        for k in range(stops):
            for _ in range(8):                                # somewhere it isn't afraid of, not where it just was
                target = d.clamp((m.rng.uniform(-d.half_x, d.half_x), m.rng.uniform(-d.half_y, d.half_y)), 0.1)
                if m.danger(target) < 0.15 and math.dist(target, last) > 0.08:
                    break
            else:                                             # everywhere it drew is scary: look near home instead
                target = m.memory.home_of("pinch") or d.home_xy
            yield ("phase", "go_look")
            yield ("goto", target, 0.05)
            if m.last_step_ok:
                last = target
            yield ("phase", "take_in")
            x, y, h = m.world.pose
            for side in ((-1, 1) if m.rng.random() < 0.5 else (1, -1)):
                yield ("look_at", (x + 0.2 * math.cos(h + 0.6 * side), y + 0.2 * math.sin(h + 0.6 * side)))
                yield ("wait", 0.8 + 1.2 * m.rng.random())
            if k == stops - 1 or m.rng.random() < 0.4:
                yield ("emote", "look_around")
            m.affect.feel("novelty", 0.6 / stops)
            if m.affect.needs.curiosity < 0.3:
                break


class Quirk(Activity):
    """Now and then - once or twice a day, at a moment nobody can predict - something a little unusual: a sneeze,
    a dance all by itself, playing dead and popping back up, clapping for itself, chasing its own claw, a big
    stretch. Never the same one as the last few (long-term HRI: rare surprises and variety keep a companion from
    wearing thin; docs/research/v2_1/longterm/notes.md). Only when it feels safe and not sad (the [wants] gate)."""
    state, max_s = "play", 40.0
    name, min_s = "quirk", 2.0
    exact_emotes = True
    GAP_H = (8.0, 30.0)                      # hours between quirks (clock time)
    QUIRKS = ("sneeze", "solo_dance", "play_dead_pop", "applause_self", "claw_chase", "big_stretch")

    @staticmethod
    def _rng(m):
        r = m.__dict__.get("_quirk_rng")
        if r is None:
            r = m.__dict__["_quirk_rng"] = np.random.default_rng(m.memory.data.get("stats", {}).get("boots", 0) + 0x9C1)
        return r

    def utility(self, m):
        now = m.world.clock()
        d = m.memory.data
        if "quirk_due" not in d:                             # (drawn once, not on every look)
            d["quirk_due"] = now + 3600.0 * self._rng(m).uniform(1.0, self.GAP_H[0])
            m.memory.changed()
        due = d["quirk_due"]
        if now < due or m.affect.quiet or m.state == "sleep":
            return 0.0
        return 0.6

    def run(self, m):
        now = m.world.clock()
        rng = self._rng(m)
        m.memory.data["quirk_due"] = now + 3600.0 * rng.uniform(*self.GAP_H)
        recent = m.memory.data.setdefault("quirks", [])[-3:]
        fresh = [q for q in self.QUIRKS if q not in recent and (q != "solo_dance" or m.affect.valence > 0.2)]
        q = fresh[int(rng.integers(len(fresh)))] if fresh else self.QUIRKS[0]
        m.memory.data["quirks"] = (m.memory.data["quirks"] + [q])[-10:]
        m.memory.changed()
        yield ("phase", q)
        if q == "sneeze":
            yield ("face", "surprised")
            yield ("emote", "sneeze")
            yield ("face", "happy")
        elif q == "solo_dance":
            yield ("face", "joy")
            yield ("sound", "happy_trill")
            yield ("emote", "dance")
        elif q == "play_dead_pop":
            yield ("face", "dead")
            yield ("emote", "play_dead")
            yield ("wait", 1.5)
            yield ("face", "surprised")
            yield ("emote", "excited")
        elif q == "applause_self":
            yield ("face", "proud")
            yield ("emote", "applause")
        elif q == "claw_chase":
            x, y, h = m.world.pose
            side = 1 if rng.random() < 0.5 else -1
            yield ("look_at", (x + 0.06 * np.cos(h + 1.2 * side), y + 0.06 * np.sin(h + 1.2 * side)))
            yield ("face", "curious")
            yield ("turn_to", (x + 0.2 * np.cos(h + 1.2 * side), y + 0.2 * np.sin(h + 1.2 * side)))
            yield ("emote", "look_around")
        else:
            yield ("emote", "stretch")
            yield ("emote", "yawn")

"""Pinch's activities: everything it can decide to do, and how much it wants to right now.

Each activity has
  utility(m)  how much Pinch wants to do it now (m = the Mind: needs, feelings, world, memory, requests)
  run(m)      a generator of steps the body carries out one after another:
                ("face", expr) ("sound", name) ("emote", name) ("walk", vx, vy, wz, s) ("goto", xy, tol)
                ("turn_to", xy) ("pose", name, s) ("push", obj, goal_xy) ("pick", obj) ("drop", goal_xy)
                ("wait", s) ("quiet", on) ("say_mood",) ("remember", kind, text) ("praise_wait", s)
The Mind (pinch/mind.py) re-scores every activity twice a second and switches when something else
matters clearly more, or at once for urgent things (a scare, a command, a meeting starting).
"""

import numpy as np

QUIET_OK = {"go_quiet", "rest", "react_touch", "comfort", "idle", "startle", "tidy", "guard_edge"}
GRASP_ENABLED = True       # the redesigned claw picks up sticks and blocks (decision 24); pushing is the fallback


class Activity:
    name = "activity"
    interruptible = True
    min_s = 2.0
    quiet_ok = False

    def utility(self, m) -> float:
        return 0.0

    def run(self, m):
        yield ("wait", 1.0)


def _owner_near(m):
    return m.world.desk.owner_xy[0], m.world.desk.owner_xy[1] + 0.32       # a spot on the desk near you


class Idle(Activity):
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


class Obey(Activity):
    """Do what my person just asked (tricks, come, sleep, ...)."""
    name, interruptible, min_s = "obey", False, 0.5

    def utility(self, m):
        return 1.6 if m.request else 0.0

    def run(self, m):
        req = m.take_request()
        if not req:
            return
        kind, arg = req
        if kind == "trick":
            yield ("face", "excited" if arg != "play_dead" else "neutral")
            if arg == "spin":
                yield ("sound", "happy_trill")
                yield ("walk", 0.0, 0.0, 0.9, 6.5)
            else:
                yield ("emote", arg)
            m.last_trick = (arg, m.t)
            yield ("face", "happy")
            yield ("praise_wait", 6.0)
        elif kind == "come":
            yield ("face", "happy")
            yield ("sound", "curious_chirp")
            yield ("goto", _owner_near(m), 0.04)
            yield ("turn_to", m.world.desk.owner_xy)
            yield ("emote", "wave")
        elif kind == "go_away":
            yield ("face", "sad")
            yield ("sound", "whimper")
            yield ("goto", m.world.desk.home_xy, 0.05)
            m.affect.feel("scolded", 0.3)
        elif kind == "stop":
            yield ("face", "surprised")
            yield ("pose", "stand", 3.0)
        elif kind == "sleep":
            yield ("face", "sleepy")
            yield ("sound", "yawn")
            yield ("goto", m.world.desk.home_xy, 0.05)
            yield ("emote", "yawn")
            m.sleeping_until = m.t + 600
        elif kind == "wake":
            m.sleeping_until = -1
            yield ("face", "surprised")
            yield ("emote", "stretch")
            yield ("face", "happy")
            yield ("sound", "greeting_beep")
        elif kind == "play":
            m.affect.feel("played_with", 0.5)
            yield ("face", "excited")
            yield ("sound", "giggle")
            yield ("emote", m.rng.choice(["excited", "dance", "beg"]))
        elif kind in ("tidy", "put_away", "fetch"):
            m.tidy_request = (kind, arg, m.t)
        elif kind == "quiet":
            m.quiet_until = m.t + 900
            yield ("quiet", True)
            yield ("emote", "shh")
        elif kind == "how_are_you":
            yield ("say_mood",)
        elif kind == "greet":
            yield ("face", "excited")
            yield ("sound", "greeting_beep")
            yield ("turn_to", m.world.desk.owner_xy)
            yield ("emote", "wave")
        elif kind == "goodbye":
            yield ("face", "sad")
            yield ("sound", "sad_tone")
            yield ("emote", "wave")
        elif kind == "teach_home":
            yield ("face", "curious")
            yield ("sound", "curious_chirp")
            yield ("emote", "point")


class GoQuiet(Activity):
    """My person is in a meeting: shh, then keep busy silently nearby."""
    name, min_s, quiet_ok = "go_quiet", 10.0, True

    def utility(self, m):
        return 2.0 if m.situation("owner_in_meeting") or m.t < m.quiet_until else 0.0

    def run(self, m):
        yield ("quiet", True)
        yield ("face", "quiet")
        yield ("turn_to", m.world.desk.owner_xy)
        yield ("emote", "shh")
        m.memory.mark_routine("meeting")
        m.memory.remember("meeting", "my person had a meeting, so I kept quiet")
        while m.situation("owner_in_meeting") or m.t < m.quiet_until:
            r = m.rng.random()
            yield ("face", "quiet" if r < 0.6 else "focused")
            if r < 0.35:
                yield ("emote", "fidget")                       # silly, silent stuff nearby
            elif r < 0.45:
                yield ("emote", "shh")
            elif r < 0.55:
                yield ("emote", "look_around")
            else:
                yield ("wait", m.rng.uniform(3, 7))
        yield ("quiet", False)


class AfterMeeting(Activity):
    """The call is over: finally! A happy little greeting."""
    name, min_s = "after_meeting", 3.0

    def utility(self, m):
        return 1.3 if m.t - m.meeting_ended_t < 15 else 0.0

    def run(self, m):
        m.meeting_ended_t = -1e9
        yield ("face", "excited")
        yield ("sound", "happy_trill")
        yield ("emote", "excited")
        yield ("goto", _owner_near(m), 0.05)
        yield ("emote", "beg")


class SeekAttention(Activity):
    """A bit needy: escalates from looking at you, to chirping, to coming over, to tricks, to sulking."""
    name, min_s = "seek_attention", 6.0

    def utility(self, m):
        n = m.affect.needs.social
        if not m.world.owner_here() or m.affect.quiet or m.world.owner.busy == "meeting":
            return 0.0
        ignored = m.affect.ignored_s
        return 0.9 * max(0.0, n - 0.45) * m.affect.personality.neediness + 0.25 * (ignored > 180)

    def run(self, m):
        level = min(4, int(m.affect.ignored_s / 120))
        yield ("turn_to", m.world.desk.owner_xy)
        yield ("face", "lonely" if level < 3 else "sad")
        yield ("sound", "attention_chirp")
        if level >= 1:
            yield ("goto", _owner_near(m), 0.06)
            yield ("emote", "wave")
        if level >= 2:
            yield ("emote", m.rng.choice(["beg", "dance", "handshake"]))
            yield ("sound", "attention_chirp")
        if level >= 4:
            yield ("face", "sad")
            yield ("sound", "whimper")
            yield ("emote", "sad")
        yield ("wait", 4.0)


class Comfort(Activity):
    """My person sounds sad / stressed / tired: come close, be soft and gentle, offer a claw."""
    name, min_s, quiet_ok = "comfort", 8.0, True

    def utility(self, m):
        s = m.situation("owner_down")
        return 1.25 * s.strength if s and m.t - m.comforted_t > 300 else 0.0

    def run(self, m):
        m.comforted_t = m.t
        mood = m.world.owner.mood
        yield ("face", "sad" if mood == "sad" else "content")
        yield ("goto", _owner_near(m), 0.05)
        yield ("turn_to", m.world.desk.owner_xy)
        yield ("sound", "purr")
        yield ("emote", "handshake")                           # offers a claw to hold
        yield ("face", "love")
        if mood == "tired":
            yield ("emote", "yawn")
        elif mood == "stressed":
            yield ("emote", "beg")                             # "take a break?"
        yield ("pose", "low", 6.0)
        m.memory.remember("comfort", f"my person was {mood}, so I stayed close")


class Celebrate(Activity):
    name, min_s = "celebrate", 3.0

    def utility(self, m):
        s = m.situation("owner_happy")
        return 0.95 * s.strength if s and m.t - m.celebrated_t > 120 else 0.0

    def run(self, m):
        m.celebrated_t = m.t
        yield ("face", "joy")
        yield ("sound", "fanfare")
        yield ("emote", m.rng.choice(["dance", "applause", "excited"]))


class ReactTouch(Activity):
    name, interruptible, min_s, quiet_ok = "react_touch", False, 1.0, True

    def utility(self, m):
        return 1.7 if m.touch_event and m.t - m.touch_event[1] < 1.0 else 0.0

    def run(self, m):
        gesture, _ = m.touch_event
        m.touch_event = None
        quiet = m.affect.quiet
        if gesture in ("stroke", "hold", "claw_hold"):
            yield ("face", "love")
            if not quiet:
                yield ("sound", "purr")
            yield ("emote", "handshake" if gesture == "claw_hold" else "fidget")
        elif gesture == "tickle":
            yield ("face", "squeeze")
            if not quiet:
                yield ("sound", "giggle")
            yield ("emote", "excited")
        elif gesture == "boop":
            yield ("face", "surprised")
            if not quiet:
                yield ("sound", "boop")
            yield ("face", "happy")
        elif gesture in ("tap", "double_tap"):
            yield ("face", "curious")
            yield ("emote", "look_around" if gesture == "tap" else "point")
        elif gesture == "ruffle":
            yield ("face", "grumpy")
            yield ("emote", "shy")
        elif gesture in ("lifted", "upside_down"):
            yield ("face", "scared")
            if not quiet:
                yield ("sound", "startled_squeak")
            yield ("pose", "stand", 1.5)
        elif gesture == "put_down":
            yield ("face", "dizzy")
            yield ("emote", "look_around")
        elif gesture == "shaken":
            yield ("face", "dizzy")
            if not quiet:
                yield ("sound", "uh_oh")
            yield ("pose", "stand", 2.0)
        elif gesture == "held_leg":
            yield ("face", "curious")
            yield ("emote", "handshake")


class Startle(Activity):
    name, interruptible, min_s, quiet_ok = "startle", False, 2.0, True

    def utility(self, m):
        return 2.2 * (m.affect.needs.safety > 0.6)

    def run(self, m):
        yield ("face", "scared")
        if not m.affect.quiet:
            yield ("sound", "startled_squeak")
        yield ("emote", "scared")
        yield ("walk", -0.04, 0.0, 0.0, 1.0)


class Tidy(Activity):
    """Put things back where they live (pens to the tray, ...). Spontaneous or asked."""
    name, min_s, quiet_ok = "tidy", 5.0, True

    def utility(self, m):
        req = m.tidy_request
        if req and m.t - req[2] < 120:
            return 1.5
        todo = [s for s in m.situations if s.kind == "object_out_of_place" and m.tidy_ok(s.target)]
        if not todo:
            return 0.0
        return 0.75 * m.affect.personality.tidiness * (0.6 + 0.6 * m.affect.needs.tidy)

    def run(self, m):
        kind, obj, _ = m.tidy_request or ("tidy", "", 0)
        m.tidy_request = None
        todo = [obj] if obj else [s.target for s in m.situations
                                  if s.kind == "object_out_of_place" and m.tidy_ok(s.target)][:1]
        if not todo:
            yield ("face", "curious")
            yield ("emote", "look_around")
            return
        for name in todo:
            ob = m.world.objects.get(name)
            if ob is None:
                yield ("face", "curious")
                yield ("emote", "look_around")
                continue
            goal = m.world.desk.owner_xy if kind == "fetch" else (m.memory.home_of(name) or m.world.desk.tray_xy)
            if kind == "fetch":
                goal = (goal[0], m.world.desk.owner_xy[1] + 0.29)
            yield ("face", "focused")
            yield ("look_at", ob.xy)
            picked = False
            if ob.graspable and GRASP_ENABLED:
                yield ("pick", name)                    # the claw first (decision 24) ...
                picked = m.last_step_ok
                if picked:
                    yield ("drop", goal)
            if not picked:
                yield ("push", name, goal)              # ... and pushing when it can't get hold of it
            ok = m.last_step_ok
            if not ok and len([1 for t, k in m.tidy_tries.get(name, []) if not k and m.t - t < 600]) >= 1:
                # second failure: ask my person for help instead of thrashing about
                yield ("face", "sad")
                yield ("turn_to", ob.xy)
                yield ("emote", "point")
                if not m.affect.quiet:
                    yield ("sound", "uh_oh")
                yield ("turn_to", m.world.desk.owner_xy)
                yield ("emote", "beg")
                m.memory.remember("help", f"asked for help with the {name}")
            m.tidy_attempt(name, ok)
            yield ("face", "proud" if ok else "sad")
            if not m.affect.quiet:
                yield ("sound", "fanfare" if ok else "uh_oh")
            if ok:
                m.affect.feel("tidied")
                m.memory.count("tidies")
                m.memory.remember("tidy", f"put the {name} back")
        if kind == "fetch":
            yield ("turn_to", m.world.desk.owner_xy)
            yield ("emote", "beg")
            yield ("praise_wait", 6.0)


class GuardEdge(Activity):
    """Something is about to fall off the desk. Pinch can't get behind it without leaving the desk itself,
    so it raises the alarm: hurries over, points at it and looks at you, worried."""
    name, min_s, quiet_ok = "guard_edge", 4.0, True

    def utility(self, m):
        s = m.situation("object_near_edge")
        return 1.35 if s and m.t - m.edge_alert_t.get(s.target, -1e9) > 120 else 0.0

    def run(self, m):
        s = m.situation("object_near_edge")
        if not s:
            return
        m.edge_alert_t[s.target] = m.t
        ob = m.world.objects[s.target]
        near = m.world.desk.clamp(np.array(ob.xy) * 0.6)
        yield ("face", "surprised")
        if not m.affect.quiet:
            yield ("sound", "uh_oh")
        yield ("goto", near, 0.05)
        yield ("turn_to", ob.xy)
        yield ("face", "scared")
        yield ("emote", "point")
        yield ("turn_to", m.world.desk.owner_xy)
        if not m.affect.quiet:
            yield ("sound", "attention_chirp")
        yield ("emote", "beg")
        m.memory.remember("edge", f"the {s.target} was about to fall off the desk")


class Play(Activity):
    name, min_s = "play", 6.0

    def utility(self, m):
        if m.affect.quiet or m.world.owner.busy == "meeting":
            return 0.0
        n = m.affect.needs
        return 0.8 * m.affect.personality.playfulness * max(0.0, n.play - 0.35) * (1 - n.sleep)

    def run(self, m):
        ball = m.world.objects.get("ball")
        yield ("face", "excited")
        yield ("sound", "giggle")
        if ball and ball.on_desk and m.world.desk.edge_distance(ball.xy) > 0.08:
            target = m.world.desk.clamp((ball.xy[0] + m.rng.uniform(-0.15, 0.15), ball.xy[1] + m.rng.uniform(-0.1, 0.1)),
                                        0.10)
            yield ("push", "ball", target)                     # dribble the ball around
        else:
            yield ("emote", m.rng.choice(["dance", "excited", "applause"]))
        m.affect.feel("played_with", 0.6)


class ShowOff(Activity):
    """Happy and watched: does a trick it has learned you like, hoping for praise."""
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
    name, min_s = "explore", 5.0

    def utility(self, m):
        if m.affect.quiet:
            return 0.0
        return 0.65 * max(0.0, m.affect.needs.curiosity - 0.35) * (1 - m.affect.needs.sleep)

    def run(self, m):
        d = m.world.desk
        target = d.clamp((m.rng.uniform(-d.half_x, d.half_x), m.rng.uniform(-d.half_y, d.half_y)), 0.1)
        yield ("face", "curious")
        yield ("goto", target, 0.05)
        yield ("emote", "look_around")
        m.affect.feel("novelty", 0.6)


class Rest(Activity):
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
            yield ("face", "asleep")
            yield ("pose", "nap", 10.0)
            m.affect.needs.sleep = max(0.0, m.affect.needs.sleep - 0.08)
            if m.affect.needs.sleep < 0.05 and m.t > m.sleeping_until:
                break
        yield ("face", "sleepy")
        yield ("emote", "stretch")


class Sulk(Activity):
    name, min_s = "sulk", 6.0

    def utility(self, m):
        return 1.1 if m.affect.last_event == "scolded" and m.t - m.affect.last_event_t < 4 else 0.0

    def run(self, m):
        yield ("face", "sad")
        if not m.affect.quiet:
            yield ("sound", "whimper")
        yield ("emote", "sad")
        yield ("face", "shy")
        yield ("wait", 3.0)


class Greet(Activity):
    name, min_s = "greet", 3.0

    def utility(self, m):
        return 1.4 if m.t - m.owner_returned_t < 8 and not m.affect.quiet else 0.0

    def run(self, m):
        m.owner_returned_t = -1e9
        yield ("face", "excited")
        yield ("sound", "greeting_beep")
        yield ("turn_to", m.world.desk.owner_xy)
        yield ("emote", "wave")
        yield ("emote", "excited")


LIBRARY = [Idle(), Obey(), GoQuiet(), AfterMeeting(), SeekAttention(), Comfort(), Celebrate(), ReactTouch(),
           Startle(), Tidy(), GuardEdge(), Play(), ShowOff(), Explore(), Rest(), Sulk(), Greet()]

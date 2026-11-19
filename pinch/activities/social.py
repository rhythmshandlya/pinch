"""With people: doing what it's asked, greeting, comforting, wanting attention, celebrating, sulking, keeping quiet during meetings, and reacting to touch."""

from pinch.activities.base import (  # noqa: F401
    Activity,
    _owner_near,
)


class Obey(Activity):
    """Do what my person just asked (tricks, come, sleep, ...)."""
    state, max_s = "social", 120.0
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


class Greet(Activity):
    state, max_s = "social", 60.0
    name, min_s = "greet", 3.0

    def utility(self, m):
        return (1.4 + 0.8 * m.greet_warmth) if m.t - m.owner_returned_t < 8 and not m.affect.quiet else 0.0

    def run(self, m):
        m.owner_returned_t = -1e9
        warm, m.greet_warmth = m.greet_warmth, 0.0
        yield ("face", "excited" if warm < 0.6 else "love")
        yield ("sound", "greeting_beep")
        yield ("turn_to", m.world.desk.owner_xy)
        yield ("emote", "wave")
        yield ("emote", "excited")
        if warm >= 0.6:                                       # missed them a lot
            yield ("sound", "happy_trill")
            yield ("emote", "dance")


class Comfort(Activity):
    """My person sounds sad / stressed / tired: come close, be soft and gentle, offer a claw."""
    state, max_s = "social", 120.0
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


class SeekAttention(Activity):
    """A bit needy: escalates from looking at you, to chirping, to coming over, to tricks, to sulking."""
    state, max_s = "social", 120.0
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


class Celebrate(Activity):
    state, max_s = "social", 60.0
    name, min_s = "celebrate", 3.0

    def utility(self, m):
        s = m.situation("owner_happy")
        return 0.95 * s.strength if s and m.t - m.celebrated_t > 120 else 0.0

    def run(self, m):
        m.celebrated_t = m.t
        yield ("face", "joy")
        yield ("sound", "fanfare")
        yield ("emote", m.rng.choice(["dance", "applause", "excited"]))


class Sulk(Activity):
    state, max_s = "social", 120.0
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


class GoQuiet(Activity):
    """My person is in a meeting: shh, then keep busy silently nearby."""
    state, max_s = "social", 7200.0
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
    state, max_s = "social", 60.0
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


class ReactTouch(Activity):
    state, max_s = "social", 60.0
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

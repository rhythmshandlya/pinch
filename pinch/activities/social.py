"""With people: doing what it's asked, greeting, comforting, wanting attention, celebrating, sulking, keeping quiet during meetings, and reacting to touch."""

from pinch.activities.base import (  # noqa: F401
    Activity,
    _owner_near,
    math,
)

# touches by a hand (the only ones it can shrink away from; being lifted, shaken or bumped has its own reactions)
HAND_TOUCHES = ("stroke", "hold", "tickle", "boop", "tap", "double_tap", "claw_hold", "ruffle")


class Obey(Activity):
    """Do what my person just asked (tricks, come, sleep, ...)."""
    state, max_s = "social", 120.0
    name, interruptible, min_s = "obey", False, 0.5
    exact_emotes = True                                    # a trick asked for by name plays as asked

    def utility(self, m):
        return 1.6 if m.request else 0.0

    def run(self, m):
        req = m.take_request()
        if not req:
            return
        kind, arg = req
        if kind == "trick":
            yield ("face", "excited" if arg != "play_dead" else "dead")
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
        elif kind == "play":                         # an invitation: a game starts next (Play), if it feels like it
            m.play_request_t = m.t                   # (credited as play only once it has played)
            m.affect.needs.play = max(m.affect.needs.play, 0.8)
            yield ("face", "excited")
            yield ("sound", "giggle")
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

    SIGNATURES = ("wave", "handshake", "excited")          # each person gets their own hello gesture
    exact_emotes = True                                    # (their gesture, as chosen: no stand-ins)

    def run(self, m):
        """A hello that depends on who it is and how long they were away (audit SOC-007; Leite et al. 2013:
        recognising and remembering the person is what keeps a robot companion engaging): a short absence gets a
        glance and a chirp, a longer one the person's own gesture, a long one with someone it loves a run over to
        them and a dance."""
        import zlib
        m.owner_returned_t = -1e9
        warm, m.greet_warmth = m.greet_warmth, 0.0
        who, away_h = getattr(m, "greet_who", None), getattr(m, "greet_away_h", 8.0)
        m.greet_who, m.greet_away_h = None, 8.0              # (used once: the next return brings its own)
        sig = self.SIGNATURES[zlib.crc32(str(who).encode()) % len(self.SIGNATURES)] if who else "wave"
        yield ("face", "excited" if warm < 0.6 else "love")
        yield ("sound", "greeting_beep")
        yield ("turn_to", m.world.desk.owner_xy)
        if away_h < 1.0:                                     # only just gone: a glance and a chirp
            yield ("emote", "curious")
            return
        if warm >= 0.6 and away_h >= 4.0:                    # missed them a lot: go to them first
            yield ("goto", _owner_near(m), 0.06)
        yield ("emote", sig)
        yield ("emote", "excited")
        if warm >= 0.6:
            yield ("sound", "happy_trill")
            yield ("emote", "dance")
        if away_h >= 24.0:                                   # a day or more: something you two share
            yield from self._callback(m)

    @staticmethod
    def _callback(m):
        """After a long absence, the trick you praised most, or a look at the toy of your favourite game: it
        remembers (long-term HRI: referring to shared history keeps a companion engaging, Leite et al. 2013)."""
        from crab.emotes import EMOTES
        likes = [(v, k) for k, v in m.memory.data.get("likes", {}).items() if v > 0.2]
        if not likes:
            return
        _, best = max(likes)
        yield ("phase", "remember")
        if best in EMOTES:
            yield ("emote", best)
        elif best.startswith("game:"):
            toy = best.split(":")[2] if best.count(":") >= 3 else None
            ob = m.world.objects.get(toy) if toy else None
            if ob is not None:
                yield ("look_at", tuple(ob.xy))
                yield ("emote", "point")


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
        reached = m.last_step_ok
        yield ("turn_to", m.world.desk.owner_xy)
        yield ("sound", "purr")
        yield ("emote", "handshake")                           # offers a claw to hold
        yield ("face", "love")
        if mood == "tired":
            yield ("emote", "yawn")
        elif mood == "stressed":
            yield ("emote", "beg")                             # "take a break?"
        yield ("pose", "low", 6.0)
        if reached:                                            # (only what happened: no false memories)
            m.memory.remember("comfort", f"my person was {mood}, so I stayed close")
        else:
            m.memory.remember("comfort", f"my person was {mood}; I couldn't get close, so I comforted them from here")


class SeekAttention(Activity):
    """A bit needy: escalates from looking at you, to chirping, to coming over, to tricks, to sulking."""
    state, max_s = "social", 120.0
    name, min_s = "seek_attention", 6.0

    def utility(self, m):
        n = m.affect.needs.social
        if not m.world.owner_here() or m.affect.quiet or m.world.owner.busy == "meeting":
            return 0.0
        ignored = m.affect.ignored_s
        if ignored < 30:                       # you just paid attention: the next bid starts patient again
            m.seek_tries = 0
        if m.t < getattr(m, "seek_next", 0.0):     # backing off after bids that got no answer (decision 94)
            return 0.0
        return 0.9 * max(0.0, n - 0.45) * m.affect.personality.neediness + 0.25 * (ignored > 180)

    BACKOFF_S, BACKOFF_MAX_S = 90.0, 1200.0     # wait 90 s, 3, 6, 12 ... up to 20 min between unanswered bids

    def run(self, m):
        m.seek_tries = getattr(m, "seek_tries", 0) + 1
        m.seek_next = m.t + min(self.BACKOFF_MAX_S, self.BACKOFF_S * 2 ** (m.seek_tries - 1))
        level = min(4, int(m.affect.ignored_s / 120))
        yield ("turn_to", m.world.desk.owner_xy)
        yield ("face", "lonely" if level < 3 else "sad")
        yield ("sound", "attention_chirp")
        if level >= 1:
            yield ("goto", _owner_near(m), 0.06)
            yield ("emote", "wave")
        if level >= 2:
            yield ("emote", m.rng.choice(["beg", "handshake"]))       # (no dance under a lonely face)
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
    """Reacts to being touched, by where (pinch/touch.py touch_place: left / right, claw, side, back, head) and by
    how much it has had lately (TouchHabituation: repeated petting wears off and rests back after a pause; a rough
    touch sharpens everything again). Asleep, only some touches wake it (WAKE_RULES, the owner's setting)."""
    state, max_s = "social", 60.0
    name, interruptible, min_s, quiet_ok = "react_touch", False, 1.0, True

    @staticmethod
    def touch_state(m):
        """Per mind (not per activity object: those are shared by every mind in the process)."""
        st = m.__dict__.get("touch_state")
        if st is None:
            from pinch.touch import TouchHabituation, WakeRules
            st = m.__dict__["touch_state"] = {"habit": TouchHabituation(), "wake": WakeRules(), "woke": None}
        return st

    @staticmethod
    def asleep(m):
        return m.activity is not None and m.activity.name == "rest" and m.face_expr == "asleep"

    def utility(self, m):
        ev = m.touch_event
        if not ev or m.t - ev[1] >= 1.0:
            return 0.0
        if self.asleep(m):
            from pinch.touch import WAKE_DEFAULT
            st = self.touch_state(m)
            mode = m.memory.data.get("settings", {}).get("wake_on_touch", WAKE_DEFAULT)
            if not st["wake"].wakes(ev, mode):
                return 0.0                                  # (felt in its sleep: the mood still took it in)
            st["woke"] = ev
        return 1.7

    def run(self, m):
        from pinch.touch import touch_place
        ev = m.touch_event
        gesture, _ = ev
        m.touch_event = None
        st = self.touch_state(m)
        woken, st["woke"] = st["woke"] == ev, None
        r = st["habit"].stimulate(gesture, m.t)             # 1 = full reaction .. 0.15 = had plenty of that
        side, region, direction = touch_place(gesture)
        g, quiet, a = str(gesture), m.affect.quiet, m.affect
        style = dict(gain=r, arousal=a.arousal, valence=a.valence, sleepy=1.0 if woken else a.needs.sleep)
        if side:
            style["side"] = side

        def emote(name, gaze=None):
            return ("emote", name, dict(style, gaze=gaze) if gaze is not None else dict(style))

        if woken:
            yield ("face", "sleepy")
        feel = a.feeling()
        unwanted = (getattr(m, "touch_from", "known") == "feared"
                    or (getattr(m, "touch_from", "known") == "stranger" and a.personality.timidity > 0.5
                        and g not in ("stroke", "hold"))
                    or (g in ("tickle", "boop", "tap", "double_tap") and feel in ("grumpy", "angry", "scared")))
        if unwanted and g in HAND_TOUCHES:
            # not wanted now (someone it fears, a stranger poking a timid crab, a poke while grumpy or scared):
            # it shrinks back from the hand instead of performing a happy reaction (configs/mind/states.toml social)
            yield ("face", "shy" if feel != "angry" else "grumpy")
            back = -0.03 * (side if side else 0.0)
            yield ("walk", -0.025, back, 0.0, 0.8)
            yield ("pose", "low", 1.5)
            return
        if direction is not None and abs(direction) > 0.3:          # eyes to the touched side
            x, y, h = m.world.pose
            yield ("look_at", (x + 0.25 * math.cos(h + direction), y + 0.25 * math.sin(h + direction)))
        if g in ("stroke", "hold"):
            yield ("face", "love" if r > 0.5 else "content")
            if not quiet:
                yield ("sound", "purr")
            if region == "back" or (g == "stroke" and region == ""):
                # a back stroke: a contented settle - sinks low, longer as it relaxes into the petting
                if r > 0.7:
                    yield emote("fidget")
                yield ("pose", "low", 1.0 + 3.0 * (1.0 - r))
            elif region == "side":
                yield emote("curious", gaze=0.12 * side)             # leans and turns toward the hand
            elif r > 0.35:
                yield emote("fidget")
        elif g == "claw_hold":
            yield ("face", "love")
            if not quiet:
                yield ("sound", "purr")
            if r > 0.3:
                yield emote("handshake")                            # shakes with the claw you hold
        elif g == "tickle":
            yield ("face", "squeeze")
            if not quiet:
                yield ("sound", "giggle")
            yield emote("excited" if r > 0.3 else "fidget")
        elif g == "boop":
            yield ("face", "surprised")
            if not quiet:
                yield ("sound", "boop")
            if r < 0.4:
                yield emote("sneeze")                               # boop, boop, boop: achoo
            yield ("face", "happy" if r >= 0.4 else "grumpy")
        elif g in ("tap", "double_tap"):
            yield ("face", "curious" if r > 0.35 else "grumpy")
            if region == "claw":                                    # the touched claw answers
                yield emote("wave" if g == "tap" else "handshake")
            elif side and r > 0.35:
                yield emote("curious", gaze=0.15 * side)            # turns toward the touch
            elif r > 0.35:
                yield emote("look_around" if g == "tap" else "point")
        elif g == "ruffle":
            yield ("face", "grumpy")
            yield emote("shy")
        elif g in ("lifted", "upside_down"):
            yield ("face", "scared")
            if not quiet:
                yield ("sound", "startled_squeak")
            yield ("pose", "stand", 1.5)
        elif g == "put_down":
            yield ("face", "dizzy")
            yield ("emote", "look_around")
        elif g == "shaken":
            yield ("face", "dizzy")
            if not quiet:
                yield ("sound", "uh_oh")
            yield ("pose", "stand", 2.0)
        elif g == "held_leg":
            yield ("face", "curious")
            yield ("emote", "handshake")
        elif g == "bumped":                                         # knocked: a flinch and a step away
            yield ("face", "surprised")
            if not quiet:
                yield ("sound", "startled_squeak")
            yield ("walk", -0.025, 0.0, 0.0, 0.6)
            yield ("pose", "low", 1.0)

"""Pinch's mind: senses in, one activity at a time out, with feelings, needs and memory in between.

    mind = Mind(memory=Memory())
    mind.hear_text("Pinch, put the pen back")       # speech (typed in the app, or speech-to-text)
    mind.hear_summary(voice_summary)                # once a second from pinch/hearing.py
    mind.touch("stroke")                            # from pinch/touch.py
    mind.world.see("pen", (0.1, 0.0))              # from vision
    motion, cancel = mind.tick(dt, body_busy, body_ok)
    # motion: the next body step to run ("emote", "wave") / ("goto", xy, tol) / ... or None
    # mind.face_expr, mind.gaze, mind.sounds (queue) drive the face screen and the speaker

Same code on the laptop (virtual pet, simulator) and on the robot.
"""

import numpy as np

from pinch.activities import LIBRARY
from pinch.affect import Affect, Personality
from pinch.hearing import MeetingDetector, OwnerVoiceMood
from pinch.language import LLMUnderstanding, understand_rules
from pinch.memory import Memory
from pinch.touch import GESTURE_EVENT
from pinch.world import WorldModel, situations

MOTION_STEPS = {"emote", "walk", "goto", "turn_to", "pose", "push", "pick", "drop"}
MOOD_REPLY = {   # "how are you?" -> face, sound, emote
    "happy": ("happy", "happy_trill", "happy"), "excited": ("excited", "giggle", "excited"),
    "content": ("content", "purr", "fidget"), "love": ("love", "purr", "handshake"),
    "lonely": ("lonely", "whimper", "beg"), "sad": ("sad", "sad_tone", "sad"), "bored": ("bored", "sleepy_hum", "yawn"),
    "sleepy": ("sleepy", "yawn", "yawn"), "scared": ("scared", "startled_squeak", "scared"),
    "curious": ("curious", "curious_chirp", "curious"), "grumpy": ("grumpy", "angry_buzz", "angry"),
    "angry": ("angry", "angry_buzz", "angry"), "neutral": ("happy", "greeting_beep", "wave"),
}


class Mind:
    def __init__(self, memory=None, personality=None, llm=True, seed=0, clock=None):
        self.memory = memory if memory is not None else Memory(path=None)
        self.affect = Affect(personality=personality or Personality())
        self.world = WorldModel()
        self.rng = np.random.default_rng(seed)
        self.llm = LLMUnderstanding() if llm else None
        if self.llm is not None and not self.llm.check():
            self.llm = None
        self.meeting = MeetingDetector()
        self.voice_mood = OwnerVoiceMood()
        self.t = 0.0
        # requests and timers the activities read
        self.request = None
        self.tidy_request = None
        self.touch_event = None
        self.quiet_until = -1e9
        self.sleeping_until = -1e9
        self.comforted_t = self.celebrated_t = self.showed_off_t = -1e9
        self.owner_returned_t = self.meeting_ended_t = -1e9
        self.last_trick = None
        self.last_step_ok = True
        self.situations = []
        self.pending_call_cue = False
        self.words_mood_t = -1e9
        self.tidy_tries = {}
        self.edge_alert_t = {}
        # outputs
        self.face_expr, self.gaze, self.sounds = "neutral", (0.0, 0.0), []
        self.activity, self.gen, self.started = None, None, 0.0
        self.waiting_motion, self.wait_until, self.praise_until = False, -1.0, -1.0
        self.next_choice = 0.0
        self.log = []                        # (t, kind, text) for the app's diary and tests
        self._was_present = False

    # ---------------------------------------------------------------- senses
    def hear_text(self, text):
        u = understand_rules(text)
        self._apply(u)
        if self.llm is not None and u["intent"] in ("chat", "none", "owner_feeling", "tell_name", "teach_home"):
            self.llm.submit(text, self.memory.summary())
        return u

    def _apply(self, u, from_llm=False):
        o, t = self.world.owner, self.t
        o.last_voice = t
        o.present = True
        in_meeting = o.busy == "meeting"
        if u["call_cue"]:
            self.pending_call_cue = True
        if u["meeting"]:
            self.meeting.told(u["meeting"] == "start")
            self._meeting_changed(u["meeting"] == "start")
        if in_meeting and not u["addressed"] and u["intent"] not in ("meeting_end", "quiet"):
            return                                   # talking to colleagues, not to me
        if u["intent"] not in ("none",):
            o.last_attention = t
        if u["owner_emotion"]:
            o.mood, o.mood_t = u["owner_emotion"], t
            self.words_mood_t = t
            ev = {"sad": "owner_sad", "tired": "owner_sad", "stressed": "owner_stressed", "angry": "owner_stressed",
                  "happy": "owner_happy", "excited": "owner_happy"}.get(u["owner_emotion"])
            if ev:
                self.affect.feel(ev)
            if u["owner_emotion"] in ("sad", "stressed", "tired"):
                self.memory.remember("owner_mood", f"my person said they felt {u['owner_emotion']}")
        if u["name"] and (u["intent"] == "tell_name" or from_llm):
            self.memory.set_owner_name(u["name"])
            self.affect.feel("greeted", 0.6)
            self._say("tell_name", f"learned my person's name: {u['name']}")
        if u["fact"] and from_llm:
            self.memory.add_fact(u["fact"])
        i = u["intent"]
        if i == "praise":
            self.affect.feel("praised")
            self.memory.count("praise")
            if self.last_trick and t - self.last_trick[1] < 12:
                self.memory.like(self.last_trick[0], +0.25)
            self.praise_until = -1
        elif i == "scold":
            self.affect.feel("scolded")
            self.memory.count("scolds")
            if self.last_trick and t - self.last_trick[1] < 12:
                self.memory.like(self.last_trick[0], -0.3)
            self._interrupt()
        elif i == "greet":
            self.affect.feel("greeted")
            self.request = ("greet", "")
        elif i == "trick" and u["trick"]:
            self.request = ("trick", u["trick"])
        elif i in ("come", "go_away", "stop", "sleep", "wake", "play", "quiet", "how_are_you", "goodbye"):
            self.request = (i, "")
            if i == "goodbye":
                o.present = False
        elif i in ("tidy", "put_away", "fetch"):
            self.request = (i, u["object"])
        elif i == "teach_home" and u["object"]:
            ob = self.world.objects.get(u["object"])
            where = ob.xy if ob else self.world.pose[:2]
            self.memory.teach_home(u["object"], where)
            self.request = ("teach_home", u["object"])
        elif i in ("chat", "owner_feeling", "tell_name"):
            self.affect.feel("spoken_to" if i == "chat" else "name_called", 0.7)
        if u["addressed"]:
            self.affect.feel("name_called", 0.5)
        if self.request:
            self._interrupt()
        self._say("heard", f"\"{u['text']}\" -> {i}" + (f" {u['trick'] or u['object']}" if u['trick'] or u['object'] else ""))

    def hear_summary(self, s, voice_dir=None):
        """Once a second from pinch/hearing.Hearing."""
        o = self.world.owner
        if s.speech > 0.2:
            o.last_voice = self.t
            o.present = True
            if voice_dir is not None:
                o.voice_dir = voice_dir
        was = self.meeting.in_meeting
        now = self.meeting.update(s, call_cue=self.pending_call_cue, usual_time=self.memory.meeting_likely())
        self.pending_call_cue = False
        if now != was:
            self._meeting_changed(now)
        label = self.voice_mood.update(s)
        if s.speech > 0.25 and self.t - self.words_mood_t > 120 and o.busy != "meeting":
            o.mood, o.mood_t = label, self.t

    def hear_event(self, kind):
        if kind == "loud_noise":
            self.affect.feel("loud_noise")
        elif kind == "double_clap":
            self.affect.feel("name_called")
            self.world.owner.last_attention = self.t
            self.request = ("come", "")

    def touch(self, gesture):
        ev = GESTURE_EVENT.get(gesture)
        if gesture == "put_down":
            self.touch_event = (gesture, self.t)
            return
        if ev:
            self.affect.feel(ev)
            self.memory.count("pets") if ev in ("petted", "stroked") else None
        o = self.world.owner
        o.last_touch = o.last_attention = self.t
        o.present = True
        self.touch_event = (gesture, self.t)
        self._interrupt()
        self._say("touch", gesture)

    def _meeting_changed(self, on):
        o = self.world.owner
        o.busy = "meeting" if on else ""
        if on:
            self._say("meeting", "my person is in a meeting - going quiet")
            self._interrupt()
        else:
            self.meeting_ended_t = self.t
            self.quiet_until = -1e9
            self._say("meeting", "the meeting is over")

    def _say(self, kind, text):
        self.log.append((round(self.t, 1), kind, text))
        self.log = self.log[-300:]

    def _interrupt(self):
        self.next_choice = 0.0
        self._force = True

    _force = False

    # ---------------------------------------------------------------- helpers for activities
    def situation(self, kind):
        return next((s for s in self.situations if s.kind == kind), None)

    def situation_strength(self, kind):
        s = self.situation(kind)
        return s.strength if s else 0.0

    def tidy_ok(self, name):
        """Not right after trying it, and not after failing three times recently (can't reach it?)."""
        tries = [(t, ok) for t, ok in self.tidy_tries.get(name, []) if self.t - t < 600]
        if tries and self.t - tries[-1][0] < 30:
            return False
        return sum(1 for _, ok in tries if not ok) < 3

    def tidy_attempt(self, name, ok):
        self.tidy_tries.setdefault(name, []).append((self.t, ok))
        if not ok and sum(1 for t, k in self.tidy_tries[name] if not k and self.t - t < 600) >= 3:
            self.memory.remember("tidy", f"I couldn't put the {name} back, I'll leave it for now")

    def take_request(self):
        r, self.request = self.request, None
        return r

    # ---------------------------------------------------------------- the loop
    def tick(self, dt, body_busy=False, body_ok=True):
        """Advance the mind by dt. Returns (motion_step or None, cancel_current_motion)."""
        self.t += dt
        self.world.t = self.t
        o = self.world.owner
        if self.llm is not None:
            r = self.llm.poll()
            if r and r["source"] == "llm":
                self._apply_llm(r)
        present = self.world.owner_here(120)
        if present and not self._was_present:
            self.owner_returned_t = self.t
            self.affect.feel("owner_returned")
            self.memory.data["stats"]["last_seen"] = self.world.clock()
            self._say("owner", "my person is back!")
        elif not present and self._was_present:
            self.affect.feel("owner_left")
            self._say("owner", "my person left")
        self._was_present = present
        self.affect.quiet = o.busy == "meeting" or self.t < self.quiet_until
        self.affect.tick(dt, owner_present=present, owner_attending=self.world.owner_attending(60),
                         battery=self.world.battery)
        if self.t >= self.next_choice or self._force:
            self.situations = situations(self.world, self.affect, self.memory)
            self.affect.needs.tidy = min(1.0, self.affect.needs.tidy + 0.02 * len(
                [s for s in self.situations if s.kind == "object_out_of_place"]))
            self.next_choice = self.t + 0.5
            cancel = self._choose()
            self._force = False
            if cancel:
                return None, True
        return self._advance(body_busy, body_ok), False

    def _choose(self):
        scores = {}
        for a in LIBRARY:
            u = a.utility(self)
            if self.affect.quiet and not a.quiet_ok:
                u = 0.0
            scores[a.name] = u + 0.02 * self.rng.random()
        self.scores = scores
        best = max(scores, key=scores.get)
        cur = self.activity
        if cur is None or self.gen is None:
            self._start(best)
            return True
        if best == cur.name:
            return False
        elapsed = self.t - self.started
        urgent = scores[best] >= 1.5 and scores[best] > scores.get(cur.name, 0) + 0.1
        better = scores[best] > scores.get(cur.name, 0) + 0.15 and elapsed > cur.min_s
        if (cur.interruptible and (urgent or better)) or (not cur.interruptible and urgent and best in ("startle",)):
            self._start(best)
            return True
        return False

    def _start(self, name):
        a = next(x for x in LIBRARY if x.name == name)
        self.activity, self.gen, self.started = a, a.run(self), self.t
        self.waiting_motion, self.wait_until = False, -1.0
        self._say("activity", name)

    def _advance(self, body_busy, body_ok):
        if self.gen is None:
            return None
        if self.waiting_motion:
            if body_busy:
                return None
            self.waiting_motion = False
            self.last_step_ok = body_ok
        if self.t < self.wait_until or self.t < self.praise_until:
            return None
        for _ in range(20):                          # run instant steps until a motion step (or the end)
            try:
                step = next(self.gen)
            except StopIteration:
                self.gen = None
                self.next_choice = 0.0
                return None
            kind = step[0]
            if kind == "face":
                self.face_expr = step[1]
            elif kind == "sound":
                if not self.affect.quiet:
                    self.sounds.append(step[1])
            elif kind == "look_at":
                self.gaze = self._gaze_to(step[1])
            elif kind == "wait":
                self.wait_until = self.t + step[1]
                return None
            elif kind == "praise_wait":
                self.praise_until = self.t + step[1]
                return None
            elif kind == "quiet":
                self.affect.quiet = step[1]
            elif kind == "remember":
                self.memory.remember(step[1], step[2])
            elif kind == "say_mood":
                f = self.affect.feeling()
                face, sound, emote = MOOD_REPLY.get(f, MOOD_REPLY["neutral"])
                self.face_expr = face
                if not self.affect.quiet:
                    self.sounds.append(sound)
                self.waiting_motion = True
                return ("emote", emote)
            elif kind in MOTION_STEPS:
                if kind in ("goto", "push", "pick", "drop"):
                    self.gaze = (0.0, 0.3)
                self.waiting_motion = True
                return step
        return None

    def _apply_llm(self, u):
        """The slow, smarter reading of the last sentence: fill in what the rules missed."""
        self._say("llm", f"{u['intent']} name={u['name']} fact={u['fact']} mood={u['owner_emotion']}")
        if u["name"]:
            self.memory.set_owner_name(u["name"])
        if u["fact"]:
            self.memory.add_fact(u["fact"])
        if u["owner_emotion"] and self.t - self.world.owner.mood_t > 5:
            self.world.owner.mood, self.world.owner.mood_t = u["owner_emotion"], self.t
        if u["intent"] in ("trick", "come", "tidy", "put_away", "fetch", "play", "sleep", "quiet") and self.request is None \
                and self.t - self.world.owner.last_voice < 30:
            if u["intent"] == "trick" and u["trick"]:
                self.request = ("trick", u["trick"])
            elif u["intent"] in ("tidy", "put_away", "fetch"):
                self.request = (u["intent"], u["object"])
            else:
                self.request = (u["intent"], "")
            self._interrupt()

    def _gaze_to(self, xy):
        x, y, h = self.world.pose
        ang = np.arctan2(xy[1] - y, xy[0] - x) - h
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        return (float(np.clip(-ang / 0.8, -1, 1)), 0.2)

    def outputs(self):
        """Face expression (the feeling unless an activity set one recently), gaze, sounds to play."""
        s, self.sounds = self.sounds, []
        return self.face_expr, self.gaze, s

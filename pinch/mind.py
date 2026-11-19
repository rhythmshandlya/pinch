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

from collections import deque

import numpy as np

from pinch.activities import LIBRARY
from pinch.affect import Affect, Personality
from pinch.hearing import MeetingDetector, OwnerVoiceMood
from pinch.language import LLMUnderstanding, understand_rules
from pinch.memory import Memory
from pinch.people import People
from pinch.play import PlayBook
from pinch.safety import NO_GO, DangerMap, hunger
from pinch.touch import GESTURE_EVENT
from pinch.world import WorldModel, situations

MOTION_STEPS = {"emote", "walk", "goto", "turn_to", "pose", "push", "pick", "drop", "release", "recover",
                "right_self"}
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
        if clock is not None:          # one clock for everything that dates things: simulation time in the twin
            self.world.clock = self.memory.clock = clock      # (exact replay); wall time on the robot
        self.rng = np.random.default_rng(seed)
        self.llm = LLMUnderstanding() if llm else None
        if self.llm is not None and not self.llm.check():
            self.llm = None
        self.meeting = MeetingDetector()
        self.voice_mood = OwnerVoiceMood()
        self.t = 0.0
        self.decisions = deque(maxlen=1000)  # recent activity switches with reasons (_record_decision)
        self.phases = deque(maxlen=1000)     # (t, activity, phase): where it is inside an activity
        self.phase = ""                      # the current activity's phase ("" until it says)
        self.suspended = None                # (activity, t): cut short by something urgent, may resume
        self.timed_out = None                # (activity, t): ended by its max_s
        self.danger_map = DangerMap(self.memory.data.get("danger"))   # learned fears of desk spots (safety.py)
        self.playbook = PlayBook(self.memory.data.get("play"))      # games it invented and how fun they were
        self.people = People(self.memory.data.get("people"))        # who it knows and how it feels about them
        self.greet_warmth = 0.0
        self.charging, self._last_battery, self._safe_check_t = False, None, 0.0
        self.lifted_since, self._lift_fear_t = None, 0.0
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
        self.stuck = False                      # StuckSense says so (a foot over the edge, caught, going nowhere)
        self.flipped_t = -1e9                   # last time BodySense said "upside down" (it says so every tick)
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
            seen = self.people.recent(self.world.clock(), within=10.0) if hasattr(self.people, "recent") else []
            if len(seen) == 1:                                # one face in view while you say your name: you
                p = self.people.get(seen[0])
                p["name"], p["owner"] = u["name"], True
                self.memory.data["people"] = self.people.to_dict()
            self.affect.feel("greeted", 0.6)
            self._say("tell_name", f"learned my person's name: {u['name']}")
        if u["fact"] and from_llm:
            self.memory.add_fact(u["fact"])
        i = u["intent"]
        if i in ("praise", "scold"):
            self._blame("praised" if i == "praise" else "scolded")
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
        if u["addressed"] and i in ("chat", "greet", "none", "trick", "come", "play"):
            self.affect.feel("name_called", 0.5)          # (not after a scold or praise: that feeling stays)
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
        if gesture in ("stuck", "unstuck"):                  # from StuckSense, not from a person
            self.stuck = gesture == "stuck"
            self.affect.feel(gesture)
            self._say("body", "I'm stuck!" if self.stuck else "free again")
            if self.stuck:
                self.memory.remember("stuck", "got stuck")
                xy, h = self.world.pose[:2], self.world.pose[2]
                over = self.world.desk.feet_margin(xy, h) < 0
                self.learn_danger(xy, "near_fall" if over else "stuck")
                self._interrupt()
            return
        if gesture == "upside_down":
            new_fall = self.t - self.flipped_t > 2.0
            self.flipped_t = self.t
            if not new_fall:                                     # (still on its back: nothing new to learn)
                return
            self.learn_danger(self.world.pose[:2], "fell")       # it fell over: that spot is scary now
            self._interrupt()
        ev = GESTURE_EVENT.get(gesture)
        if ev:
            self._blame(ev)
        if gesture == "lifted":
            self.people.lifted_by_stranger(self.world.clock())
        if gesture == "lifted":
            self.lifted_since = self.t
        if gesture == "put_down" and self.lifted_since is not None:
            held = self.t - self.lifted_since                  # relief, bigger after a long scary hold
            self.lifted_since = None
            self.affect.feel("put_down_relief", min(1.0, 0.4 + held / 20.0))
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

    # ---------------------------------------------------------------- the decision log (why it did what it did)
    decision_log_path = None             # a file to append the decisions to as JSON lines (None: keep in memory)
    DECISION_LOG_MAX = 5_000_000         # bytes: the file is rotated (to .1) past this

    def _record_decision(self, name, reason):
        """One line per activity switch: from what, to what, why, the strongest candidates, what it noticed
        and how it felt - so any choice can be explained after the fact."""
        scores = getattr(self, "scores", {})
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:4]
        nxt = next(x for x in LIBRARY if x.name == name)
        rec = dict(t=round(self.t, 2), clock=round(self.world.clock(), 2),
                   prev=self.activity.name if self.activity else None, to=name, reason=reason,
                   state=[self.state, nxt.state],
                   top=[[k, round(v, 3)] for k, v in top],
                   noticed=[[s.kind, s.target, round(s.strength, 2)] for s in self.situations[:5]],
                   feeling=self.affect.feeling(), valence=round(self.affect.valence, 3),
                   needs={k: round(v, 2) for k, v in self.affect.needs.as_dict().items()})
        self.decisions.append(rec)
        if self.decision_log_path:
            import json
            from pathlib import Path
            p = Path(self.decision_log_path)
            try:
                if p.exists() and p.stat().st_size > self.DECISION_LOG_MAX:
                    p.replace(p.with_suffix(p.suffix + ".1"))
                with open(p, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            except OSError:
                pass                                         # (a full or read-only disk must not stop the pet)

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

    @property
    def flipped(self):
        """On its back right now (BodySense reports it every tick while it lasts)."""
        return self.t - self.flipped_t < 0.6

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
        self._self_preservation(dt)
        timed_out = self._timeout()
        if self.t >= self.next_choice or self._force:
            self.situations = situations(self.world, self.affect, self.memory)
            self.affect.needs.tidy = min(1.0, self.affect.needs.tidy + 0.02 * len(
                [s for s in self.situations if s.kind == "object_out_of_place"]))
            self.next_choice = self.t + 0.5
            cancel = self._choose()
            self._force = False
            if cancel or timed_out:
                return None, True
        return self._advance(body_busy, body_ok), False

    # ------------------------------------------------------------------ people (people.py)
    def see_person(self, pid, confidence=1.0, name="", owner=False):
        """A person seen (on the robot: a recognised face; in the twin and the app: simulated)."""
        now = self.world.clock()
        known = pid in self.people.people
        away_h = self.people.see(pid, confidence, now, name=name, owner=owner)
        p = self.people.get(pid)
        if p["R"] > 0.45:                                     # someone it's afraid of
            self.affect.feel("loud_noise", 0.6)
            self._say("people", f"{p['name'] or pid} is here - scared ({self.people.describe(pid)})")
            self._interrupt()
        elif p["W"] > 0.5 and (not known or away_h > 0.25):   # a favourite back: greet, harder after longer
            self.greet_warmth = min(1.0, p["W"] * min(1.0, away_h / 8.0) + 0.2)
            self.owner_returned_t = self.t
            self._say("people", f"{p['name'] or pid} is back after {away_h:.1f} h!")
        self.memory.data["people"] = self.people.to_dict()

    def forget_person(self, pid):
        self.people.forget(pid)
        self.memory.data["people"] = self.people.to_dict()

    def _blame(self, event):
        pid = self.people.blame(event, self.world.clock())
        if pid:
            self.memory.data["people"] = self.people.to_dict()

    # ------------------------------------------------------------------ self-preservation (safety.py)
    def danger(self, xy):
        return self.danger_map.danger(xy)

    def hunger(self):
        return hunger(self.world.battery)

    def learn_danger(self, xy, what):
        spot = self.danger_map.learn(xy, what)
        self.danger_map.cap(self.world.desk)
        self.memory.data["danger"] = self.danger_map.to_list()
        self.memory.remember("danger", f"{what} near ({spot['x']:.2f}, {spot['y']:.2f}): fear {spot['v']:.2f}")
        self._say("danger", f"learned: {what} here is dangerous (fear {spot['v']:.2f})")

    def _self_preservation(self, dt):
        self.danger_map.decay(dt / 3600.0)
        if self.t - self._safe_check_t > 5.0:                 # walking through a feared spot safely unlearns it
            self._safe_check_t = self.t
            if not self.stuck and self.danger(self.world.pose[:2]) > 0.05:
                self.danger_map.safe_pass(self.world.pose[:2])
        # held up in the air: fear grows the longer it lasts (fear of heights), calmed by being petted
        if self.lifted_since is not None and self.t - self._lift_fear_t > 2.0:
            self._lift_fear_t = self.t
            held = self.t - self.lifted_since
            calm = 0.6 if self.t - self.world.owner.last_touch < 5.0 else 0.0   # a trusted hand on it
            self.affect.feel("high_up", (1 - calm) * min(1.0, 0.25 + held / 20.0))
        b = self.world.battery
        if self._last_battery is not None and b > self._last_battery + 1e-5 and not self.charging:
            self.charging = True                              # put on charge: relief, scaled by how hungry it was
            self.affect.feel("fed", 0.5 + 1.5 * hunger(self._last_battery))
            self.memory.remember("charged", f"charged from {self._last_battery:.0%}")
            self._say("battery", "charging - ahh")
        elif self._last_battery is not None and b < self._last_battery - 1e-5:
            self.charging = False
        self._last_battery = b

    RESUME_S, RESUME_BONUS = 60.0, 0.1   # an activity cut short by something urgent is preferred back for 60 s

    def _choose(self):
        scores = {}
        for a in LIBRARY:
            u = a.utility(self)
            if self.affect.quiet and not a.quiet_ok:
                u = 0.0
            if u > 0 and self.suspended and self.suspended[0] == a.name and self.t - self.suspended[1] < self.RESUME_S:
                u += self.RESUME_BONUS
            scores[a.name] = u + 0.02 * self.rng.random()
        cur = self.activity
        # commitment: an activity under way keeps most of the priority it started with, so a job doesn't
        # evaporate halfway (e.g. a pushed cap stops counting as "settled" the moment Pinch moves it)
        if cur is not None and self.gen is not None and not (self.affect.quiet and not cur.quiet_ok):
            scores[cur.name] = max(scores.get(cur.name, 0.0), 0.9 * self.start_score)
        self.scores = scores
        best = max(scores, key=scores.get)
        if cur is None or self.gen is None:
            self._start(best, "first" if cur is None else "previous activity finished")
            return True
        if best == cur.name:
            return False
        elapsed = self.t - self.started
        urgent = scores[best] >= 1.5 and scores[best] > scores.get(cur.name, 0) + 0.1
        better = scores[best] > scores.get(cur.name, 0) + 0.15 and elapsed > cur.min_s
        if (cur.interruptible and (urgent or better)) or (not cur.interruptible and urgent and best in ("startle", "right_self")):
            if urgent and cur.name != best:
                self.suspended = (cur.name, self.t)              # (it may come back to it afterwards)
            self._start(best, ("urgent" if urgent else "clearly better") + f" than {cur.name}"
                        + (" (after an interrupt)" if self._force else ""))
            return True
        return False

    def _timeout(self):
        """An activity that has run past its max_s ends now (reason "timeout"): nothing stays stuck in a state."""
        cur = self.activity
        if cur is None or self.gen is None or self.t - self.started <= cur.max_s:
            return False
        self._say("timeout", f"{cur.name} ran {self.t - self.started:.0f} s (limit {cur.max_s:.0f} s)")
        self.gen = None
        self.timed_out = (cur.name, self.t)
        self._interrupt()
        return True

    @property
    def state(self):
        """The top-level behaviour state now (activities/base.py STATES)."""
        return self.activity.state if self.activity is not None else "idle"

    start_score = 0.0

    def _start(self, name, reason=""):
        if self.suspended and self.suspended[0] == name:
            reason += " (resumed)"
            self.suspended = None
        if self.timed_out and self.timed_out[0] == name and self.t - self.timed_out[1] < 1.0:
            reason = "restarted after a timeout"
        self._record_decision(name, reason)
        a = next(x for x in LIBRARY if x.name == name)
        self.phase = ""
        self.start_score = getattr(self, "scores", {}).get(name, 0.0)
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
            elif kind == "phase":                                # which part of the activity it is in
                if step[1] != self.phase:
                    self.phase = step[1]
                    self.phases.append((round(self.t, 2), self.activity.name, step[1]))
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

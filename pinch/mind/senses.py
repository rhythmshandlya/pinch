"""What it hears and feels: speech (rules and the local LLM), the sound of the room (meetings), touch gestures."""

from pinch.mind.common import (  # noqa: F401
    GESTURE_EVENT,
    understand_rules,
)


class Senses:
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
        elif i == "estop":
            self.estop_request = True                     # the firmware acts on it (PinchFirmware.estop)
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
        elif i == "quiet_hours" and u.get("quiet_hours"):
            if u["quiet_hours"] == "clear":
                self.routine.clear_quiet_hours()
                self._say("quiet_hours", "no quiet hours any more")
            else:
                self.routine.set_quiet_hours(*u["quiet_hours"])
                self._say("quiet_hours", "quiet from {:05.2f} to {:05.2f}".format(*u["quiet_hours"]))
            self.memory.changed()                         # (the settings live in the memory file)
            self.affect.feel("spoken_to", 0.5)
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
                self.attention.notice("voice", self.t, strength=min(1.0, s.speech * 1.5), direction=voice_dir)
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
            self.attention.notice("loud", self.t, 1.0, direction=self.world.owner.voice_dir)
            self.affect.feel("loud_noise")
        elif kind == "double_clap":
            self.affect.feel("name_called")
            self.world.owner.last_attention = self.t
            self.request = ("come", "")

    TOUCH_DIR = {"stroke": 0.0, "tickle": 0.0, "boop": 0.0, "tap": 0.0, "hold": 0.0, "claw_hold": 0.0}

    def touch(self, gesture):
        if gesture in self.TOUCH_DIR:
            self.attention.notice("touch", self.t, 1.0, direction=self.TOUCH_DIR[gesture], key=f"touch:{gesture}")
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

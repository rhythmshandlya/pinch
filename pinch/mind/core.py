"""The mind's frame: its state, the tick loop (perceive -> choose -> advance), what it says, and helpers the activities use."""

from pinch.mind.common import (  # noqa: F401
    Affect,
    DangerMap,
    LLMUnderstanding,
    MeetingDetector,
    Memory,
    OwnerVoiceMood,
    People,
    Personality,
    PlayBook,
    Routine,
    WorldModel,
    deque,
    np,
    situations,
)
from pinch.mind.expression import face_now, sound_for
from pinch.mind.learning import CFG as LEARN_CFG
from pinch.mind.learning import own_block

TASTE = {"play": "play", "explore": "curiosity"}       # learned activity -> the need that drives it


class MindCore:
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
        self.routine = Routine(self.memory.data.setdefault("settings", {}))   # time of day, quiet hours
        self.meeting = MeetingDetector()
        self.voice_mood = OwnerVoiceMood()
        self.t = 0.0
        self.decisions = deque(maxlen=1000)  # recent activity switches with reasons (_record_decision)
        self.phases = deque(maxlen=1000)     # (t, activity, phase): where it is inside an activity
        self.phase = ""                      # the current activity's phase ("" until it says)
        self.suspended = None                # (activity, t): cut short by something urgent, may resume
        self.resume_stack = []               # older interrupted activities, newest last (nested interruptions)
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
        self.flipped_t = -1e9
        self.estopped = False                    # an emergency stop is on: the mind issues no motion
        from pinch.mind.learning import FeedbackLearner
        self.learner = FeedbackLearner(self.memory.data.get("learned"))   # what its person's feedback taught it
        from pinch.attention import Attention
        self.attention = Attention()             # what it is paying attention to (pinch/attention.py)
        self.gaze_set_t = -1e9                   # when an activity last aimed the eyes (attention waits)
        self.estop_request = False               # someone said "freeze": the firmware acts on it                   # last time BodySense said "upside down" (it says so every tick)
        self.next_choice = 0.0
        self.log = []                        # (t, kind, text) for the app's diary and tests
        self._was_present = False

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
        now = self.world.clock()
        self.affect.quiet = o.busy == "meeting" or self.t < self.quiet_until or self.routine.quiet_hours(now)
        n = self.affect.needs
        before = {need: getattr(n, need) for need in TASTE.values()}
        self.affect.tick(dt, owner_present=present, owner_attending=self.world.owner_attending(60),
                         battery=self.world.battery, sleep_drive=self.routine.sleep_drive(now))
        self._acquired_taste(before)
        self._self_preservation(dt)
        self._fatigue(dt, body_busy)
        timed_out = self._timeout()
        if self.t >= self.next_choice or self._force:
            self.situations = situations(self.world, self.affect, self.memory)
            self.affect.needs.tidy = min(1.0, self.affect.needs.tidy + 0.02 * len(
                [s for s in self.situations if s.kind == "object_out_of_place"]))
            self.next_choice = self.t + 0.5
            cancel = self._choose()
            self._force = False
            if cancel or timed_out:
                self.face_expr = face_now(self)
                return None, True
        step = self._advance(body_busy, body_ok)
        self.face_expr = face_now(self)                  # the face follows the feeling (pinch/mind/expression.py)
        return step, False

    FATIGUE_UP_S, FATIGUE_DOWN_S, FATIGUE_REST_S = 4500.0, 1800.0, 600.0   # s to go 0 -> 1 working, 1 -> 0 idle / resting

    def _acquired_taste(self, before):
        """What its person praised it for, it comes to want more often, and less what it was scolded for: the need
        behind a learned activity builds up to taste_gain x faster or slower (configs/mind/learning.toml). Without
        this, a drive with no outlet only builds until it gets one, and scolding would just delay it."""
        ctx, person, n, clock = self.learn_context(), self.feedback_person(), self.affect.needs, self.world.clock()
        for activity, need in TASTE.items():
            grew = getattr(n, need) - before[need]
            if grew > 0 and not own_block(activity, clock):             # (in its own stretches, just itself)
                k = 1.0 + LEARN_CFG["taste_gain"] * self.learner.bias(activity, ctx, person) / LEARN_CFG["max_bias"]
                setattr(n, need, before[need] + grew * max(0.0, k))

    def _fatigue(self, dt, body_busy):
        """ARC-002: physical tiredness builds while the body works (about an hour of non-stop motion to the top) and
        drains when still, three times faster while resting or asleep; it makes rest more attractive."""
        n = self.affect.needs
        if body_busy:
            n.fatigue = min(1.0, n.fatigue + dt / self.FATIGUE_UP_S)
        else:
            resting = self.activity is not None and self.activity.state == "rest"
            n.fatigue = max(0.0, n.fatigue - dt / (self.FATIGUE_REST_S if resting else self.FATIGUE_DOWN_S))

    def _gaze_to(self, xy):
        x, y, h = self.world.pose
        ang = np.arctan2(xy[1] - y, xy[0] - x) - h
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        return (float(np.clip(-ang / 0.8, -1, 1)), 0.2)

    gaze_target_xy = None
    ORIENT_S, ORIENT_SALIENCE = 1.2, 0.25           # a fresh stimulus holds the eyes this long (research: orienting)

    def _orienting(self):
        """The eyes' answer to a new stimulus (docs/research/v2_1/aliveness: contingent reaction, eyes before body):
        a sound, a touch or a thing that just appeared, noticed in the last ORIENT_S, pulls the eyes to it, over
        whatever an activity aimed them at. None when nothing fresh is there."""
        fresh = [s for s in self.attention.stimuli if s.t0 is not None and self.t - s.t0 <= self.ORIENT_S
                 and s.salience0 >= self.ORIENT_SALIENCE and (s.xy is not None or s.direction is not None)]
        if not fresh:
            return None
        from pinch.attention import direction_of
        s = max(fresh, key=lambda x: x.t0)
        d = direction_of(s.xy, self.world.pose) if s.xy is not None else s.direction
        return (float(max(-1.0, min(1.0, -d / 0.8))), 0.2)

    OUTCOME_STATES = ("task", "play", "explore", "social", "idle")

    def react_to_outcome(self, ok, kind):
        """A goal-directed motion just ended (research: reacting to outcomes raised perceived competence 3.86 ->
        4.72, Takayama et al. 2011). Only for the things it set out to do: never while hiding, fleeing, on alert or
        getting up (a smile after running away reads as wrong). A failure adds frustration (grumpy after a few), a
        pick, push or drop that worked adds satisfaction; the face is asked for and shows only if it fits."""
        a = self.activity
        if a is None or a.state not in self.OUTCOME_STATES:
            return
        if not ok or kind != "goto":
            self.affect.feel("succeeded" if ok else "failed")
        face = ("happy" if kind in ("pick", "drop", "push") else "content") if ok else "confused"
        self.outcome_intent = (face, self.t, self.started)   # its own slot: the activity's next look waits a moment
        self.face_set_t = self.t
        if not ok:
            self.sound("uh_oh")
        elif face_now(self) == self.face_expr and sound_for("happy_trill", self.affect.feeling()) == "happy_trill":
            self.sound("happy_trill")                # the face can't show it: a small pleased sound, if that fits

    def sound(self, name):
        """Play a sound that fits the feeling: one that says the opposite becomes the feeling's own sound
        (configs/mind/expression.toml [sound]); nothing in quiet mode."""
        if self.affect.quiet:
            return
        s = sound_for(name, self.affect.feeling())
        if s:
            self.sounds.append(s)
            self.sound_set_t = self.t

    def outputs(self):
        """Face expression (the feeling unless an activity set one recently), gaze, sounds to play. Gaze: where
        an activity aimed it in the last 3 s, otherwise the focus of attention, otherwise ahead. A target with a
        place on the desk is re-aimed from the current pose every tick, so the eyes stay on it while it walks."""
        from pinch.attention import direction_of
        s, self.sounds = self.sounds, []
        self.face_expr = face_now(self)
        new = self._orienting()
        if new is not None:                                  # something just happened: the eyes go to it first
            self.gaze = new
            return self.face_expr, self.gaze, s
        if self.t - self.gaze_set_t <= 3.0:
            if self.gaze_target_xy is not None:
                self.gaze = self._gaze_to(self.gaze_target_xy)
        else:
            f, sal = self.attention.update(self.t)
            if f is not None and sal > 0.1 and (f.xy is not None or f.direction is not None):
                d = direction_of(f.xy, self.world.pose) if f.xy is not None else f.direction
                self.gaze = (float(max(-1.0, min(1.0, -d / 0.8))), 0.2)
            elif f is None:
                self.gaze = self._glance()
        return self.face_expr, self.gaze, s

    _glance_xy, _glance_until = None, -1e9

    def _glance(self):
        """Nothing calls its attention: the eyes wander between things it knows around it - its person's spot, the
        desk things in view - a few seconds each, now and then straight ahead (states.toml idle: "wander"). Own
        random stream, so the activity choices replay as before."""
        if self.affect.quiet or self.state == "sleep":
            return (0.0, 0.0)
        if self.t >= self._glance_until:
            rng = self.__dict__.get("_glance_rng")
            if rng is None:
                rng = self._glance_rng = np.random.default_rng(0x5EED)
            x, y, h = self.world.pose
            seen = [xy for _, xy in self.world.things_on_desk()] + [self.world.desk.owner_xy]
            near = [xy for xy in seen
                    if abs((np.arctan2(xy[1] - y, xy[0] - x) - h + np.pi) % (2 * np.pi) - np.pi) < 0.8]
            self._glance_xy = None if not near or rng.random() < 0.2 else tuple(near[rng.integers(len(near))])
            self._glance_until = self.t + rng.uniform(1.5, 4.0)
        return (0.0, 0.0) if self._glance_xy is None else self._gaze_to(self._glance_xy)

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
    WorldModel,
    deque,
    np,
    situations,
)


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
        self.flipped_t = -1e9
        self.estopped = False                    # an emergency stop is on: the mind issues no motion
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

    def _gaze_to(self, xy):
        x, y, h = self.world.pose
        ang = np.arctan2(xy[1] - y, xy[0] - x) - h
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        return (float(np.clip(-ang / 0.8, -1, 1)), 0.2)

    def outputs(self):
        """Face expression (the feeling unless an activity set one recently), gaze, sounds to play."""
        s, self.sounds = self.sounds, []
        return self.face_expr, self.gaze, s

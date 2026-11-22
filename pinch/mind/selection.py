"""Choosing what to do: utility scores, the state layer (timeouts, resume), stepping through the activity, and the decision log."""

from pinch.mind.behaviour import allowed, category, overrides_uninterruptible, rank, states
from pinch.mind.common import (  # noqa: F401
    LIBRARY,
    MOOD_REPLY,
    MOTION_STEPS,
    np,
    situations,
)
from pinch.mind.expression import EXPR

WANTS = EXPR.get("wants", {})


class Selection:
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
                   priority=[category(self.activity.name) if self.activity else None, category(name)],
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

    @property
    def _own_jitter(self):
        """The tie-breaking jitter of activities with own_jitter (a separate stream: adding such an activity doesn't
        reshuffle the random choices every other activity makes, so seeded runs of the rest replay as before)."""
        r = self.__dict__.get("_own_jitter_rng")
        if r is None:
            r = self._own_jitter_rng = np.random.default_rng(0x9E3779B9)
        return r

    RESUME_S, RESUME_BONUS = 60.0, 0.15  # an activity cut short by something urgent is preferred back for 60 s
                                         # (0.15: enough to beat idling's 0.25 plus jitter when it still wants it)

    def _choose(self):
        scores = {}
        for a in LIBRARY:
            u = a.utility(self)
            if self.affect.quiet and not a.quiet_ok:
                u = 0.0
            if u > 0 and a.name in WANTS:
                u *= self._mood_gate(WANTS[a.name])
            if u > 0 and self.suspended and self.suspended[0] == a.name and self.t - self.suspended[1] < self.RESUME_S:
                u += self.RESUME_BONUS
            scores[a.name] = u + 0.02 * (self._own_jitter.random() if a.own_jitter else self.rng.random())
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
        # the current state says how important an urgent newcomer must be (configs/mind/states.toml interrupt):
        # e.g. a greeting doesn't wake it from sleep, a touch or its person's request does
        may_cut_in = rank(best) <= states()[self.state]["interrupt"]
        urgent = urgent and may_cut_in
        better = scores[best] > scores.get(cur.name, 0) + 0.15 and elapsed > cur.min_s and may_cut_in
        if (cur.interruptible and (urgent or better)) or (not cur.interruptible and urgent and overrides_uninterruptible(best)):
            if urgent and cur.name != best:
                self._suspend(cur.name)                          # (it may come back to it afterwards)
            self._start(best, ("urgent" if urgent else "clearly better") + f" than {cur.name}"
                        + (" (after an interrupt)" if self._force else ""))
            return True
        return False

    def _mood_gate(self, w):
        """How much the feeling makes it want this (configs/mind/expression.toml [wants]; Loyall/Oz: emotion shown
        through every choice, docs/research/v2_1/believable)."""
        a = self.affect
        e = a.emotions
        lv = dict(fear=max(a.needs.safety, e.get("scared", 0.0)),
                  sad=max(0.0, -a.valence, e.get("sad", 0.0), e.get("lonely", 0.0)),
                  joy=max(0.0, a.valence), fatigue=a.needs.fatigue)
        return min(2.0, max(0.1, 1.0 + sum(k * lv[n] for n, k in w.items())))

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
        """The top-level behaviour state now (activities/base.py STATES; configs/mind/states.toml): the activity's,
        or "sleep" while a resting activity is in its asleep phase."""
        if self.activity is None:
            return "idle"
        if self.activity.state == "rest" and self.phase == "asleep":
            return "sleep"
        return self.activity.state

    start_score = 0.0

    def motion_owner(self):
        """(owner, layer) for the motion the mind hands the body (pinch/body/arbiter.py layers): the top two
        priority categories as reflexes, what its person asked as a command, hazards and human contact as
        reactions, everything else as the activity."""
        a = self.activity
        if a is None:
            return "mind", "activity"
        r = rank(a.name)
        layer = "reflex" if r <= 2 else "command" if a.name == "obey" else "reaction" if r <= 4 else "activity"
        return a.name, layer

    def channel_owners(self):
        """Who drives the eyes, the face and the sound now: the activity when it set them recently, else ambient
        life (the attention focus, the feeling, silence)."""
        name = self.activity.name if self.activity else None
        return {"gaze": name if name and self.t - self.gaze_set_t <= 3.0 else "ambient:attention",
                "face": name if name and self.t - self.face_set_t <= 3.0 else "ambient:feeling",
                "sound": name if name and self.t - self.sound_set_t <= 1.0 else "ambient"}

    face_set_t = sound_set_t = -1e9
    face_intent = None
    _motion_kind = None

    RESUME_DEPTH = 3                     # interruptions remembered at once (explore -> investigate -> guard_edge)

    def _suspend(self, name):
        """Remember an interrupted activity. One already waiting moves down the resume stack instead of being
        forgotten, so nested interruptions unwind in order."""
        if self.suspended and self.suspended[0] != name:
            self.resume_stack = (self.resume_stack + [self.suspended])[-self.RESUME_DEPTH:]
        self.suspended = (name, self.t)

    def _start(self, name, reason=""):
        if self.suspended and self.suspended[0] == name:
            reason += " (resumed)"
            self.suspended = None
            while self.resume_stack and self.suspended is None:    # the next one down gets its own window
                older, _ = self.resume_stack.pop()
                if older != name:
                    self.suspended = (older, self.t)
        if self.timed_out and self.timed_out[0] == name and self.t - self.timed_out[1] < 1.0:
            reason = "restarted after a timeout"
        self._record_decision(name, reason)
        a = next(x for x in LIBRARY if x.name == name)
        self.phase = ""
        self.start_score = getattr(self, "scores", {}).get(name, 0.0)
        self.activity, self.gen, self.started = a, a.run(self), self.t
        self.waiting_motion, self.wait_until, self._lead_step = False, -1.0, None
        self._say("activity", name)

    def _advance(self, body_busy, body_ok):
        if self.gen is None or self.estopped:               # (an emergency stop: no motion of any kind)
            return None
        if self.waiting_motion:
            if body_busy:
                return None
            self.waiting_motion = False
            self.last_step_ok = body_ok
            if self._motion_kind in ("goto", "push", "pick", "drop"):
                self.react_to_outcome(body_ok, self._motion_kind)
        if self.t < self.wait_until or self.t < self.praise_until:
            return None
        if self._lead_step is not None:                   # it has looked at the goal: now it goes
            step, self._lead_step = self._lead_step, None
            self.gaze_set_t = self.t - 3.0 + self.LEAD_HOLD_S   # (then the eyes are free for what catches them)
            return self._issue(step)
        for _ in range(20):                          # run instant steps until a motion step (or the end)
            try:
                step = next(self.gen)
            except StopIteration:
                self.gen = None
                self.next_choice = 0.0
                return None
            kind = step[0]
            if kind == "face":                       # an asked-for look: shown only if it fits the feeling
                self.face_intent = (step[1], self.t, self.started)     # (pinch/mind/expression.py)
                self.face_set_t = self.t
            elif kind == "sound":
                self.sound(step[1])
            elif kind == "look_at":
                self.gaze = self._gaze_to(step[1])
                self.gaze_target_xy = tuple(step[1])        # re-aimed every tick as the body moves (outputs)
                self.gaze_set_t = self.t
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
                self.face_intent = (face, self.t, self.started)
                if not self.affect.quiet:
                    self.sounds.append(sound)
                self.waiting_motion = True
                return ("emote", emote)
            elif kind in MOTION_STEPS:
                if not allowed(self.state, kind):            # the state's rules (configs/mind/states.toml)
                    self._say("refused", f"{kind} is not allowed in {self.state} ({self.activity.name})")
                    self.last_step_ok = False
                    continue
                goal = self._goal_xy(step)
                if goal is not None and not self._eyes_on(goal):
                    # forethought (Takayama et al. 2011): eyes to the goal first, the legs a moment later
                    self.gaze_target_xy, self.gaze_set_t = tuple(goal), self.t
                    self.gaze = self._gaze_to(goal)
                    self._lead_step, self.wait_until = step, self.t + self.LEAD_S
                    return None
                return self._issue(step)
        return None

    LEAD_S = 0.4                       # eyes lead a goal-directed move by this much
    LEAD_HOLD_S = 0.0                  # then, once the legs go, the eyes are free again (attention, ARC-005)
    _lead_step = None

    def _issue(self, step):
        kind = step[0]
        if kind in ("pick", "drop"):
            self.gaze = (0.0, 0.3)                       # (looks down at the claw's work)
        self.waiting_motion = True
        self._motion_kind = kind
        return step

    def _goal_xy(self, step):
        """Where a goal-directed motion step is headed (None for other steps)."""
        kind = step[0]
        if kind not in ("goto", "push", "pick", "drop") or len(step) < 2:
            return None
        a = step[1]
        if isinstance(a, str):
            o = self.world.objects.get(a)
            return tuple(o.xy) if o is not None else None
        return tuple(a[:2]) if isinstance(a, (tuple, list)) and len(a) >= 2 else None

    def _eyes_on(self, xy):
        """Were the eyes already aimed at xy (an activity looked there in the last 3 s)?"""
        g = self.gaze_target_xy
        return g is not None and self.t - self.gaze_set_t <= 3.0 and self.t - self.gaze_set_t >= self.LEAD_S - 0.05             and (g[0] - xy[0]) ** 2 + (g[1] - xy[1]) ** 2 < 0.03 ** 2

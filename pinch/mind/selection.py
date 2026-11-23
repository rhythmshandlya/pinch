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
from pinch.mind.learning import LEARNABLE, own_block  # noqa: E402


class Selection:
    # ---------------------------------------------------------------- the decision log (why it did what it did)
    decision_log_path = None             # a file to append the decisions to as JSON lines (None: keep in memory)

    DECISION_LOG_MAX = 5_000_000         # bytes: the file is rotated (to .1) past this

    def _record_decision(self, name, reason):
        """One line per activity switch: from what, to what, why, the strongest candidates, what it noticed
        and how it felt - so any choice can be explained after the fact."""
        scores = self.scores
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:4]
        nxt = next(x for x in LIBRARY if x.name == name)
        self.decision_id += 1
        rec = dict(t=round(self.t, 2), id=self.decision_id, clock=round(self.world.clock(), 2),
                   prev=self.activity.name if self.activity else None, to=name, reason=reason,
                   state=[self.state, nxt.state],
                   priority=[category(self.activity.name) if self.activity else None, category(name)],
                   top=[[k, round(v, 3)] for k, v in top],
                   why={k: self.score_parts.get(k, {}) for k, _ in top},
                   noticed=[[s.kind, s.target, round(s.strength, 2)] for s in self.situations[:5]],
                   feeling=self.affect.feeling(), valence=round(self.affect.valence, 3),
                   needs={k: round(v, 2) for k, v in self.affect.needs.as_dict().items()},
                   # everything a learned chooser would see (docs/research: behaviour cloning / offline RL from logs)
                   scores={k: round(v, 3) for k, v in scores.items() if v > 0.005},
                   parts={k: v for k, v in self.score_parts.items() if v.get("want", 0) > 0},
                   affect=self.affect_snapshot(), context=self.context_snapshot())
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

    def affect_snapshot(self):
        a = self.affect
        return dict(arousal=round(a.arousal, 3), mood=round(a.mood if a.mood is not None else a.valence, 3),
                    emotions={k: round(v, 3) for k, v in a.emotions.items()}, quiet=bool(a.quiet))

    def context_snapshot(self):
        """The situation around a choice, for explaining it and for learning from it later."""
        w, now = self.world, self.world.clock()
        o = w.owner
        return dict(part_of_day=self.routine.part_of_day(now), owner_here=bool(w.owner_here(120)),
                    owner_attending=bool(w.owner_attending(60)), owner_busy=o.busy, owner_mood=o.mood,
                    people={pid: dict(W=round(p.get("W", 0), 2), F=round(p.get("F", 0), 2), R=round(p.get("R", 0), 2))
                            for pid, p in self.people.people.items() if pid in self.people.present(now)},
                    battery=None if w.battery is None else round(float(w.battery), 3), flipped=bool(self.flipped),
                    stuck=bool(self.stuck), request=self.request[0] if self.request else None,
                    suspended=self.suspended[0] if self.suspended else None, state=self.state)

    def _open_episode(self, rec):
        a = self.affect
        self.episode = dict(id=rec["id"], activity=rec["to"], t0=self.t, valence0=a.valence,
                            needs0=a.needs.as_dict(), steps_ok=0, steps_failed=0, feedback=[])

    def _close_episode(self, end):
        """How the activity that is ending went: the record a learned chooser is trained on (its outcome)."""
        e = self.episode
        if e is None:
            return
        a = self.affect
        self.episodes.append(dict(
            t=round(self.t, 2), id=e["id"], activity=e["activity"], end=end, seconds=round(self.t - e["t0"], 2),
            steps_ok=e["steps_ok"], steps_failed=e["steps_failed"], feedback=e["feedback"],
            dvalence=round(a.valence - e["valence0"], 3),
            dneeds={k: round(v - e["needs0"][k], 3) for k, v in a.needs.as_dict().items()}))
        self.episode = None

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

    def learn_context(self):
        """The coarse situation learned feedback is filed under: part of the day, and whether its person is here."""
        here = "here" if self.world.owner_here(120) else "away"
        return f"{self.routine.part_of_day(self.world.clock())}:{here}"

    def owner_id(self):
        """Its own person's id (marked as the owner), or "owner" before it has seen them."""
        return next((pid for pid, p in self.people.people.items() if p.get("owner")), "owner")

    def feedback_person(self):
        """Whose teaching counts now: the person seen most recently if it knows them (familiarity >= 0.3), else
        its own person - strangers don't teach it, and its person's lessons don't vanish when a visitor is about.
        None: a stranger is the only one here (their feedback teaches nothing)."""
        now = self.world.clock()
        here = self.people.present(now)
        known = [p for p in here if self.people.people.get(p, {}).get("F", 0.0) >= 0.3]
        if known:
            return max(known, key=lambda k: self.people.seen[k][0])
        if here:                                         # only people it hardly knows in view: not its person's lesson
            return None
        return self.owner_id()

    def _choose(self):
        scores = {}
        ctx, person, clock = self.learn_context(), self.feedback_person(), self.world.clock()
        daytime = EXPR.get("daytime", {}).get(self.routine.part_of_day(clock), {})
        parts = {}
        for a in LIBRARY:                                         # book-keeping first: scoring changes nothing
            a.prepare(self)
        env = dict(ctx=ctx, person=person, clock=clock, daytime=daytime)
        for a in LIBRARY:
            u = a.utility(self)
            why = dict(want=round(u, 3))                          # what made the score (the decision log keeps it)
            for stage in self.score_stages():
                u = stage(a, u, why, env)
            parts[a.name] = why
            scores[a.name] = u + 0.02 * (self._own_jitter.random() if a.own_jitter else self.rng.random())
        cur = self.activity
        # commitment: an activity under way keeps most of the priority it started with, so a job doesn't
        # evaporate halfway (e.g. a pushed cap stops counting as "settled" the moment Pinch moves it)
        if cur is not None and self.gen is not None and not (self.affect.quiet and not cur.quiet_ok):
            scores[cur.name] = max(scores.get(cur.name, 0.0), 0.9 * self.start_score)
        self.scores, self.score_parts = scores, parts
        best = max(scores, key=scores.get)
        if self.choice_log is not None:
            self.choice_log.append(dict(t=round(self.t, 2), current=cur.name if cur else None, best=best,
                                        scores={k: round(v, 3) for k, v in scores.items() if v > 0.005},
                                        affect=self.affect_snapshot(), needs=self.affect.needs.as_dict()))
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
        # clearly better and the same kind of thing (a new object while exploring): it may take over even where the
        # state lets only higher kinds cut in (level of interest, Blumberg's Hamsterdam); other kinds follow the state
        same_kind = rank(best) == rank(cur.name)
        better = scores[best] > scores.get(cur.name, 0) + 0.15 and elapsed > cur.min_s and (may_cut_in or same_kind)
        if (cur.interruptible and (urgent or better)) or (not cur.interruptible and urgent and overrides_uninterruptible(best)):
            if cur.name != best:
                self._suspend(cur.name)                          # (it may come back to it afterwards)
            self._start(best, ("urgent" if urgent else "clearly better") + f" than {cur.name}"
                        + (" (after an interrupt)" if self._force else ""))
            return True
        return False

    # ---------------------------------------------------------------- the score stages
    # A choice is each activity's want passed through these stages in order; each may change the score and says
    # why (the decision log keeps it). extra_stages (empty today) is where a learned scorer or the language-model
    # director plugs in, after what it was taught and before the resume bonus: same signature, capped by itself.
    extra_stages = ()

    def score_stages(self):
        return (self._stage_quiet, self._stage_mood, self._stage_daytime, self._stage_learned,
                *self.extra_stages, self._stage_resume)

    def _stage_quiet(self, a, u, why, env):
        if self.affect.quiet and not a.quiet_ok:
            why["quiet"] = True
            return 0.0
        return u

    def _stage_mood(self, a, u, why, env):
        if u > 0 and a.name in WANTS:
            g = self._mood_gate(WANTS[a.name])
            why["mood"] = round(g, 2)
            return u * g
        return u

    def _stage_daytime(self, a, u, why, env):
        k = env["daytime"].get(a.name, 1.0)
        if u > 0 and k != 1.0:                                    # the day's rhythm
            why["day"] = k
            return u * k
        return u

    def _stage_learned(self, a, u, why, env):
        if u > 0 and not own_block(a.name, env["clock"]):
            u2 = self.learner.shape(a.name, u, env["ctx"], env["person"] or self.owner_id(), env["clock"])
            if u2 != u:
                why["learned"] = round(u2 - u, 3)                 # what its person's feedback taught it
            return u2
        if u > 0 and a.name in LEARNABLE:
            why["as_itself"] = True                               # (one of its own stretches: learning set aside)
        return u

    def _stage_resume(self, a, u, why, env):
        if u > 0 and self.suspended and self.suspended[0] == a.name and self.t - self.suspended[1] < self.RESUME_S:
            why["resume"] = self.RESUME_BONUS
            return u + self.RESUME_BONUS
        return u

    def learn_from(self, event):
        """Its person's feedback (praise, scolding, petting) teaches it about what it is doing or just did."""
        if self.episode is not None:
            self.episode["feedback"].append([round(self.t - self.episode["t0"], 2), event])
        person = self.feedback_person()
        if person is None:
            return
        cur = self.activity.name if self.activity is not None else None
        out = self.learner.feedback(event, self.t, self.world.clock(), cur, self.learn_context(), person,
                                    current_age=self.t - self.started)
        self.memory.data["learned"] = self.learner.to_dict()           # (saved every time: the fade too)
        self.memory.changed()
        if out:
            self._say("learned", f"{event}: " + ", ".join(f"{a} {d:+.3f}" for a, d in out.items()))

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
        timed_out = bool(self.timed_out and self.activity is not None and self.timed_out[0] == self.activity.name
                         and self.t - self.timed_out[1] < 1.0)
        self._close_episode("timeout" if timed_out else "finished" if reason.startswith("previous activity finished")
                            else f"replaced by {name}")
        self._record_decision(name, reason)
        self._open_episode(self.decisions[-1])
        a = next(x for x in LIBRARY if x.name == name)
        self.phase = ""
        if self.activity is not None:
            self.learner.ended(self.t, self.activity.name)            # (it can still be praised for it a moment)

        self.start_score = self.scores.get(name, 0.0)
        self.activity, self.gen, self.started = a, a.run(self), self.t
        self.waiting_motion, self.wait_until, self._lead_step, self._motion_kind = False, -1.0, None, None
        self._say("activity", name)

    def _advance(self, body_busy, body_ok):
        if self.gen is None or self.estopped:               # (an emergency stop: no motion of any kind)
            return None
        if self.waiting_motion:
            if body_busy:
                if self._motion_kind in ("pick", "drop") and self.gaze_target_xy is not None \
                        and not self.body_reflex:
                    self.gaze_set_t = self.t                 # eyes stay on the thing / the spot (not through a reflex)
                return None
            self.waiting_motion = False
            self.last_step_ok = body_ok
            if self.episode is not None:
                self.episode["steps_ok" if body_ok else "steps_failed"] += 1
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
                if not allowed(self.state, "emote"):         # the state's rules apply to the reply too
                    self._say("refused", f"emote is not allowed in {self.state} ({self.activity.name})")
                    continue
                return self._issue(("emote", emote))
            elif kind in MOTION_STEPS:
                if not allowed(self.state, kind):            # the state's rules (configs/mind/states.toml)
                    self._say("refused", f"{kind} is not allowed in {self.state} ({self.activity.name})")
                    self.last_step_ok = False
                    continue
                goal = self._goal_xy(step)
                if goal is not None and rank(self.activity.name) <= 3:   # urgent (a hazard): eyes there at once, no pause
                    self.gaze_target_xy, self.gaze_set_t = tuple(goal), self.t
                    self.gaze = self._gaze_to(goal)
                elif goal is not None and not self._eyes_on(goal):
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

    def motion_cancelled(self):
        """The body dropped the step it was doing from outside (an emergency stop): it did not succeed, and nothing
        it was about to start goes ahead."""
        if self.waiting_motion:
            self.last_step_ok = False
        self.waiting_motion, self._lead_step, self._motion_kind = False, None, None

    def _vary_emote(self, step):
        """An emote it just did gives way to an unused sibling from its family (configs/mind/expression.toml
        [emote_family]), so asking twice doesn't look the same twice. Own random stream: replays exactly."""
        fam_cfg = EXPR.get("emote_family", {})
        name, recent = step[1], self.__dict__.setdefault("_recent_emotes", [])
        fam = next((v for k, v in fam_cfg.items() if k != "recent" and name in v), None)
        if fam and name in recent[-fam_cfg.get("recent", 3):]:
            fresh = [e for e in fam if e not in recent[-fam_cfg.get("recent", 3):]]
            if fresh:
                rng = self.__dict__.get("_emote_rng")
                if rng is None:
                    rng = self._emote_rng = np.random.default_rng(self.rng.bit_generator.seed_seq.spawn(1)[0])
                name = fresh[int(rng.integers(len(fresh)))]
                step = (step[0], name) + tuple(step[2:])
        self._recent_emotes = (recent + [name])[-8:]
        return step

    def _issue(self, step):
        kind = step[0]
        if kind == "emote" and len(step) > 1 and not getattr(self.activity, "exact_emotes", False):
            step = self._vary_emote(step)                   # (a trick asked for by name plays as asked)
        if kind in ("pick", "drop"):
            goal = self._goal_xy(step)                   # watches the thing it picks up, or where it puts it down
            self.gaze_target_xy = tuple(goal) if goal is not None else None
            if goal is not None:
                self.gaze_set_t = self.t
                self.gaze = self._gaze_to(goal)
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
        age = self.t - self.gaze_set_t
        return g is not None and self.LEAD_S - 0.05 <= age <= 3.0 and (g[0] - xy[0]) ** 2 + (g[1] - xy[1]) ** 2 < 0.03 ** 2

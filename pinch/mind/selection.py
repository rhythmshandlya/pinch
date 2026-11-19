"""Choosing what to do: utility scores, the state layer (timeouts, resume), stepping through the activity, and the decision log."""

from pinch.mind.common import (  # noqa: F401
    LIBRARY,
    MOOD_REPLY,
    MOTION_STEPS,
    situations,
)


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
        if self.gen is None or self.estopped:               # (an emergency stop: no motion of any kind)
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

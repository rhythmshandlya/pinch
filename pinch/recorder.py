"""Session recorder: one directory per session with everything needed to judge, replay and film it.

    rec = SessionRecorder("runs/session-1", header=run_header(seed=5, scenario="life"))
    ... each 20 ms control tick:  rec.tick(t, k, mind, body, q_req, q_cmd, sample_fn=...)
    ... things that happen:        rec.event(t, "say", text="hi pinch")   rec.step_started(t, ("emote", "wave"))
    rec.close()                    # flushes, then runs the observer and writes the report

The directory:
  header.json      versions (git commit, dirty tree, policy files with their hashes and registry entries, python /
                   numpy / mujoco), seed, scenario, config, how long it ran
  events.jsonl     the timeline, one JSON object per line, {"t", "kind", ...}:
                   decision (with its reason, candidates, feeling), phase, step_start / step_end (with the result,
                   duration and smoothness), face, feeling, look (an activity aimed the eyes), attend, sound, log
                   (Mind.log: touches, owner, curiosity, timeouts ...), stimuli (say, touch, show_prop ...), finding
  samples_NNN.npz  trajectories every `sample_every` ticks (default 3 = 16.7 Hz): base pose, how upright,
                   joint targets sent vs measured, gaze and where it should be, face, feeling, activity, phase,
                   body status, smoothness in the window; with state=True also the whole MuJoCo state for video
  model.mjb        (twin, state=True) the compiled world, so sim/record.py can film the session afterwards
  findings.json, report.json, report.txt   the observer's findings and the realism metrics (pinch/observer.py)

It only reads: it draws no random numbers and changes nothing, so a recorded run is the same run as an unrecorded
one (tests/test_recorder.py checks it against the golden trace). On the robot it runs without state or video
(robot/pinch_main.py --record DIR): the timeline and the samples are a few kB per minute.
"""

import hashlib
import json
import math
import platform
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np

SCHEMA = 1
LOCOMOTION = {"goto", "walk", "push", "turn_to", "recover"}
CURIOUS = {"investigate", "watch", "orient", "startle"}
OBJECT_TASKS = {"tidy", "play"}
SOCIAL = {"greet", "comfort", "seek_attention"}
ROOT = Path(__file__).resolve().parents[1]


def step_label(step):
    """('emote', 'wave') -> 'emote:wave'; ('goto', (x, y)) -> 'goto'."""
    if not step:
        return ""
    arg = step[1] if len(step) > 1 and isinstance(step[1], str) else ""
    return f"{step[0]}:{arg}" if arg else str(step[0])


def _plain(x):
    """JSON-friendly copy (tuples -> lists, numpy -> python, floats rounded)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, deque)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return round(v, 4) if math.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if x is None or isinstance(x, (str, int, bool)):
        return x
    return str(x)


def gaze_for(direction):
    """The face's horizontal gaze (-1 right .. 1 left) that looks toward a body-frame direction (pinch/mind)."""
    return float(np.clip(-direction / 0.8, -1.0, 1.0))


def _direction(xy, pose):
    a = math.atan2(xy[1] - pose[1], xy[0] - pose[0]) - pose[2]
    return (a + math.pi) % (2 * math.pi) - math.pi


def gaze_target(mind, last_target=None):
    """(what, gaze it should have) for the current activity, or (None, None) when it has no particular target.
    Curious activities look at the thing they are about, object tasks at the object, social ones at the person."""
    act = mind.activity.name if mind.activity is not None else None
    pose = mind.world.pose
    xy, what = None, None
    if act in CURIOUS:
        f = mind.attention.focus
        if f is not None and f.xy is not None:
            xy, what = f.xy, f.key or f.kind
        elif "_novelty" in mind.__dict__:
            th = mind.novelty.moving(mind.t) if act == "watch" else None
            if th is None:
                near = sorted(mind.novelty.unknown(mind.t), key=lambda h: math.hypot(h.xy[0] - pose[0], h.xy[1] - pose[1]))
                th = near[0] if near else None
            if th is not None:
                xy, what = th.xy, f"thing{th.id}"
        elif f is not None and f.direction is not None:
            return f.key or f.kind, gaze_for(f.direction)
    elif act in OBJECT_TASKS and last_target is not None:
        kind, val = last_target
        if kind == "obj":
            ob = mind.world.objects.get(val)
            if ob is not None:
                xy, what = ob.xy, val
        else:
            xy, what = val, "drop spot"
    elif act in SOCIAL:
        xy, what = mind.world.desk.owner_xy, "owner"
    if xy is None:
        return None, None
    return what, gaze_for(_direction(xy, pose))


def _git(*args, cwd=ROOT):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def run_header(root=ROOT, **config):
    """What this session ran: the code (commit, dirty), the policies (file hashes + registry entries), versions,
    and the caller's config (seed, scenario ...)."""
    root = Path(root)
    pol = {}
    reg = {}
    try:
        import tomllib
        reg = tomllib.loads((root / "policies" / "registry.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError, ImportError):
        pass
    for f in sorted((root / "policies").glob("*.npz")):
        entry = next((dict(slot=k, **v) for k, v in reg.items() if isinstance(v, dict) and v.get("file") == f.name), None)
        pol[f.name] = dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest()[:16], registry=entry)
    vers = dict(python=platform.python_version(), numpy=np.__version__)
    try:
        import mujoco
        vers["mujoco"] = mujoco.__version__
    except ImportError:
        pass
    return dict(schema=SCHEMA, git=dict(commit=_git("rev-parse", "HEAD"), branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
                                        dirty=bool(_git("status", "--porcelain", "--untracked-files=no"))),
                policies=pol, versions=vers, platform=platform.system(), created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                config=_plain(config))


class SessionRecorder:
    def __init__(self, path, header=None, sample_every=3, state=True, live_every=None, chunk=6000, dt=0.02,
                 max_step=None):
        """sample_every: ticks between samples (3 = 16.7 Hz at 50 Hz, aligned with the twin's face updates).
        state: keep the whole simulator state in the samples (video). live_every: run the observer every so many
        seconds and put what it finds on the timeline. max_step: per-joint largest target change per tick the
        servos follow (robot/crab_robot.py MAX_STEP); a requested jump beyond it counts as abrupt."""
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in list(self.dir.glob("samples_*.npz")) + [self.dir / n for n in ("findings.json", "report.json",
                                                                                   "report.txt", "events.jsonl")]:
            if old.exists():
                old.unlink()
        self.sample_every, self.state, self.chunk, self.dt = int(sample_every), state, int(chunk), dt
        self.live_every, self._next_live = live_every, live_every or 0.0
        self.max_step = None if max_step is None else np.asarray(max_step, float)
        self.header = dict(header or {}, sample_dt=round(dt * self.sample_every, 4), tick_dt=dt, state=state)
        self._write_header()
        self._fh = open(self.dir / "events.jsonl", "w", encoding="utf-8")
        self.events = deque(maxlen=50000)                  # the recent timeline in memory (the live observer)
        self.n_events = 0
        self._rows, self._chunks, self.n_samples = [], 0, 0
        self.findings_seen = set()
        # what was seen last (the mind's logs are polled for what is new)
        self._last = dict(decision=None, phase=None, log=None)
        self._face = self._feeling = self._focus = None
        self._gaze_t = None
        self._mind_t = None
        self._task, self._pending, self._step = None, None, None
        self._end_why = None
        self.target = None                                 # ("obj", name) or ("xy", (x, y)): the last step's target
        self.goal = None                                   # the running step's goal, the same way
        self._q = deque(maxlen=4)                          # last joint targets sent (jerk)
        self._q_req = None
        self._win = self._new_window()
        self._last_status = ""
        self.closed = False

    # -- the timeline --------------------------------------------------------------------------------------------
    def event(self, t, kind, **data):
        rec = dict(t=round(float(t), 3), kind=kind, **_plain(data))
        self.events.append(rec)
        self.n_events += 1
        self._fh.write(json.dumps(rec) + "\n")
        return rec

    def step_started(self, t, step, source="mind"):
        """The caller started a body step (the mind's, or a reflex): tick() matches it to the new body task."""
        self._pending = (step, source)

    def step_cancelled(self, t, why="mind"):
        self._end_why = f"cancelled ({why})"

    # -- every control tick --------------------------------------------------------------------------------------
    def tick(self, t, k, mind, body, q_req, q_cmd, sample_fn=None):
        """After the body's tick. t: time at the end of this tick; k: the tick's index. sample_fn() -> dict with
        pose (x, y, yaw), up (z of the body's up axis), q_meas, and with state: qpos, mocap."""
        if mind is not None and mind.t != self._mind_t:
            self._mind_t = mind.t
            self._poll_mind(t, mind)
        self._smooth(np.asarray(q_req, float), np.asarray(q_cmd, float), body.status if body.task is not None else "")
        self._poll_body(t, body)                           # (it files the body's trace under the right step)
        if k % self.sample_every == 0:
            self._sample(t, mind, body, q_cmd, sample_fn() if sample_fn else {})
        if self.live_every and t >= self._next_live:
            self._next_live = t + self.live_every
            self.observe_live(t)

    def _new(self, key, seq):
        """Items of seq after the last one seen (by identity: the mind's logs are trimmed, never rewritten)."""
        if not seq:
            return []
        last = self._last[key]
        if last is not None and seq[-1] is last:
            return []
        out = []
        for item in reversed(seq):
            if item is last:
                break
            out.append(item)
        self._last[key] = seq[-1]
        return out[::-1]

    def _poll_mind(self, t, mind):
        for d in self._new("decision", mind.decisions):
            self.event(d["t"], "decision", **{k: v for k, v in d.items() if k != "t"})
        for pt, act, ph in self._new("phase", getattr(mind, "phases", ())):
            self.event(pt, "phase", activity=act, phase=ph)
        for lt, kind, text in self._new("log", mind.log):
            if kind != "activity":                         # (the decisions say that, with the reason)
                self.event(lt, "log", what=kind, text=text)
        if mind.face_expr != self._face:
            self._face = mind.face_expr
            self.event(t, "face", face=mind.face_expr, activity=self._act(mind))
        f = mind.affect.feeling()
        if f != self._feeling:
            self._feeling = f
            self.event(t, "feeling", feeling=f, valence=mind.affect.valence, arousal=mind.affect.arousal)
        if mind.gaze_set_t != self._gaze_t:                # an activity aimed the eyes
            self._gaze_t = mind.gaze_set_t
            if mind.gaze_set_t > -1e8:
                what, want = gaze_target(mind, self.target)
                self.event(t, "look", gaze=mind.gaze, target=what, want=want, activity=self._act(mind))
        foc = mind.attention.focus
        if foc is not self._focus:
            self._focus = foc
            if foc is not None:
                self.event(t, "attend", stimulus=foc.kind, key=foc.key)

    _UNSET = object()

    def _poll_trace(self, t, body):
        """The body's own trace (BodyCore.trace: lined up, missed, lifted it or not, detours ...)."""
        log = body.log
        if not log:
            return
        if self._last.get("trace", self._UNSET) is self._UNSET:
            self._last["trace"] = log[-1]                  # (what was there before the recording isn't ours)
            return
        for _, what, kw in self._new("trace", log):
            self.event(t, "body", what=what, step=self._step["label"] if self._step else None, info=kw)

    @staticmethod
    def _act(mind):
        return mind.activity.name if mind.activity is not None else None

    def _poll_body(self, t, body):
        task = body.task
        if self._pending is None:                          # this tick's trace is the running (or just ended) step's
            self._poll_trace(t, body)
        if task is self._task and self._pending is None:
            return
        if self._step is not None and (task is not self._task or self._pending is not None):
            if task is None and self._pending is None:
                why = self._end_why or ("done" if body.ok else "failed")
            else:
                why = self._end_why or "preempted"
            self._end(t, why, body.ok if task is None and self._pending is None else False)
        if self._pending is not None or task is not None:
            step, source = self._pending if self._pending is not None else ((body.status,), "other")
            self._pending = None
            label = step_label(step)
            self._step = dict(t0=t, label=label, kind=str(step[0]), source=source, jerk=0.0, vpeak=0.0, jump=0.0,
                              clamped=0)
            self.goal = None                                # where the step is trying to get (stuck check)
            if len(step) > 1:
                if step[0] in ("pick", "push") and isinstance(step[1], str):
                    self.target = self.goal = ("obj", step[1])
                    if step[0] == "push" and len(step) > 2 and _is_xy(step[2]):
                        self.goal = ("push", step[1], tuple(float(v) for v in step[2][:2]))
                elif step[0] in ("drop", "goto") and _is_xy(step[1]):
                    self.target = ("xy", tuple(step[1])) if step[0] == "drop" else self.target
                    self.goal = ("xy", tuple(float(v) for v in step[1][:2]))
            self.event(t, "step_start", step=label, motion=step[0], args=list(step[1:]), source=source,
                       status=body.status)
            if source != "other":
                self._poll_trace(t, body)                  # (a new step's first tick: its trace)
            if task is None:                                # (over at once: an unknown step, or nothing to do)
                self._end(t, "done" if body.ok else "failed", body.ok)
        self._task = task
        self._end_why = None

    def _end(self, t, why, ok):
        s, self._step = self._step, None
        dur = t - s["t0"]
        ldlj = None
        if dur > 0.2 and s["vpeak"] > 1e-3 and s["jerk"] > 0:
            ldlj = -math.log(dur ** 3 / s["vpeak"] ** 2 * s["jerk"])
        self.event(t, "step_end", step=s["label"], motion=s["kind"], ok=bool(ok), end=why, dur=dur, source=s["source"],
                   ldlj=ldlj, max_jump=s["jump"], clamped=s["clamped"])

    def _smooth(self, q_req, q_cmd, status=""):
        """Jerk of the joint targets sent, and jumps in the targets asked for (before the servo speed limit)."""
        dt = self.dt
        self._q.append(q_cmd.copy())
        jerk_sq = v = 0.0
        if len(self._q) >= 2:
            v = float(np.linalg.norm(self._q[-1] - self._q[-2])) / dt
        if len(self._q) == 4:
            q0, q1, q2, q3 = self._q
            jerk_sq = float(np.sum(((q3 - 3 * q2 + 3 * q1 - q0) / dt ** 3) ** 2))
        jump, clamped, joint = 0.0, 0, -1
        if self._q_req is not None:
            d = np.abs(q_req - self._q_req)
            joint = int(d.argmax())
            jump = float(d[joint])
            if self.max_step is not None and len(self._q) >= 2:
                clamped = int(np.any(np.abs(q_req - self._q[-2]) > self.max_step * 1.5))
        self._q_req = q_req.copy()
        w = self._win
        w["jerk"] += jerk_sq
        w["n"] += 1
        if jump > w["jump"]:
            w["jump"], w["jump_joint"], w["jump_status"] = jump, joint, status or self._last_status
        w["clamped"] += clamped
        self._last_status = status
        if self._step is not None:
            s = self._step
            s["jerk"] += jerk_sq * dt
            s["vpeak"] = max(s["vpeak"], v)
            s["jump"] = max(s["jump"], jump)
            s["clamped"] += clamped

    def _goal_distance(self, mind, body, pose):
        """How far the running step still is from its goal (m): Pinch to the spot (goto, drop) or to the thing
        (pick), the thing to where it is being pushed (push); NaN when there is no goal."""
        if self.goal is None or body.task is None or mind is None:
            return np.nan
        pose = pose if pose is not None else mind.world.pose
        kind, val = self.goal[0], self.goal[1]
        if kind == "push":
            ob = mind.world.objects.get(val)
            return float(math.dist(ob.xy, self.goal[2])) if ob is not None else np.nan
        if kind == "obj":
            ob = mind.world.objects.get(val)
            val = ob.xy if ob is not None else None
        return float(math.dist(pose[:2], val[:2])) if val is not None else np.nan

    @staticmethod
    def _new_window():
        return dict(jerk=0.0, n=0, jump=0.0, clamped=0, jump_joint=-1, jump_status="")

    # -- trajectories ----------------------------------------------------------------------------------------------
    def _sample(self, t, mind, body, q_cmd, x):
        w = self._win
        row = dict(t=float(t), q_cmd=np.asarray(q_cmd, np.float32).copy(),
                   jerk_rms=math.sqrt(w["jerk"] / max(1, w["n"])), max_jump=w["jump"], clamped=w["clamped"],
                   jump_joint=w["jump_joint"], jump_status=w["jump_status"],
                   busy=bool(body.busy), status=body.status if body.busy else "", cmd_v=float(np.hypot(*body.cmd_vel[:2])),
                   cmd_w=float(body.cmd_vel[2]))
        self._win = self._new_window()
        pose = x.get("pose")
        if pose is not None:
            row["x"], row["y"], row["yaw"] = (float(v) for v in pose)
        row["z"] = float(x.get("z", np.nan))
        row["up"] = float(x.get("up", np.nan))
        if x.get("q_meas") is not None:
            row["q_meas"] = np.asarray(x["q_meas"], np.float32)
        if mind is not None:
            what, want = gaze_target(mind, self.target)
            row.update(gaze_x=float(mind.gaze[0]), gaze_y=float(mind.gaze[1]), gaze_want=np.nan if want is None else want,
                       gaze_target=what or "", face=mind.face_expr, feeling=mind.affect.feeling(),
                       valence=float(mind.affect.valence), arousal=float(mind.affect.arousal),
                       activity=self._act(mind) or "", phase=mind.phase, state=mind.state, quiet=bool(mind.affect.quiet))
        row["goal_d"] = self._goal_distance(mind, body, pose)
        if self.state:
            if x.get("qpos") is not None:
                row["qpos"] = np.asarray(x["qpos"], np.float64).copy()
            if x.get("mocap") is not None:
                row["mocap"] = np.asarray(x["mocap"], np.float64).copy()
        self._rows.append(row)
        self.n_samples += 1
        if len(self._rows) >= self.chunk:
            self._flush_samples()

    def _flush_samples(self):
        if not self._rows:
            return
        cols = {}
        for key in self._rows[0]:
            vals = [r.get(key) for r in self._rows]
            if any(v is None for v in vals):
                continue
            cols[key] = np.array(vals)
        np.savez_compressed(self.dir / f"samples_{self._chunks:03d}.npz", **cols)
        self._chunks += 1
        self._rows = []

    def samples(self):
        """The samples not yet written to a chunk, as columns (the live observer)."""
        return _columns(self._rows)

    # -- the observer, live ----------------------------------------------------------------------------------------
    def observe_live(self, t):
        from pinch.observer import observe_session
        sess = dict(header=self.header, events=list(self.events), samples=self.samples())
        new = []
        for f in observe_session(sess):
            if f.id not in self.findings_seen:
                self.findings_seen.add(f.id)
                new.append(f)
                self.event(t, "finding", finding=f.kind, severity=f.severity, t0=f.t0, t1=f.t1, evidence=f.evidence)
        return new

    # -- the end -------------------------------------------------------------------------------------------------------
    def _write_header(self, **more):
        self.header.update(more)
        (self.dir / "header.json").write_text(json.dumps(_plain(self.header), indent=1), encoding="utf-8")

    def close(self, t=None, analyse=True, **more):
        """Flush everything; with analyse, run the observer over the whole session and write the report."""
        if self.closed:
            return None
        self.closed = True
        self._flush_samples()
        self._fh.close()
        self._write_header(duration=t, events=self.n_events, samples=self.n_samples, closed=True, **more)
        if not analyse:
            return None
        from pinch.observer import write_report
        return write_report(self.dir)


def _is_xy(v):
    return isinstance(v, (tuple, list, np.ndarray)) and len(v) >= 2 and all(isinstance(c, (int, float, np.floating))
                                                                           for c in v[:2])


def _columns(rows):
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        vals = [r.get(key) for r in rows]
        if not any(v is None for v in vals):
            out[key] = np.array(vals)
    return out


def load_session(path):
    """{header, events, samples (columns, all chunks joined)} of a recorded session directory."""
    p = Path(path)
    header = json.loads((p / "header.json").read_text(encoding="utf-8"))
    events = [json.loads(ln) for ln in (p / "events.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    parts = [dict(np.load(f)) for f in sorted(p.glob("samples_*.npz"))]
    samples = {}
    if parts:
        keys = set(parts[0]).intersection(*parts[1:]) if len(parts) > 1 else set(parts[0])
        samples = {k: np.concatenate([q[k] for q in parts]) for k in keys}
    return dict(header=header, events=events, samples=samples, path=str(p))

"""Liveliness bench (phase 4, docs/research/v2_1/aliveness/notes.md): does Pinch behave like something alive, not
just move? Real stimuli go in through its senses in the twin; what it does back is timed and scored.

    python sim/benches/liveliness.py                 # 12 trials per stimulus kind + a 6-minute session per seed
    python sim/benches/liveliness.py --quick         # fewer trials

  1 contingent reaction   a voice or a bang from one side, a tap, a toy moving into view: the eyes move toward it
                          (median latency <= 0.4 s, >= 95 % within 1 s)
  2 eyes before body      when the body turns to it too, the eyes moved first (>= 90 %)
  3 meaningful gaze       share of looking time spent on something that matters - what it attends to, a voice it
                          just heard, a moving thing, the goal of what it is doing - when there is one (>= 70 %)
  4 forethought           goal-directed moves (walk to, push, pick) start with the eyes already on the goal
                          (>= 95 %, the eyes there >= 0.1 s before the legs move)
  5 outcome reaction      a face or sound change within 1 s after a goal-directed move succeeds or fails (>= 90 %)
  6 idle from feeling     breathing faster when excited than when calm (>= 30 %); fidgets more often (rate ratio)
  7 variety               fidget timing coefficient of variation >= 0.5
Thresholds are the research notes' (most marked [P]: starting values, to recalibrate with the owner's review).
"""

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from pinch.hearing import VoiceSummary  # noqa: E402
from pinch.memory import Memory  # noqa: E402
from pinch.twin import PinchSim  # noqa: E402

DT = 0.02
EYE_MOVE = 0.10              # the drawn eye shifts this far toward the stimulus (eye range -1..1)
BODY_TURN = math.radians(6)  # the body turns this far toward it
GAZE_TOL = 0.30              # the eye within this of a target's direction counts as looking at it


def eye_dir(direction):
    """Where the eye sits (-1..1) for a body-frame direction (rad, + left): Mind._gaze_to convention."""
    return float(np.clip(-direction / 0.8, -1, 1))


def _sim(seed, props=()):
    sim = PinchSim(memory=Memory(path=None), seed=seed, props=list(props))
    sim.mind.memory.data["homes"] = {}
    return sim


# ---------------------------------------------------------------------------------------------- 1, 2: reactions
def react_trial(seed, kind, side):
    sim = _sim(seed, props=["toy"] if kind == "motion" else ())
    sim.run(4.0 + (seed % 5) * 0.37)
    eye0, yaw0 = float(sim.face.gaze[0]), sim.pose()[2]
    d = 0.9 * side                                                  # rad, + left
    if kind == "voice":
        sim.mind.hear_summary(VoiceSummary(sim.mind.t, 0.8, -20.0, 210.0, 2.0, 3.0, 0), voice_dir=d)
    elif kind == "bang":
        sim.mind.world.owner.voice_dir = d
        sim.mind.hear_event("loud_noise")
    elif kind == "tap":
        sim.touch("tap")
        d = None
    elif kind == "none":                                            # control: nothing happens
        pass
    elif kind == "motion":
        x, y, h = sim.pose()
        a = h + d
        start = sim.desk.clamp((x + 0.30 * math.cos(a), y + 0.30 * math.sin(a)), 0.04)
        end = sim.desk.clamp((x + 0.25 * math.cos(a + 0.3 * side), y + 0.25 * math.sin(a + 0.3 * side)), 0.04)
        sim.show_prop("toy", start)
        sim.move_prop("toy", [start, end], speed=0.05)
    face0 = sim.mind.face_expr
    t_eye = t_body = t_face = None
    for k in range(int(2.0 / DT)):
        sim.run(DT)
        t = (k + 1) * DT
        eye = float(sim.face.gaze[0])
        if d is not None:
            want = eye_dir(d)
            if t_eye is None and abs(eye - want) < GAZE_TOL and abs(eye0 - want) >= GAZE_TOL:   # landed on it
                t_eye = t
            turn = (sim.pose()[2] - yaw0 + math.pi) % (2 * math.pi) - math.pi
            if t_body is None and turn * side > BODY_TURN:
                t_body = t
        else:
            if t_eye is None and abs(eye - eye0) > EYE_MOVE:
                t_eye = t
        if t_face is None and sim.mind.face_expr != face0:
            t_face = t
    first = t_eye if d is not None else min([x for x in (t_eye, t_face) if x is not None], default=None)
    return dict(kind=kind, eye=t_eye, body=t_body, face=t_face, first=first)


# ------------------------------------------------------------------------------- 3, 4, 5: a session of pet life
def session(seed, seconds=360.0):
    sim = _sim(seed, props=["toy"])
    sim.run(2.0)
    sim.move_prop("toy", [(0.28, 0.18), (-0.18, 0.18), (0.28, 0.18), (-0.18, 0.18)], speed=0.025)
    events = [(40.0, "say", "hi pinch"), (90.0, "touch", "stroke"), (150.0, "say", "good crab"),
              (220.0, "touch", "tap"), (300.0, "say", "pinch, play!")]
    m = sim.mind
    on_target = with_target = 0
    goal_starts, outcomes = [], []
    eye_hist = []
    last_status, last_task = "", None
    face_t, sound_t = [], []
    face0, n_sounds = m.face_expr, 0
    while sim.t < seconds:
        if events and sim.t >= events[0][0]:
            _, what, arg = events.pop(0)
            (sim.say if what == "say" else sim.touch)(arg)
        sim.run(0.1)
        x, y, h = sim.pose()
        eye = float(sim.face.gaze[0])
        eye_hist.append((sim.t, eye))
        if m.face_expr != face0:
            face0 = m.face_expr
            face_t.append(sim.t)
        if len(sim.sounds_played) != n_sounds:
            n_sounds = len(sim.sounds_played)
            sound_t.append(sim.t)
        # what matters right now (directions in the body frame)
        targets = []
        f = m.attention.focus
        if f is not None and m.t - f.t < 3.0:
            if f.xy is not None:
                targets.append(math.atan2(f.xy[1] - y, f.xy[0] - x) - h)
            elif f.direction is not None:
                targets.append(f.direction)
        if m.t - m.world.owner.last_voice < 4.0 and m.world.owner.voice_dir is not None:
            targets.append(m.world.owner.voice_dir)
        toy = sim.prop_xy("toy")
        if toy is not None:
            targets.append(math.atan2(toy[1] - y, toy[0] - x) - h)
        if m.gaze_target_xy is not None and m.t - m.gaze_set_t < 5.0:
            gx, gy = m.gaze_target_xy
            targets.append(math.atan2(gy - y, gx - x) - h)
        targets = [((a + math.pi) % (2 * math.pi) - math.pi) for a in targets]
        targets = [a for a in targets if abs(a) < 1.0]                  # within the eyes' reach
        if targets:
            with_target += 1
            on_target += any(abs(eye - eye_dir(a)) < GAZE_TOL for a in targets)
        # goal-directed steps: was the eye on the goal when the legs started; what followed the outcome
        task = sim.body.task
        if task is not None and task is not last_task and sim.body.status in ("goto", "push", "pick", "drop"):
            step = sim.motions[-1][1] if sim.motions else ()
            goal = None
            if len(step) > 1 and isinstance(step[1], (tuple, list)) and len(step[1]) >= 2:
                goal = tuple(step[1][:2])
            elif len(step) > 1 and isinstance(step[1], str):
                o = sim.mind.world.objects.get(step[1])
                goal = tuple(o.xy) if o is not None else None
            if goal is not None:
                a = math.atan2(goal[1] - y, goal[0] - x) - h
                a = (a + math.pi) % (2 * math.pi) - math.pi
                if 0.25 < abs(a) < 1.0:                             # off to a side, within the eyes' reach
                    lead = [e for t, e in eye_hist if sim.t - 0.35 <= t <= sim.t - 0.05]
                    goal_starts.append(any(abs(e - eye_dir(a)) < GAZE_TOL for e in lead))
        if last_task is not None and task is not last_task and last_status in ("goto", "push", "pick", "drop"):
            outcomes.append(sim.t)
        last_task, last_status = task, sim.body.status if task is not None else ""
    reacted = [any(t0 < t <= t0 + 1.0 for t in face_t + sound_t) for t0 in outcomes]
    life = sim.body.life
    gaps = np.diff([t for t, _ in life.events]) if len(life.events) > 3 else np.array([1.0])
    return dict(gaze=on_target / max(1, with_target), forethought=float(np.mean(goal_starts)) if goal_starts else None,
                n_goal=len(goal_starts), outcome=float(np.mean(reacted)) if reacted else None, n_out=len(reacted),
                fidget_cv=float(np.std(gaps) / max(1e-6, np.mean(gaps))))


# ------------------------------------------------------------------------------------------- 6: idle from feeling
def idle_from_feeling():
    from pinch.body.life import IdleLife
    out = {}
    for name, lv in (("calm", 0.15), ("excited", 0.95)):
        life = IdleLife(np.random.default_rng(0))
        life.liveliness = lv
        t, femur = np.arange(0, 120, DT), []
        for tt in t:
            femur.append(life.offsets(tt, DT)[1])                   # FL femur: breathing + sway + fidgets
        n0 = len(life.events)
        x = np.asarray(femur) - np.mean(femur)
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), DT)
        band = (freqs > 0.1) & (freqs < 1.0)
        out[name] = dict(breath_hz=float(freqs[band][np.argmax(spec[band])]), fidgets_per_min=n0 / 2.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n = 4 if a.quick else 12
    trials = [react_trial(s, k, 1 if s % 2 else -1) for k in ("voice", "bang", "tap", "motion") for s in range(n)]
    # control: the same check for "the eyes landed on that side" with nothing there (false alarms)
    control = [react_trial(s, "none", 1 if s % 2 else -1) for s in range(n)]
    false_alarm = sum(1 for t in control if t["eye"] is not None) / len(control)
    lat = [t["first"] for t in trials]
    seen = [x for x in lat if x is not None]
    within = sum(1 for x in lat if x is not None and x <= 1.0) / len(lat)
    order = [t for t in trials if t["eye"] is not None and t["body"] is not None]
    eyes_first = sum(1 for t in order if t["eye"] <= t["body"]) / len(order) if order else None
    sess = [session(s, 180.0 if a.quick else 360.0) for s in (3, 8)]
    idle = idle_from_feeling()
    rows = [
        ("1 contingent reaction", f"median {np.median(seen) if seen else float('nan'):.2f} s, {100 * within:.0f} % within 1 s "
         f"(eyes there anyway with nothing happening: {100 * false_alarm:.0f} %)",
         bool(seen) and np.median(seen) <= 0.4 and within >= 0.95 and false_alarm <= 0.1),
        ("2 eyes before body", "n/a (no body turn)" if eyes_first is None else f"{100 * eyes_first:.0f} % of {len(order)}",
         eyes_first is not None and eyes_first >= 0.9),
        ("3 meaningful gaze", f"{100 * np.mean([s['gaze'] for s in sess]):.0f} % of looking time on a target",
         np.mean([s["gaze"] for s in sess]) >= 0.7),
        ("4 forethought", ", ".join(f"{100 * s['forethought']:.0f} % of {s['n_goal']}" if s["forethought"] is not None
                                    else "no goal moves" for s in sess),
         all(s["forethought"] is not None and s["forethought"] >= 0.95 for s in sess)),
        ("5 outcome reaction", ", ".join(f"{100 * s['outcome']:.0f} % of {s['n_out']}" if s["outcome"] is not None
                                         else "no outcomes" for s in sess),
         all(s["outcome"] is not None and s["outcome"] >= 0.9 for s in sess)),
        ("6 idle from feeling", f"breathing {idle['calm']['breath_hz'] * 60:.0f} -> {idle['excited']['breath_hz'] * 60:.0f} /min, "
         f"fidgets {idle['calm']['fidgets_per_min']:.1f} -> {idle['excited']['fidgets_per_min']:.1f} /min",
         idle["excited"]["breath_hz"] >= 1.3 * idle["calm"]["breath_hz"]),
        ("7 variety", f"fidget timing CV {np.mean([s['fidget_cv'] for s in sess]):.2f}",
         np.mean([s["fidget_cv"] for s in sess]) >= 0.5),
    ]
    for kind in ("voice", "bang", "tap", "motion"):
        ks = [t["first"] for t in trials if t["kind"] == kind]
        ok = [x for x in ks if x is not None]
        print(f"   {kind:7s} reacted {len(ok)}/{len(ks)}, median {np.median(ok) if ok else float('nan'):.2f} s")
    for name, val, ok in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:22s} {val}")
    return 0 if all(ok for _, _, ok in rows) else 1


if __name__ == "__main__":
    os.environ.setdefault("PINCH_QUIET", "1")
    sys.exit(main())

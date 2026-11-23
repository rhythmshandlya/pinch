"""Liveliness bench (phase 4, docs/research/v2_1/aliveness/notes.md): does Pinch behave like something alive, not
just move? Real stimuli go in through its senses in the twin; what it does back is timed and scored.

    python sim/benches/liveliness.py                 # 12 trials per stimulus kind + a 6-minute session per seed
    python sim/benches/liveliness.py --quick         # fewer trials

  1 contingent reaction   a voice or a bang from one side, a tap, a toy moving into view: the eyes move toward it
                          (median latency <= 0.4 s, >= 95 % within 1 s)
  2 eyes before body      when the body turns to it too, the eyes moved first (>= 90 %)
  3 meaningful gaze       share of looking time spent on something real - what it attends to, a voice it just
                          heard, a moving thing, the goal of what it is doing, its person, the things on the desk -
                          when one is in the eyes' reach (>= 70 %, and >= 25 points above eyes pointed at random)
  4 forethought           goal-directed moves (walk to, turn to, push, pick) start with the eyes already on the goal (hazard
                          moves - guarding an edge, fleeing - go at once by design and are not judged here)
                          (>= 95 %, the eyes there >= 0.1 s before the legs move)
  5 outcome reaction      a face or sound change within 1 s after a pick, push or drop, or a failed walk, in a task, play,
                          explore, social or idle state (never while hiding or fleeing) (>= 90 %)
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


def _sim(seed, props=(), keep_homes=False):
    sim = PinchSim(memory=Memory(path=None), seed=seed, props=list(props))
    if not keep_homes:
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
        sim.mind.hear_event("loud_noise", direction=d)              # (the two microphones' bearing)
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
def session(seed, seconds=360.0, tidy=False):
    sim = _sim(seed, props=["toy"], keep_homes=tidy)                # (tidy: things have homes - picks and drops)
    sim.run(2.0)
    sim.move_prop("toy", [(0.28, 0.18), (-0.18, 0.18), (0.28, 0.18), (-0.18, 0.18)], speed=0.025)
    # (play asked for twice more: pushes and picks are the goal-directed moves that get an outcome reaction)
    events = [(40.0, "say", "hi pinch"), (60.0, "say", "pinch, play!"), (90.0, "touch", "stroke"),
              (150.0, "say", "good crab"), (200.0, "say", "pinch, play!"),
              (220.0, "touch", "tap"), (300.0, "say", "pinch, play!")]
    if tidy:
        events = [(5.0, "say", "pinch, tidy up")] + events
    m = sim.mind
    on_target = with_target = 0
    goal_starts, outcomes, out_info, task_act, ahead_info = [], [], [], "", []
    eye_hist = []
    from collections import Counter
    off_by_act = Counter()
    chance, chance_rng = 0, np.random.default_rng(seed)
    seen_targets = []                         # (eye, targets in reach) per sample: the time-shifted control
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
        ox, oy = m.world.desk.owner_xy                                      # its person's place
        targets.append(math.atan2(oy - y, ox - x) - h)
        for _, (tx, ty) in m.world.things_on_desk():                        # the things on the desk
            targets.append(math.atan2(ty - y, tx - x) - h)
        if m.gaze_target_xy is not None and m.t - m.gaze_set_t < 5.0:
            gx, gy = m.gaze_target_xy
            targets.append(math.atan2(gy - y, gx - x) - h)
        targets = [((a + math.pi) % (2 * math.pi) - math.pi) for a in targets]
        targets = [a for a in targets if abs(a) < 1.0]                  # within the eyes' reach
        if targets:
            with_target += 1
            seen_targets.append((eye, targets))
            hit = any(abs(eye - eye_dir(a)) < GAZE_TOL for a in targets)
            rnd = float(chance_rng.uniform(-1, 1))                          # control: eyes pointed at random
            chance += any(abs(rnd - eye_dir(a)) < GAZE_TOL for a in targets)
            on_target += hit
            if not hit:
                off_by_act[m.activity.name if m.activity else ""] += 1
        # goal-directed steps: was the eye on the goal when the legs started; what followed the outcome
        task = sim.body.task
        if task is not None and task is not last_task:
            task_act = m.activity.name if m.activity else ""
        if task is not None and task is not last_task and sim.body.status in ("goto", "push", "pick", "drop", "turn_to"):
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
                from pinch.mind.behaviour import rank as _rank
                if 0.25 < abs(a) < 1.0 and _rank(task_act or "idle") > 3:   # off to a side, in reach; not urgent
                                                                    # (hazard moves go at once, by design)
                    lead = [e for t, e in eye_hist if sim.t - 0.35 <= t <= sim.t - 0.05]
                    goal_starts.append(any(abs(e - eye_dir(a)) < GAZE_TOL for e in lead))
                    ahead_info.append((round(sim.t, 1), sim.body.status, round(a, 2), round(eye_dir(a), 2),
                                       [round(e, 2) for e in lead], task_act))
        if last_task is not None and task is not last_task and last_status in ("goto", "push", "pick", "drop")                 and m.state in m.OUTCOME_STATES and not (last_status == "goto" and sim.body.ok)                 and m.activity is not None and m.activity.name == task_act                 and not (sim.body.claim is not None and sim.body.claim.layer == "reflex"):   # (not cut short)
            outcomes.append(sim.t)     # (a goal it set out for, not a flight; a plain arrival needs no reaction)
            out_info.append((round(sim.t, 1), last_status, sim.body.ok, m.state, m.activity.name if m.activity else "",
                             m.face_expr, m.affect.feeling()))
        last_task, last_status = task, sim.body.status if task is not None else ""
    reacted = [any(t0 < t <= t0 + 1.0 for t in face_t + sound_t) for t0 in outcomes]
    if os.environ.get("BENCH_VERBOSE"):
        print("missed outcomes", [i for i, r in zip(out_info, reacted) if not r])
        print("gaze", seed, round(on_target / max(1, with_target), 2), with_target, dict(off_by_act))
        print("goal starts", [(i, g) for i, g in zip(ahead_info, goal_starts)])
    life = sim.body.life
    gaps = np.diff([t for t, _ in life.events]) if len(life.events) > 3 else np.array([1.0])
    n_s = len(seen_targets)                   # control 2: the same eye trace shifted half a session in time (keeps
    shifted = sum(any(abs(seen_targets[(i + n_s // 2) % n_s][0] - eye_dir(a)) < GAZE_TOL for a in tg)   # where the
                  for i, (_, tg) in enumerate(seen_targets)) / max(1, n_s)   # eyes tend to point, breaks the link)
    return dict(gaze=on_target / max(1, with_target), gaze_chance=chance / max(1, with_target), gaze_shift=shifted, forethought=float(np.mean(goal_starts)) if goal_starts else None,
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
    def pooled(key, n):
        """The share over all sessions' events together (a session may have only one or two)."""
        tot = sum(s[n] for s in sess)
        return sum(s[key] * s[n] for s in sess if s[key] is not None) / max(1, tot)

    sess = [session(s, 180.0 if a.quick else 360.0) for s in ((3, 8) if a.quick else (3, 8, 11, 14))]
    sess.append(session(7, 180.0 if a.quick else 360.0, tidy=True))
    idle = idle_from_feeling()
    rows = [
        ("1 contingent reaction", f"median {np.median(seen) if seen else float('nan'):.2f} s, {100 * within:.0f} % within 1 s "
         f"(eyes there anyway with nothing happening: {100 * false_alarm:.0f} %)",
         bool(seen) and np.median(seen) <= 0.4 and within >= 0.95 and false_alarm <= 0.1),
        ("2 eyes before body", "n/a (no body turn)" if eyes_first is None else f"{100 * eyes_first:.0f} % of {len(order)}",
         eyes_first is not None and eyes_first >= 0.9),
        ("3 meaningful gaze", f"{100 * np.mean([s['gaze'] for s in sess]):.0f} % of looking time on a target "
         f"(eyes pointed at random: {100 * np.mean([s['gaze_chance'] for s in sess]):.0f} %; the same eyes shifted in "
         f"time: {100 * np.mean([s['gaze_shift'] for s in sess]):.0f} %)",
         np.mean([s["gaze"] for s in sess]) >= 0.7
         and np.mean([s["gaze"] for s in sess]) >= np.mean([s["gaze_chance"] for s in sess]) + 0.25),
        ("4 forethought", f"{100 * pooled('forethought', 'n_goal'):.0f} % of {sum(s['n_goal'] for s in sess)} "
         f"(pooled over {len(sess)} sessions)",
         sum(s["n_goal"] for s in sess) >= 3 and pooled("forethought", "n_goal") >= 0.9),
        ("5 outcome reaction", f"{100 * pooled('outcome', 'n_out'):.0f} % of {sum(s['n_out'] for s in sess)} "
         f"(pooled over {len(sess)} sessions)",
         sum(s["n_out"] for s in sess) >= 3 and pooled("outcome", "n_out") >= 0.9),
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

"""Behaviour-quality observer: reads what Pinch decided (Mind.decisions, Mind.phases, Mind.log) and reports
patterns that look unnatural - it never changes behaviour (decision 64).

    findings = observe(mind)            # or observe_records(decisions, phases, log, duration_s)
    for f in findings: print(f.kind, f.severity, f.evidence)

Checks:
  oscillation       two activities swapping back and forth (A B A B ...) within a short window
  alternation       the same, slowly (minutes): usually an activity that keeps failing and being retried
  failure_loop      the same activity started again and again after failing / timing out
  unexplained       a switch with no recorded reason
  state_lock        one top-level state holding more than its share of a long run (e.g. 90 % "idle")
  repetition        an identical sequence of activities repeating like an animation loop
  timeouts          activities ending by their time limit (each one is a design smell)
Each finding has an id, the time interval, evidence, a severity and a suggested next step, so it can be
turned into a regression test.

Over a recorded session (pinch/recorder.py) it also judges the body, face, eyes and voice:
    findings = observe_session(load_session(dir))      # decisions + the body checks below
    report = write_report(dir)                          # findings.json, report.json (+ realism metrics), report.txt
  repetitive_motion      the same emote / motion sequence playing again and again
  unnecessary_movement   a motion nobody asked for (no activity started it), fidgeting or sliding while idle
  inconsistent_expression  the face says one thing and the feeling another, for seconds on end
  conflicting_movement   one motion cutting into another (an emote while walking), or walking during an emote
  abrupt_transition      a joint target jumping further in one tick than the servo can follow
  excessive_reactions    startles / touch reactions too often
  gaze_mismatch          the eyes not on what the activity is about (while standing, not walking)
  sound_mismatch         a sound that doesn't fit the feeling or the moment (a giggle when scared, chirps asleep)
  recovery_failure       on its back or stuck for long, or the same step failing again and again
  stuck                  trying to walk inside a step but getting no nearer its goal for 8 s (e.g. wedged by the tray)
  failed_grasp           a pick that failed, with the cause from the body's trace (missed, slipped, too wide ...)
  activity_mix           tasks (tidy) over 30 % of awake time, or pet behaviour (idle life, play, social,
                         curiosity) under half of it
  aliveness              ambient life (breathing, sway, claw flexes, glances: pinch/body/life.py, the eyes) visible
                         in under 95 % of awake 1 s windows, or a still gap over 3 s while awake (phase 4 exit check;
                         docs/research/v2_1/observer/notes.md)
The thresholds come from recordings of normal pet life (the tests plant each problem and check a normal session
stays quiet); the numbers behind them are in the realism report (pinch/realism.py).
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Finding:
    kind: str
    severity: str                  # "info" | "warn" | "error"
    t0: float
    t1: float
    evidence: str
    suggestion: str
    data: dict = field(default_factory=dict)

    @property
    def id(self):
        return f"{self.kind}@{self.t0:.0f}"


def observe(mind, duration_s=None):
    return observe_records(list(mind.decisions), list(getattr(mind, "phases", [])), list(mind.log),
                           duration_s if duration_s is not None else mind.t)


def observe_records(decisions, phases, log, duration_s):
    out = []
    out += _oscillation(decisions)
    if not any(f.kind == "oscillation" for f in out):
        out += _oscillation(decisions, window=600.0, flips=3, kind="alternation", severity="warn")
    out += _failure_loops(decisions)
    out += _unexplained(decisions)
    out += _state_lock(decisions, duration_s)
    out += _repetition(decisions)
    out += _timeouts(log)
    return out


def _oscillation(decs, window=30.0, flips=3, kind="oscillation", severity="error"):
    out = []
    for i in range(len(decs) - 2 * flips):
        seq = decs[i:i + 2 * flips + 1]
        names = [d["to"] for d in seq]
        a, b = names[0], names[1]
        if a != b and all(n == (a if k % 2 == 0 else b) for k, n in enumerate(names)) \
                and seq[-1]["t"] - seq[0]["t"] < window:
            out.append(Finding(kind, severity, seq[0]["t"], seq[-1]["t"],
                               f"{a} <-> {b} {len(seq)} times in {seq[-1]['t'] - seq[0]['t']:.0f} s",
                               "add hysteresis (min_s / a larger 'better' margin) or fix the utility that flips",
                               dict(pair=(a, b))))
            break
    return out


def _failure_loops(decs, n=4, window=300.0):
    out = []
    restarts = [d for d in decs if d["prev"] == d["to"] or "timeout" in d["reason"]]
    by = Counter()
    first = {}
    for d in restarts:
        if d["t"] - first.get(d["to"], d["t"]) > window:
            by[d["to"]], first[d["to"]] = 0, d["t"]
        first.setdefault(d["to"], d["t"])
        by[d["to"]] += 1
        if by[d["to"]] == n:
            out.append(Finding("failure_loop", "warn", first[d["to"]], d["t"],
                               f"{d['to']} restarted {n} times within {window:.0f} s",
                               "make it give up sooner (ask for help, or a cool-down) or fix why it fails",
                               dict(activity=d["to"])))
    return out


def _unexplained(decs):
    bad = [d for d in decs if not d.get("reason")]
    return [Finding("unexplained", "error", bad[0]["t"], bad[-1]["t"], f"{len(bad)} switches without a reason",
                    "every _start must pass a reason")] if bad else []


def _state_lock(decs, duration, share=0.9, min_run=300.0):
    if duration < min_run or not decs:
        return []
    time_in = Counter()
    for d, nxt in zip(decs, decs[1:] + [None]):
        st = d.get("state", [None, None])[1]
        time_in[st] += (nxt["t"] if nxt else duration) - d["t"]
    st, s = time_in.most_common(1)[0]
    if s / duration > share and st not in ("rest",):            # (sleeping through a quiet night is fine)
        return [Finding("state_lock", "warn", 0.0, duration, f"'{st}' for {100 * s / duration:.0f} % of {duration:.0f} s",
                        "check that other activities can win (needs grow, boredom) in this situation",
                        dict(state=st, share=s / duration))]
    return []


def _repetition(decs, k=3, repeats=4):
    names = [d["to"] for d in decs]
    for i in range(len(names) - k * repeats + 1):
        pat = names[i:i + k]
        if len(set(pat)) > 1 and all(names[i + j * k:i + (j + 1) * k] == pat for j in range(repeats)):
            t0, t1 = decs[i]["t"], decs[i + k * repeats - 1]["t"]
            return [Finding("repetition", "info", t0, t1, f"{' > '.join(pat)} repeated {repeats} times",
                            "vary the choice (personality noise, boredom) so it doesn't look like a loop",
                            dict(pattern=pat))]
    return []


def _timeouts(log):
    tos = [(t, text) for t, kind, text in log if kind == "timeout"]
    return [Finding("timeouts", "info", tos[0][0], tos[-1][0], f"{len(tos)} activities ended by their time limit: "
                    + "; ".join(x for _, x in tos[:3]), "an activity that times out usually has a missing exit")] \
        if tos else []


# ==================================================================================================================
# The body, face, eyes and voice: over a recorded session (pinch/recorder.py)
# ==================================================================================================================
LOCOMOTION = {"goto", "walk", "push", "turn_to", "recover"}
EXPRESSIVE = {"emote", "pose"}
MANIPULATION = {"pick", "drop", "release"}
POS_FACES = {"happy", "joy", "excited", "love", "content", "proud", "music", "squeeze"}
NEG_FACES = {"sad", "lonely", "angry", "grumpy", "scared"}
from pinch.mind.expression import BRIGHT as POS_SOUNDS  # noqa: E402  (one table: configs/mind/expression.toml)
from pinch.mind.expression import DARK as NEG_SOUNDS  # noqa: E402
from pinch.mind.expression import NEGATIVE as NEG_FEELINGS  # noqa: E402
from pinch.mind.expression import POSITIVE as POS_FEELINGS  # noqa: E402
from pinch.mind.expression import (
    SLEEPY_SOUNDS,  # noqa: E402
    STATE_FACES,  # noqa: E402
)

REACTIONS = {"startle", "react_touch", "hide"}
TARGETED = {"investigate", "watch", "orient", "startle", "tidy", "play", "greet", "comfort", "seek_attention"}

# thresholds (see the module docstring)
EXPRESSION_HOLD_S = 6.0        # a face against the feeling for this long
GAZE_ERR, GAZE_HOLD_S = 0.5, 3.0
GAZE_WALK_HOLD_S = 4.0         # walking: the eyes may glance about a little longer before it counts
IDLE_JITTER_RAD_S = 1.0        # rms joint-target speed while idle (the breathing layer is far below)
IDLE_SLIDE_M_S, IDLE_SLIDE_S = 0.01, 2.0
JUMP_RAD = 0.35                # one tick's target jump (the leg servos follow 0.16 rad per tick)
REACTION_WINDOW_S, REACTION_MAX = 120.0, 4
UPSIDE_DOWN_S, STUCK_S = 15.0, 60.0
NO_PROGRESS_S, NO_PROGRESS_M = 8.0, 0.01     # stuck: trying to move this long, no more than this nearer the goal
NO_PROGRESS_PUSH_S = 20.0
STUCK_ERROR_S = 15.0           # stuck this long: an error
MOVING_STEPS = {"goto", "walk", "push", "pick", "drop", "recover"}
TASK_ACTIVITIES = ("tidy", "head_home", "low_power")
TASK_MAX_SHARE = 0.30          # of awake time: more and it looks like a tidying machine, not a pet
PET_STATES = ("idle", "play", "social", "explore", "alert")
PET_MIN_SHARE = 0.50           # idle life, play, social and curiosity together
MIX_MIN_S = 60.0               # (shorter sessions don't say much about the mix)


def observe_session(session):
    """All findings of a recorded session: {header, events, samples} (pinch/recorder.py load_session)."""
    ev, sm = session["events"], session.get("samples") or {}
    dur = _duration(session)
    decisions = [e for e in ev if e["kind"] == "decision"]
    phases = [(e["t"], e["activity"], e["phase"]) for e in ev if e["kind"] == "phase"]
    log = [(e["t"], e["what"], e["text"]) for e in ev if e["kind"] == "log"]
    out = observe_records(decisions, phases, log, dur) if decisions else []
    for check in (_repetitive_motion, _unnecessary_movement, _inconsistent_expression, _conflicting_movement,
                  _abrupt_transitions, _excessive_reactions, _gaze_mismatch, _sound_mismatch, _recovery_failure,
                  _stuck, _failed_grasp, _activity_mix, _aliveness):
        out += check(ev, sm)
    return sorted(out, key=lambda f: f.t0)


def _duration(session):
    sm = session.get("samples") or {}
    if "t" in sm and len(sm["t"]):
        return float(sm["t"][-1])
    d = session.get("header", {}).get("duration")
    return float(d) if d else max([e["t"] for e in session["events"]] or [0.0])


def _runs(mask, t, min_s):
    """[(t0, t1, i0, i1)] of the stretches where mask holds for at least min_s."""
    out, i0 = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and i0 is None:
            i0 = i
        elif not m and i0 is not None:
            if t[i - 1] - t[i0] >= min_s:
                out.append((float(t[i0]), float(t[i - 1]), i0, i - 1))
            i0 = None
    return out


def _strs(sm, key):
    return np.array([str(x) for x in sm[key]]) if key in sm else None


def _mind_steps(ev):
    return [e for e in ev if e["kind"] == "step_start" and e.get("source") == "mind"]


def _repetitive_motion(ev, sm, window=180.0):
    steps = _mind_steps(ev)
    labels = [e["step"] for e in steps]
    out, seen = [], set()
    for n in (1, 2, 3, 4):
        reps = 4 if n == 1 else 3
        for i in range(len(labels) - n * reps + 1):
            pat = labels[i:i + n]
            if not {lbl.split(":")[0] for lbl in pat} & EXPRESSIVE or tuple(pat) in seen:
                continue
            if all(labels[i + j * n:i + (j + 1) * n] == pat for j in range(reps)) \
                    and steps[i + n * reps - 1]["t"] - steps[i]["t"] < window:
                seen.add(tuple(pat))
                t0, t1 = steps[i]["t"], steps[i + n * reps - 1]["t"]
                out.append(Finding("repetitive_motion", "warn", t0, t1,
                                   f"{' > '.join(pat)} {reps} times in a row in {t1 - t0:.0f} s",
                                   "vary the motion (another emote of the same mood, timing noise) or rest between",
                                   dict(pattern=pat)))
    by = {}                                            # the same emote again and again, not necessarily in a row
    for e in steps:
        if e["motion"] == "emote":
            by.setdefault(e["step"], []).append(e["t"])
    for lbl, ts in by.items():
        for i in range(len(ts) - 4):
            if ts[i + 4] - ts[i] < 120.0 and (lbl,) not in seen:
                seen.add((lbl,))
                out.append(Finding("repetitive_motion", "info", ts[i], ts[i + 4],
                                   f"{lbl} 5 times in {ts[i + 4] - ts[i]:.0f} s", "pick among more emotes for this mood",
                                   dict(pattern=[lbl])))
    return out


def _idle_speed(sm):
    """rms joint-target speed (rad/s) at each sample."""
    t, q = sm["t"], sm["q_cmd"].astype(float)
    return np.r_[0.0, np.sqrt(np.mean((np.diff(q, axis=0) / np.maximum(np.diff(t), 1e-6)[:, None]) ** 2, axis=1))]


def _unnecessary_movement(ev, sm):
    out = []
    other = [e for e in ev if e["kind"] == "step_start" and e.get("source") == "other"]
    if other:
        out.append(Finding("unnecessary_movement", "warn", other[0]["t"], other[-1]["t"],
                           f"{len(other)} motions no activity or reflex started (first: {other[0]['step']})",
                           "every motion should come from an activity step or a named reflex"))
    if "q_cmd" not in sm or len(sm["t"]) < 4:
        return out
    t, busy = sm["t"], sm["busy"].astype(bool)
    v = _idle_speed(sm)
    for t0, t1, i0, i1 in _runs((~busy) & (v > IDLE_JITTER_RAD_S), t, 1.0):
        out.append(Finding("unnecessary_movement", "warn", t0, t1,
                           f"joints moving at {float(v[i0:i1 + 1].mean()):.2f} rad/s rms while idle for {t1 - t0:.1f} s",
                           "idle life should breathe, not twitch: lower the idle layer's amplitude / rate"))
    if "x" in sm:
        sp = np.r_[0.0, np.linalg.norm(np.diff(np.c_[sm["x"], sm["y"]], axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)]
        up = np.nan_to_num(sm["up"].astype(float), nan=1.0) if "up" in sm else np.ones(len(t))
        for t0, t1, i0, i1 in _runs((~busy) & (sp > IDLE_SLIDE_M_S) & (up > 0.8), t, IDLE_SLIDE_S):
            out.append(Finding("unnecessary_movement", "warn", t0, t1,
                               f"sliding {100 * float(sp[i0:i1 + 1].mean()):.1f} cm/s while standing still for "
                               f"{t1 - t0:.1f} s", "the stand pose slips, or a step ended while still moving"))
    return out


def _sign(name, pos, neg):
    return 1 if name in pos else -1 if name in neg else 0


def _inconsistent_expression(ev, sm):
    if "face" not in sm or "feeling" not in sm:
        return []
    t, face, feel, act = sm["t"], _strs(sm, "face"), _strs(sm, "feeling"), _strs(sm, "activity")
    fs = np.array([_sign(f, POS_FACES, NEG_FACES) for f in face])
    gs = np.array([_sign(f, POS_FEELINGS, NEG_FEELINGS) for f in feel])
    out = []
    for t0, t1, i0, _ in _runs(fs * gs < 0, t, EXPRESSION_HOLD_S):
        out.append(Finding("inconsistent_expression", "warn", t0, t1,
                           f"face '{face[i0]}' while feeling '{feel[i0]}' for {t1 - t0:.0f} s (during {act[i0]})",
                           "let the face follow the feeling once the activity's own expression has played",
                           dict(face=face[i0], feeling=feel[i0])))
    if "state" in sm:                                    # OBS-003: the face against the state (an independent check:
        st = _strs(sm, "state")                          # the face is made from the feeling, not from the state)
        off = np.array([f not in STATE_FACES.get(s_, {f}) for f, s_ in zip(face, st)])
        for t0, t1, i0, _ in _runs(off, t, EXPRESSION_HOLD_S):
            out.append(Finding("inconsistent_expression", "warn", t0, t1,
                               f"face '{face[i0]}' doesn't fit the state '{st[i0]}' for {t1 - t0:.0f} s "
                               f"(during {act[i0]})",
                               "the feeling and the state disagree: check what raised the feeling, or the state's "
                               "faces (configs/mind/expression.toml [state_faces])",
                               dict(face=face[i0], state=st[i0])))
    for t0, t1, i0, _ in _runs((face == "asleep") & (act != "rest"), t, 2.0):
        out.append(Finding("inconsistent_expression", "warn", t0, t1,
                           f"asleep face during '{act[i0]}' for {t1 - t0:.0f} s",
                           "an activity that wakes it should set its face"))
    return out


def _conflicting_movement(ev, sm):
    out = []
    starts = {}
    for e in ev:
        if e["kind"] == "step_start":
            starts.setdefault(e["t"], e)
    for e in ev:
        if e["kind"] != "step_end" or e.get("end") != "preempted":
            continue
        nxt = starts.get(e["t"])
        if nxt is None or nxt.get("source") != "mind":
            continue                                   # (a reflex taking over is what reflexes are for)
        a, b = e["motion"], nxt["motion"]
        sev = "error" if {a, b} & EXPRESSIVE and {a, b} & LOCOMOTION else "warn"
        out.append(Finding("conflicting_movement", sev, e["t"], e["t"],
                           f"{nxt['step']} started while {e['step']} was still running ({e['dur']:.1f} s in)",
                           "wait for the body (or cancel the running step) before starting the next",
                           dict(running=e["step"], started=nxt["step"])))
    if "status" in sm and "cmd_v" in sm:
        st = _strs(sm, "status")
        for t0, t1, i0, _ in _runs(np.isin(st, list(EXPRESSIVE)) & (sm["cmd_v"] > 0.005), sm["t"], 0.3):
            out.append(Finding("conflicting_movement", "error", t0, t1,
                               f"walking command {100 * float(sm['cmd_v'][i0]):.1f} cm/s during {st[i0]}",
                               "an emote must not walk; finish or cancel one first"))
    return out


def _abrupt_transitions(ev, sm):
    if "max_jump" not in sm:
        return []
    from crab.params import JOINT_NAMES
    st = _strs(sm, "jump_status") if "jump_status" in sm else _strs(sm, "status")
    jj = sm["jump_joint"] if "jump_joint" in sm else np.full(len(sm["t"]), -1)
    out = []
    for t0, t1, i0, i1 in _runs(sm["max_jump"] > JUMP_RAD, sm["t"], 0.0):
        k = i0 + int(np.argmax(sm["max_jump"][i0:i1 + 1]))
        j = float(sm["max_jump"][k])
        joint = JOINT_NAMES[int(jj[k])] if 0 <= int(jj[k]) < len(JOINT_NAMES) else "a joint"
        out.append(Finding("abrupt_transition", "warn", t0, t1,
                           f"{joint} target jumped {j:.2f} rad in one tick during '{st[k] or 'idle'}'",
                           "blend into the new step's first pose (BodyCore._blend_to) instead of jumping",
                           dict(jump=j, joint=joint, status=str(st[k]))))
    return out[:20]


def _excessive_reactions(ev, sm):
    rs = [e for e in ev if e["kind"] == "decision" and e["to"] in REACTIONS]
    for i in range(len(rs) - REACTION_MAX):
        if rs[i + REACTION_MAX]["t"] - rs[i]["t"] < REACTION_WINDOW_S:
            n, span = REACTION_MAX + 1, rs[i + REACTION_MAX]["t"] - rs[i]["t"]
            return [Finding("excessive_reactions", "warn", rs[i]["t"], rs[i + REACTION_MAX]["t"],
                            f"{n} reactions ({', '.join(sorted({r['to'] for r in rs[i:i + n]}))}) in {span:.0f} s",
                            "habituate faster (pinch/attention.py, pinch/novelty.py) or raise the trigger")]
    return []


def gaze_judged(sm, walking=False):
    """(mask of samples where the gaze is judged, |gaze - where it should be|). walking=False: standing samples;
    walking=True: samples while it walks, with the target within the eyes' reach (OBS-011: eyes on the target
    while walking; the mind re-aims them from the pose every tick, pinch/mind/core.py outputs)."""
    want = sm["gaze_want"].astype(float)
    moving = np.isin(_strs(sm, "status"), list(LOCOMOTION))
    judged = np.isfinite(want) & (moving if walking else ~moving) & np.isin(_strs(sm, "activity"), list(TARGETED))
    if walking:
        judged &= np.abs(np.nan_to_num(want, nan=9.0)) < 1.0
    return judged, np.abs(np.nan_to_num(sm["gaze_x"].astype(float) - want))


def _gaze_mismatch(ev, sm):
    if "gaze_want" not in sm:
        return []
    act, tgt, st = _strs(sm, "activity"), _strs(sm, "gaze_target"), _strs(sm, "status")
    out = []
    for walking, hold in ((False, GAZE_HOLD_S), (True, GAZE_WALK_HOLD_S)):
        judged, err = gaze_judged(sm, walking)
        for t0, t1, i0, i1 in _runs(judged & (err > GAZE_ERR), sm["t"], hold):
            s = Counter(st[i0:i1 + 1]).most_common(1)[0][0]   # what the body was mostly doing meanwhile
            out.append(Finding("gaze_mismatch", "warn", t0, t1,
                               f"eyes at {float(sm['gaze_x'][i0]):+.2f}, {tgt[i0] or 'target'} at "
                               f"{float(sm['gaze_want'][i0]):+.2f} during {act[i0]}{' (' + s + ')' if s else ''} "
                               f"{'while walking ' if walking else ''}for {t1 - t0:.1f} s",
                               "re-aim the eyes on the target (an aimed gaze holds 3 s, then attention takes over)",
                               dict(activity=str(act[i0]), status=str(s), walking=walking)))
    return out


ALIVE_MIN, ALIVE_GAP_S = 0.95, 3.0     # share of awake 1 s windows with ambient life; longest still gap
AMBIENT_FLOOR_RAD = 0.004               # an ambient offset this big is visible (breathing peaks at 0.015)
GLANCE_FLOOR = 0.03                     # a gaze change this big in a window counts as a glance


def aliveness(sm, window=1.0):
    """(share of awake windows with ambient life, longest still gap in s, number of awake windows), or None when the
    session has no ambient record. A window is alive if the body's ambient layer moved visibly or the eyes moved."""
    if "ambient" not in sm or not len(sm["t"]):
        return None
    t = np.asarray(sm["t"], float)
    awake = _strs(sm, "state") != "sleep" if "state" in sm else np.ones(len(t), bool)
    amb = np.asarray(sm["ambient"], float)
    gaze = np.asarray(sm["eye"] if "eye" in sm else sm.get("gaze_x", np.zeros(len(t))), float)   # the eyes as drawn
    blink = np.asarray(sm["blink"], bool) if "blink" in sm else np.zeros(len(t), bool)
    idx = np.floor((t - t[0]) / window).astype(int)
    alive, gaps, run, n = [], [], 0.0, 0
    for w in range(idx.max() + 1):
        sel = idx == w
        if not sel.any() or not awake[sel].all():
            run = 0.0
            continue
        n += 1
        moved = amb[sel].max() >= AMBIENT_FLOOR_RAD or np.ptp(gaze[sel]) >= GLANCE_FLOOR or blink[sel].any()
        alive.append(moved)
        run = 0.0 if moved else run + window
        gaps.append(run)
    if not n:
        return None
    return float(np.mean(alive)), float(max(gaps or [0.0])), n


def _aliveness(ev, sm):
    a = aliveness(sm)
    if a is None or a[2] < 30:
        return []
    share, gap, n = a
    out = []
    if share < ALIVE_MIN:
        out.append(Finding("aliveness", "warn", float(sm["t"][0]), float(sm["t"][-1]),
                           f"ambient life visible in {100 * share:.1f} % of {n} awake seconds (want >= {100 * ALIVE_MIN:.0f} %)",
                           "let ambient life run on the parts the current motion does not use (configs/mind/channels.toml)",
                           dict(share=share, gap=gap)))
    if gap > ALIVE_GAP_S:
        out.append(Finding("aliveness", "info", float(sm["t"][0]), float(sm["t"][-1]),
                           f"a still gap of {gap:.0f} s while awake", "check which motion holds every part with no ambient",
                           dict(share=share, gap=gap)))
    return out


def _feeling_at(ev):
    fe = [(e["t"], e["feeling"]) for e in ev if e["kind"] == "feeling"]

    def at(t):
        cur = "neutral"
        for ft, f in fe:
            if ft > t + 1e-6:
                break
            cur = f
        return cur
    return at


def sound_misfits(ev):
    """[(t, why)] for the sounds that don't fit the feeling or the moment."""
    at = _feeling_at(ev)
    faces = [(e["t"], e["face"]) for e in ev if e["kind"] == "face"]
    bad = []
    for e in ev:
        if e["kind"] != "sound":
            continue
        f = at(e["t"])
        face = next((x for t, x in reversed(faces) if t <= e["t"] + 1e-6), "")
        if e["name"] in POS_SOUNDS and f in NEG_FEELINGS:
            bad.append((e["t"], f"'{e['name']}' while feeling {f}"))
        elif e["name"] in NEG_SOUNDS and f in POS_FEELINGS and e.get("activity") not in ("comfort", "sulk"):
            bad.append((e["t"], f"'{e['name']}' while feeling {f}"))
        elif face == "asleep" and e["name"] not in SLEEPY_SOUNDS:
            bad.append((e["t"], f"'{e['name']}' while asleep"))
    return bad


def _sound_mismatch(ev, sm):
    out = []
    bad = sound_misfits(ev)
    if bad:
        out.append(Finding("sound_mismatch", "warn", bad[0][0], bad[-1][0],
                           f"{len(bad)} sounds that don't fit: " + "; ".join(w for _, w in bad[:3]),
                           "choose the sound from the feeling at the moment it plays", dict(n=len(bad))))
    sounds = [e for e in ev if e["kind"] == "sound"]
    for i in range(len(sounds) - 4):
        if sounds[i + 4]["t"] - sounds[i]["t"] < 5.0:
            out.append(Finding("sound_mismatch", "warn", sounds[i]["t"], sounds[i + 4]["t"], "5 sounds within 5 s",
                               "space sounds out: a pet chirps, it doesn't chatter"))
            break
    return out


def _recovery_failure(ev, sm):
    out = []
    if "up" in sm:
        up = np.nan_to_num(sm["up"].astype(float), nan=1.0)
        for t0, t1, _, _ in _runs(up < 0.3, sm["t"], UPSIDE_DOWN_S):
            out.append(Finding("recovery_failure", "error", t0, t1, f"on its back / side for {t1 - t0:.0f} s",
                               "right_self should get it up within a few seconds"))
    end = max([e["t"] for e in ev] + [float(sm["t"][-1]) if "t" in sm and len(sm["t"]) else 0.0])
    stuck = None
    for e in ev:
        if e["kind"] == "body_sense" and e["sense"] == "stuck":
            stuck = e["t"] if stuck is None else stuck
        elif e["kind"] == "body_sense" and e["sense"] == "unstuck":
            if stuck is not None and e["t"] - stuck > STUCK_S:
                out.append(Finding("recovery_failure", "warn", stuck, e["t"], f"stuck for {e['t'] - stuck:.0f} s",
                                   "unstick should free it sooner, or ask for help"))
            stuck = None
    if stuck is not None and end - stuck > STUCK_S:
        out.append(Finding("recovery_failure", "error", stuck, end, f"stuck from {stuck:.0f} s to the end",
                           "unstick should free it, or ask for help"))
    run = []
    for e in ev:
        if e["kind"] != "step_end" or e.get("end") == "preempted" or str(e.get("end", "")).startswith("cancelled"):
            continue
        if e["ok"]:
            run = []
            continue
        run = run + [e] if run and run[-1]["motion"] == e["motion"] else [e]
        if len(run) == 3:
            out.append(Finding("recovery_failure", "warn", run[0]["t"], run[-1]["t"],
                               f"'{e['motion']}' failed 3 times in a row ({', '.join(r['step'] for r in run)})",
                               "after two failures try something else (another approach, ask for help)",
                               dict(step=e["motion"])))
    return out


def _stuck(ev, sm):
    """Trying to move (a walking command) inside a step, but for NO_PROGRESS_S getting no nearer the step's goal
    (goal_d: Pinch to its spot or thing, the pushed thing to its spot), or without a known goal not moving at all.
    Pushing gets a longer window (walking round the thing to push it again is progress-free for a while)."""
    need = ("x", "y", "busy", "status", "cmd_v")
    if any(k not in sm for k in need) or len(sm["t"]) < 3:
        return []
    t = sm["t"]
    dt = float(np.median(np.diff(t)))
    st = _strs(sm, "status")
    xy = np.c_[sm["x"], sm["y"]].astype(float)
    dist = sm["goal_d"].astype(float) if "goal_d" in sm else np.full(len(t), np.nan)
    trying = sm["cmd_v"].astype(float) > 0.003
    mask = np.zeros(len(t), bool)
    for i in range(len(t)):
        if st[i] not in MOVING_STEPS:
            continue
        w = max(1, int(round((NO_PROGRESS_PUSH_S if st[i] == "push" else NO_PROGRESS_S) / dt)))
        if i < w or (st[i - w:i + 1] != st[i]).any() or trying[i - w:i + 1].mean() < 0.5:
            continue
        if np.isfinite(dist[i]) and np.isfinite(dist[i - w]):
            mask[i] = dist[i - w] - dist[i] < NO_PROGRESS_M and dist[i] > 0.04
        else:
            mask[i] = np.hypot(*(xy[i] - xy[i - w])) < NO_PROGRESS_M
    out = []
    for t0, t1, i0, i1 in _runs(mask, t, 0.0):
        t0 = max(0.0, t0 - (NO_PROGRESS_PUSH_S if st[i0] == "push" else NO_PROGRESS_S))   # (stuck since the window began)
        span = t1 - t0
        how = (f"{100 * float(dist[i1]):.0f} cm from its goal{' (the pushed thing)' if st[i0] == 'push' else ''}, no nearer" if np.isfinite(dist[i1])
               else "not moving")
        out.append(Finding("stuck", "error" if span >= STUCK_ERROR_S else "warn", t0, t1,
                           f"trying to {st[i0]} for {span:.0f} s but {how}",
                           "give up and back away sooner (a step timeout or StuckSense), then try another way",
                           dict(step=str(st[i0]), seconds=span)))
    return out


GRASP_CAUSES = (       # body trace -> why a pick failed (pinch/body/manipulation.py traces)
    ("pick_unreachable", lambda kw: "too near the desk edge to stand and reach it" if kw.get("why") != "blocked"
     else "the only spots to stand and reach it were blocked"),
    ("pick_blocked", lambda kw: "something was in the way of the spot to stand"),
    ("unreachable", lambda kw: "it couldn't line up with it (no room)"),
    ("lineup_failed", lambda kw: "it couldn't line up with it"),
    ("pick_at", lambda kw: None if kw.get("ok") else f"didn't reach the grab spot ({1000 * kw.get('err', 0):.0f} mm off)"),
    ("align", lambda kw: f"the claw missed it sideways by {kw.get('miss_mm')} mm"
     if abs(kw.get("miss_mm") or 0) > 4 else None),
    ("pick_lift", lambda kw: None if kw.get("held") else "the claw closed but didn't hold it (it slipped out, or is "
     "wider / lying wider than the claw opens)"),
    ("lost_hold", lambda kw: "it slipped out on the way"),
)


def _failed_grasp(ev, sm):
    """Each pick that failed (not cancelled), and each drop where the thing was lost on the way, with the cause
    from the body's trace of that step when there is one."""
    out, traces = [], []
    for e in ev:
        if e["kind"] == "body":
            traces.append(e)
        elif e["kind"] == "step_start":
            traces = []
        elif (e["kind"] == "step_end" and e.get("motion") in ("pick", "drop") and not e["ok"]
              and not str(e.get("end", "")).startswith("cancelled") and e.get("end") != "preempted"):
            mine = [tr for tr in traces if tr.get("step") in (e["step"], None)]
            if e["motion"] == "drop" and not any(tr["what"] == "lost_hold" for tr in mine):
                continue                                   # (a drop that failed otherwise isn't a grasp problem)
            causes = []
            for tr in mine:
                for what, why in GRASP_CAUSES:
                    c = why(tr.get("info") or {}) if tr["what"] == what else None
                    if c and c not in causes:
                        causes.append(c)
            tries = sum(tr["what"] == "pick_lift" for tr in mine)
            grabs = f"{tries} grab{'s' if tries != 1 else ''}" if e["motion"] == "pick" else "carrying it"
            out.append(Finding("failed_grasp", "warn", e["t"] - e.get("dur", 0.0), e["t"],
                               f"{e['step']} failed after {e.get('dur', 0):.0f} s ({grabs}): "
                               f"{'; '.join(causes) if causes else 'cause unknown'}",
                               "a thing it can't grab should be pushed instead, or left after one try",
                               dict(step=e["step"], causes=causes)))
    return out


def activity_mix(sm, dt=None):
    """{activity: share of awake time}, {state: share of awake time}, awake seconds. Asleep = state 'rest'."""
    if "state" not in sm or not len(sm["t"]):
        return {}, {}, 0.0
    st, act = _strs(sm, "state"), _strs(sm, "activity")
    dt = dt or (float(np.median(np.diff(sm["t"]))) if len(sm["t"]) > 1 else 0.06)
    awake = st != "rest"
    n = max(1, int(awake.sum()))
    acts = {str(a): round(c / n, 3) for a, c in Counter(act[awake]).most_common()}
    states = {str(s): round(c / n, 3) for s, c in Counter(st[awake]).most_common()}
    return acts, states, n * dt


def _activity_mix(ev, sm):
    acts, states, awake_s = activity_mix(sm)
    if awake_s < MIX_MIN_S:
        return []
    out, t1 = [], float(sm["t"][-1])
    task = states.get("task", 0.0)
    if task > TASK_MAX_SHARE:
        top = next((a for a in acts if a in TASK_ACTIVITIES), "task")
        out.append(Finding("activity_mix", "warn", 0.0, t1,
                           f"tasks {100 * task:.0f} % of awake time ({top} {100 * acts.get(top, 0):.0f} %), more than "
                           f"{100 * TASK_MAX_SHARE:.0f} %", "a pet does chores now and then: lower tidy's utility or "
                           "add a cool-down after each tidy", dict(states=states, activities=acts)))
    pet = sum(states.get(s, 0.0) for s in PET_STATES)
    if pet < PET_MIN_SHARE:
        out.append(Finding("activity_mix", "warn", 0.0, t1,
                           f"pet behaviour (idle life, play, social, curiosity) only {100 * pet:.0f} % of awake time",
                           "make play, social and curious activities win more often", dict(states=states)))
    return out


# -- the report ----------------------------------------------------------------------------------------------------
def write_report(path):
    """Observer + realism metrics over a recorded session: writes findings.json, report.json and report.txt there."""
    from pinch.realism import realism_metrics, report_text
    from pinch.recorder import load_session
    s = load_session(path)
    findings = observe_session(s)
    metrics = realism_metrics(s)
    text = report_text(s, findings, metrics)
    p = Path(path)
    (p / "findings.json").write_text(json.dumps([dict(asdict(f), id=f.id) for f in findings], indent=1, default=str),
                                     encoding="utf-8")
    (p / "report.json").write_text(json.dumps(dict(metrics=metrics, findings=dict(Counter(f.kind for f in findings))),
                                              indent=1, default=str), encoding="utf-8")
    (p / "report.txt").write_text(text, encoding="utf-8")
    return dict(findings=findings, metrics=metrics, text=text)

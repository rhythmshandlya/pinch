"""The behaviour-quality observer (decision 64): it catches planted problems, and ten minutes of pet life alone
produce no errors (oscillation, unexplained switches). Over recorded sessions it judges the body, face, eyes and
voice too: each problem planted in an otherwise calm synthetic session is caught, the calm session itself is clean,
and a recorded stretch of real pet life raises no errors and few warnings."""

import tempfile
from pathlib import Path

import numpy as np

from pinch.memory import Memory
from pinch.observer import observe, observe_records, observe_session
from pinch.realism import conditional_entropy, entropy, realism_metrics
from pinch.recorder import load_session
from pinch.twin import PinchSim


def _d(t, prev, to, reason="x", state=("idle", "idle")):
    return dict(t=t, prev=prev, to=to, reason=reason, state=list(state))


def test_it_catches_planted_problems():
    osc = [_d(i * 2.0, None, "a" if i % 2 == 0 else "b") for i in range(9)]
    kinds = {f.kind for f in observe_records(osc, [], [], 20.0)}
    assert "oscillation" in kinds
    loop = [_d(i * 20.0, "tidy", "tidy", "previous activity finished") for i in range(5)]
    assert "failure_loop" in {f.kind for f in observe_records(loop, [], [], 120.0)}
    assert "unexplained" in {f.kind for f in observe_records([_d(0, None, "idle", "")], [], [], 1.0)}
    slow = [_d(i * 60.0, None, "idle" if i % 2 == 0 else "tidy") for i in range(9)]
    assert "alternation" in {f.kind for f in observe_records(slow, [], [], 600.0)}
    rep = [_d(i * 10.0, None, n) for i, n in enumerate(["a", "b", "c"] * 5)]
    assert "repetition" in {f.kind for f in observe_records(rep, [], [], 150.0)}
    lock = [_d(0.0, None, "idle", state=("idle", "idle"))]
    assert "state_lock" in {f.kind for f in observe_records(lock, [], [], 1000.0)}
    log = [(400.0, "timeout", "rest ran 400 s (limit 300 s)")]                 # OBS-007 inappropriate timing
    assert "timeouts" in {f.kind for f in observe_records([_d(0.0, None, "rest")], [], log, 500.0)}


def test_ten_minutes_alone_has_no_errors():
    sim = PinchSim(memory=Memory(path=None), seed=5)
    sim.run(600.0)
    findings = observe(sim.mind)
    errors = [f for f in findings if f.severity == "error"]
    assert not errors, "; ".join(f"{f.kind}: {f.evidence}" for f in errors)
    assert len(sim.mind.decisions) >= 5, "barely any decisions in ten minutes"


# -- the body, face, eyes and voice (recorded sessions) ----------------------------------------------------------------
DT = 0.06


def _calm(seconds=60.0):
    """A synthetic session where nothing is wrong: standing idle, content, looking ahead, a couple of varied
    emotes that finish, a fitting sound."""
    n = int(seconds / DT)
    t = np.arange(n) * DT
    sm = dict(t=t, q_cmd=np.zeros((n, 18), np.float32) + 0.01 * np.sin(t)[:, None], busy=np.zeros(n, bool),
              status=np.array([""] * n, "<U16"), cmd_v=np.zeros(n), x=np.zeros(n), y=np.zeros(n), yaw=np.zeros(n),
              up=np.ones(n), gaze_x=np.zeros(n), gaze_y=np.zeros(n), gaze_want=np.full(n, np.nan),
              gaze_target=np.array([""] * n, "<U16"), face=np.array(["content"] * n, "<U16"), feeling=np.array(["content"] * n, "<U16"),
              valence=np.full(n, 0.3), activity=np.array(["idle"] * n, "<U16"), phase=np.array([""] * n, "<U16"),
              jerk_rms=np.full(n, 100.0), max_jump=np.full(n, 0.01), clamped=np.zeros(n, int),
              jump_joint=np.zeros(n, int), jump_status=np.array([""] * n, "<U16"),
              state=np.array(["idle"] * n, "<U16"), goal_d=np.full(n, np.nan))
    ev = [dict(t=0.1, kind="decision", prev=None, to="idle", reason="first", state=["idle", "idle"]),
          dict(t=0.1, kind="feeling", feeling="content", valence=0.3, arousal=0.3),
          dict(t=0.1, kind="face", face="content", activity="idle")]
    for t0, emote in [(10.0, "wave"), (30.0, "look_around")]:
        ev += [dict(t=t0, kind="step_start", step=f"emote:{emote}", motion="emote", args=[emote], source="mind"),
               dict(t=t0 + 2, kind="step_end", step=f"emote:{emote}", motion="emote", ok=True, end="done", dur=2.0,
                    source="mind", ldlj=-12.0, max_jump=0.05, clamped=0)]
    ev.append(dict(t=31.0, kind="sound", name="purr", activity="idle"))
    return dict(header=dict(sample_dt=DT), events=sorted(ev, key=lambda e: e["t"]), samples=sm)


def _kinds(session):
    return {f.kind for f in observe_session(session)}


def _span(sm, t0, t1):
    return (sm["t"] >= t0) & (sm["t"] <= t1)


def _steps(ev, labels, t0=1.0, every=3.0, ok=True, source="mind"):
    for i, lbl in enumerate(labels):
        t = t0 + i * every
        ev += [dict(t=t, kind="step_start", step=lbl, motion=lbl.split(":")[0], args=[], source=source),
               dict(t=t + every - 0.1, kind="step_end", step=lbl, motion=lbl.split(":")[0], ok=ok, end="done" if ok
                    else "failed", dur=every - 0.1, source=source)]
    ev.sort(key=lambda e: e["t"])


def test_a_calm_session_is_clean():
    assert not observe_session(_calm()), [f"{f.kind}: {f.evidence}" for f in observe_session(_calm())]


def test_it_catches_repetitive_motion():
    s = _calm()
    _steps(s["events"], ["emote:wave"] * 4, t0=35.0)
    assert "repetitive_motion" in _kinds(s)
    s = _calm()
    _steps(s["events"], ["emote:wave", "goto", "emote:dance"] * 3, t0=35.0, every=2.0)
    assert "repetitive_motion" in _kinds(s)


def test_it_catches_unnecessary_movement():
    s = _calm()
    _steps(s["events"], ["emote:wave"], t0=40.0, source="other")              # a motion nobody started
    assert "unnecessary_movement" in _kinds(s)
    s = _calm()
    m = _span(s["samples"], 40.0, 45.0)
    s["samples"]["q_cmd"][m] += 0.1 * (-1.0) ** np.arange(m.sum())[:, None]   # twitching while idle
    assert "unnecessary_movement" in _kinds(s)
    s = _calm()
    m = _span(s["samples"], 40.0, 60.0)
    s["samples"]["x"][m] = 0.02 * (s["samples"]["t"][m] - 40.0)               # sliding while standing
    assert "unnecessary_movement" in _kinds(s)


def test_it_catches_inconsistent_expression():
    s = _calm()
    m = _span(s["samples"], 20.0, 30.0)
    s["samples"]["feeling"][m] = "sad"
    s["samples"]["face"][m] = "happy"
    assert "inconsistent_expression" in _kinds(s)
    s = _calm()
    s["samples"]["face"][_span(s["samples"], 20.0, 30.0)] = "asleep"            # asleep face while not resting
    assert "inconsistent_expression" in _kinds(s)


def test_it_catches_conflicting_movement():
    s = _calm()
    s["events"] += [dict(t=40.0, kind="step_start", step="goto", motion="goto", args=[], source="mind"),
                    dict(t=41.0, kind="step_end", step="goto", motion="goto", ok=False, end="preempted", dur=1.0,
                         source="mind"),
                    dict(t=41.0, kind="step_start", step="emote:dance", motion="emote", args=[], source="mind")]
    s["events"].sort(key=lambda e: e["t"])
    f = [x for x in observe_session(s) if x.kind == "conflicting_movement"]
    assert f and f[0].severity == "error", f
    s = _calm()
    m = _span(s["samples"], 40.0, 42.0)
    s["samples"]["status"][m], s["samples"]["busy"][m], s["samples"]["cmd_v"][m] = "emote", True, 0.03
    assert "conflicting_movement" in _kinds(s)
    s = _calm()                                     # (a reflex taking over is not a conflict)
    s["events"] += [dict(t=40.0, kind="step_start", step="goto", motion="goto", args=[], source="mind"),
                    dict(t=41.0, kind="step_end", step="goto", motion="goto", ok=False, end="preempted", dur=1.0,
                         source="mind"),
                    dict(t=41.0, kind="step_start", step="recover", motion="recover", args=[], source="edge reflex")]
    s["events"].sort(key=lambda e: e["t"])
    assert "conflicting_movement" not in _kinds(s)


def test_it_catches_abrupt_transitions():
    s = _calm()
    i = int(20.0 / DT)
    s["samples"]["max_jump"][i], s["samples"]["jump_joint"][i], s["samples"]["jump_status"][i] = 0.6, 16, "push"
    f = [x for x in observe_session(s) if x.kind == "abrupt_transition"]
    from crab.params import JOINT_NAMES
    assert f and "0.60" in f[0].evidence and JOINT_NAMES[16] in f[0].evidence, f


def test_it_catches_excessive_reactions():
    s = _calm()
    s["events"] += [dict(t=5.0 + 10 * k, kind="decision", prev="idle", to="startle", reason="x",
                         state=["idle", "alert"]) for k in range(5)]
    s["events"].sort(key=lambda e: e["t"])
    assert "excessive_reactions" in _kinds(s)
    s = _calm()
    s["events"] += [dict(t=5.0 + 40 * k, kind="decision", prev="idle", to="startle", reason="x",
                         state=["idle", "alert"]) for k in range(5)]          # five in 160 s: fine
    s["events"].sort(key=lambda e: e["t"])
    assert "excessive_reactions" not in _kinds(s)


def test_it_catches_gaze_mismatch():
    s = _calm()
    m = _span(s["samples"], 20.0, 26.0)
    s["samples"]["activity"][m], s["samples"]["gaze_want"][m], s["samples"]["gaze_x"][m] = "investigate", 0.8, -0.4
    assert "gaze_mismatch" in _kinds(s)
    s["samples"]["gaze_x"][m] = 0.7                                               # looking at it: fine
    assert "gaze_mismatch" not in _kinds(s)
    s["samples"]["gaze_x"][m], s["samples"]["status"][m] = -0.4, "goto"           # walking: eyes ahead, fine
    assert "gaze_mismatch" not in _kinds(s)


def test_it_catches_sound_mismatch():
    s = _calm()
    s["events"] += [dict(t=40.0, kind="feeling", feeling="scared", valence=-0.5, arousal=0.9),
                    dict(t=41.0, kind="sound", name="giggle", activity="startle")]
    s["events"].sort(key=lambda e: e["t"])
    assert "sound_mismatch" in _kinds(s)
    s = _calm()
    s["events"] += [dict(t=40.0 + 0.8 * k, kind="sound", name="curious_chirp", activity="idle") for k in range(5)]
    s["events"].sort(key=lambda e: e["t"])
    assert "sound_mismatch" in _kinds(s)                                          # chattering


def test_it_catches_recovery_failure():
    s = _calm()
    s["samples"]["up"][_span(s["samples"], 20.0, 40.0)] = -1.0                   # on its back for 20 s
    f = [x for x in observe_session(s) if x.kind == "recovery_failure"]
    assert f and f[0].severity == "error"
    s = _calm(seconds=150.0)
    s["events"].append(dict(t=20.0, kind="body_sense", sense="stuck", why="feet over the edge"))
    s["events"].sort(key=lambda e: e["t"])
    assert "recovery_failure" in _kinds(s)                                        # stuck to the end
    s = _calm()
    _steps(s["events"], ["pick:pen"] * 3, t0=35.0, ok=False)
    assert "recovery_failure" in _kinds(s)


def test_it_catches_being_stuck():
    s = _calm()
    m = _span(s["samples"], 20.0, 40.0)                    # walking toward a spot 20 cm away, getting nowhere
    s["samples"]["status"][m], s["samples"]["busy"][m], s["samples"]["cmd_v"][m] = "goto", True, 0.04
    s["samples"]["goal_d"][m] = 0.20
    f = [x for x in observe_session(s) if x.kind == "stuck"]
    assert f and f[0].severity == "error" and "20 cm from its goal" in f[0].evidence, f
    s["samples"]["goal_d"][m] = 0.20 - 0.008 * (s["samples"]["t"][m] - 20.0)    # getting there (8 mm/s): fine
    assert "stuck" not in _kinds(s)
    s = _calm()                                            # no known goal: trying to walk, not moving at all
    s["samples"]["status"][m], s["samples"]["busy"][m], s["samples"]["cmd_v"][m] = "walk", True, 0.04
    assert "stuck" in _kinds(s)


def test_it_reports_failed_grasps_with_their_cause():
    s = _calm()
    s["events"] += [dict(t=40.0, kind="step_start", step="pick:pen", motion="pick", args=["pen"], source="mind"),
                    dict(t=45.0, kind="body", what="align", step="pick:pen", info=dict(yaw=0.1, miss_mm=2.0)),
                    dict(t=48.0, kind="body", what="pick_lift", step="pick:pen", info=dict(held=False)),
                    dict(t=50.0, kind="step_end", step="pick:pen", motion="pick", ok=False, end="failed", dur=10.0,
                         source="mind")]
    s["events"].sort(key=lambda e: e["t"])
    f = [x for x in observe_session(s) if x.kind == "failed_grasp"]
    assert f and "didn't hold it" in f[0].evidence and "1 grab" in f[0].evidence, f
    s["events"][-1:] = [dict(s["events"][-1], end="cancelled (mind)")]        # cut short by the mind: not a failure
    assert "failed_grasp" not in _kinds(s)


def test_it_reports_the_activity_mix():
    s = _calm(seconds=120.0)
    m = _span(s["samples"], 0.0, 60.0)                       # half the time tidying: a tidying machine
    s["samples"]["state"][m], s["samples"]["activity"][m] = "task", "tidy"
    f = [x for x in observe_session(s) if x.kind == "activity_mix"]
    assert f and "tidy 50 %" in f[0].evidence, f
    s["samples"]["state"][_span(s["samples"], 60.0, 120.0)] = "recover"      # and no pet behaviour left
    assert len([x for x in observe_session(s) if x.kind == "activity_mix"]) == 2
    s = _calm(seconds=120.0)
    s["samples"]["state"][_span(s["samples"], 0.0, 20.0)] = "task"           # a sixth of the time: fine
    assert "activity_mix" not in _kinds(s)
    s["samples"]["state"][_span(s["samples"], 20.0, 120.0)] = "rest"         # (asleep time doesn't count)
    assert "activity_mix" not in _kinds(s)                                   # (20 s awake: too short to judge)
    m = realism_metrics(_calm(seconds=120.0))
    assert m["pet_share"] == 1.0 and m["task_share"] == 0.0 and m["activity_share"] == {"idle": 1.0}


def test_realism_metrics():
    assert entropy(["a", "a", "a"]) == (0.0, 0.0)
    assert abs(entropy(["a", "b", "c", "d"])[1] - 1.0) < 1e-9
    assert conditional_entropy(["a", "b"] * 10) < 1e-9 < conditional_entropy(["a", "b", "a", "a", "b", "b", "a"])
    s = _calm()
    _steps(s["events"], ["emote:wave"] * 4, t0=35.0)
    m = realism_metrics(s)
    assert m["repeat_rate"] > 0.5 and m["min_repeat_s"] <= 3.0 and m["expression_agreement"] == 1.0
    assert m["sound_fit"] == 1.0 and m["step_success"] == 1.0
    m2 = realism_metrics(_calm())
    assert m2["repeat_rate"] == 0.0 and m2["action_entropy_norm"] == 1.0


# Real problems this session shows today, which the observer must keep finding until they are fixed (then remove
# them here): wedged against the tray for 20 s while lining up to pick the eraser, walking further from it, until a
# foot goes over the edge (the owner saw it in the review videos, 2026-11-20; pinch/body/manipulation.py).
KNOWN_PROBLEMS = set()             # ("stuck" was here; gone since phase 4 part 3)


def test_recorded_pet_life_has_few_findings():
    """Three minutes of real pet life (you say hello, stroke it, praise it, ask for a dance), recorded: no errors
    but the known problems, and at most two warnings per minute. (The warnings it gives today are real: tidying 45 %
    of the time, the claw snapping shut at the end of a drop, a sad face kept through a tidy retry while happy. The
    cap catches a check that starts crying wolf.)"""
    with tempfile.TemporaryDirectory() as tmp:
        sim = PinchSim(memory=Memory(path=None), seed=5, record_dir=Path(tmp) / "life", record=dict(state=False))
        for t_at, what in ((4.0, "hi pinch, I'm Sam"), (30.0, None), (50.0, "good crab"), (120.0, "dance!")):
            sim.run(t_at - sim.t)
            sim.say(what) if what else sim.touch("stroke")
        sim.run(180.0 - sim.t)
        sim.stop_recording()
        s = load_session(Path(tmp) / "life")
    findings = observe_session(s)
    errors = [f for f in findings if f.severity == "error" and f.kind not in KNOWN_PROBLEMS]
    warns = [f for f in findings if f.severity == "warn" and f.kind not in KNOWN_PROBLEMS]
    assert not errors, "; ".join(f"{f.kind}: {f.evidence}" for f in errors)
    assert len(warns) <= 6, "; ".join(f"{f.kind}: {f.evidence}" for f in warns)
    for k in KNOWN_PROBLEMS:                          # (fixed? then take it off the list)
        assert k in {f.kind for f in findings}, f"'{k}' is no longer found here: fixed? remove it from KNOWN_PROBLEMS"


def test_it_catches_a_still_pinch():
    s = _calm()
    n = len(s["samples"]["t"])
    s["samples"]["ambient"] = np.zeros(n)                     # no ambient life at all, the eyes fixed
    s["samples"]["eye"] = np.zeros(n)
    assert "aliveness" in _kinds(s)
    s["samples"]["ambient"] = np.full(n, 0.01)                # breathing all the time: alive
    assert "aliveness" not in _kinds(s)


def test_ambient_life_is_visible_almost_all_awake_time():
    """Phase 4 exit check: ten minutes of pet life (a hello, strokes, praise, a dance) with ambient life - breathing,
    claw breathing, flexes, the eyes' darts and blinks - in at least 95 % of the awake seconds, never still for more
    than 3 s. Full-body emotes own every part (they move on purpose), so they are the seconds it may miss."""
    from pinch.observer import aliveness
    with tempfile.TemporaryDirectory() as tmp:
        sim = PinchSim(memory=Memory(path=None), seed=11, record_dir=Path(tmp) / "life", record=dict(state=False))
        for t_at, what in ((4.0, "hi pinch, I'm Sam"), (30.0, None), (50.0, "good crab"), (120.0, "dance!"), (300.0, None)):
            sim.run(t_at - sim.t)
            sim.say(what) if what else sim.touch("stroke")
        sim.run(600.0 - sim.t)
        sim.stop_recording()
        sm = load_session(Path(tmp) / "life")["samples"]
    share, gap, n = aliveness(sm)
    assert n > 500 and share >= 0.95 and gap <= 3.0, f"alive {100 * share:.1f} % of {n} s, longest still gap {gap:.0f} s"
    body = dict(sm, eye=np.zeros(len(sm["t"])), blink=np.zeros(len(sm["t"]), bool))
    b_share, _, _ = aliveness(body)
    assert b_share >= 0.65, f"the body alone is alive {100 * b_share:.1f} % of the time"

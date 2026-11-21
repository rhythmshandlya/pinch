"""The session recorder (pinch/recorder.py): opt-in and read-only - the golden session recorded is still the golden
session; a short recording has every stream; the same seed and inputs give the same recording; the firmware
records too (no video); and a recording can be filmed afterwards."""

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from pinch.memory import Memory
from pinch.recorder import load_session
from pinch.twin import PinchSim


def _record(path, seed=3, seconds=20.0, state=True, props=("box",)):
    sim = PinchSim(memory=Memory(path=None), seed=seed, props=props, record_dir=path,
                   record=dict(header=dict(scenario="test"), state=state))
    script = {2.0: ("say", "hi pinch, I'm Sam"), 6.0: ("touch", "stroke"), 9.0: ("show_prop", ("box", (0.15, 0.0)))}
    done = set()
    while sim.t < seconds:
        for t0, (kind, arg) in script.items():
            if t0 not in done and sim.t >= t0:
                done.add(t0)
                if kind == "say":
                    sim.say(arg)
                elif kind == "touch":
                    sim.touch(arg)
                else:
                    sim.show_prop(*arg)
        sim.run(0.5)
    rep = sim.stop_recording()
    return sim, rep


def test_recording_changes_nothing_golden_trace():
    """The golden session (tests/test_golden.py) with the recorder on matches the golden trace exactly."""
    import test_golden as g
    want = json.loads(g.GOLDEN.read_text(encoding="utf-8"))
    if want.get("platform", "Windows") != platform.system():
        print(f"    (skipped: the golden trace was recorded on {want.get('platform', 'Windows')})")
        return
    with tempfile.TemporaryDirectory() as tmp:
        orig, made = g.PinchSim, []

        def recorded(**kw):
            made.append(PinchSim(record_dir=Path(tmp) / "golden", record=dict(state=False), **kw))
            return made[-1]
        g.PinchSim = recorded
        try:
            got = json.loads(json.dumps(g.session()))
        finally:
            g.PinchSim = orig
            for sim in made:
                sim.stop_recording(analyse=False)
        for k in ("decisions", "motions", "sounds", "pose", "cap"):
            assert got[k] == want[k], f"recording changed '{k}' of the golden session"
        s = load_session(Path(tmp) / "golden")
        assert sum(e["kind"] == "decision" for e in s["events"]) == len(want["decisions"]), \
            "the recording missed decisions the golden trace has"


def test_short_recording_has_all_streams():
    with tempfile.TemporaryDirectory() as tmp:
        sim, rep = _record(Path(tmp) / "s")
        s = load_session(Path(tmp) / "s")
        h = s["header"]
        assert h["closed"] and h["config"]["seed"] == 3 and h["config"]["scenario"] == "test"
        assert h["git"]["commit"] and "gait.npz" in h["policies"] and h["policies"]["gait.npz"]["registry"]
        assert "mujoco" in h["versions"]
        kinds = {e["kind"] for e in s["events"]}
        for k in ("decision", "step_start", "step_end", "face", "feeling", "say", "touch", "show_prop", "sound", "log",
                  "attend", "body"):
            assert k in kinds, f"no '{k}' events in a recording that had one: {sorted(kinds)}"
        decs = [e for e in s["events"] if e["kind"] == "decision"]
        assert [d["to"] for d in decs] == [d["to"] for d in sim.mind.decisions] and all(d["reason"] for d in decs)
        assert sum(e["kind"] == "sound" for e in s["events"]) == len(sim.sounds_played)
        mind_steps = [e for e in s["events"] if e["kind"] == "step_start" and e["source"] == "mind"]
        assert len(mind_steps) == len(sim.motions)
        ends = [e for e in s["events"] if e["kind"] == "step_end"]
        assert ends and all({"ok", "dur", "end"} <= set(e) for e in ends)
        sm = s["samples"]
        for k in ("t", "x", "y", "yaw", "up", "q_cmd", "q_meas", "gaze_x", "gaze_want", "face", "feeling", "activity",
                  "phase", "state", "status", "goal_d", "jerk_rms", "max_jump", "qpos", "mocap"):
            assert k in sm, f"samples lack '{k}'"
        rate = (len(sm["t"]) - 1) / (sm["t"][-1] - sm["t"][0])
        assert 10.0 <= rate <= 20.0, f"{rate:.1f} Hz"
        assert sm["q_cmd"].shape[1] == sm["q_meas"].shape[1] == 18
        assert np.abs(sm["q_cmd"] - sm["q_meas"]).mean() < 0.2, "targets and measured joints far apart"
        assert np.isfinite(sm["gaze_want"]).any(), "never knew where it should be looking"
        assert (Path(tmp) / "s" / "model.mjb").exists()
        for f in ("findings.json", "report.json", "report.txt"):
            assert (Path(tmp) / "s" / f).exists()
        assert "metrics" in json.loads((Path(tmp) / "s" / "report.json").read_text(encoding="utf-8"))
        assert rep["metrics"]["duration_s"] >= 19.0


def test_same_seed_same_recording():
    with tempfile.TemporaryDirectory() as tmp:
        _record(Path(tmp) / "a", seed=4, seconds=25.0, state=False)
        _record(Path(tmp) / "b", seed=4, seconds=25.0, state=False)
        a, b = load_session(Path(tmp) / "a"), load_session(Path(tmp) / "b")
        assert a["events"] == b["events"], "the same seed and inputs gave different event streams"
        for k in a["samples"]:
            x, y = a["samples"][k], b["samples"][k]
            assert x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind == "f"
                                           else np.array_equal(x, y)), f"samples '{k}' differ"


def test_firmware_records_without_video():
    from robot.hw_emulated import EmulatedHardware
    from robot.pinch_main import PinchFirmware
    with tempfile.TemporaryDirectory() as tmp:
        fw = PinchFirmware(EmulatedHardware(), llm=False, memory_path=False, record_dir=Path(tmp) / "robot")
        fw.say("hi pinch")
        fw.run(8.0, realtime=False)
        s = load_session(Path(tmp) / "robot")
        assert s["header"]["closed"] and not s["header"]["state"]
        kinds = {e["kind"] for e in s["events"]}
        assert {"decision", "say", "step_start", "face"} <= kinds, kinds
        assert "qpos" not in s["samples"] and len(s["samples"]["t"]) >= 30
        assert not (Path(tmp) / "robot" / "model.mjb").exists()
        assert (Path(tmp) / "robot" / "report.txt").exists()


def _can_render():
    """Try the renderer in a child process. Without a display MuJoCo does not raise: it aborts the interpreter,
    which took the whole test run down on the Linux CI machine."""
    probe = "import mujoco; mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'), 8, 8).close()"
    try:
        return subprocess.run([sys.executable, "-c", probe], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def test_a_recording_can_be_filmed():
    if not _can_render():
        assert not os.environ.get("PINCH_REQUIRE_RENDER"), "no renderer, but PINCH_REQUIRE_RENDER is set (CI)"
        print("    (skipped: no renderer here)")
        return
    from sim.record import render
    with tempfile.TemporaryDirectory() as tmp:
        sim = PinchSim(memory=Memory(path=None), seed=1, record_dir=Path(tmp) / "f")
        sim.run(2.0)
        sim.stop_recording()
        out = render(Path(tmp) / "f", Path(tmp) / "f.gif", gif=True, speed=2)
        assert out.exists() and out.stat().st_size > 1000

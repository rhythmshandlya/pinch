"""Golden trace: a fixed, seeded 2-minute session (talk, a trick, a tidy request, being stroked, feeling sad)
must reproduce the recorded run exactly - every decision, motion, sound and the final body pose. It guards
refactors: moving code around must not change what Pinch does. When behaviour changes on purpose, re-record:

    python tests/test_golden.py --record
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinch.memory import Memory  # noqa: E402
from pinch.twin import PinchSim  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "session_seed7.json"
SCRIPT = {3.0: ("say", "hi pinch, I'm Sam"), 12.0: ("say", "dance!"), 30.0: ("say", "pinch, put the cap away"),
          70.0: ("stroke", None), 90.0: ("say", "I'm sad today")}


def session(seconds=120.0):
    sim = PinchSim(memory=Memory(path=None), seed=7)
    sim.place("cap", (-0.06, 0.0))
    sim.mind.world.see("cap", (-0.06, 0.0), graspable=False)
    sim.mind.memory.teach_home("cap", sim.desk.tray_xy)
    done = set()
    while sim.t < seconds:
        for t0, (kind, arg) in SCRIPT.items():
            if t0 not in done and sim.t >= t0:
                done.add(t0)
                if kind == "say":
                    sim.say(arg)
                else:
                    sim.mind.touch("stroke")
        sim.run(0.5)
    return dict(decisions=[[r["t"], r["prev"], r["to"], r["reason"]] for r in sim.mind.decisions],
                motions=[[round(t, 2), str(m)] for t, m in sim.motions],
                sounds=[[round(t, 2), s] for t, s in sim.sounds_played],
                pose=[round(float(v), 9) for v in sim.d.qpos[:7]],
                cap=[round(float(v), 9) for v in sim.obj("cap")[0][:2]])


def test_session_matches_golden_trace():
    assert GOLDEN.exists(), "no golden trace yet: python tests/test_golden.py --record"
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    got = json.loads(json.dumps(session()))
    for k in ("decisions", "motions", "sounds", "pose", "cap"):
        if got[k] != want[k]:
            a, b = got[k], want[k]
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            raise AssertionError(f"{k} differs from the golden trace at item {i}: got {a[i:i + 2]}, recorded "
                                 f"{b[i:i + 2]}")
    assert np.allclose(got["pose"], want["pose"])


if __name__ == "__main__":
    if "--record" in sys.argv:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(session(), indent=0), encoding="utf-8")
        print("recorded", GOLDEN)

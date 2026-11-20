"""Pick-up bench: the pen or the eraser at random spots on the desk (not by an edge), Pinch picks it up with
its claw and drops it in the tray, through the robot's own Body in the full twin. Mind off: this measures the
skill, not the choice to use it.

    python sim/benches/pick.py                       # scripted grab
    python sim/benches/pick.py --grasp <name>        # with a learned grab (policies/<name>.npz)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.benches.common import run_parallel, save_json  # noqa: E402


def one(args):
    thing, seed, grasp = args
    from crab.policy_np import POLICY_DIR, NumpyPolicy
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    rng = np.random.default_rng(seed)
    sim = PinchSim(memory=Memory(path=None), seed=seed)
    sim.body.grasp_policy = NumpyPolicy.load(POLICY_DIR / f"{grasp}.npz") if grasp else None
    sim.autonomous = False
    sim.run(1.5)
    while True:                                   # a spot >= 9 cm from every edge, off the tray, not on Pinch
        xy = (rng.uniform(-0.25, 0.25), rng.uniform(-0.13, 0.13))
        if not sim.desk.in_tray(xy) and np.hypot(*xy) > 0.13 and sim.desk.edge_distance(xy) > 0.09:
            break
    sim.place(thing, xy, rng.uniform(-np.pi, np.pi))
    sim.run(0.5)
    t0 = sim.t
    sim.body.start(("pick", thing))
    while sim.body.busy and sim.t - t0 < 90:
        sim.run(0.2)
    held = bool(sim.body.held)
    in_tray = False
    if held:
        sim.body.start(("drop", sim.desk.tray_xy))
        while sim.body.busy and sim.t - t0 < 150:
            sim.run(0.2)
        sim.run(1.0)
        in_tray = sim.in_tray(thing)
    return dict(thing=thing, seed=seed, held=held, in_tray=bool(in_tray), t=round(sim.t - t0), falls=sim.falls,
                off=sim.obj(thing)[1] < -0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--grasp", default=None, help="policies/<name>.npz learned grab (default: scripted)")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--json", default=None, help="also write the results here")
    a = ap.parse_args()
    jobs = [(th, s, a.grasp) for th in ("pen", "eraser") for s in range(a.n)]
    res = run_parallel(one, jobs, a.jobs)
    save_json(a.json, res)
    print(f"grab: {a.grasp or 'scripted'}")
    for th in ("pen", "eraser"):
        r = [x for x in res if x["thing"] == th]
        print(f"  {th:6s} picked {sum(x['held'] for x in r)}/{len(r)}, in the tray {sum(x['in_tray'] for x in r)}/{len(r)}"
              f"  falls {sum(x['falls'] for x in r)}  knocked off {sum(x['off'] for x in r)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

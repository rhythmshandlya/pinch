"""Tidy bench: the two tidy scenarios (put the cap away when asked; notice the eraser is out of place and tidy
it unasked) over several seeds, in parallel. Reports success, time, falls, and desk things knocked off the desk
along the way. Seed 0 is the scenario layout; the others jitter where everything starts.

    python sim/benches/tidy.py [--seeds 8] [--jobs 16]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.benches.common import run_parallel, save_json  # noqa: E402


def one(args):
    case, seed = args
    import numpy as np

    from pinch.memory import Memory
    from pinch.twin import PinchSim
    from sim.desk import OBJECTS, START
    rng = np.random.default_rng(1000 + seed)
    # every seed moves the desk things a little (+-2 cm, +-0.3 rad): one layout is one trajectory, and the
    # twin's contacts are chaotic - a pass rate over layouts is what says whether tidying works
    start = {n: (x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02), yaw + rng.uniform(-0.3, 0.3))
             for n, (x, y, yaw) in START.items()} if seed else None
    sim = PinchSim(memory=Memory(path=None), seed=seed, start=start)
    sim.run(2.0)
    name, xy, limit = ("cap", (-0.06, 0.0), 200) if case == "cap" else ("eraser", (-0.05, 0.02), 240)
    if seed:
        xy = (xy[0] + rng.uniform(-0.02, 0.02), xy[1] + rng.uniform(-0.02, 0.02))
    sim.place(name, xy)
    sim.mind.world.see(name, xy, graspable=False)
    sim.mind.memory.teach_home(name, sim.desk.tray_xy)
    if case == "cap":
        sim.say("pinch, put the cap away")
    t0 = sim.t
    for _ in range(limit):
        sim.run(1.0)
        if sim.in_tray(name):
            break
    off = [o for o in OBJECTS if sim.obj(o)[1] < -0.02]
    return dict(case=case, seed=seed, ok=bool(sim.in_tray(name)), t=round(sim.t - t0), falls=sim.falls, off=off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--offset", type=int, default=0, help="first seed (split a big run across machines)")
    ap.add_argument("--json", default=None, help="also write the per-run results here")
    a = ap.parse_args()
    jobs = [(c, s) for c in ("cap", "eraser") for s in range(a.offset, a.offset + a.seeds)]
    res = run_parallel(one, jobs, a.jobs)
    save_json(a.json, res)
    for c in ("cap", "eraser"):
        rs = [r for r in res if r["case"] == c]
        ok = [r for r in rs if r["ok"]]
        print(f"{c:7s} {len(ok)}/{len(rs)} tidied  mean time {sum(r['t'] for r in ok) / max(1, len(ok)):.0f} s  "
              f"falls {sum(r['falls'] for r in rs)}  knocked off: {sum(len(r['off']) for r in rs)} "
              f"{[(r['seed'], r['off']) for r in rs if r['off']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

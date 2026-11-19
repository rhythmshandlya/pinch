"""Caught on something: Pinch walks (forward, then sideways) at a thing fixed to the desk - a 5 mm cable, a
pen, an 8 mm book edge - in the full twin, through the robot's own Body. If the walk stops short, recover()
runs (as the Unstick activity does when the stuck sense fires), which climbs over with the high-step crawl.

    python sim/obstacle_test.py
Prints, per case: over or not, how (walked / climbed), time, falls.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

THINGS = {"cable": ("cylinder", 0.0025), "pen": ("cylinder", 0.0045), "book": ("box", 0.004)}


def case(args):
    thing, way = args
    from pinch.memory import Memory
    from pinch.sim_runner import PinchSim
    kind, r = THINGS[thing]
    # Pinch at (0, 0.12) facing -y (forward) or facing +x (then -y is to its right: sideways)
    heading = -np.pi / 2 if way == "forward" else 0.0
    if kind == "cylinder":
        extra = dict(name=thing, type="cylinder", size=[r, 0.30], pos=[0.0, 0.06, r], euler=[0, np.pi / 2, 0])
    else:                                                     # a book's edge: a 15 cm-deep slab to climb onto
        extra = dict(name=thing, type="box", size=[0.30, 0.075, r], pos=[0.0, 0.06 - 0.075, r])
    sim = PinchSim(memory=Memory(path=None), extras=[extra], policies=True)
    sim.autonomous = False
    q = sim.d.qpos
    q[0:2] = [0.0, 0.12]
    q[3:7] = [np.cos(heading / 2), 0, 0, np.sin(heading / 2)]
    mujoco.mj_forward(sim.m, sim.d)
    sim.run(1.5)
    goal = (0.0, -0.05)
    how, t0 = "walked", sim.t

    def go():
        sim.body.start(("goto", goal, 0.02, heading))
        while sim.body.busy and sim.t - t0 < 120:
            sim.run(0.2)

    go()
    for attempt in range(2):
        if np.hypot(*(np.array(sim.pose()[:2]) - goal)) < 0.03:
            break
        how = "climbed"
        sim.body.start(("recover",))
        while sim.body.busy:
            sim.run(0.2)
        go()
    x, y, h = sim.pose()
    over = y < 0.06 - 0.06
    return dict(thing=thing, way=way, over=bool(over), how=how, t=round(sim.t - t0), falls=sim.falls,
                end=(round(x, 3), round(y, 3)), climbs=sum(1 for e in sim.body.log if e[1] == "climb"))


def main():
    jobs = [(t, w) for t in THINGS for w in ("forward", "sideways")]
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(case, jobs))
    for r in res:
        print(f"{r['thing']:6s} {r['way']:8s} {'OVER' if r['over'] else 'not over':8s} {r['how']:8s} "
              f"{r['t']:4d} s  falls {r['falls']}  climbs {r['climbs']}  end {r['end']}")
    return 0 if all(r["over"] and not r["falls"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())

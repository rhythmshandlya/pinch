"""Sweep check of the v2 assembly: swing every joint over its range and find where its parts first meet another part.

    python cad/v2/sweep_check.py                 # one left leg front and rear, the left arm (the right side mirrors)
    python cad/v2/sweep_check.py --steps 41

For each joint (the others at the stand pose), every part that moves with it is intersected with every part that does
not, at evenly spaced angles over the joint's range in sim/models/crab_v2.xml. The clear range is the stretch around
the stand angle with no overlap of 1 mm3 or more. It must contain the range walking and the moves use (with a 0.1 rad
margin; .ops/bench/used_ranges.npy when present). The shell is left out: cad/v2/body.py cuts its joint openings from
the same sweeps (cad/v2/body_check.py checks them). A claw's jaws meeting each other at its closed end is the jaw
shutting, not a clash: the claw passes if its clear range starts within JAWS_MEET of the closed end (the servo then
stalls on the pads, at its 40 % torque limit; the simulator collides the two jaws' pads).
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CRAB_DESIGN", "v2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from build123d import Location, Plane  # noqa: E402

from cad.v2.assembly import MODEL, groups  # noqa: E402
from crab.params import JOINT_NAMES  # noqa: E402

LIMIT = 1.0
MARGIN = 0.1
JAWS_MEET = 0.12            # rad of claw travel (about 2 mm at the pads)
JOINTS = ("FL_coxa", "FL_femur", "FL_tibia", "RL_coxa", "RL_femur", "RL_tibia",
          "L_shoulder_yaw", "L_shoulder_pitch", "L_claw")


class Poser:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("stand").id)
        self.stand = self.d.qpos.copy()
        self.stand[:7] = [0, 0, 0, 1, 0, 0, 0]

    def poses(self, joint=None, q=None):
        d = self.d
        d.qpos[:] = self.stand
        if joint is not None:
            d.qpos[self.m.jnt_qposadr[self.m.joint(joint).id]] = q
        mujoco.mj_kinematics(self.m, d)
        out = {}
        for name in groups(proxy=True):
            bid = self.m.body(name).id
            R = d.xmat[bid].reshape(3, 3)
            out[name] = Location(Plane(origin=tuple(d.xpos[bid] * 1000.0), x_dir=tuple(R[:, 0]), z_dir=tuple(R[:, 2])))
        return out

    def subtree(self, joint):
        b = self.m.jnt_bodyid[self.m.joint(joint).id]
        out = set()
        for i in range(self.m.nbody):
            j = i
            while j > 0 and j != b:
                j = self.m.body_parentid[j]
            if j == b:
                out.add(self.m.body(i).name)
        return out


def _hits(moving, still):
    found = []
    for n1, s1, b1 in moving:
        for n2, s2, b2 in still:
            if (b1.max.X < b2.min.X or b2.max.X < b1.min.X or b1.max.Y < b2.min.Y or b2.max.Y < b1.min.Y
                    or b1.max.Z < b2.min.Z or b2.max.Z < b1.min.Z):
                continue
            try:
                v = (s1 & s2).volume
            except Exception:                                  # noqa: BLE001
                v = float("nan")
            if not v < LIMIT:
                found.append((n1, n2, v))
    return found


def sweep(poser, joint, steps):
    """[(q, [(moving part, other part, mm3)])] over the joint's range."""
    g = groups(proxy=True)
    sub = poser.subtree(joint)
    lo, hi = poser.m.jnt_range[poser.m.joint(joint).id]
    out = []
    for q in np.linspace(lo, hi, steps):
        P = poser.poses(joint, q)
        moving, still = [], []
        for body, items in g.items():
            for label, solid, _, _ in items:
                if label in ("shell", "tray", "cover") or label.startswith("wires_"):
                    continue
                s = P[body] * solid
                (moving if body in sub else still).append((f"{body}:{label}", s, s.bounding_box()))
        out.append((float(q), _hits(moving, still)))
    return out


def clear_range(rows, q0):
    """The largest stretch of clear angles that contains q0 (or the nearest clear angle)."""
    qs = np.array([q for q, _ in rows])
    ok = np.array([not h for _, h in rows])
    i0 = int(np.argmin(np.abs(qs - q0)))
    if not ok[i0]:
        return None
    a = b = i0
    while a > 0 and ok[a - 1]:
        a -= 1
    while b < len(qs) - 1 and ok[b + 1]:
        b += 1
    return float(qs[a]), float(qs[b]), a == 0, b == len(qs) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--joints", default=",".join(JOINTS))
    a = ap.parse_args()
    poser = Poser()
    used_f = ROOT / ".ops" / "bench" / "used_ranges.npy"
    used = np.load(used_f) if used_f.exists() else None
    fails = 0
    for joint in a.joints.split(","):
        rows = sweep(poser, joint, a.steps)
        q0 = poser.stand[poser.m.jnt_qposadr[poser.m.joint(joint).id]]
        cr = clear_range(rows, q0)
        lo, hi = poser.m.jnt_range[poser.m.joint(joint).id]
        need = None
        if used is not None:
            i = JOINT_NAMES.index(joint)
            need = (max(lo, used[0, i] - MARGIN), min(hi, used[1, i] + MARGIN))
        if cr is None:
            ok = False
            msg = "blocked at the stand pose"
        else:
            lo_ok = cr[0] <= need[0] + 1e-9 or cr[2] if need else True
            if joint.endswith("_claw") and cr[0] - lo <= JAWS_MEET:
                lo_ok = True                                 # the jaws meet: the claw is shut
            ok = need is None or lo_ok and (cr[1] >= need[1] - 1e-9 or cr[3])
            msg = f"clear {cr[0]:+.2f}..{cr[1]:+.2f} rad of {lo:+.2f}..{hi:+.2f}"
        fails += not ok
        print(f"{joint:18s} {'PASS' if ok else 'FAIL'}  {msg}" + (f"  (needs {need[0]:+.2f}..{need[1]:+.2f})" if need else ""))
        for q, hits in rows:
            if hits:
                worst = max(hits, key=lambda h: h[2] if h[2] == h[2] else 1e9)
                print(f"    q {q:+.2f}: {len(hits)} hit(s), e.g. {worst[0]} x {worst[1]} {worst[2]:.0f} mm3")
    print("all joints clear over the used range" if not fails else f"{fails} joint(s) blocked inside the used range")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

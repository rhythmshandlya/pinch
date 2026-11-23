"""Finger-trap check for design v2 (R-25, children under 6): every joint swept over its full range.

    python sim/gap_check.py              # all joints
    python sim/gap_check.py --verbose    # every part pair

For each joint, the gap between each part that moves with it and each part it closes against (its parent link and
the shell) is measured over the whole range (MuJoCo's geometric distance, meshes by their convex hulls), with the
other joints at the stand pose. The rule (EN 71-1, summarised by toy-test labs; the standard text was not read): a gap
that ever admits a 5 mm rod must admit a 12 mm rod at every position. So a pair passes if its gap stays under 5 mm over
the whole travel (no finger gets in) or never closes below 12 mm (a finger never gets caught). A gap that does not
change over the travel (less than 1 mm) cannot pinch and is not counted; nor are the electronics inside the closed shell.
"""

import argparse
import sys
from pathlib import Path

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.params import JOINT_NAMES, PARAMS  # noqa: E402

SMALL, BIG = 0.005, 0.012
CLOSING = 0.001                                      # a gap that changes less than this over the travel does not pinch
STEPS = 40


def subtree_bodies(m, b):
    return {i for i in range(m.nbody) if _ancestor(m, i, b)}


def _ancestor(m, i, b):
    while i > 0:
        if i == b:
            return True
        i = m.body_parentid[i]
    return i == b


def geoms_of(m, bodies):
    """The parts a finger can reach: everything but the electronics inside the closed shell (layout parts, `part_*`)."""
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
            and m.geom_size[g].max() > 0.002 and not (m.geom(g).name or "").startswith("part_")]


def check_joint(m, d, joint, stand, rng=None):
    j = m.joint(joint).id
    child = m.jnt_bodyid[j]
    moving = subtree_bodies(m, child)
    parent = m.body_parentid[child]
    against = {parent, m.body("body").id} - moving
    gm, ga = geoms_of(m, moving), geoms_of(m, against)
    lo, hi = m.jnt_range[j] if rng is None else rng
    qa = m.jnt_qposadr[j]
    gaps = {}
    for q in np.linspace(lo, hi, STEPS):
        d.qpos[:] = stand
        d.qpos[qa] = q
        mujoco.mj_kinematics(m, d)
        for g1 in gm:
            for g2 in ga:
                dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.05, None)
                gaps.setdefault((g1, g2), []).append(max(dist, 0.0))
    out = []
    for (g1, g2), v in gaps.items():
        v = np.array(v)
        if v.min() >= BIG or v.max() < SMALL or v.max() - v.min() < CLOSING:
            continue                                     # never closes on a finger, or never lets one in, or does not move
        name = lambda g: m.geom(g).name or f"{m.body(m.geom_bodyid[g]).name}#{g}"   # noqa: E731
        out.append((name(g1), name(g2), float(v.min()), float(v.max())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--used", default=None, help="a .npy of [min, max] per joint (the range walking and the moves use), "
                    "swept with a 0.1 rad margin instead of the joints' full range")
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "models" / PARAMS.model_file))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    stand = d.qpos.copy()
    fails = 0
    used = np.load(a.used) if a.used else None
    for i, joint in enumerate(JOINT_NAMES):
        rng = None
        if used is not None:
            lo, hi = m.jnt_range[m.joint(joint).id]
            rng = (max(lo, used[0, i] - 0.1), min(hi, used[1, i] + 0.1))
        bad = check_joint(m, d, joint, stand, rng)
        fails += len(bad)
        print(f"{joint:18s} {'PASS' if not bad else f'{len(bad)} trap(s)'}")
        for g1, g2, lo, hi in (bad if a.verbose else bad[:3]):
            print(f"    {g1} vs {g2}: gap {lo * 1000:.1f}-{hi * 1000:.1f} mm over the travel")
    print(f"finger traps: {fails} part pairs open past 5 mm and close below 12 mm" if fails else "no finger traps")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

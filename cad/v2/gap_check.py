"""Finger-trap check on the real CAD shapes (the gap rule for small children): every joint swept over the range it is used in.

    python cad/v2/gap_check.py                  # one front leg, one rear leg, the left arm (the right side mirrors)
    python cad/v2/gap_check.py --full           # the joints' whole mechanical range

For each joint (the others at the stand pose), the exact distance between every part that moves with it and every
part it closes against (its parent link and the body) is measured at evenly spaced angles (OpenCascade's minimum
distance between the solids). The rule (EN 71-1, as toy-test labs summarise it): a gap that ever admits a 5 mm rod
must admit a 12 mm rod at every position. A pair passes if its gap stays under 5 mm over the travel, never closes
below 12 mm, or does not change (less than 1 mm: it cannot close on a finger). The claw's jaws are meant to close
on things: they are reported separately, with the claw's force limit (STS3032 at 40 % torque) as their safeguard.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cad.v2.assembly import groups  # noqa: E402
from cad.v2.sweep_check import JOINTS, MARGIN, Poser  # noqa: E402
from crab.params import JOINT_NAMES  # noqa: E402

SMALL, BIG, CLOSING = 5.0, 12.0, 1.0      # mm


def gaps(poser, joint, qs):
    g = groups(proxy=True)
    sub = poser.subtree(joint)
    parent = poser.m.body(poser.m.body_parentid[poser.m.jnt_bodyid[poser.m.joint(joint).id]]).name
    against = {parent, "body"} - sub
    out = {}
    for q in qs:
        P = poser.poses(joint, q)
        moving = [(f"{b}:{lab}", P[b] * s) for b, items in g.items() if b in sub for lab, s, _, _ in items]
        still = [(f"{b}:{lab}", P[b] * s) for b, items in g.items() if b in against for lab, s, _, _ in items]
        for n1, s1 in moving:
            b1 = s1.bounding_box()
            for n2, s2 in still:
                b2 = s2.bounding_box()
                far = max(b2.min.X - b1.max.X, b1.min.X - b2.max.X, b2.min.Y - b1.max.Y, b1.min.Y - b2.max.Y,
                          b2.min.Z - b1.max.Z, b1.min.Z - b2.max.Z)
                d = BIG + 1 if far > BIG else s1.distance_to(s2)
                out.setdefault((n1, n2), []).append(d)
    return out


def traps(d):
    bad = []
    for (n1, n2), v in d.items():
        v = np.array(v)
        if v.min() >= BIG or v.max() < SMALL or v.max() - v.min() < CLOSING:
            continue
        bad.append((n1, n2, float(v.min()), float(v.max())))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="the whole mechanical range, not the used range")
    ap.add_argument("--steps", type=int, default=13)
    ap.add_argument("--joints", default=",".join(JOINTS))
    a = ap.parse_args()
    poser = Poser()
    used_f = ROOT / ".ops" / "bench" / "used_ranges.npy"
    used = np.load(used_f) if used_f.exists() and not a.full else None
    total = 0
    for joint in a.joints.split(","):
        lo, hi = poser.m.jnt_range[poser.m.joint(joint).id]
        if used is not None:
            i = JOINT_NAMES.index(joint)
            lo, hi = max(lo, used[0, i] - MARGIN), min(hi, used[1, i] + MARGIN)
        bad = traps(gaps(poser, joint, np.linspace(lo, hi, a.steps)))
        jaws = [b for b in bad if joint.endswith("_claw") and ("dactyl" in b[0] and "upper" in b[1])]
        bad = [b for b in bad if b not in jaws]
        total += len(bad)
        print(f"{joint:18s} {'PASS' if not bad else f'{len(bad)} trap(s)'}  ({lo:+.2f}..{hi:+.2f} rad)"
              + (f"  jaws: closing gap {jaws[0][2]:.1f}-{jaws[0][3]:.1f} mm, force-limited" if jaws else ""))
        for n1, n2, g0, g1 in bad:
            print(f"    {n1} vs {n2}: {g0:.1f}-{g1:.1f} mm over the travel")
    print("no finger traps" if not total else f"{total} finger trap(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())

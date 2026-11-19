"""Find collision-free joint limits from the CAD (bisection from the stand pose).

    python cad/limits.py

For each joint end that collides, bisect between the stand value and the nominal
limit until the boundary is known to 1 degree, then subtract a 3 degree margin.
Prints a params.py-ready summary.
"""

import sys
from math import degrees, radians
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cad import assembly as A  # noqa: E402
from cad.check import joint_ranges  # noqa: E402
from crab.emotes import STAND  # noqa: E402
from crab.params import JOINT_NAMES  # noqa: E402

JOINTS = ["FL_coxa", "RL_coxa", "FL_femur", "RL_femur", "FL_tibia", "RL_tibia", "L_shoulder_yaw",
          "L_shoulder_pitch", "L_claw"]
MARGIN = radians(3)


def collides(name, value):
    q = STAND.copy()
    q[JOINT_NAMES.index(name)] = value
    return bool(A.interferences(q))


def boundary(name, start, end, tol=radians(1)):
    """Largest |value - start| toward `end` that is collision-free."""
    if not collides(name, end):
        return end
    ok, bad = start, end
    while abs(bad - ok) > tol:
        mid = (ok + bad) / 2
        if collides(name, mid):
            bad = mid
        else:
            ok = mid
    return ok - MARGIN * (1 if end > start else -1)


def main():
    ranges = joint_ranges()
    print(f"{'joint':18s} {'nominal (deg)':>16s} {'collision-free (deg)':>22s}")
    for n in JOINTS:
        lo, hi = ranges[n]
        s = STAND[JOINT_NAMES.index(n)]
        new_lo, new_hi = boundary(n, s, lo), boundary(n, s, hi)
        flag = "" if (abs(new_lo - lo) < 1e-6 and abs(new_hi - hi) < 1e-6) else "  <- tightened"
        print(f"{n:18s} {degrees(lo):7.0f} .. {degrees(hi):5.0f}   {degrees(new_lo):9.0f} .. {degrees(new_hi):5.0f}{flag}",
              flush=True)


if __name__ == "__main__":
    main()

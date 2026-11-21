"""Interference check of the v2 assembly: do any two parts overlap at the stand pose?

    python cad/v2/interference.py              # every pair whose bounding boxes overlap is intersected exactly
Overlaps under 1 mm3 are touching contacts (a plate on a horn face) and are ignored. Exits 1 if any remain.
"""

import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("CRAB_DESIGN", "v2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cad.v2.assembly import body_poses, groups  # noqa: E402

LIMIT = 1.0          # mm3


def posed_parts():
    poses = body_poses()
    out = []
    for body, items in groups(proxy=True).items():
        for label, solid, _, _ in items:
            out.append((f"{body}:{label}", poses[body] * solid))
    return out


def overlaps(parts):
    boxes = [(n, s, s.bounding_box()) for n, s in parts]
    found = []
    for (n1, s1, b1), (n2, s2, b2) in combinations(boxes, 2):
        if (b1.max.X < b2.min.X or b2.max.X < b1.min.X or b1.max.Y < b2.min.Y or b2.max.Y < b1.min.Y
                or b1.max.Z < b2.min.Z or b2.max.Z < b1.min.Z):
            continue
        where = None
        try:
            common = s1 & s2
            v = common.volume
            if v >= LIMIT:
                bb = common.bounding_box()
                where = tuple(round(c, 1) for c in (bb.min.X, bb.max.X, bb.min.Y, bb.max.Y, bb.min.Z, bb.max.Z))
        except Exception:                                  # noqa: BLE001 (a failed boolean is reported, not hidden)
            v = float("nan")
        if not v < LIMIT:
            found.append((n1, n2, v, where))
    return found


if __name__ == "__main__":
    bad = overlaps(posed_parts())
    for n1, n2, v, where in sorted(bad, key=lambda r: -(r[2] if r[2] == r[2] else 1e9)):
        print(f"  {n1:32s} x {n2:32s} {v:9.1f} mm3  {where or ''}")
    print(f"{len(bad)} overlapping pairs" if bad else "no overlaps")
    sys.exit(1 if bad else 0)

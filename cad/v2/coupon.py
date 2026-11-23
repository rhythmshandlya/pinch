"""The print test coupon: print it first, on the printer and filament the robot will be printed with, and keep the
sizes that fit (the CAD's hole and fit sizes are then set from it).

    python cad/v2/coupon.py            # -> cad/out/v2/print/coupon.stl (and its peg, coupon_peg.stl)

Rows (left to right in each row, the size grows; one notch on the edge marks the first):
1. M2.5 heat-set insert holes 3.4 / 3.6 / 3.8 mm, 6 mm deep, in 8 mm bosses (cad/v2/body.py M25_INSERT = 3.6)
2. M3 heat-set insert holes 4.0 / 4.2 / 4.4 mm, 7 mm deep (cad/v2/lid.py M3_INSERT = 4.2)
3. M2 self-tapping pilots 1.6 / 1.7 / 1.8 mm (body.py M2_PILOT = 1.7) and M2.5 clearance 2.6 / 2.7 / 2.8 mm
4. sockets for the separate 6.0 mm peg at 0.1 / 0.2 / 0.3 mm clearance (sliding fits: battery door hook, lid back)
Rules: hole = insert OD - 0.3 mm, deeper than the insert (UW-Madison ME design lab); walls in 0.4 mm line widths.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build123d import Box, Cylinder, Pos, export_stl  # noqa: E402

from cad.v2.finish import RBox  # noqa: E402

OUT = ROOT / "cad" / "out" / "v2" / "print"
T = 8.0                                      # plate thickness
PITCH = 12.0


def coupon():
    plate = Pos(0, 0, T / 2) * RBox(4 * PITCH + 6.0, 4 * PITCH + 4.0, T, r=1.2)
    rows = [((3.4, 3.6, 3.8), 6.0), ((4.0, 4.2, 4.4), 7.0), ((1.6, 1.7, 1.8, 2.6, 2.7, 2.8), T + 1.0),
            ((6.2, 6.4, 6.6), 6.0)]
    y0 = 1.5 * PITCH
    for r, (dias, depth) in enumerate(rows):
        y = y0 - r * PITCH
        n = len(dias)
        step = (4 * PITCH - 6.0) / max(n - 1, 1)
        for k, d in enumerate(dias):
            x = -2 * PITCH + 3.0 + k * step
            plate = plate - Pos(x, y, T - depth / 2 + 0.01) * Cylinder(d / 2, depth + 0.02)
        plate = plate - Pos(-2 * PITCH - 1.0, y, T) * Box(2.0, 1.0, 1.0)      # the notch: the first of the row
    return plate


def peg():
    """A 6.0 mm peg, 10 mm long, with a 0.5 mm chamfer-like taper at the tip, printed standing up."""
    return Pos(0, 0, 5.0) * Cylinder(3.0, 10.0) + Pos(0, 0, 0.5) * Box(10.0, 10.0, 1.0)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    export_stl(coupon(), str(OUT / "coupon.stl"), tolerance=0.05)
    export_stl(peg(), str(OUT / "coupon_peg.stl"), tolerance=0.05)
    print(OUT / "coupon.stl")

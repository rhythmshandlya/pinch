"""Servo models for the v2 CAD: the Feetech STS3215 from its real STEP model, the STS3032 as a measured block.

STS3215 (cad/parts_ext/STS3215_03a.step, SO-ARM100, Apache-2.0), in its own frame: case x -22.7..22.7 (length),
y -12.4..12.4, z -19.4 (idler boss) .. 20.2 (horn face); output shaft +z through (11.25, 0); horn holes dia 2.5 on a
7 mm radius at z = 17.4; idler holes at z = -16.6; case screw holes dia 1.5 at x = 5.0 / -15.7, y = +-10.2 on the top
and bottom faces (measured from the STEP's cylindrical faces, cad/v2/servos.py `measure`).
We use the SERVO FRAME of cad/servo.py for both: shaft = +Z through the origin, horn face at +Z, case length along X
with the output end at +X. For the STS3215 that is the STEP moved by (-11.25, 0, 0).

STS3032 (no 3D model exists: Feetech drawing in cad/servo.py): case 23 x 12 x 27.5 mm, ears 32 mm tip to tip.
"""

import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build123d import Align, Box, Cylinder, Pos, import_step  # noqa: E402

from cad.servo import STS3032  # noqa: E402

STEP = ROOT / "cad" / "parts_ext" / "STS3215_03a.step"
SHAFT_X = 11.25                       # shaft from the STEP's case centre, along the case length

# STS3215 in the servo frame (shaft at the origin), mm
STS3215 = dict(x_min=-22.7 - SHAFT_X, x_max=22.7 - SHAFT_X, half_w=12.4, z_min=-19.4, z_max=20.2,
               case_top=17.4, case_bottom=-16.6, horn_r=10.0, horn_pcd=7.0,
               case_holes=((5.0 - SHAFT_X, 10.2), (5.0 - SHAFT_X, -10.2), (-15.7 - SHAFT_X, 10.2), (-15.7 - SHAFT_X, -10.2)),
               hole_d=1.5, mass=0.055)


@cache
def sts3215():
    """The real STS3215 solid in the servo frame."""
    return Pos(-SHAFT_X, 0, 0) * import_step(str(STEP))


@cache
def sts3215_envelope(clear=0.5):
    """A box around the STS3215 case (no horn), for clearance cuts, in the servo frame."""
    s = STS3215
    return Pos((s["x_min"] + s["x_max"]) / 2, 0, (s["case_bottom"] + s["case_top"]) / 2) * Box(
        s["x_max"] - s["x_min"] + 2 * clear, 2 * s["half_w"] + 2 * clear, s["case_top"] - s["case_bottom"] + 2 * clear)


@cache
def sts3215_proxy():
    """The STS3215 as simple solids of its real outside sizes (case, horn disc, idler boss), in the servo frame. Booleans
    on the imported STEP fail in OpenCascade, so the interference checks use this; renders and masses use the STEP."""
    s = STS3215
    case = Pos((s["x_min"] + s["x_max"]) / 2, 0, (s["case_bottom"] + s["case_top"]) / 2) * Box(
        s["x_max"] - s["x_min"], 2 * s["half_w"], s["case_top"] - s["case_bottom"])
    horn = Pos(0, 0, s["case_top"]) * Cylinder(s["horn_r"], s["z_max"] - s["case_top"], align=(Align.CENTER, Align.CENTER, Align.MIN))
    boss = Pos(0, 0, s["z_min"]) * Cylinder(6.0, s["case_bottom"] - s["z_min"], align=(Align.CENTER, Align.CENTER, Align.MIN))
    return case + horn + boss


@cache
def sts3032():
    """The STS3032 as a block: case, ears and horn disc, in the servo frame (cad/servo.py numbers)."""
    p = STS3032
    case = Pos((p.x_rear + p.x_out) / 2 * 0 + (p.x_out - p.L / 2), 0, 0) * Box(p.L, p.W, p.T)
    ears = Pos(p.x_out - p.L / 2, 0, p.ear_z0 + p.ear_t / 2 - p.T / 2 + p.T / 2) * Box(p.L + 2 * p.ear_len, p.ear_w, p.ear_t)
    horn = Pos(0, 0, p.T / 2) * Cylinder(p.horn_r, p.horn_face - p.T / 2, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return case + ears + horn


if __name__ == "__main__":
    s = sts3215()
    bb = s.bounding_box()
    print("STS3215 servo frame bbox", [round(v, 1) for v in (bb.min.X, bb.min.Y, bb.min.Z)],
          [round(v, 1) for v in (bb.max.X, bb.max.Y, bb.max.Z)])
    b = sts3032().bounding_box()
    print("STS3032 block bbox", [round(v, 1) for v in (b.min.X, b.min.Y, b.min.Z)], [round(v, 1) for v in (b.max.X, b.max.Y, b.max.Z)])

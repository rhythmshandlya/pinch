"""The v2 arms (style B): shoulder bracket, upper arm with the palm, and the moving jaw (dactyl), in build123d.

Frames are the simulator's (sim/models/crab_v2.xml): shoulder frame at the yaw axis (crab/params.py arm_mount), the
upper arm frame at the lift (pitch) axis 24 mm ahead (axis -y), the dactyl frame at the claw hinge 45 mm further
(axis +z for the left arm, -z for the right; geometry is built for the left arm and mirrored in y for the right).
All three arm servos are Feetech STS3032 (servo frame of cad/v2/servos.py: shaft +z, horn face +19.2, case x
-17.29..5.71, z -13.75..13.75, ears z 6.75..8.25, 4.5 mm past each end, holes at x 8.21 / -19.79, y +-3.25).

Servo poses:
  yaw   (in the body): upside down, shaft vertical through the shoulder point, horn face at shoulder z = +9.5
  lift  (in the shoulder bracket): shaft at (24, 0, 0) along -y, case along +x, its ears on a wall behind them
  claw  (in the palm): shaft at the hinge (45, 0, 0 upper frame) vertical, horn up, servo origin at z = CLAW_Z
The jaws are the simulator's own geometry (sim/build_model.py jaw_geoms and v2_pads: backing strips, blades, TPU pads
with the pen groove, the ball cup and the tip lips), so the printed claw and the simulated one are the same shape.
Units mm.
"""

import os
import sys
from functools import cache
from math import cos, radians, sin
from pathlib import Path

os.environ.setdefault("CRAB_DESIGN", "v2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from build123d import Align, Box, Cylinder, Location, Plane, Pos  # noqa: E402

from cad.servo import STS3032 as SP  # noqa: E402
from crab.params import PARAMS  # noqa: E402

LS, LU = PARAMS.arm.shoulder_offset * 1000, PARAMS.arm.upper_len * 1000      # 24, 45
PLATE, WALL = 3.0, 2.5
HUB_R = 11.0
HORN_PCD, HORN_HOLE = SP.horn_pcd_r, 2.2
YAW_HORN_Z = 9.5           # shoulder frame: the upside-down yaw servo's horn face (body z = -5.4)
CLAW_Z = -6.0              # upper frame: the claw servo's origin (horn face at CLAW_Z + 19.2)
CASE = dict(x0=SP.x_rear, x1=SP.x_out, hw=SP.W / 2, z0=-SP.T / 2, z1=SP.T / 2)   # servo frame, mm


def _box(center, half, quat=None):
    """A box at a centre (mm) with half sizes (mm) and an optional quaternion (w, x, y, z)."""
    b = Box(2 * half[0], 2 * half[1], 2 * half[2])
    if quat is None:
        return Pos(*center) * b
    w, x, y, z = quat
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return Location(Plane(origin=tuple(center), x_dir=tuple(R[:, 0]), z_dir=tuple(R[:, 2]))) * b


def _jaws():
    """The simulator's jaw boxes for the left arm, in mm (sim/build_model.py)."""
    import sim.build_model as B
    return B.jaw_geoms(1)


def _horn_plate(z0, t=PLATE, r=HUB_R):
    """A disc on a horn, servo frame from z0 upward, with the horn holes and a centre hole."""
    d = Pos(0, 0, z0) * Cylinder(r, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for a in (45, 135, 225, 315):
        d -= Pos(HORN_PCD * cos(radians(a)), HORN_PCD * sin(radians(a)), z0) * Cylinder(
            HORN_HOLE / 2, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return d - Pos(0, 0, z0) * Cylinder(2.4, t, align=(Align.CENTER, Align.CENTER, Align.MIN))


@cache
def shoulder_link():
    """Bracket under the yaw servo's horn: a plate on the horn, and a frame hanging from it that the lift servo's ears
    screw onto (the case passes through the frame's window). Shoulder frame."""
    plate = Location((0, 0, YAW_HORN_Z), (180, 0, 0)) * _horn_plate(0.0)          # under the upside-down horn
    # the lift servo: shaft (24, 0, 0) along -y, case along +x: servo x -> +x, servo y -> +z, servo z -> -y
    case_x0, case_x1 = LS + CASE["x0"], LS + CASE["x1"]
    ear_y = -(SP.ear_z0 + SP.ear_t)                                                  # the ears' outer face (-8.25)
    fx0, fx1 = case_x0 - SP.ear_len - 2.0, case_x1 + SP.ear_len + 2.0
    fz0, fz1 = -SP.W / 2 - 4.0, YAW_HORN_Z
    frame = Pos((fx0 + fx1) / 2, ear_y - WALL / 2, (fz0 + fz1) / 2) * Box(fx1 - fx0, WALL, fz1 - fz0)
    frame -= Pos((case_x0 + case_x1) / 2, ear_y - WALL / 2, 0) * Box(case_x1 - case_x0 + 0.6, WALL + 1, SP.W + 0.6)
    for hx in SP.ear_holes_x:
        for hy in SP.ear_holes_y:
            frame -= Pos(LS + hx, ear_y - WALL / 2, hy) * Location((0, 0, 0), (90, 0, 0)) * Cylinder(1.1, WALL + 1)
    y0, y1 = ear_y - WALL, HUB_R                                   # stops short of the lift servo's horn (y < -13.75)
    top = Pos((fx1 - HUB_R) / 2, (y0 + y1) / 2, YAW_HORN_Z - PLATE / 2) * Box(fx1 + HUB_R, y1 - y0, PLATE)
    return plate + frame + top


@cache
def upper_arm():
    """Upper arm: a plate on the lift servo's horn and a bearing plate on its boss, straps forward to the palm, the
    palm around the claw servo (its ears on the palm's rim), and the fixed jaw (pollex). Upper frame."""
    horn_y, boss_y = -SP.horn_face, -CASE["z0"] + SP.boss_h                          # -19.2, +14.75
    horn = Location((0, horn_y - PLATE / 2, 0), (90, 0, 0)) * Cylinder(HUB_R, PLATE)
    boss = Location((0, boss_y + PLATE / 2, 0), (90, 0, 0)) * Cylinder(HUB_R, PLATE)
    boss -= Location((0, boss_y + PLATE / 2, 0), (90, 0, 0)) * Cylinder(SP.boss_r + 0.3, PLATE + 1)
    for a in (45, 135, 225, 315):
        horn -= Location((HORN_PCD * cos(radians(a)), horn_y - PLATE / 2, HORN_PCD * sin(radians(a))), (90, 0, 0)) * Cylinder(
            HORN_HOLE / 2, PLATE + 1)
    # the palm: a sleeve round the claw servo (vertical, horn up, case along +x so the output end is at the hinge)
    cx0, cx1 = LU + CASE["x0"] - WALL, LU + CASE["x1"] + WALL
    cz0, cz1 = CLAW_Z + CASE["z0"] - WALL, CLAW_Z + SP.ear_z0
    palm = Pos((cx0 + cx1) / 2, 0, (cz0 + cz1) / 2) * Box(cx1 - cx0, SP.W + 2 * WALL, cz1 - cz0)
    palm -= Pos(LU + (CASE["x0"] + CASE["x1"]) / 2, 0, (cz0 + WALL + cz1) / 2 + 1) * Box(SP.L + 0.4, SP.W + 0.4, cz1 - cz0)
    strap_len = cx0 + 1
    straps = (Pos(strap_len / 2, horn_y - PLATE / 2, 0) * Box(strap_len, PLATE, 2 * HUB_R)
              + Pos(strap_len / 2, boss_y + PLATE / 2, 0) * Box(strap_len, PLATE, 2 * HUB_R))
    bx0, bx1 = cx0 - 2.0, LU + CASE["x0"] - 0.3                       # behind the claw servo's case
    bridge = Pos((bx0 + bx1) / 2, (horn_y + boss_y) / 2, 0) * Box(bx1 - bx0, boss_y - horn_y + 2 * PLATE, 2 * HUB_R)
    ear_x0 = LU + CASE["x0"] - SP.ear_len - 0.3                       # the claw servo's rear ear rests on the palm rim
    bridge -= Pos((ear_x0 + bx1 + 1) / 2, 0, cz1 - 0.25 + 6) * Box(bx1 + 1 - ear_x0, SP.ear_w + 0.6, 12.0)
    palm -= Pos(LU, 0, cz0 - 1) * Cylinder(SP.boss_r + 0.5, 2 * WALL + 2)   # the idler boss through the palm floor
    J = _jaws()
    m = 1000.0
    pollex = _box(np.array(J["strip"][0]) * m, np.array(J["strip"][1]) * m, _quat_y(-J["pitch"]))
    pollex += _box(np.array(J["blade"][0]) * m, np.array(J["blade"][1]) * m)
    for c, h, q in J["pads"]["pollex"]:
        pollex += _box(c, h, q)
    # a beam from the palm's front to the pollex's root
    root = np.array(J["strip"][0]) * m
    beam_x0, beam_x1 = cx1 - 1, root[0] - np.array(J["strip"][1])[0] * m + 1
    beam = Pos((beam_x0 + beam_x1) / 2, root[1], root[2] + 4) * Box(beam_x1 - beam_x0, 5.4, 14.0)
    return horn + boss + straps + bridge + palm + pollex + beam


def _quat_y(a):
    return (cos(-a / 2), 0.0, sin(-a / 2), 0.0)


@cache
def dactyl():
    """The moving jaw: a plate on the claw servo's horn, an arm forward over the palm, a drop down to the jaw strip,
    the strip with its blade and pads. Dactyl frame (hinge at the origin, the jaw at q = 0: parallel, shut)."""
    J = _jaws()
    m = 1000.0
    hub_z = CLAW_Z + SP.horn_face
    hub = _horn_plate(hub_z)
    strip_c, strip_h, strip_q = np.array(J["d_strip"][0]) * m, np.array(J["d_strip"][1]) * m, J["d_strip"][2]
    jaw = _box(strip_c, strip_h, strip_q) + _box(np.array(J["d_blade"][0]) * m, np.array(J["d_blade"][1]) * m, J["d_blade"][2])
    for c, h, q in J["pads"]["dactyl"]:
        jaw += _box(c, h, q)
    x_drop = strip_c[0] - strip_h[0] + 3.0                     # the drop rises from the jaw strip's back end
    arm = Pos(x_drop / 2 + 2, strip_c[1] / 2, hub_z + PLATE / 2) * Box(x_drop + 4, abs(strip_c[1]) + 2 * 4, PLATE)
    top_of_strip = strip_c[2] + strip_h[2]
    drop = Pos(x_drop, strip_c[1], (top_of_strip + hub_z + PLATE) / 2) * Box(6.0, 5.4, hub_z + PLATE - top_of_strip + 2)
    return hub + arm + drop + jaw


@cache
def arm_envelope(z_shoulder, x_max=45.0, steps=7):
    """What swings with a shoulder yaw joint, as boxes for cad/v2/body.py's openings: (name, x0, x1, half width, z0,
    z1), x forward from the yaw axis, z in the body frame (the shoulder frame sits at body z = z_shoulder). The bracket
    with the lift servo, and the upper arm with its claw servo at `steps` lift angles, clipped at x_max."""
    from math import degrees

    from cad.v2.servos import sts3032
    lift = Location((LS, 0, 0), (90, 0, 0)) * sts3032()

    def box(name, solid):
        bb = (solid & Pos(x_max - 500, 0, 0) * Box(1000, 1000, 1000)).bounding_box()     # the part at x <= x_max
        return (name, bb.min.X, bb.max.X, max(abs(bb.min.Y), abs(bb.max.Y)), bb.min.Z + z_shoulder, bb.max.Z + z_shoulder)
    out = [box("bracket", shoulder_link() + lift)]
    upper = upper_arm() + Pos(LU, 0, CLAW_Z) * sts3032()
    lo, hi = PARAMS.arm.shoulder_pitch_range
    for k in range(steps):
        q = lo + (hi - lo) * k / (steps - 1)
        out.append(box(f"upper{k}", Pos(LS, 0, 0) * Location((0, 0, 0), (0, degrees(-q), 0)) * upper))
    return tuple(out)


if __name__ == "__main__":
    import time
    for name, fn in (("shoulder_link", shoulder_link), ("upper_arm", upper_arm), ("dactyl", dactyl)):
        t = time.time()
        s = fn()
        bb = s.bounding_box()
        print(f"{name:14s} {s.volume / 1000:6.1f} cm3  bbox x {bb.min.X:6.1f}..{bb.max.X:6.1f}  y {bb.min.Y:6.1f}..{bb.max.Y:6.1f}"
              f"  z {bb.min.Z:6.1f}..{bb.max.Z:6.1f}  ({time.time() - t:.1f} s)")

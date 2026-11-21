"""The v2 legs (style B) in build123d: hip U-bracket (coxa link), thigh (femur link), shin (tibia) and foot.

Frames are the simulator's (sim/models/crab_v2.xml): each link frame has its joint at the origin; the coxa link turns
about z, the femur and tibia about -y; the next joint is at +x (coxa 35.2 mm, femur 56 mm, foot 80 mm). Every servo
shaft lies exactly on its joint axis, so the CAD and the simulation share one skeleton (the SO-ARM100 pattern:
U-brackets clamp a servo's horn and idler; a sleeve screwed to the next servo's side faces carries it).

Servo poses (servo frame of cad/v2/servos.py: shaft +z through the origin, case along x, output end +x):
  hip   (in the body): shaft vertical through the hip point, horn up at z = +15.0, case along body y running inward
  thigh (in the coxa link):  shaft at (35.2, 0, 0) along -y (horn toward -y), case hanging (servo x -> coxa +z)
  knee  (in the femur link): shaft at (56, 0, 0) along -y, case standing up from the knee (servo x -> femur -z), so
        the shin folds back under the thigh (walking uses -142 deg) without meeting it; crab knees stand high
Units mm. Printed in PETG; masses come from the solids (cad/v2/assembly.py).
"""

import os
import sys
from functools import cache
from math import cos, radians, sin
from pathlib import Path

os.environ.setdefault("CRAB_DESIGN", "v2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build123d import Align, Box, Cylinder, Location, Pos, Rot, Sphere  # noqa: E402

from cad.v2.servos import STS3215  # noqa: E402
from crab.params import PARAMS  # noqa: E402

STRAP_Z = (6.0, 26.0)       # femur straps run above the knee hub (the shin folds back below them)
LC, LF, LT = (v * 1000 for v in (PARAMS.leg.coxa_len, PARAMS.leg.femur_len, PARAMS.leg.tibia_len))
FOOT_R = PARAMS.leg.foot_radius * 1000
PLATE = 3.0                 # bracket plates on a horn or idler
WALL = 2.5                  # sleeves around a servo case
HORN_FACE, IDLER_FACE = STS3215["z_max"], STS3215["z_min"]     # 20.2, -19.4 (servo frame)
HUB_R = 13.0                # plate disc around a horn
HORN_HOLE, HORN_PCD = 2.6, STS3215["horn_pcd"]
SIDE_HOLES = ((-16.5 - 11.25, 15.2), (4.2 - 11.25, 15.2), (-16.5 - 11.25, -15.2), (4.2 - 11.25, -15.2))  # (x, z) on +-y faces
SIDE_HOLE = 2.2             # M2 through the sleeve into the case's side holes
HIP_HORN_Z, HIP_IDLER_Z = 15.0, -24.6     # agreed with cad/v2/body.py (the body seats the hip servo)
PLATE_Y = (-(HORN_FACE + PLATE / 2), -(IDLER_FACE - PLATE / 2))   # y of the plates on a horn and on an idler (link frame)
YOKE_X, SHIN_R = 26.0, 9.0                # the shin's yoke behind the knee hub, the shin's radius

# servo frame -> link frame: build123d Location rotations are extrinsic x, y, z in degrees
THIGH_SERVO = Location((LC, 0, 0), (-90, -180, -90))   # servo x -> +z, servo y -> -x, servo z -> -y
KNEE_SERVO = Location((LF, 0, 0), (-90, -180, 90))     # servo x -> -z, servo y -> +x, servo z -> -y
HUB_FRAME = Location((0, 0, 0), (90, 0, 0))             # a servo on this link's own joint: servo z -> -y


def axis_map(loc):
    """Where a Location sends the servo frame's x, y and z axes (unit vectors), for checks."""
    out = []
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        v = (loc * Location(axis)).position - loc.position
        out.append((round(v.X, 6), round(v.Y, 6), round(v.Z, 6)))
    return out


def hub_plate(t, outward):
    """A plate on a horn (outward=+1, beyond the horn face) or on an idler (-1), in the servo frame: a disc with the
    four horn holes and a centre hole for the horn screw."""
    z0 = HORN_FACE if outward > 0 else IDLER_FACE - t
    disc = Pos(0, 0, z0) * Cylinder(HUB_R, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for a in (45, 135, 225, 315):
        disc -= Pos(HORN_PCD * cos(radians(a)), HORN_PCD * sin(radians(a)), z0) * Cylinder(
            HORN_HOLE / 2, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    disc -= Pos(0, 0, z0) * Cylinder(2.6, t, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return disc


def sleeve():
    """A rectangular tube round the STS3215 case (its four sides parallel to the shaft), screwed to the side faces
    (y = +-12.4), in the servo frame. Open at the horn and idler faces so the next bracket turns freely."""
    s = STS3215
    x0, x1 = s["x_min"] - WALL, s["x_max"] + WALL
    z0, z1 = s["case_bottom"], s["case_top"]
    outer = Pos((x0 + x1) / 2, 0, (z0 + z1) / 2) * Box(x1 - x0, 2 * (s["half_w"] + WALL), z1 - z0)
    inner = Pos((s["x_min"] + s["x_max"]) / 2, 0, (z0 + z1) / 2) * Box(
        s["x_max"] - s["x_min"] + 0.4, 2 * s["half_w"] + 0.4, z1 - z0 + 2)
    t = outer - inner
    for x, z in SIDE_HOLES:
        t -= Pos(x, 0, z) * Rot(90, 0, 0) * Cylinder(SIDE_HOLE / 2, 2 * (s["half_w"] + WALL) + 2)
    t -= Pos(s["x_max"] + WALL / 2, 0, 0) * Box(WALL + 1, 10.0, 8.0)          # cable exit at the output end
    return t


def _plate_holes(z0):
    out = None
    for a in (45, 135, 225, 315):
        h = Pos(HORN_PCD * cos(radians(a)), HORN_PCD * sin(radians(a)), z0) * Cylinder(
            HORN_HOLE / 2, PLATE, align=(Align.CENTER, Align.CENTER, Align.MIN))
        out = h if out is None else out + h
    return out


@cache
def coxa_link():
    """Hip U-bracket: plates on the hip servo's horn (top) and idler (bottom), a back wall outside the hip servo's swing,
    and the thigh servo's sleeve. Coxa frame (hip point at the origin, x outward)."""
    top = Pos(0, 0, HIP_HORN_Z) * Cylinder(HUB_R, PLATE, align=(Align.CENTER, Align.CENTER, Align.MIN))
    bot = Pos(0, 0, HIP_IDLER_Z - PLATE) * Cylinder(HUB_R, PLATE, align=(Align.CENTER, Align.CENTER, Align.MIN))
    top -= _plate_holes(HIP_HORN_Z)
    bot -= _plate_holes(HIP_IDLER_Z - PLATE)
    w0 = 18.5                                                   # the hip servo's case corner swings at 16.9 mm
    th_x0 = LC - STS3215["half_w"] - WALL                       # the thigh sleeve's inner side (coxa x)
    arm_top = Pos(th_x0 / 2, 0, HIP_HORN_Z + PLATE / 2) * Box(th_x0, 26.0, PLATE)
    arm_bot = Pos((w0 + WALL) / 2, 0, HIP_IDLER_Z - PLATE / 2) * Box(w0 + WALL, 26.0, PLATE)
    wall = Pos(w0 + WALL / 2, 0, (HIP_IDLER_Z + HIP_HORN_Z) / 2) * Box(WALL, 26.0, HIP_HORN_Z - HIP_IDLER_Z + 2 * PLATE)
    return top + bot + arm_top + arm_bot + wall + THIGH_SERVO * sleeve()


@cache
def femur_link():
    """Thigh: plates on the thigh servo's horn (-y side) and idler (+y side), straps along the thigh above the knee hub,
    a web across to the back wall of the knee servo's sleeve. Femur frame (thigh axis at the origin, knee at x = LF)."""
    plates = HUB_FRAME * hub_plate(PLATE, +1) + HUB_FRAME * hub_plate(PLATE, -1)
    y_h, y_i = -(HORN_FACE + PLATE / 2), -(IDLER_FACE - PLATE / 2)
    x0, x1 = LF - STS3215["half_w"] - WALL, LF - STS3215["half_w"] - 0.4    # the sleeve's back wall (femur x)
    z0, z1 = STRAP_Z
    straps = (Pos(x1 / 2, y_h, (z0 + z1) / 2) * Box(x1, PLATE, z1 - z0)
              + Pos(x1 / 2, y_i, (z0 + z1) / 2) * Box(x1, PLATE, z1 - z0))
    web = Pos((x0 + x1) / 2, (y_h + y_i) / 2, (z0 + z1) / 2) * Box(x1 - x0, y_i - y_h + PLATE, z1 - z0)
    return plates + straps + web + KNEE_SERVO * sleeve()


# the crab shin (owner, 2026-11-22: no round tube and ball foot): a flattened, tapering segment, broad in the plane it
# swings in and thinner across, with a slight outward belly, ending in a blunt point under a TPU tip cap. Its reach
# (tip cap's far side 94 mm from the knee) is the old shin + ball's, so the kinematics and stand height are unchanged.
# Sections along the shin: (x, half width across y, half height in z, centre z), mm, tibia frame.
SHIN_SECTIONS = ((22.0, 8.5, 12.5, 0.0), (40.0, 7.5, 10.5, 1.0), (58.0, 6.2, 8.0, 1.2), (74.0, 4.8, 5.6, 0.6),
                 (86.0, 3.4, 3.8, 0.0))
TIP_R = 5.0                                          # TPU tip cap (child-safe: a blunt 10 mm dome)
TIP_X = LT + FOOT_R - TIP_R                          # 89 mm: same reach as the 80 mm shin + 14 mm ball


def _shin():
    from build123d import Ellipse, Plane, loft
    y0 = sum(PLATE_Y) / 2
    secs = [Plane(origin=(x, y0, zc), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * Ellipse(hy, hz) for x, hy, hz, zc in SHIN_SECTIONS]
    return loft(secs)


@cache
def tibia():
    """Shin: plates on the knee servo's horn and idler, a yoke past the hub, the crab shin to the tip. Tibia frame."""
    plates = HUB_FRAME * hub_plate(PLATE, +1) + HUB_FRAME * hub_plate(PLATE, -1)
    y_h, y_i = PLATE_Y
    yoke_x = YOKE_X
    arms = (Pos(yoke_x / 2, y_h, 0) * Box(yoke_x, PLATE, 2 * HUB_R) + Pos(yoke_x / 2, y_i, 0) * Box(yoke_x, PLATE, 2 * HUB_R))
    yoke = Pos(yoke_x, (y_h + y_i) / 2, 0) * Box(6.0, y_i - y_h + PLATE, 2 * HUB_R)
    return plates + arms + yoke + _shin()


@cache
def foot():
    """The TPU tip cap over the shin's point, with the point's socket (tibia frame)."""
    return Pos(TIP_X, sum(PLATE_Y) / 2, 0) * Sphere(TIP_R) - tibia()


if __name__ == "__main__":
    import time
    print("thigh servo axes (x, y, z) ->", axis_map(THIGH_SERVO))
    print("knee servo axes  (x, y, z) ->", axis_map(KNEE_SERVO))
    print("hub frame axes   (x, y, z) ->", axis_map(HUB_FRAME))
    for name, fn in (("coxa_link", coxa_link), ("femur_link", femur_link), ("tibia", tibia), ("foot", foot)):
        t = time.time()
        s = fn()
        bb = s.bounding_box()
        print(f"{name:10s} {s.volume / 1000:6.1f} cm3  bbox {bb.size.X:5.1f} x {bb.size.Y:5.1f} x {bb.size.Z:5.1f} mm  "
              f"({time.time() - t:.1f} s)")

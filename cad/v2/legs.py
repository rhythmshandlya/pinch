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

import sys
from functools import cache
from math import cos, radians, sin
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build123d import Align, Axis, Box, Cylinder, Location, Pos, Rot, fillet  # noqa: E402

from cad.v2.finish import RBox  # noqa: E402
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


CAP_H = 2.5                 # the knuckle cap's height over its plate


def horn_cap(outward):
    """A smooth domed knuckle cap over a hub plate's outer face (servo frame): it hides the horn screws and holes, so
    each joint reads as one round knuckle. Pressed on over the plate's centre hole, removed for servicing."""
    h, r = CAP_H, HUB_R - 0.5
    cap = Cylinder(r, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    cap = fillet(cap.edges().sort_by(Axis.Z)[-1], h * 0.9)
    if outward > 0:
        return Pos(0, 0, HORN_FACE + PLATE) * cap
    return Pos(0, 0, IDLER_FACE - PLATE) * Rot(180, 0, 0) * cap


def knuckles():
    """Both caps of a joint's hub plates (servo frame)."""
    return horn_cap(+1) + horn_cap(-1)


def sleeve():
    """A rectangular tube round the STS3215 case (its four sides parallel to the shaft), screwed to the side faces
    (y = +-12.4), in the servo frame. Open at the horn and idler faces so the next bracket turns freely."""
    s = STS3215
    x0, x1 = s["x_min"] - WALL, s["x_max"] + WALL
    z0, z1 = s["case_bottom"], s["case_top"]
    outer = Pos((x0 + x1) / 2, 0, (z0 + z1) / 2) * RBox(x1 - x0, 2 * (s["half_w"] + WALL), z1 - z0, r=4.5)   # a rounded housing
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
    arm_top = Pos(th_x0 / 2, 0, HIP_HORN_Z + PLATE / 2) * RBox(th_x0, 26.0, PLATE)
    arm_bot = Pos((w0 + WALL) / 2, 0, HIP_IDLER_Z - PLATE / 2) * RBox(w0 + WALL, 26.0, PLATE)
    wall = Pos(w0 + WALL / 2, 0, (HIP_IDLER_Z + HIP_HORN_Z) / 2) * RBox(WALL, 26.0, HIP_HORN_Z - HIP_IDLER_Z + 2 * PLATE)
    cap = Cylinder(HUB_R - 0.5, CAP_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    cap = fillet(cap.edges().sort_by(Axis.Z)[-1], CAP_H * 0.9)
    caps = Pos(0, 0, HIP_HORN_Z + PLATE) * cap + Pos(0, 0, HIP_IDLER_Z - PLATE) * Rot(180, 0, 0) * cap   # hip knuckles
    return top + bot + caps + arm_top + arm_bot + wall + THIGH_SERVO * sleeve()


@cache
def femur_link():
    """Thigh: plates on the thigh servo's horn (-y side) and idler (+y side), straps along the thigh above the knee hub,
    a web across to the back wall of the knee servo's sleeve. Femur frame (thigh axis at the origin, knee at x = LF)."""
    plates = HUB_FRAME * (hub_plate(PLATE, +1) + hub_plate(PLATE, -1) + knuckles())
    y_h, y_i = -(HORN_FACE + PLATE / 2), -(IDLER_FACE - PLATE / 2)
    x0, x1 = LF - STS3215["half_w"] - WALL, LF - STS3215["half_w"] - 0.4    # the sleeve's back wall (femur x)
    z0, z1 = STRAP_Z
    straps = (Pos(x1 / 2, y_h, (z0 + z1) / 2) * RBox(x1, PLATE, z1 - z0)
              + Pos(x1 / 2, y_i, (z0 + z1) / 2) * RBox(x1, PLATE, z1 - z0))
    web = Pos((x0 + x1) / 2, (y_h + y_i) / 2, (z0 + z1) / 2) * RBox(x1 - x0, y_i - y_h + PLATE, z1 - z0)
    return plates + straps + web + KNEE_SERVO * sleeve()


# the shin (owner, 2026-11-23: "do not make legs chicken like ... it does not get smaller and smaller radius wise"):
# a straight oval segment of near-constant section, broad in the plane it swings in, easing only slightly towards the
# foot, into a replaceable TPU sole boot. Reach (the sole's far side 94 mm from the knee) is the old shin + ball's, so
# the kinematics and stand height are unchanged. Sections: (x, half width across y, half height in z, centre z), mm.
SHIN_SECTIONS = ((22.0, 8.5, 12.0, 0.0), (40.0, 8.2, 11.4, 0.6), (58.0, 7.9, 10.8, 0.6), (74.0, 7.6, 10.2, 0.3),
                 (80.0, 6.6, 8.6, 0.0), (85.0, 6.4, 8.2, 0.0))   # the last two hide inside the sole boot (its socket)
SOLE_R = 8.0                                         # the sole's contact radius (the model's foot sphere)
SOLE_X = LT + FOOT_R - SOLE_R                        # 86 mm: the sole's centre; its far side at 94 mm as before
SOLE_TOP = 76.0                                      # where the boot starts up the shin
SOLE_WALL = 1.6                                      # TPU round the socket
SOLE_FLAT = 1.0                                      # the toe's flat contact patch: 1 mm off the round
TIP_R, TIP_X = SOLE_R, SOLE_X - SOLE_FLAT            # (the model's foot sphere: its far side on the flat)


def _shin():
    from build123d import Ellipse, Plane, loft
    y0 = sum(PLATE_Y) / 2
    secs = [Plane(origin=(x, y0, zc), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * Ellipse(hy, hz) for x, hy, hz, zc in SHIN_SECTIONS]
    return loft(secs)


@cache
def tibia():
    """Shin: plates on the knee servo's horn and idler, a yoke past the hub, the crab shin to the tip. Tibia frame."""
    plates = HUB_FRAME * (hub_plate(PLATE, +1) + hub_plate(PLATE, -1) + knuckles())
    y_h, y_i = PLATE_Y                                   # (the sole's pin hole goes through the shin's end: below)
    yoke_x = YOKE_X
    arms = (Pos(yoke_x / 2, y_h, 0) * RBox(yoke_x, PLATE, 2 * HUB_R) + Pos(yoke_x / 2, y_i, 0) * RBox(yoke_x, PLATE, 2 * HUB_R))
    yoke = Pos(yoke_x, (y_h + y_i) / 2, 0) * RBox(6.0, y_i - y_h + PLATE, 2 * HUB_R)
    return plates + arms + yoke + _shin() - sole_pin()


@cache
def foot():
    """The TPU sole boot over the shin's end (tibia frame): an oval sleeve that continues the shin's outline from
    SOLE_TOP, closed by a rounded oval toe whose bottom is the sole, three 0.6 mm grip grooves round it; the shin's end
    is its socket (a press fit, a drop of glue: pulls off to replace)."""
    from build123d import Ellipse, Plane, extrude, loft
    y0 = sum(PLATE_Y) / 2
    hy, hz = SHIN_SECTIONS[3][1] + 0.9, SHIN_SECTIONS[3][2] + 0.6          # flush with the shin above it
    sleeve = extrude(Plane(origin=(SOLE_TOP, y0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * Ellipse(hy, hz), SOLE_X - SOLE_TOP)
    for k in range(3):                                   # grip grooves round the sleeve, cut before the toe joins
        x = SOLE_X - 2.0 - 2.5 * k
        ring = extrude(Plane(origin=(x - 0.4, y0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * Ellipse(hy + 2, hz + 2), 0.8)
        core = extrude(Plane(origin=(x - 0.4, y0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * Ellipse(hy - 0.6, hz - 0.6), 0.8)
        sleeve = sleeve - (ring - core)
    # the rounded toe: an oval dome lofted through shrinking sections (a scaled sphere does not fuse reliably), its tip
    # cut flat for the contact patch (about 8 x 10 mm)
    import math
    secs = []
    for t in (0.0, 20.0, 40.0, 55.0, 67.0, 75.0):
        k = math.cos(math.radians(t))
        x = min(SOLE_X + SOLE_R * math.sin(math.radians(t)), SOLE_X + SOLE_R - SOLE_FLAT)
        secs.append(Plane(origin=(x, y0, 0), x_dir=(0, 1, 0), z_dir=(1, 0, 0)) * Ellipse(hy * k, hz * k))
    toe = loft(secs)
    return sleeve + toe - tibia() - sole_pin()


def sole_pin():
    """An M2 pin (a screw) across the boot and the shin's end: the boot cannot pull off, and comes off to replace
    (audit I7: glue alone)."""
    y0 = sum(PLATE_Y) / 2
    return Pos(SOLE_TOP + 4.5, y0, 0) * Rot(90, 0, 0) * Cylinder(1.1, 40.0)


def _bbox(solid, x_max=None):
    """[x0, x1, half width, z0, z1] of the part of a solid at x <= x_max."""
    if x_max is not None:
        solid = solid & Pos(x_max - 500, 0, 0) * Box(1000, 1000, 1000)
    bb = solid.bounding_box()
    return [bb.min.X, bb.max.X, max(abs(bb.min.Y), abs(bb.max.Y)), bb.min.Z, bb.max.Z]


@cache
def coxa_envelope(x_max=75.0, steps=9):
    """What swings with a hip (coxa) joint, as boxes for cad/v2/body.py's joint openings: (name, x0, x1, half width,
    z0, z1) in the coxa frame (x out along the link from the hip axis, z = body z). The coxa bracket with the thigh
    servo, and the thigh with its knee servo at `steps` angles over the widest thigh range (front and rear legs),
    clipped at x_max (farther out is clear of the shell)."""
    from math import degrees

    from cad.v2.servos import sts3215_proxy
    cx = coxa_link()
    slab = lambda z0, z1: Pos(0, 0, (z0 + z1) / 2) * Box(1000, 1000, z1 - z0)     # noqa: E731
    inner = Pos(-500 + LC - STS3215["half_w"] - WALL, 0, 0) * Box(1000, 1000, 1000)    # inside the thigh sleeve
    out = [("coxa_top", *_bbox(cx & slab(HIP_HORN_Z - 0.1, 100) & inner, x_max)),  # the plate on the hip horn
           ("coxa_bottom", *_bbox(cx & slab(-100, HIP_IDLER_Z + 0.1) & inner, x_max)),   # the plate under the idler
           ("coxa_wall", *_bbox(cx & slab(HIP_IDLER_Z, HIP_HORN_Z) & Pos(0, 0, 0) * Box(2 * (LC - STS3215["half_w"] - WALL) - 0.2, 1000, 1000), x_max)),
           ("thigh_servo", *_bbox(THIGH_SERVO * (sleeve() + sts3215_proxy()), x_max))]
    lo = min(PARAMS.leg.femur_range[0], PARAMS.leg.femur_range_for("FL")[0])
    hi = max(PARAMS.leg.femur_range[1], PARAMS.leg.femur_range_for("FL")[1])
    # the thigh as two boxes in its own frame (the plates and straps; the knee servo standing up), tilted to each angle
    # the thigh as its real pieces (a hub disc and a strap on each side, the web, the knee housing): one box over all of
    # them would sweep the empty space between the side plates over the shell
    knee = _bbox(KNEE_SERVO * (sleeve() + sts3215_proxy()))
    (y_h, y_i), t = PLATE_Y, PLATE / 2 + 0.2
    x0w, x1w = LF - STS3215["half_w"] - WALL, LF - STS3215["half_w"] - 0.4
    z0s, z1s = STRAP_Z
    pieces = [("hub_h", -HUB_R, HUB_R, (y_h - t, y_h + t), -HUB_R, HUB_R), ("hub_i", -HUB_R, HUB_R, (y_i - t, y_i + t), -HUB_R, HUB_R),
              ("strap_h", 0.0, x1w, (y_h - t, y_h + t), z0s, z1s), ("strap_i", 0.0, x1w, (y_i - t, y_i + t), z0s, z1s),
              ("web", x0w, x1w, (y_h - t, y_i + t), z0s, z1s), ("knee", *knee)]
    for k in range(steps):
        q = lo + (hi - lo) * k / (steps - 1)
        tilt = (degrees(-q), LC, 0.0)                                    # the thigh turns about -y at the thigh axis
        out += [(f"{n}{k}", *box, *tilt) for n, *box in pieces]
    return tuple(out)


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

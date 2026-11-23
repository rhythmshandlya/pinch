"""The printable v2 body: the pill shell split into a bottom TRAY and a top COVER, with every mount, window and opening.

    python cad/v2/body.py            # build, export cad/out/v2/print/{tray,cover}.stl, render the review picture

Units mm, body frame (x forward, y left, z up, hip plane z = 0). Everything is placed from the project's own numbers:
the shell from cad/v2/shell.py, the lid hinge from cad/v2/lid.py, the parts from configs/layout_v2.toml (crab.electronics
Layout), the joint ranges from sim/models/crab_v2.xml, and the servo interfaces agreed with the limbs CAD (below).

How it is built (all cached, a full build takes about a minute):
  tray  = lower shell (z <= SPLIT_Z) + a 2 mm lip + every internal mount, minus the cut-outs and the joint openings
  cover = upper shell + the lid's fixed hinge knuckle, minus the cut-outs (mic holes, grille, pad recesses, cable slot)
Six M2.5 heat-set inserts in tray bosses at the split, screws from the top through the cover's bosses.
`features()` lists every added or cut solid with the layout part it belongs to, so body_check.py can tell a part's
own mount from anything else in its way.
"""

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build123d import Align, Axis, Box, Cone, Cylinder, Location, Plane, Polyline, Pos, Solid, extrude, import_step, make_face, revolve  # noqa: E402, I001

from cad.servo import STS3032  # noqa: E402
from cad.servo import case as servo_case  # noqa: E402
from cad.v2 import arms as ARMS_CAD  # noqa: E402
from cad.v2 import legs as LEGS_CAD  # noqa: E402
from cad.v2 import lid as LID  # noqa: E402
from cad.v2 import shell as SHELL  # noqa: E402
from crab.electronics import Layout  # noqa: E402
from crab.shapes import BodyPlan, Shell  # noqa: E402

OUT = ROOT / "cad" / "out" / "v2" / "print"
RENDER = ROOT / "media" / "review" / "phase-02" / "cad_body.png"
MODEL_XML = ROOT / "sim" / "models" / "crab_v2.xml"

# ---------------------------------------------------------------------------------------------------- design numbers
WALL = SHELL.WALL
# the arm yaw servo (upside down): its origin from the shoulder point (layout) and the limbs CAD (horn face height)
SHOULDER_Z0 = float(Layout()["shoulder_l"].d["joint_mm"][2]) + ARMS_CAD.YAW_HORN_Z - ARMS_CAD.YAW_SIGN * STS3032.horn_face
SHOULDER_SX = -1 if ARMS_CAD.YAW_UP else 1          # horn up, the arm yaw servo's case runs forward from its shaft
SPLIT_Z = max(28.0, SHOULDER_Z0 + STS3032.T / 2 + 0.5)   # tray / cover parting plane: above the shoulder servo cases, so every
#                            servo, board and joint opening is in the tray and the cover is a lid with mics and pads
LIP_H, LIP_T, LIP_GAP = 2.0, 1.2, 0.2      # the tray's lip rises 2 mm inside the cover's wall, 0.2 mm slip fit
FIT = 0.3                  # clearance around bought parts (FDM +-0.2 mm, plus unknowns; cad/servo.py uses the same)
CLEAR = 2.0                # clearance around every moving part at a joint opening
SWEEP_STEP_DEG = 4.0       # joint openings: the moving parts' boxes stepped over the joint's range this often
PETG_G_CM3 = 1.27

M2_PILOT, M2_CLEAR = 1.7, 2.2              # self-tapping M2 into PETG / clearance (cad/servo.py)
M25_PILOT, M25_CLEAR = 2.1, 2.7            # thread-forming M2.5 into PETG / clearance
M25_INSERT, M25_INSERT_L = 3.6, 5.0        # hole for an M2.5 x 4 brass heat-set insert (4.0 OD - 0.3, deeper than the insert:
#                                            uwisc ME design lab rule); print a test coupon first
M25_HEAD = 5.0                             # counterbore for an M2.5 cap / pan head
BOSS_R, BOSS_INSET = 4.0, 3.0              # split bosses: radius, centre this far inside the inner wall
POST_R = {"M2": 2.25, "M2.5": 2.75}
PILOT = {"M2": M2_PILOT, "M2.5": M25_PILOT}

# Hip (leg yaw) servos, Feetech STS3215: the real model, cad/parts_ext/STS3215_03a.step (SO-ARM100, Apache-2.0), and the
# interface agreed with the limbs CAD (coordinator, 2026-11-22, revised): shaft vertical through the hip point, horn up,
# STEP origin at body z -5.2 (horn top face z +15.0, idler boss bottom z -24.6), the case running along body x toward
# the middle: front hips STEP +x -> body +x, rear hips rotated 180 deg; the STEP is moved by -STEP_SHAFT_X first.
STS3215_STEP = ROOT / "cad" / "parts_ext" / "STS3215_03a.step"
STEP_Z0 = -5.2
STEP_SHAFT_X = 11.25       # the interface's shaft position in the STEP (cad/v2/servos.py on the limbs branch)
STEP_SHAFT_MEASURED = 12.5  # where the STEP's horn and idler holes actually centre (7 mm circle): body_check reports it
STEP_CASE = (-22.7, 22.7, -12.4, 12.4, -17.7, 14.4)   # main case box (STEP frame): bottom face -17.7, top face 14.4
STEP_BOTTOM_HOLES = ((4.2, 10.25), (4.2, -10.25), (-20.3, 10.25), (-20.3, -10.25))    # dia 1.5, in recesses
HIP_KEEPOUT_R = 34.0       # coxa U-bracket swings within this radius ...
HIP_TAB_T = 5.0            # the hip seat's bottom tab under the case (audit B4: it was 1.5 mm, 0.3 mm under the heads)
HIP_PLATES = ((15.0, 18.0), (-27.6, -24.6))      # ... as a top plate on the horn and a bottom plate on the idler boss
# Moving envelope of one coxa (local x along the link from the hip axis, z in the body frame), (name, x0, x1, half
# width, z0, z1), from the limbs CAD (cad/v2/legs.py coxa_envelope): the coxa bracket with the thigh servo, and the thigh with the
# knee servo at nine angles over its range.
LEG_ENVELOPE = LEGS_CAD.coxa_envelope()

# Shoulder (arm yaw) servos, STS3032 (cad/servo.py profile), interface from the limbs CAD: shaft vertical through the
# shoulder point, mounted upside down (servo x -> body x, y -> -y, z -> -z), servo origin at SHOULDER_Z0 (the horn
# face 19.2 mm below it), ears screwed from above. Moving envelope of one arm about the yaw axis, from the limbs CAD (cad/v2/arms.py arm_envelope): the bracket with the lift servo, the upper arm at seven lift angles.
ARM_ENVELOPE = ARMS_CAD.arm_envelope(float(Layout()["shoulder_l"].d["joint_mm"][2]))

# Wall-mounted sensor apertures: radius of the window at the sensor face (the cone grows from it)
APERTURE_R = {"camera_lens": 8.5, "tof8": 4.0, "mic_l": 0.25, "mic_r": 0.25}   # mics: a 1.1 mm port (+ FIT) through the
#   2 mm wall over a foam gasket ring (audit I3; MEMS makers: a 1.0-1.2 mm port, short, sealed to the board)
TOF_APERTURE_R = 3.0       # VL53L0X emitter + receiver (both inside a 6 mm circle on the GY board)
WINDOW_CHECK_MM = 20.0     # every window is clear this far along its direction

STANDOFFS = {             # board -> (screw, inset of the holes from the board's edges); ESTIMATE where no drawing
    "radxa": ("M2.5", 3.5), "driver": ("M2.5", 3.0), "hub": ("M2.5", 3.0), "reg_a": ("M2", 2.5),
    "reg_b": ("M2", 2.5), "ina": ("M2.5", 2.5), "imu": ("M2.5", 2.5), "mpr121": ("M2.5", 2.5),
    "amp": ("M2.5", 2.5), "dock_board": ("M2", 2.0)}
WALL_BOARDS = ("tof8", "tof_fl", "tof_fr", "tof_rear", "tof_left", "tof_right")
CRADLES = ("fuse",)                              # (the XT60 rides in the battery bay on the pack's lead)
PADS = ("pad_head", "pad_back_left", "pad_back_right", "pad_side_left", "pad_side_right", "pad_face")
PAD_RECESS = 0.4           # copper tape recess depth
BOSS_XY = ((40.0, 50.0), (40.0, -50.0), (-25.0, 51.5), (-25.0, -51.5), (-58.0, 34.0), (-58.0, -34.0))   # the
#                            cover's six screws (audit B1): in the corners of the boxed body's solid walls at the seam:
#                            front corners behind the arm ports, mid-sides between the hips, the rear
VENT_X = (-6.0, -1.0, 4.0, 9.0, 14.0, 19.0)                 # belly intake slots under the computer
EXHAUST_X = (30.0, 34.0, 38.0, 42.0, 46.0)                  # top exhaust slots behind the lid hinge
NO_MOUNT = {"radxa_heatsink": "stuck on the radxa", "oled": "in the lid (lid.py)", "visor": "the lid (lid.py)",
            "harness": "mass only"}


# ---------------------------------------------------------------------------------------------------- small helpers
@dataclass(frozen=True)
class Feature:
    name: str
    owner: str             # layout part (or joint) it belongs to; "" = structure
    half: str              # "tray" | "cover"
    op: str                # "add" | "cut"
    solid: object
    clip: bool = True      # adds: keep only the part inside the shell's outer surface (fuses posts into the wall)
    trimmed: float = 0.0   # mm^3 the joint sweeps and the bracket zones took away from this add (body_check reports it)


def loc(center, R):
    R = np.asarray(R, float)
    return Location(Plane(origin=tuple(map(float, center)), x_dir=tuple(R[:, 0]), z_dir=tuple(R[:, 2])))


def obox(center, size, R=None):
    """A box of `size` centred at `center`, its axes the columns of R."""
    return loc(center, np.eye(3) if R is None else R) * Box(*map(float, size))


def abox(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return obox((lo + hi) / 2, hi - lo)


def zcyl(x, y, r, z0, z1):
    return Pos(x, y, (z0 + z1) / 2) * Cylinder(r, z1 - z0)


def axis_cyl(p, d, r, length, back=0.0):
    """Cylinder of radius r from p - back*d along unit d."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    base = np.asarray(p, float) - back * d
    return Location(Plane(origin=tuple(base), z_dir=tuple(d))) * Cylinder(r, length + back, align=(Align.CENTER, Align.CENTER, Align.MIN))


def cone(p, d, r0, half_deg, length, back=0.0):
    """The view cone of a window: radius r0 at p, opening at half_deg along unit d for `length`; starts `back` behind."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    t = math.tan(math.radians(half_deg))
    base = np.asarray(p, float) - back * d
    r_base = max(r0 - back * t, 0.05)
    return Location(Plane(origin=tuple(base), z_dir=tuple(d))) * Cone(r_base, r0 + length * t, length + back,
                                                                     align=(Align.CENTER, Align.CENTER, Align.MIN))


def fuse(solids):
    solids = [s for s in solids if s is not None]
    if not solids:
        return None
    out = solids[0]
    for s in solids[1:]:
        out = out + s
    return out


def volume(s):
    return 0.0 if s is None else float(sum(abs(x.volume) for x in s.solids())) if hasattr(s, "solids") else 0.0


# ---------------------------------------------------------------------------------------------------- inputs
@cache
def layout():
    return Layout()


@cache
def shell_math():
    return Shell.from_chassis(layout().chassis)


@cache
def joints():
    """name -> dict(pos mm, base yaw rad, axis sign, range rad) for the four coxae and the two shoulder yaws."""
    root = ET.parse(MODEL_XML).getroot()
    out = {}
    for body in root.iter("body"):
        name = body.get("name", "")
        if not (name.endswith("_coxa_link") or name.endswith("_shoulder")):
            continue
        j = body.find("joint")
        pos = np.array([float(v) for v in body.get("pos").split()]) * 1000.0
        yaw = float(body.get("euler", "0 0 0").split()[2])
        sign = float(j.get("axis").split()[2])
        lo, hi = (float(v) for v in j.get("range").split())
        key = name.split("_")[0]                         # FL, FR, RL, RR, L, R
        out[key] = dict(joint=j.get("name"), pos=pos, yaw=yaw, sign=sign, range=(lo, hi),
                        world=tuple(sorted((yaw + sign * lo, yaw + sign * hi))))
    return out


LEGS = ("FL", "FR", "RL", "RR")
ARMS = ("L", "R")


# The body in plan (owner, 2026-11-23: a clean front, no jagged cuts; audit B1, B7): the pill's top and bottom curves,
# cut by vertical walls to a rounded box between the leg bays, a short nose in front between the arms (the face plate:
# camera and the front ToF), and a chin under the arms (the front cliff sensors, the arm servos' seats). The legs and
# arms swing outside it, so the shell is no longer notched by their sweeps. Like Mini Pupper's and Petoi Bittle's box
# bodies and v1's look: legs outside a closed body, a domed lid on top.
_PLAN = Layout().chassis["plan"]   # the body's outline in plan (configs/layout_v2.toml chassis.plan; crab/shapes.py)
BODY_Y, FRONT_X = _PLAN["body_y"], _PLAN["front_x"]
NOSE_X, NOSE_Y = _PLAN["nose_x"], _PLAN["nose_y"]
CHIN_Y, CHIN_Z = _PLAN["chin_y"], _PLAN["chin_z"]
PLAN_R = 8.0               # corner radius in plan


def plan_box(d=0.0):
    """The body's plan as boxes (x0, x1, y0, y1, z0, z1), each inset by d: main body, nose, chin."""
    return BodyPlan(**_PLAN).boxes(d)


def plan_contains(P, d=0.0):
    """Points (..., 3) inside the plan prisms inset by d (numpy; for the cable router)."""
    return BodyPlan(**_PLAN).contains(P, d)


@cache
def plan_prism(d=0.0):
    """The plan's prisms inset by d, with rounded vertical edges (corner radius PLAN_R - d)."""
    from build123d import fillet
    parts = []
    for x0, x1, y0, y1, z0, z1 in plan_box(d):
        b = abox((x0, y0, z0), (x1, y1, z1))
        r = max(0.5, PLAN_R - d)
        edges = b.edges().filter_by(Axis.Z)
        try:
            b = fillet(edges, r)
        except Exception:                               # noqa: BLE001 (a thin box: keep it square)
            pass
        parts.append(b)
    return fuse(parts)


@cache
def outer():
    return _inset(0.0)


@cache
def inner():
    return _inset(WALL)


CHAMFER_H = 14.0           # the belly's edge cut at 45 deg up to this height above the flat (printability: the pill's
#                            belly curves up too gently from the bed to print without support; 45 deg prints clean)


@cache
def _bottom_chamfer(d=0.0):
    """Everything above a 45 deg chamfer round the flat belly, shrunk by d: per plan box a frustum from the flat (inset
    by CHAMFER_H + d) to CHAMFER_H above it (inset by d), and the full prism above that."""
    z0 = -51.7 + 1.0 + d                                   # the flat (Z_FLAT) shrunk by d
    parts = []
    for x0, x1, y0, y1, _zb, zt in plan_box(d):
        zt = min(zt, 300.0)
        if zt <= z0 + 0.5:
            continue                                     # the box (the chin, shrunk) ends below the flat
        h = min(CHAMFER_H, (min(x1 - x0, y1 - y0) - 2.0) / 2, zt - z0 - 0.5)   # narrow or short boxes: smaller
        if h > 0.5:
            lo = Polyline((x0 + h, y0 + h, z0), (x1 - h, y0 + h, z0), (x1 - h, y1 - h, z0), (x0 + h, y1 - h, z0), close=True)
            hi = Polyline((x0, y0, z0 + h), (x1, y0, z0 + h), (x1, y1, z0 + h), (x0, y1, z0 + h), close=True)
            parts.append(Solid.make_loft([lo, hi], ruled=True))
        else:
            h = 0.0
        parts.append(abox((x0, y0, z0 + h), (x1, y1, zt)))
    return fuse(parts)


@cache
def _notch(d=0.0):
    """The chin's notch in front of the camera grown by d (configs/layout_v2.toml chassis.plan notch): the wedge
    |y| < y + d, x > x - d, from the chin's top (shrunk by d) down to the plane falling at deg from the lens's lower
    edge (lowered by d)."""
    n = _PLAN.get("notch")
    if not n:
        return None
    t = math.tan(math.radians(n["deg"]))
    drop = d / math.cos(math.radians(n["deg"]))
    x0, x1, ztop = n["x"] - d, 200.0, CHIN_Z - d
    zp = lambda x: n["z"] - (x - n["x"]) * t - drop                      # noqa: E731
    face = make_face(Polyline((x0, ztop), (x1, ztop), (x1, zp(x1)), (x0, zp(x0)), close=True))
    w = n["y"] + d
    return Plane.XZ.offset(-w) * extrude(face, amount=2 * w)


@cache
def _inset(d):
    """The body's solid shrunk by d: the pill shrunk by d, inside the plan prisms shrunk by d, its belly edge
    chamfered at 45 deg, the camera's notch cut from the chin."""
    s = SHELL._solid(d) & plan_prism(d) & _bottom_chamfer(d)
    n = _notch(d)
    return s - n if n is not None else s


def below(z):
    return Pos(0, 0, z - 150) * Box(400, 400, 300)


def above(z):
    return Pos(0, 0, z + 150) * Box(400, 400, 300)


def section_point(angle_deg, z, inset):
    """Point on the shell's horizontal section at height z, `inset` mm inside the outer surface (approx.), at the
    polar angle (deg, about the shell centre) - the superellipse parameter v equals the polar angle here."""
    a, b, c, ctr, ev, eh = SHELL.dims(inset)
    w = abs((z - ctr[2]) / c)
    cu = max(1.0 - w ** (2 / ev), 0.0) ** (ev / 2)
    v = math.radians(angle_deg)
    sp = lambda q, e: math.copysign(abs(q) ** e, q)  # noqa: E731
    return np.array([ctr[0] + a * cu * sp(math.cos(v), eh), ctr[1] + b * cu * sp(math.sin(v), eh), z])


# ---------------------------------------------------------------------------------------------------- servos
@cache
def _sts3215_step():
    return import_step(str(STS3215_STEP))


def _hip_dir(leg):
    """+1 when the STEP's +x runs along body +x (front hips), -1 for the rear hips (rotated 180 deg)."""
    return 1.0 if joints()[leg]["pos"][0] > 0 else -1.0


def hip_location(leg):
    p, d = joints()[leg]["pos"], _hip_dir(leg)
    return Location((p[0], p[1], STEP_Z0), (0, 0, 0.0 if d > 0 else 180.0)) * Pos(-STEP_SHAFT_X, 0, 0)


def hip_to_body(leg, q):
    p, d = joints()[leg]["pos"], _hip_dir(leg)
    x, y, z = q
    return np.array([p[0] + d * (x - STEP_SHAFT_X), p[1] + d * y, z + STEP_Z0])


@cache
def hip_servo(leg):
    return hip_location(leg) * _sts3215_step()


def hip_case_aabb(leg):
    x0, x1, y0, y1, z0, z1 = STEP_CASE
    pts = np.array([hip_to_body(leg, (x, y, z)) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    return pts.min(0), pts.max(0)


def shoulder_location(arm):
    p = joints()[arm]["pos"]
    return Location((p[0], p[1], SHOULDER_Z0), (0, 0, 180.0) if ARMS_CAD.YAW_UP else (180.0, 0, 0))


@cache
def shoulder_servo(arm):
    return shoulder_location(arm) * servo_case(STS3032)


def shoulder_case_aabb(arm, ears=False):
    p, P = joints()[arm]["pos"], STS3032
    ext = P.ear_len if ears else 0.0
    xa, xb = sorted((p[0] + SHOULDER_SX * (P.x_rear - ext), p[0] + SHOULDER_SX * (P.x_out + ext)))
    return (np.array([xa, p[1] - P.half_w, SHOULDER_Z0 - P.face]),
            np.array([xb, p[1] + P.half_w, SHOULDER_Z0 + P.face]))


# ---------------------------------------------------------------------------------------------------- joint sweeps
def _sector(cx, cy, r0, r1, a0, a1, z0, z1, step_deg=2.0):
    n = max(2, int(math.ceil((a1 - a0) / math.radians(step_deg))) + 1)
    ts = np.linspace(a0, a1, n)
    rc = r1 / math.cos((a1 - a0) / (n - 1) / 2)          # circumscribed: the polygon never cuts inside the arc
    pts = [(cx + rc * math.cos(t), cy + rc * math.sin(t)) for t in ts]
    pts += [(cx + r0 * math.cos(t), cy + r0 * math.sin(t)) for t in ts[::-1]] if r0 > 0 else [(cx, cy)]
    face = make_face(Polyline(*[(x, y, z0) for x, y in pts], close=True))
    return extrude(face, amount=z1 - z0)


def _env_box(cx, cy, t, x0, x1, hw, z0, z1, tilt=None):
    """A moving part's box at joint yaw t. tilt = (deg about y, pivot x, pivot z): the box is in a link frame tilted
    about the pivot (a thigh at one of its angles), so it follows the part instead of a box square to the hip."""
    loc = Location((cx, cy, 0), (0, 0, math.degrees(t)))
    if tilt:
        deg, px, pz = tilt
        loc = loc * Location((px, 0, pz), (0, deg, 0))
    y0, y1 = hw if isinstance(hw, tuple) else (-hw, hw)
    return loc * abox((x0, y0, z0), (x1, y1, z1))


def envelope_at(key, q, grow=0.0):
    """The moving envelope of a joint (leg or arm) at joint angle q (rad), grown by `grow` mm."""
    j = joints()[key]
    t = j["yaw"] + j["sign"] * q
    cx, cy = j["pos"][:2]
    parts = []
    for e in (LEG_ENVELOPE if key in LEGS else ARM_ENVELOPE):
        if e[1] == "cyl":
            parts.append(zcyl(cx, cy, e[2] + grow, e[3] - grow, e[4] + grow))
        else:
            _, x0, x1, hw, z0, z1, *tilt = e
            parts.append(_env_box(cx, cy, t, x0 - grow, x1 + grow, _grow_y(hw, grow), z0 - grow, z1 + grow, tilt or None))
    return fuse(parts)


@cache
def sweep(key, grow=0.0):
    """Volume swept by a joint's moving envelope over its range (grown by `grow`): each box stepped over the range."""
    j = joints()[key]
    a0, a1 = j["world"]
    cx, cy = j["pos"][:2]
    parts = []
    for e in (LEG_ENVELOPE if key in LEGS else ARM_ENVELOPE):
        if e[1] == "cyl":
            parts.append(zcyl(cx, cy, e[2] + grow, e[3] - grow, e[4] + grow))
            continue
        _, x0, x1, hw, z0, z1, *tilt = e
        parts.append(_revolved(cx, cy, a0, a1, (x0 - grow, x1 + grow, _grow_y(hw, grow), z0 - grow, z1 + grow), tilt or None))
    return fuse(parts)


def _grow_y(hw, grow):
    """A box's sideways extent, symmetric (half width) or (y0, y1), grown by `grow`."""
    return (hw[0] - grow, hw[1] + grow) if isinstance(hw, tuple) else hw + grow


def _hull2d(pts):
    """Convex hull of 2-D points (monotone chain), counter-clockwise."""
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def _revolved(cx, cy, a0, a1, box, tilt=None):
    """The volume a box (in the joint's frame, optionally tilted about a pivot) sweeps as the joint turns from a0 to a1
    about the vertical axis through (cx, cy): its outline in (radius, z) revolved over the turn plus the box's own
    angular width. Exact up to the convex outline; one solid instead of a box per step."""
    x0, x1, hw, z0, z1 = box
    y0, y1 = hw if isinstance(hw, tuple) else (-hw, hw)
    g = np.linspace(0.0, 1.0, 7)
    P = np.array([(x0 + (x1 - x0) * u, y0 + (y1 - y0) * v, z0 + (z1 - z0) * w) for u in g for v in g for w in g
                  if min(u, 1 - u) == 0 or min(v, 1 - v) == 0 or min(w, 1 - w) == 0])
    if tilt:
        deg, px, pz = tilt
        c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        R = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
        P = P @ R.T + np.array([px, 0.0, pz])
    r, z, phi = np.hypot(P[:, 0], P[:, 1]), P[:, 2], np.arctan2(P[:, 1], P[:, 0])
    if phi.max() - phi.min() > math.pi:
        return _revolved_rings(cx, cy, a0, a1, box, tilt)       # it straddles the axis: ring by ring
    outline = _hull2d([(round(float(a), 4), round(float(b), 4)) for a, b in zip(r, z)])
    if (a1 - a0) + (phi.max() - phi.min()) >= 2 * math.pi - 1e-3:
        start, arc = 0.0, 2 * math.pi
    else:
        start, arc = a0 + phi.min(), (a1 - a0) + (phi.max() - phi.min())
    # a ruled loft through the outline every <= 10 deg (OpenCascade's booleans of a true revolve against the shell's
    # spline surface fail silently for some poses); the outline is pushed out by the chord's sag so the loft covers
    # the true sweep
    n = max(2, int(math.ceil(math.degrees(arc) / 10.0)) + 1)
    sag = 1.0 / math.cos(arc / (n - 1) / 2)
    wires = []
    for k in range(n):
        t = start + arc * k / (n - 1)
        c, s_ = math.cos(t), math.sin(t)
        wires.append(Polyline(*[(a * sag * c, a * sag * s_, b) for a, b in outline], close=True))
    if arc >= 2 * math.pi - 1e-6:                             # a full ring: revolve is fine there (no end faces)
        face = make_face(Polyline(*[(a * sag, 0.0, b) for a, b in outline], close=True))
        return Pos(cx, cy, 0) * revolve(face, Axis.Z, 360)
    return Pos(cx, cy, 0) * Solid.make_loft(wires, ruled=True)



RING_MM = 4.0              # a part straddling its joint's axis is swept ring by ring this wide


def _sector(r0, r1, z0, z1, start, arc):
    """An annular sector (radii r0..r1, heights z0..z1, from angle `start` over `arc`), about the z axis; the outer
    radius pushed out by the chord's sag (ruled loft every <= 10 deg; a full ring is revolved)."""
    if arc >= 2 * math.pi - 1e-6:
        face = make_face(Polyline((r0, 0.0, z0), (r1, 0.0, z0), (r1, 0.0, z1), (r0, 0.0, z1), close=True))
        return revolve(face, Axis.Z, 360)
    n = max(2, int(math.ceil(math.degrees(arc) / 10.0)) + 1)
    r1 = r1 / math.cos(arc / (n - 1) / 2)
    wires = []
    for k in range(n):
        t = start + arc * k / (n - 1)
        c, s_ = math.cos(t), math.sin(t)
        wires.append(Polyline(*[(a * c, a * s_, b) for a, b in ((r0, z0), (r1, z0), (r1, z1), (r0, z1))], close=True))
    return Solid.make_loft(wires, ruled=True)


def _revolved_rings(cx, cy, a0, a1, box, tilt=None):
    """The sweep of a box that straddles the joint's axis (a bracket plate round the shaft, an arm's root): revolving
    its whole outline would be a full disc out to its far end, so it is cut into rings about the axis, each swept only
    over its own angular width (sampled through the box's volume) plus the turn."""
    x0, x1, hw, z0, z1 = box
    y0, y1 = hw if isinstance(hw, tuple) else (-hw, hw)
    g = np.linspace(0.0, 1.0, 17)
    U, V, W = np.meshgrid(g, g, g, indexing="ij")
    P = np.column_stack([(x0 + (x1 - x0) * U).ravel(), (y0 + (y1 - y0) * V).ravel(), (z0 + (z1 - z0) * W).ravel()])
    if tilt:
        deg, px, pz = tilt
        c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        P = P @ np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]]).T + np.array([px, 0.0, pz])
    r, z, phi = np.hypot(P[:, 0], P[:, 1]), P[:, 2], np.arctan2(P[:, 1], P[:, 0])
    step = max(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)) / 16.0         # the sampling pitch: widen each ring by it
    edges = np.arange(0.0, r.max() + RING_MM, RING_MM)
    parts = []
    for ra, rb in zip(edges[:-1], edges[1:]):
        k = (r >= ra - step) & (r <= rb + step)
        if not k.any():
            continue
        ph = np.sort(phi[k])
        gaps = np.diff(np.concatenate([ph, ph[:1] + 2 * math.pi]))        # the band's angular extent: the circle less its
        j = int(np.argmax(gaps))                                          # largest empty gap
        width = 2 * math.pi - gaps[j] + 2 * math.asin(min(1.0, step / max(ra, step)))
        lo = ph[(j + 1) % len(ph)] - math.asin(min(1.0, step / max(ra, step)))
        arc = (a1 - a0) + width
        za, zb = float(z[k].min()) - step / 2, float(z[k].max()) + step / 2
        parts.append(_sector(float(ra), float(rb), za, zb, a0 + lo, min(arc, 2 * math.pi)))
    return Pos(cx, cy, 0) * fuse(parts)

@cache
def hip_keepouts(leg, grow=0.0):
    """The free zones around a hip: the volumes the U-bracket's two plates (the limbs CAD's coxa_top and coxa_bottom)
    sweep over the leg's turn, ring by ring about the shaft. (They were full 36 mm cylinders round the shaft, which
    cut a slot along the whole flank above and below the hip servos; the plates only turn through the leg's range.)"""
    j = joints()[leg]
    cx, cy = j["pos"][:2]
    a0, a1 = j["world"]
    out = []
    for e in LEG_ENVELOPE:
        if e[0] in ("coxa_top", "coxa_bottom"):
            _, x0, x1, hw, z0, z1, *tilt = e
            out.append(_revolved(cx, cy, a0, a1, (x0 - grow, x1 + grow, _grow_y(hw, grow), z0 - grow, z1 + grow),
                                 tilt or None))
    return out


# ---------------------------------------------------------------------------------------------------- features
def _column_ok(lo_xy, hi_xy, z_top, owner, z_bot=-60.0):
    """A vertical column (an axis-aligned footprint, from the floor - or z_bot - to z_top) stays 0.5 mm clear of
    every other part and servo, outside the hip brackets' bottom-plate zone and the shoulder brackets."""
    lay = layout()
    lo = np.array([lo_xy[0] - 0.5, lo_xy[1] - 0.5, z_bot])
    hi = np.array([hi_xy[0] + 0.5, hi_xy[1] + 0.5, z_top - 0.01])
    for n, p in lay.parts.items():
        if n == owner or p.mass_only or p.d.get("in_model") or n in ("oled", "visor") or p.on == owner:
            continue
        c = p.corners()
        if np.all(c.max(0) > lo) and np.all(c.min(0) < hi):
            return False
    for blo, bhi in [hip_case_aabb(leg) for leg in LEGS] + [shoulder_case_aabb(a, ears=True) for a in ARMS]:
        if np.all(bhi > lo) and np.all(blo < hi):
            return False
    ctr, rad = (lo[:2] + hi[:2]) / 2, float(np.hypot(*(hi[:2] - lo[:2]))) / 2
    for leg in LEGS:                                   # the bottom plate's free zone (columns cross it on their way down)
        p = joints()[leg]["pos"]
        if (math.hypot(ctr[0] - p[0], ctr[1] - p[1]) < HIP_KEEPOUT_R + rad and z_top > HIP_PLATES[1][0] - CLEAR
                and z_bot < HIP_PLATES[1][1] + CLEAR):
            return False
    for arm in ARMS:
        p = joints()[arm]["pos"]
        if math.hypot(ctr[0] - p[0], ctr[1] - p[1]) < 13.0 + CLEAR + rad:
            return False
    return True


def _post_ok(xy, r, z_top, owner):
    return _column_ok((xy[0] - r, xy[1] - r), (xy[0] + r, xy[1] + r), z_top, owner)


def _edge_ribs(p, name, bottom, width=1.6, step=2.0, min_len=6.0):
    """Ribs (1.6 mm walls from the floor) under a board's edge strips, where nothing stands below them."""
    out = []
    for ax in (0, 1):                                  # the rib runs along axis `ax` of the part, on side sg of 1-ax
        for sg in (-1, 1):
            n = int(2 * p.half[ax] // step)
            ok = []
            for k in range(n):
                q0, q1 = np.zeros(3), np.zeros(3)
                q0[ax], q1[ax] = -p.half[ax] + k * step, -p.half[ax] + (k + 1) * step
                q0[1 - ax], q1[1 - ax] = sorted((sg * (p.half[1 - ax] - width), sg * p.half[1 - ax]))
                w = np.array([p.world(q0), p.world(q1)])
                ok.append(_column_ok(w.min(0)[:2], w.max(0)[:2], bottom, name))
            k = 0
            while k < n:
                if not ok[k]:
                    k += 1
                    continue
                k0 = k
                while k < n and ok[k]:
                    k += 1
                if (k - k0) * step >= min_len:
                    q0, q1 = np.zeros(3), np.zeros(3)
                    q0[ax], q1[ax] = -p.half[ax] + k0 * step, -p.half[ax] + k * step
                    q0[1 - ax], q1[1 - ax] = sorted((sg * (p.half[1 - ax] - width), sg * p.half[1 - ax]))
                    q0[2], q1[2] = -p.half[2] - 80.0, -p.half[2]
                    out.append(obox(p.world((q0 + q1) / 2), q1 - q0, p.R))
    return out


def standoffs(name):
    """Posts under a board's corner holes; for a board stacked over other parts (fewer than two free corners), ribs
    under the edge strips that have nothing below them (the board is glued or zip-tied to them)."""
    p = layout()[name]
    screw, inset = STANDOFFS[name]
    r = POST_R[screw]
    bottom = p.corners()[:, 2].min()
    out, skipped = [], []
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = p.world((sx * (p.half[0] - inset), sy * (p.half[1] - inset), -p.half[2]))
            if _post_ok(c[:2], r, bottom, name):
                post = zcyl(c[0], c[1], r, -60.0, bottom) - zcyl(c[0], c[1], PILOT[screw] / 2, bottom - 6.0, bottom + 0.1)
                out.append(Feature(f"{name}_post{len(out)}", name, "tray", "add", post))
            else:
                skipped.append((sx, sy))
    if len(out) >= 2:
        return out, skipped
    for rib in _edge_ribs(p, name, bottom):            # boards stacked over other parts: ribs under the free edges
        out.append(Feature(f"{name}_rib{len(out)}", name, "tray", "add", rib))
    if len(out) < 2:                                   # still not held: beams under it to pillars beside the parts below
        for beam in _bridges(p, name, bottom):
            out.append(Feature(f"{name}_bridge{len(out)}", name, "tray", "add", beam))
    return out, skipped


DECK_Z = (-15.5, -14.0)    # the electronics deck: a 1.5 mm plate across the body over the hub, monitor, driver and pack
DECK_X = (-60.0, 44.0)
DECK_BOARDS = ("radxa", "imu", "mpr121", "reg_a", "reg_b", "amp", "dock_board")   # screwed to the deck, not the floor
DECK_SLOT = 6.0            # a cable slot's half size round each crossing


@cache
def deck():
    """The electronics deck (audit B3: most boards were glued to one rib or held by nothing), a separate part printed
    flat on the bed with no supports (printability: inside the tray it would be an 80 mm bridge): a plate across the
    body over the boards on the floor, every board above it screwed to posts printed on it; a slot wherever a routed
    cable crosses its height (cad/v2/routing.py); a step round any part that passes it; four holes onto the tray's
    deck posts (M2.5 into heat-set inserts). Body frame, as fitted."""
    from cad.v2 import wiring
    plate = abox((DECK_X[0], -BODY_Y, DECK_Z[0]), (DECK_X[1], BODY_Y, DECK_Z[1])) & _inset(WALL + 0.4)
    slots = []
    for _name, _kind, pts, dia, *_ in wiring.cables():
        P = np.asarray(pts, float)
        for a, b in zip(P[:-1], P[1:]):
            if min(a[2], b[2]) < DECK_Z[1] and max(a[2], b[2]) > DECK_Z[0]:
                t = np.clip((sum(DECK_Z) / 2 - a[2]) / (b[2] - a[2]), 0.0, 1.0) if b[2] != a[2] else 0.5
                q = a + (b - a) * t
                h = max(DECK_SLOT, dia / 2 + 2.0)
                slots.append(abox((q[0] - h, q[1] - h, DECK_Z[0] - 1), (q[0] + h, q[1] + h, DECK_Z[1] + 1)))
    lay = layout()
    for n, part in lay.parts.items():                   # parts that pass through the deck's height
        if part.mass_only or n in ("oled", "visor") or part.d.get("in_model") or n in DECK_BOARDS:
            continue
        c = part.corners()
        if c[:, 2].min() < DECK_Z[1] + FIT and c[:, 2].max() > DECK_Z[0] - FIT:
            slots.append(obox(part.pos, part.size + 2 * (FIT + 0.5), part.R))
    for leg in LEGS:                                     # and the hip servo cases at the walls
        lo, hi = hip_case_aabb(leg)
        slots.append(abox(lo - FIT - 0.5, hi + FIT + 0.5))
    d = plate - fuse(slots) if slots else plate
    kept = []
    for n in DECK_BOARDS:                                # a post stands only where the deck is solid under it
        ps = [f.solid for f in deck_posts(n) if not _empty(f.solid & d)]
        if len(ps) < 2:
            print(f"deck: {n} has {len(ps)} post(s) on solid deck", file=sys.stderr)
        kept += ps
    holes = fuse([zcyl(x, y, M25_CLEAR / 2, DECK_Z[0] - 1.0, DECK_Z[1] + 1.0) for x, y in deck_mounts()])
    return (d + fuse(kept) if kept else d) - holes


@cache
def deck_mounts():
    """Where the deck screws down: four points spread over it whose columns to the floor are free."""
    cands = []
    for x in np.arange(DECK_X[0] + 6.0, DECK_X[1] - 5.0, 2.0):
        for y in np.arange(-BODY_Y + WALL + 5.0, BODY_Y - WALL - 4.0, 2.0):
            if _post_ok((x, y), BOSS_R, DECK_Z[0], "deck") and bool(plan_contains(np.array([x, y, DECK_Z[0]]), WALL + BOSS_R)):
                cands.append((float(x), float(y)))
    out = []
    for tx, ty in ((DECK_X[0], -BODY_Y), (DECK_X[0], BODY_Y), (DECK_X[1], -BODY_Y), (DECK_X[1], BODY_Y)):
        free = [c for c in cands if all(math.hypot(c[0] - o[0], c[1] - o[1]) > 25.0 for o in out)]
        if free:
            out.append(min(free, key=lambda c: (c[0] - tx) ** 2 + (c[1] - ty) ** 2))
    return out


def deck_supports():
    """The tray's four posts under the deck, each with a brass M2.5 heat-set insert at its top."""
    out = []
    for k, (x, y) in enumerate(deck_mounts()):
        post = zcyl(x, y, BOSS_R, -60.0, DECK_Z[0]) - zcyl(x, y, M25_INSERT / 2, DECK_Z[0] - M25_INSERT_L, DECK_Z[0] + 0.1)
        out.append(Feature(f"deck_post{k}", "", "tray", "add", post))
    return out


def deck_posts(name):
    """Posts from the deck up to a board's corner holes, with pilots for its screws."""
    p = layout()[name]
    screw, inset = STANDOFFS[name]
    r = POST_R[screw]
    bottom = p.corners()[:, 2].min()
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = p.world((sx * (p.half[0] - inset), sy * (p.half[1] - inset), -p.half[2]))
            if bottom - DECK_Z[1] < 0.5:
                continue
            post = zcyl(c[0], c[1], r, DECK_Z[1] - 0.5, bottom) - zcyl(c[0], c[1], PILOT[screw] / 2, bottom - 6.0, bottom + 0.1)
            out.append(Feature(f"{name}_deckpost{len(out)}", name, "tray", "add", post))
    return out

def _bridges(p, name, bottom, depth=2.5, width=3.0, reach=40.0, step=2.0):
    """Up to two beams under a board (depth mm under its bottom face, along one of its axes), each carried by two
    pillars found along the beam's line beyond the board's ends, where a column to the floor is free."""
    out = []
    for ax in (0, 1):
        for off in (-1, 1):
            lat = off * (p.half[1 - ax] - width / 2 - 1.0)
            ends = []
            for sg in (-1, 1):
                for k in range(int(reach / step) + 1):
                    q = np.zeros(3)
                    q[ax] = sg * (p.half[ax] - 2.5 + k * step)
                    q[1 - ax] = lat
                    c = p.world(q)
                    if _post_ok(c[:2], 2.5, bottom, name):
                        ends.append(sg * (p.half[ax] - 2.5 + k * step))
                        break
            if len(ends) < 2:
                continue
            q0, q1 = np.zeros(3), np.zeros(3)
            q0[ax], q1[ax] = ends[0] - 2.5, ends[1] + 2.5
            q0[1 - ax], q1[1 - ax] = lat - width / 2, lat + width / 2
            q0[2], q1[2] = -p.half[2] - depth, -p.half[2]
            w = np.array([p.world(q0), p.world(q1)])
            if not _column_ok(w.min(0)[:2], w.max(0)[:2], bottom, name, z_bot=bottom - depth):
                continue
            beam = obox(p.world((q0 + q1) / 2), q1 - q0, p.R)
            for e in ends:
                q = np.zeros(3)
                q[ax], q[1 - ax] = e, lat
                c = p.world(q)
                beam = beam + zcyl(c[0], c[1], 2.5, -60.0, bottom - depth + 0.5)
            out.append(beam)
            if len(out) == 2:
                return out
        if out:
            return out
    return out


def wall_board(name):
    """A sensor board sitting in the wall (window flush outside): a pocket through the wall, the view cone, and a
    backing bar across each end of the board's inner face; the board is pressed in from outside, a drop of glue."""
    p = layout()[name]
    wp, wd = p.window_pose()
    k = int(np.argmax(np.abs(np.asarray(p.window["dir"], float))))
    k_out = float(np.sign(p.window["dir"][k]))
    i, j = sorted((a for a in range(3) if a != k), key=lambda a: -p.half[a])
    h = p.half
    half = "tray" if p.pos[2] < SPLIT_Z else "cover"
    feats = [Feature(f"{name}_pocket", name, half, "cut", obox(p.pos, p.size + 2 * FIT, p.R))]
    r0 = APERTURE_R.get(name, TOF_APERTURE_R)
    feats.append(Feature(f"{name}_window", name, half, "cut",
                         cone(wp, wd, r0 + FIT, p.window.get("cone_deg", 10.0), WINDOW_CHECK_MM + 2, back=0.5)))
    for si in (-1, 1):
        lo, hi = np.zeros(3), np.zeros(3)
        lo[i], hi[i] = sorted((si * (h[i] - 4.0), si * h[i]))
        lo[j], hi[j] = -(h[j] + FIT + 2.0), h[j] + FIT + 2.0
        back = sorted((-k_out * (h[k] + FIT), -k_out * (h[k] + FIT + 1.5)))
        lo[k], hi[k] = back
        bar = obox(p.world((lo + hi) / 2), hi - lo, p.R)
        for sj in (-1, 1):                      # legs from the bar out into the wall
            llo, lhi = lo.copy(), hi.copy()
            llo[j], lhi[j] = sorted((sj * (h[j] + FIT), sj * (h[j] + FIT + 2.0)))
            llo[k], lhi[k] = sorted((back[0] if k_out > 0 else back[1], k_out * (h[k] + 4.0)))
            bar = bar + obox(p.world((llo + lhi) / 2), lhi - llo, p.R)
        feats.append(Feature(f"{name}_bar{si:+d}", name, half, "add", bar))
    return feats


def camera_mount():
    cam, lens = layout()["camera"], layout()["camera_lens"]
    feats = []
    x = cam.pos[0]
    top = cam.corners()[:, 2].max()
    bottom = cam.corners()[:, 2].min()
    for s in (-1, 1):                               # two slotted posts hold the board's edges
        y0 = s * (cam.half[1] - 1.5)
        post = abox((x - 2.5, min(y0, y0 + s * 5.5), -60), (x + 2.5, max(y0, y0 + s * 5.5), top - 4.0))
        slot = abox((x - cam.half[0] - FIT, min(y0 - s, s * (cam.half[1] + FIT)), bottom - 1.0),
                    (x + cam.half[0] + FIT, max(y0 - s, s * (cam.half[1] + FIT)), top + 1.0))
        feats.append(Feature(f"camera_post{s:+d}", "camera", "tray", "add", post - slot))
    wp, wd = lens.window_pose()
    feats.append(Feature("camera_lens_pocket", "camera_lens", "tray", "cut", obox(lens.pos, lens.size + 2 * FIT, lens.R)))
    feats.append(Feature("camera_window", "camera_lens", "tray", "cut",
                         cone(wp, wd, APERTURE_R["camera_lens"] + FIT, lens.window["cone_deg"], WINDOW_CHECK_MM + 2, back=0.5)))
    return feats


def cradle(name):
    """A U-channel for a power part (XT60 pair, fuse holder): a block from the floor, two side walls to half height,
    a zip-tie tunnel under the part."""
    p = layout()[name]
    h = p.half
    i = int(np.argmax(h[:2]))                    # long horizontal axis (part frame)
    j = 1 - i
    lo, hi = -h - FIT, h + FIT
    feats = []
    base_lo, base_hi = lo.copy(), hi.copy()
    base_lo[i], base_hi[i] = -(h[i] - 2.0), h[i] - 2.0
    base_lo[j], base_hi[j] = -(h[j] + FIT + 1.6), h[j] + FIT + 1.6
    base_lo[2], base_hi[2] = -h[2] - 60.0, -h[2]
    base = obox(p.world((base_lo + base_hi) / 2), base_hi - base_lo, p.R)
    for s in (-1, 1):
        wlo, whi = base_lo.copy(), base_hi.copy()
        wlo[j], whi[j] = sorted((s * (h[j] + FIT), s * (h[j] + FIT + 1.6)))
        wlo[2], whi[2] = -h[2], 0.0
        base = base + obox(p.world((wlo + whi) / 2), whi - wlo, p.R)
    tlo, thi = np.zeros(3), np.zeros(3)
    tlo[i], thi[i] = -2.0, 2.0
    tlo[j], thi[j] = -(h[j] + 6.0), h[j] + 6.0
    tlo[2], thi[2] = -h[2] - 4.0, -h[2] - 1.5
    base = base - obox(p.world((tlo + thi) / 2), thi - tlo, p.R)
    feats.append(Feature(f"{name}_cradle", name, "tray", "add", base))
    return feats


HATCH_BAND, HATCH_GAP, HATCH_FRAME = 4.0, 0.25, 2.0     # rebate width round the opening, door clearance, frame thickness


def _hatch_regions():
    """The battery hatch under the pack: the through opening (small) and the rebate the door's rim sits in (large),
    as boxes below the pack's underside, and the four screw positions in the rebate."""
    lay = layout()
    c = np.vstack([lay["pack"].corners(), lay["xt60"].corners()])   # the pack and its plug come out together
    lo, hi = c.min(0), c.max(0)
    top = lay["pack"].corners()[:, 2].min() - 0.5
    small = (lo[0] - 2.0, lo[1] - 2.0, hi[0] + 2.0, hi[1] + 2.0)
    large = (small[0] - HATCH_BAND, small[1] - HATCH_BAND, small[2] + HATCH_BAND, small[3] + HATCH_BAND)
    m = HATCH_BAND / 2
    yc = (small[1] + small[3]) / 2
    screws = [(small[2] + m, yc)]          # one M2.5 into a heat-set insert at the front end; a hook at the rear end
    return small, large, top, screws


def _region(r, top, grow=0.0):
    return abox((r[0] - grow, r[1] - grow, -90.0), (r[2] + grow, r[3] + grow, top))


def battery_cradle():
    """The battery bay: walls round the pack to its top (a fire barrier between the Li-ion pack and the boards, and it
    cannot shift in a fall), a strap slot over it, and a hatch in the floor so the pack comes out from below without
    opening the shell: a rebated opening with a 2 mm frame inside for four M2 screws; the door (battery_door) carries
    the pack's support rails."""
    p = layout()["pack"]
    lo, hi = p.corners().min(0), p.corners().max(0)
    small, large, top, screws = _hatch_regions()
    # the walls stand on the solid rim round the hatch opening (2 mm round the pack: a 1 mm foam pad each side)
    walls = []
    for y0, y1 in ((small[1] - 2.0, small[1]), (small[3], small[3] + 2.0)):
        walls.append(abox((small[0] - 2.0, y0, -60.0), (small[2] + 2.0, y1, hi[2] + 1.0)))
    for x0, x1 in ((small[0] - 2.0, small[0]), (small[2], small[2] + 2.0)):
        walls.append(abox((x0, small[1] - 2.0, -60.0), (x1, small[3] + 2.0, hi[2] + 1.0)))
    wire = abox((small[2] - 0.1, p.pos[1] - 6.0, hi[2] - 9.0), (small[2] + 2.1, p.pos[1] + 6.0, hi[2] + 1.1))  # the lead out
    strap = abox((p.pos[0] - 11.0, lo[1] - 6.0, hi[2] - 6.0), (p.pos[0] + 11.0, hi[1] + 6.0, hi[2] - 3.0))
    frame = (_inset(WALL) - _inset(WALL + HATCH_FRAME)) & _region(large, top, 1.0) - _region(small, top)
    # the door's screw: a boss inside the frame with a brass M2.5 heat-set insert pressed in from below (audit B2:
    # self-tapping M2 into 3 mm of PETG strips after a few swaps)
    bosses = fuse([zcyl(x, y, BOSS_R, -90.0, top) & _inset(WALL - 0.5) for x, y in screws])
    holes = fuse([zcyl(x, y, M25_INSERT / 2, -90.0, top - 1.5) for x, y in screws])
    hook_slot = abox((small[0] - 2.6, (small[1] + small[3]) / 2 - 7.0, top - 0.2),
                     (small[0] + 0.1, (small[1] + small[3]) / 2 + 7.0, top + 2.2))   # the door's hook passes under the bay wall
    rebate = (outer() - _inset(WALL / 2)) & _region(large, top)
    opening = _region(small, top)
    walls = [w - hook_slot for w in walls]
    return [Feature("pack_bay", "pack", "tray", "add", fuse(walls) - wire - strap),
            Feature("pack_hatch_frame", "pack", "tray", "add", frame + bosses - holes),
            Feature("pack_hatch_rebate", "pack", "tray", "cut", rebate),
            Feature("pack_hatch_opening", "pack", "tray", "cut", opening)]


@cache
def battery_door():
    """The hatch door under the battery: the shell's skin over the opening, its outer half spread over the rebate, two
    rails the pack rests on, four M2 clearance holes. Printed separately (body frame, as fitted)."""
    p = layout()["pack"]
    lo, hi = p.corners().min(0), p.corners().max(0)
    small, large, top, screws = _hatch_regions()
    g = -HATCH_GAP
    skin = ((outer() - _inset(WALL)) & _region(small, top, g)) + ((outer() - _inset(WALL / 2)) & _region(large, top, g))
    rails = fuse([abox((lo[0] + 4.0, yc - 1.5, -90.0), (hi[0] - 4.0, yc + 1.5, top)) for yc in (p.pos[1] - 12.0, p.pos[1] + 12.0)])
    rails = rails & _inset(WALL / 2)
    skin = skin & above(Z_FLAT)                          # flush with the flat belly
    holes = fuse([zcyl(x, y, M25_CLEAR / 2, -90.0, top) for x, y in screws])
    # the hook: a tongue at the rear end that goes in first under the frame, then the door swings shut onto its screw
    yc = (small[1] + small[3]) / 2
    hook = abox((small[0] - 2.4, yc - 6.0, top - 0.1), (small[0] + 4.0, yc + 6.0, top + 1.8))
    return skin + rails + hook - holes


def speaker_mount():
    """A ring round the speaker on the cover's inside (it hangs from the cover, face against the skin over a 1 mm foam
    gasket, glued round its rim): sound goes straight out of the grille instead of a 40 mm box (audit I2)."""
    p = layout()["speaker"]
    r_in = p.half[0] + FIT
    n = p.R[:, 2]
    face = p.pos + n * p.half[2]
    ring = (axis_cyl(face - n * (p.size[2] + 1.0), n, r_in + 1.6, p.size[2] + 12.0)
            - axis_cyl(face - n * (p.size[2] + 2.0), n, r_in, p.size[2] + 14.0)) & _inset(WALL - 0.5)
    return [Feature("speaker_ring", "speaker", "cover", "add", ring)]


def grille():
    """Speaker grille in the cover over the speaker, along the speaker's axis: 1.5 mm holes on a 2.2 mm hex grid over
    the cone (about 40 % open; audit I2 asked >= 25 %), kept off the pads and mics."""
    lay = layout()
    p = lay["speaker"]
    n, u, v = p.R[:, 2], p.R[:, 0], p.R[:, 1]
    face = p.pos + n * p.half[2]
    keep_off = [lay[k].corners() for k in ("pad_back_left", "pad_back_right", "mic_l", "mic_r")]
    holes = []
    pitch = 2.2
    for row in range(-8, 9):
        for col in range(-8, 9):
            a = pitch * col + (pitch / 2 if row % 2 else 0.0)
            b = pitch * row * math.sqrt(3) / 2
            if math.hypot(a, b) > 11.5:
                continue
            c = face + a * u + b * v
            if any(k[:, 0].min() - 2.5 < c[0] < k[:, 0].max() + 2.5 and k[:, 1].min() - 2.5 < c[1] < k[:, 1].max() + 2.5
                   for k in keep_off):
                continue
            holes.append(axis_cyl(c - n * 0.5, n, 0.75, 8.0))
    return [Feature("speaker_grille", "speaker", "cover", "cut", fuse(holes))], len(holes)


def mic_mount(name):
    p = layout()[name]
    wp, wd = p.window_pose()
    rim = obox(p.pos, p.size + 2 * (FIT + 1.2) + np.array([0, 0, 4.0]), p.R) - obox(p.pos, p.size + 2 * FIT + np.array([0, 0, 10.0]), p.R)
    return [Feature(f"{name}_rim", name, "cover", "add", rim),
            Feature(f"{name}_pocket", name, "cover", "cut", obox(p.pos, p.size + 2 * FIT, p.R)),
            Feature(f"{name}_hole", name, "cover", "cut",
                    cone(wp, wd, APERTURE_R[name] + FIT, p.window["cone_deg"], WINDOW_CHECK_MM + 2, back=0.5))]


def pad_recess(name):
    """A shallow recess on the outside for the copper tape, and a 1.8 mm hole for its lead."""
    p = layout()[name]
    n = p.R[:, 2]
    if n @ (p.pos - shell_math().c) < 0:
        n = -n
    t = shell_math().ray(p.pos + 40.0 * n, -n)
    s = p.pos + (40.0 - t) * n
    half = "tray" if s[2] < SPLIT_Z else "cover"
    recess = obox(s, (p.size[0] + 1.0, p.size[1] + 1.0, 2 * PAD_RECESS), p.R)
    lead = axis_cyl(s, -n, 0.9, 8.0, back=1.0)
    return [Feature(f"{name}_recess", name, half, "cut", recess + lead)]


def switch_cut():
    p = layout()["switch"]
    lo, hi = p.corners().min(0) - FIT, p.corners().max(0) + FIT
    lo[0] -= 20.0
    return [Feature("switch_cutout", "switch", "tray", "cut", abox(lo, hi))]


def pogo_mount(name):
    p = layout()[name]
    lo, hi = p.corners().min(0), p.corners().max(0)
    block = abox((lo[0] - 10.0, p.pos[1] - 5.0, p.pos[2] - 4.0), (hi[0] - 4.0, p.pos[1] + 5.0, p.pos[2] + 4.0))
    pins = fuse([axis_cyl((lo[0] - 15.0, p.pos[1] + dy, p.pos[2]), (1, 0, 0), 0.7, 40.0) for dy in (-2.0, 2.0)])
    return [Feature(f"{name}_block", name, "tray", "add", block - pins),
            Feature(f"{name}_holes", name, "tray", "cut", pins)]


def hip_seat(leg):
    """Seat of an STS3215 below z 12: a rail along the case's inner long side, an end rib at the middle (x = 0, shared
    with the other hip on that side; outside the 34 mm free radius, so it may reach the floor) and a bottom tab at
    the inner end with two M2 screws from below into the case's bottom holes. The case's outer side stays open: the
    coxa web swings there."""
    p, d = joints()[leg]["pos"], _hip_dir(leg)
    s = 1.0 if p[1] > 0 else -1.0
    lo, hi = hip_case_aabb(leg)
    zb = lo[2]                                           # case bottom face (STEP z -17.7)
    x_in = p[0] + d * (STEP_CASE[0] - STEP_SHAFT_X)     # inner end of the case, near x = 0
    y_in = s * (abs(p[1]) - STEP_CASE[3])               # inner long side
    w = 2.5
    xr = sorted((x_in - d * FIT, p[0]))
    rail = abox((xr[0], min(y_in - s * (FIT + w), y_in - s * FIT), -24.4),
                (xr[1], max(y_in - s * (FIT + w), y_in - s * FIT), 6.0))
    ys = sorted((y_in - s * (FIT + w), s * abs(hi[1] if s > 0 else lo[1])))
    xe = sorted((x_in - d * FIT, 0.0))
    end = abox((xe[0], ys[0], -60.0), (xe[1], ys[1], 6.0))
    xt = sorted((x_in - d * FIT, x_in + d * 14.0))
    tab = abox((xt[0], ys[0], zb - HIP_TAB_T), (xt[1], ys[1], zb))
    seat = fuse([rail, end, tab]) - abox(lo - FIT, hi + FIT)
    for n in ("tof_left", "tof_right"):                  # the side cliff sensors stand between the hips
        q = layout()[n]
        seat = seat - obox(q.pos, q.size + 2 * (FIT + 1.0), q.R)
    owner = f"hip_{leg.lower()}"
    feats = []
    for hx, hy in STEP_BOTTOM_HOLES[2:]:
        c = hip_to_body(leg, (hx, hy, 0))                # the screw heads sit in counterbores with 3 mm under them
        seat = seat - zcyl(c[0], c[1], M2_CLEAR / 2, -70.0, zb + 0.1) - zcyl(c[0], c[1], 2.1, -70.0, zb - 3.0)
        feats.append(Feature(f"{owner}_access{len(feats)}", owner, "tray", "cut", zcyl(c[0], c[1], 3.0, -70.0, zb - HIP_TAB_T)))
    # the seat stands outside the side wall where the servo is (the hip servos stand through the boxed body's walls):
    # not clipped to the shell; the joint zones still trim it
    return [Feature(f"{owner}_seat", owner, "tray", "add", seat, clip=False)] + feats


def shoulder_seat(arm):
    """The upside-down STS3032: a post under the rear ear (2 x M2 from above) inside the shell, a chin pad under the
    front ear outside the shell's front, and the pocket where the case passes through the front wall."""
    P, p = STS3032, joints()[arm]["pos"]
    ear_lo = SHOULDER_Z0 + P.ear_z0 if ARMS_CAD.YAW_UP else SHOULDER_Z0 - P.ear_z0 - P.ear_t     # the ears' underside
    sx = SHOULDER_SX                                   # horn up, the case runs forward: servo x -> body -x

    def xs(a, b):
        return sorted((p[0] + sx * a, p[0] + sx * b))
    (r0, r1), (c0, c1) = xs(P.x_rear - P.ear_len - 1.0, P.x_rear + 1.0), xs(0.0, P.x_out + P.ear_len + 1.0)
    rear = abox((r0, p[1] - P.half_w - 1.0, -60.0), (r1, p[1] + P.half_w + 1.0, ear_lo))
    chin = abox((c0, p[1] - P.half_w - 1.0, ear_lo - 6.0), (c1, p[1] + P.half_w + 1.0, ear_lo))
    holes = []
    for ex in P.ear_holes_x:                       # servo x -> body sx * x, servo y -> body -y
        for ey in P.ear_holes_y:
            holes.append(zcyl(p[0] + sx * ex, p[1] - ey, M2_PILOT / 2, ear_lo - 8.0, ear_lo + 0.1))
    lo, hi = shoulder_case_aabb(arm)
    pocket = abox(lo - FIT, hi + FIT)
    elo, ehi = shoulder_case_aabb(arm, ears=True)
    pocket = pocket + abox((elo[0] - FIT, elo[1] - FIT, ear_lo), (ehi[0] + FIT, ehi[1] + FIT, ear_lo + P.ear_t + 2 * FIT))
    boss_z = (lo[2] - P.boss_h - FIT, lo[2]) if ARMS_CAD.YAW_UP else (hi[2], hi[2] + P.boss_h + FIT)   # the idler boss
    pocket = pocket + zcyl(p[0], p[1], P.boss_r + FIT, *boss_z)
    owner = f"shoulder_{arm.lower()}"
    return [Feature(f"{owner}_rear_post", owner, "tray", "add", rear - fuse(holes) - pocket),
            Feature(f"{owner}_chin", owner, "tray", "add", chin - fuse(holes) - pocket, clip=False),
            Feature(f"{owner}_pocket", owner, "tray", "cut", pocket)]


def split_bosses():
    """Six joints across the split: a tray boss with an M2.5 heat-set insert at the lip, a cover boss with a
    counterbored clearance hole, the screw from the top."""
    feats = []
    for k, (bx, by) in enumerate(BOSS_XY):
        c = (bx, by)
        tray = (zcyl(c[0], c[1], BOSS_R, SPLIT_Z - 10.0, SPLIT_Z + LIP_H)
                + Pos(c[0], c[1], SPLIT_Z - 10.0) * Cone(0.5, BOSS_R, 8.0, align=(Align.CENTER, Align.CENTER, Align.MAX)))
        tray = tray - zcyl(c[0], c[1], M25_INSERT / 2, SPLIT_Z + LIP_H - M25_INSERT_L, SPLIT_Z + LIP_H + 0.1)
        cov = zcyl(c[0], c[1], BOSS_R, SPLIT_Z + LIP_H, SPLIT_Z + 30.0)
        hole = zcyl(c[0], c[1], M25_CLEAR / 2, SPLIT_Z + LIP_H - 0.1, SPLIT_Z + 40.0)
        cbore = zcyl(c[0], c[1], M25_HEAD / 2 + 0.2, SPLIT_Z + LIP_H + 4.0, SPLIT_Z + 40.0)
        feats.append(Feature(f"boss{k}_tray", "", "tray", "add", tray))
        feats.append(Feature(f"boss{k}_cover", "", "cover", "add", cov))
        feats.append(Feature(f"boss{k}_hole", "", "cover", "cut", hole + cbore))
    return feats


def side_vents():
    """Cooling (audit B5: the old flank vents cut nothing; the Radxa and two bucks sit in a closed shell): six 2 x 15 mm
    intake slots in the belly under the computer, clear of the battery hatch and the cliff sensors, and five exhaust
    slots in the top behind the face lid's hinge, so warm air rises through the body and out."""
    skin = outer() - _inset(WALL + 0.01)
    intake = fuse([abox((x - 1.0, -7.5, -90.0), (x + 1.0, 7.5, -20.0)) for x in VENT_X])
    exhaust = fuse([abox((x - 1.0, -10.0, SPLIT_Z + LIP_H + 4.0), (x + 1.0, 10.0, 90.0)) for x in EXHAUST_X])
    floor = abox((-200.0, -200.0, Z_FLAT - 1.0), (200.0, 200.0, Z_FLAT + 3.0))      # through the belly pad as well
    return [Feature("vents_belly", "radxa", "tray", "cut", intake & (skin + floor)),
            Feature("vents_top", "radxa", "cover", "cut", exhaust & skin)]


def lid_knuckle():
    hx, hy, hz = LID.hinge_location().position
    k = Pos(hx, hy, hz) * LID.shell_knuckle()            # no tilt: the shell's knuckle stays fixed
    slot = Pos(hx + LID.LID_T / 2, 0, hz - 6.0) * Box(5.0, LID.CABLE_SLOT[0] + 1.0, 28.0)
    return [Feature("lid_knuckle", "", "cover", "add", k - inner(), clip=False),
            Feature("hdmi_slot", "oled", "cover", "cut", slot)]


WAIST_Y = BODY_Y - WALL    # the side walls' inner face (the cable router's bound)
LEAD_Z = -8.0              # the limbs' cable ports: height of the lead slots in the walls
ARM_PORT_Y = 30.0          # the arm servo bundles' ports in the front face, |y|


PORT_W, PORT_H = 12.0, 7.0     # a limb's cable port: three servo leads side by side (3 x 2.6 mm) with room, round ends


def cable_port(c, through):
    """A stadium-shaped cable port centred at c, through a wall along `through` ("x" or "y"): rounded ends so the
    bundle never rides on a sharp edge (small-robot practice: a guide hole with margin, its entry rounded)."""
    r = PORT_H / 2
    L = PORT_W - PORT_H
    rot = (90, 0, 0) if through == "y" else (0, 90, 0)
    core = Box(L, 20.0, PORT_H) if through == "y" else Box(20.0, L, PORT_H)
    ends = [Location((c[0] + (dx if through == "y" else 0), c[1] + (dx if through == "x" else 0), c[2]), rot)
            * Cylinder(r, 20.0) for dx in (-L / 2, L / 2)]
    return Pos(*c) * core + ends[0] + ends[1]


TIE_SLOT = (4.2, 1.6)          # a 3.6 x 1.2 mm cable tie through each anchor


def cable_ties():
    """A printed tie anchor wherever the harness is tied (cad/v2/routing.py tie_points: within 20 mm of every
    connector and every 40 mm along a channel): a block from the nearest wall or floor up to the bundle, with a slot
    for a cable tie across the run."""
    from cad.v2 import routing, wiring
    free = routing.grid()[0]
    feats = []
    for k, (q, d) in enumerate(routing.tie_points([c[2] for c in wiring.cables()])):
        c0 = routing._cell(q)
        best = None
        for axis, sgn in ((2, -1), (1, 1), (1, -1), (2, 1), (0, 1), (0, -1)):
            for n in range(1, 6):                       # the nearest surface within 10 mm
                c = list(c0)
                c[axis] += sgn * n
                if not (0 <= c[axis] < free.shape[axis]) or not free[tuple(c)]:
                    if best is None or n < best[2]:
                        best = (axis, sgn, n)
                    break
        if best is None:
            continue
        axis, sgn, n = best
        gap = n * routing.STEP
        lo, hi = np.array(q, float) - 4.0, np.array(q, float) + 4.0
        lo[axis], hi[axis] = sorted((q[axis] + sgn * 2.5, q[axis] + sgn * (gap + 1.5)))   # surface to under the bundle
        block = abox(lo, hi)
        dd = np.abs(np.asarray(d, float))
        dd[axis] = -1.0
        run = int(np.argmax(dd))                        # the slot runs across the cable, parallel to the surface
        across = [i for i in range(3) if i not in (axis, run)][0]
        slo, shi = np.array(q, float), np.array(q, float)
        mid = q[axis] + sgn * (2.5 + min(gap, 6.0) / 2)
        slo[axis], shi[axis] = mid - TIE_SLOT[1] / 2, mid + TIE_SLOT[1] / 2
        slo[run], shi[run] = q[run] - TIE_SLOT[0] / 2, q[run] + TIE_SLOT[0] / 2
        slo[across], shi[across] = q[across] - 6.0, q[across] + 6.0
        block = block - abox(slo, shi)
        half = "tray" if q[2] < SPLIT_Z else "cover"
        feats.append(Feature(f"tie{k}", "", half, "add", block))
    return feats

def joint_openings():
    """Every joint's exact sweep (2 mm clear) cut where it passes inside the body; the hip servo cases' pockets through
    the side walls; a rounded cable port for each limb's servo bundle (by each hip in the side wall, beside each arm
    yaw servo in the front face)."""
    feats = []
    for key in LEGS + ARMS:
        feats.append(Feature(f"opening_{key}", key, "tray", "cut", sweep(key, CLEAR)))
    for leg in LEGS:
        lo, hi = hip_case_aabb(leg)
        p = joints()[leg]["pos"]
        s_ = 1.0 if p[1] > 0 else -1.0
        cut = abox(lo - FIT, hi + FIT) + cable_port((p[0], s_ * (BODY_Y - WALL / 2), LEAD_Z), "y")
        feats.append(Feature(f"hip_pocket_{leg}", f"hip_{leg.lower()}", "tray", "cut", cut))
    for side in (1, -1):
        feats.append(Feature(f"arm_port_{'left' if side > 0 else 'right'}", "", "tray", "cut",
                             cable_port((FRONT_X - WALL / 2, side * ARM_PORT_Y, LEAD_Z), "x")))
    return feats


LINER_T = 1.2              # the bay liners' wall (closing only)


def body_core():
    """The closed body's inside (within the shell's wall)."""
    return _inset(WALL - 0.5)


def bay_liners():
    """Where a leg or arm swings inside the body's walls (a front thigh swung forward passes inside the waist line at
    the body's front corners; the hind thighs at the back), a 2 mm liner follows the swept volume 2 mm clear of it
    (the joint opening's own cut hollows it), so the body stays closed and nothing inside shows; split at the parting
    plane between the tray and the cover."""
    feats = []
    core = body_core()
    for key in LEGS + ARMS:
        liner = sweep(key, CLEAR + LINER_T) & core
        if liner is None or volume(liner) < 1.0:
            continue
        feats.append(Feature(f"liner_{key}_low", "", "tray", "add", liner & below(SPLIT_Z + LIP_H), clip=False))
        feats.append(Feature(f"liner_{key}_high", "", "cover", "add", liner & above(SPLIT_Z), clip=False))
    return feats

@cache
def features():
    lay = layout()
    f = []
    for leg in LEGS:
        f += hip_seat(leg)
    for arm in ARMS:
        f += shoulder_seat(arm)
    for name in STANDOFFS:
        if name not in DECK_BOARDS:                     # (the deck boards' posts are printed on the deck)
            f += standoffs(name)[0]
    f += deck_supports()
    for name in WALL_BOARDS:
        f += wall_board(name)
    for name in CRADLES:
        f += cradle(name)
    f += battery_cradle() + camera_mount() + speaker_mount() + grille()[0]
    f += mic_mount("mic_l") + mic_mount("mic_r")
    for name in PADS:
        f += pad_recess(name)
    f += switch_cut() + pogo_mount("pogo_l") + pogo_mount("pogo_r")
    f += split_bosses() + side_vents() + lid_knuckle() + bay_liners() + cable_ties() + belly_pad()
    dropped = [x.name for x in f if _empty(x.solid)]
    if dropped:                                          # a feature wholly outside the body's outline (clipped away)
        print("features outside the body, left out:", ", ".join(dropped), file=sys.stderr)
    f = [t for t in (_trim(x) for x in f if not _empty(x.solid)) if not _empty(t.solid)] + joint_openings()
    assert all(n in lay.parts for n in {x.owner for x in f} if n and n not in LEGS + ARMS)
    return tuple(f)


@cache
def joint_zones():
    """Where no printed feature may be: every joint's swept volume + 2 mm, and the hip brackets' plate zones."""
    return fuse([sweep(k, CLEAR) for k in LEGS + ARMS]
                + [z for leg in LEGS for z in hip_keepouts(leg)])


def _overlaps(a, b):
    ba, bb = a.bounding_box(), b.bounding_box()
    return not (ba.max.X < bb.min.X or bb.max.X < ba.min.X or ba.max.Y < bb.min.Y or bb.max.Y < ba.min.Y
                or ba.max.Z < bb.min.Z or bb.max.Z < ba.min.Z)


@cache
def part_boxes():
    """name -> the room each layout part takes (its box + FIT; the servos' case boxes + FIT), for shaping features."""
    out = {}
    for n, p in layout().parts.items():
        if p.mass_only or n in ("oled", "visor") or n in PADS or p.d.get("in_model"):
            continue
        out[n] = obox(p.pos, p.size + 2 * FIT, p.R)
    for leg in LEGS:
        lo, hi = hip_case_aabb(leg)
        out[f"hip_{leg.lower()}"] = abox(lo - FIT, hi + FIT)
    for arm in ARMS:
        lo, hi = shoulder_case_aabb(arm, ears=True)
        out[f"shoulder_{arm.lower()}"] = abox(lo - FIT, hi + FIT)
    return out


def _empty(s):
    """A solid that is missing or holds no shape (a boolean that removed everything)."""
    if s is None or getattr(s, "_wrapped", None) is None:
        return True
    try:
        return s.volume < 1e-3
    except Exception:                                    # noqa: BLE001
        return True


def _trim(f):
    """Shape an added feature around the joint zones and around every other part's room; record what was taken."""
    if f.op != "add":
        return f
    lay = layout()
    own = {f.owner} | ({lay[f.owner].on} if f.owner in lay.parts and lay[f.owner].on else set())
    own |= {n for n, p in lay.parts.items() if p.on and p.on == f.owner}
    zones = [s for s in joint_zones().solids() if _overlaps(f.solid, s)]
    zones += [b for n, b in part_boxes().items() if n not in own and _overlaps(f.solid, b)]
    if not zones:
        return f
    zone = fuse(zones)
    taken = volume((f.solid & outer()) & zone if f.clip else f.solid & zone)
    if taken < 0.5:
        return f
    return Feature(f.name, f.owner, f.half, f.op, f.solid - zone, f.clip, taken)


# ---------------------------------------------------------------------------------------------------- the two halves
Z_FLAT = -50.7             # the tray's belly cut flat 1 mm up: a flat datum to print on (audit M2: it rocked)


def belly_pad():
    """Floor stiffening over the flat: the cut leaves 1 mm of wall at the lowest point, the pad brings it to 2.5 mm."""
    pad = abox((-45.0, -45.0, Z_FLAT), (45.0, 45.0, Z_FLAT + 2.5)) & outer()
    return [Feature("belly_pad", "", "tray", "add", pad, clip=False)]


@cache
def tray_shell():
    lower = (outer() - inner()) & below(SPLIT_Z) & above(Z_FLAT)
    lip = (_inset(WALL + LIP_GAP) - _inset(WALL + LIP_GAP + LIP_T)) & above(SPLIT_Z) & below(SPLIT_Z + LIP_H)
    return lower + lip


@cache
def cover_shell():
    return (outer() - inner()) & above(SPLIT_Z)


def _assemble(base, half):
    fs = [x for x in features() if x.half == half]
    adds_in = fuse([x.solid for x in fs if x.op == "add" and x.clip])
    adds_out = fuse([x.solid for x in fs if x.op == "add" and not x.clip])
    body = base
    if adds_in is not None:
        body = body + (adds_in & outer())
    if adds_out is not None:
        body = body + adds_out
    cuts = fuse([x.solid for x in features() if x.op == "cut"])
    body = body - cuts
    if half == "tray":
        body = body & below(SPLIT_Z + LIP_H + 0.01)
    else:
        body = body & above(SPLIT_Z)
    return body


@cache
def tray():
    return _assemble(tray_shell(), "tray")


@cache
def cover():
    return _assemble(cover_shell(), "cover")


def mass_g(s):
    return volume(s) / 1000.0 * PETG_G_CM3


# ---------------------------------------------------------------------------------------------------- export, render
def export():
    from build123d import export_stl
    OUT.mkdir(parents=True, exist_ok=True)
    for name, s in (("tray", tray()), ("cover", cover())):
        export_stl(s, str(OUT / f"{name}.stl"), tolerance=0.02, angular_tolerance=0.1)
    return OUT


def _mesh(shape, tol=0.3):
    v, t = shape.tessellate(tol, 0.3)
    return np.array([(p.X, p.Y, p.Z) for p in v]), np.array(t, int)


def render(path=RENDER):
    """Three MuJoCo renders side by side: the tray with the servos (cover off), the cover from below, the closed body."""
    import mujoco
    from PIL import Image
    orange, light_orange, dark = (0.95, 0.50, 0.15, 1), (0.98, 0.64, 0.32, 1), (0.13, 0.13, 0.15, 1)
    servos = [hip_servo(leg) for leg in LEGS] + [shoulder_servo(a) for a in ARMS]
    scenes = [
        ("tray and servos, cover off", [(tray(), orange, 0)] + [(sv, dark, 0) for sv in servos], (10, 0, -15), 0.33, 125, -42),
        ("cover from below", [(cover(), light_orange, 0)], (0, 0, 40), 0.31, 150, 40),
        ("closed, front left", [(tray(), orange, 0), (cover(), light_orange, 0)] + [(sv, dark, 0) for sv in servos],
         (10, 0, 0), 0.36, 145, -18),
    ]
    tiles = []
    for title, items, lookat, dist, azim, elev in scenes:
        spec = mujoco.MjSpec()
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 900, 700
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[1.0, 1.0, 1.0], rgb2=[0.82, 0.86, 0.92], width=256, height=256)
        spec.worldbody.add_light(pos=[0.3, -0.4, 0.8], dir=[-0.3, 0.4, -0.8], diffuse=[0.5, 0.5, 0.5])
        for k, (shape, rgba, dz) in enumerate(items):
            v, t = _mesh(shape)
            name = f"m{k}"
            spec.add_mesh(name=name, uservert=((v + [0, 0, dz]) / 1000.0).ravel().tolist(), userface=t.ravel().tolist(),
                          inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=rgba, contype=0, conaffinity=0)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = np.array(lookat) / 1000.0
        cam.distance, cam.azimuth, cam.elevation = dist, azim, elev
        with mujoco.Renderer(model, 700, 900) as r:
            r.update_scene(data, camera=cam)
            img = r.render()
        tiles.append((title, Image.fromarray(img)))
    from PIL import ImageDraw
    W = sum(t.width for _, t in tiles)
    out = Image.new("RGB", (W, 760), "white")
    d = ImageDraw.Draw(out)
    d.text((12, 8), f"Pinch v2 printable body (cad/v2/body.py): tray {mass_g(tray()):.0f} g + cover {mass_g(cover()):.0f} g "
                    f"PETG, split at z = {SPLIT_Z:.0f} mm; servos dark (STS3215 STEP, STS3032 blocks)", fill="black")
    x = 0
    for title, t in tiles:
        out.paste(t, (x, 50))
        d.text((x + 12, 32), title, fill="black")
        x += t.width
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path


if __name__ == "__main__":
    import time
    t0 = time.time()
    tr, cv = tray(), cover()
    print(f"tray {mass_g(tr):.1f} g, cover {mass_g(cv):.1f} g PETG ({time.time() - t0:.0f} s)")
    print("STL:", export())
    print("render:", render(), f"({time.time() - t0:.0f} s)")

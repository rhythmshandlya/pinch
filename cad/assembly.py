"""Assemble the CAD parts on the simulator's kinematic skeleton.

Each simulator body gets a rigid "group": the printed part(s) attached to it,
plus the servo cases and horns that move with it. MuJoCo computes body poses
for any joint vector, so CAD placement always matches the simulator exactly.
Right-side groups are mirror images of the left (local y -> -y).

Units: CAD in mm, MuJoCo in m.
"""

import sys
from functools import cache
from pathlib import Path

import numpy as np
from build123d import Location, Plane, mirror

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cad import parts as P  # noqa: E402
from cad.servo import GS4308, STS3032, case, horn  # noqa: E402
from crab.params import ARM_IDS, LEG_IDS  # noqa: E402

MODEL = ROOT / "sim" / "models" / "crab.xml"

# material densities (g/cm^3) and effective fill (walls + infill) per part
PETG, TPU = 1.27, 1.21
PRINTED = {   # part -> (builder, material density, fill factor, material name, qty per robot)
    "chassis":    (P.chassis, PETG, 0.75, "PETG", 1),
    "top_shell":  (P.top_shell, PETG, 1.00, "PETG", 1),
    "belly_door": (P.belly_door, PETG, 1.00, "PETG", 1),
    "pi_tray":    (P.pi_tray, PETG, 0.90, "PETG", 1),
    "desk_tray":  (P.desk_tray, PETG, 0.50, "PETG", 1),
    "coxa_link":  (P.coxa_link, PETG, 0.85, "PETG", 4),
    "femur_link": (P.femur_link, PETG, 0.85, "PETG", 4),
    "tibia":      (P.tibia, PETG, 0.90, "PETG", 4),
    "foot_tip":   (P.foot_tip, TPU, 0.70, "TPU", 4),
    "shoulder":   (P.shoulder, PETG, 0.85, "PETG", 2),
    "upper_arm":  (P.upper_arm, PETG, 0.85, "PETG", 2),
    "dactyl_L":   (lambda: P.dactyl(+1), PETG, 0.90, "PETG", 1),
    "dactyl_R":   (lambda: P.dactyl(-1), PETG, 0.90, "PETG", 1),
    "jaw_pad":    (P.jaw_pad, TPU, 1.00, "TPU", 4),
}
from crab.params import ARM_SERVO, FEMUR_SERVO, MICRO_SERVO, SERVOS  # noqa: E402

# servo masses follow the servos the simulator is set to (the envelopes are the STS3032's)
SC_MASS, HD_MASS = SERVOS[MICRO_SERVO].mass, SERVOS[FEMUR_SERVO].mass
SC_HORN = HD_HORN = 0.0025                       # aluminium disc horn
GS_MASS, GS_HORN = SERVOS[ARM_SERVO].mass, 0.0004  # tiny arm servos, plastic arm horn

# electronics carried by the body (kg, centre in body frame mm, box size mm)
ELECTRONICS = [
    ("battery 2S 850 mAh (bay under the deck, crosswise)", 0.046, (0, 0, -18.7), (30, 57, 17)),
    ("Radxa ZERO 3W (Pi Zero size)", 0.012, (5.5, 0, 14.0), (65, 30, 5)),
    ("camera (front wall)", 0.003, (48, 0, 1.0), (5, 12, 12)),
    ("FE-URT-1 bus adapter + bus hubs", 0.010, (-6, 0, -3.0), (40, 25, 8)),
    ("2 x 5 V 5 A regulators", 0.010, (8, 0, -3.0), (25, 40, 8)),
    ("IMU, mic, amp, cliff + touch sensors", 0.010, (0, 0, 0), (60, 60, 10)),
    ("speaker 28 mm (in the dome)", 0.006, (-15, 0, 22), (28, 28, 5)),
    ("wiring, switch, fuse, screws, magnets", 0.030, (0, 0, -8), (70, 90, 10)),
]


def servo_at(origin, x_dir, z_dir):
    return Location(Plane(origin=origin, x_dir=x_dir, z_dir=z_dir))


@cache
def part(name):
    return PRINTED[name][0]()


def part_mass(name):
    _, rho, fill, _, _ = PRINTED[name]
    return part(name).volume * 1e-3 * rho * fill / 1000.0      # kg


# ---------------------------------------------------------------- groups (left side)
def _left_groups():
    """body name pattern -> list of (label, solid in body frame mm, mass kg)."""
    H = P.H_LOC
    yaw_cases = [(f"{n} yaw servo", P.yaw_servo_loc(n, x, y, z, yaw) * case(P.yaw_profile(n)),
                  GS_MASS if n in ("L", "R") else SC_MASS) for n, x, y, z, yaw in P.yaw_mounts()]
    return {
        "body": [("chassis", part("chassis"), part_mass("chassis")),
                 ("top_shell", part("top_shell"), part_mass("top_shell")),
                 ("belly_door", part("belly_door"), part_mass("belly_door")),
                 ("pi_tray", part("pi_tray"), part_mass("pi_tray"))] + yaw_cases,
        "coxa": [("coxa_link", part("coxa_link"), part_mass("coxa_link")),
                 ("yaw horn", H["yaw"] * horn(STS3032), SC_HORN),
                 ("femur servo", P.L_FEMUR * case(STS3032), HD_MASS)],
        "femur": [("femur_link", part("femur_link"), part_mass("femur_link")),
                  ("femur horn", H["femur"] * horn(STS3032), HD_HORN),
                  ("knee servo", P.L_KNEE * case(STS3032), SC_MASS)],
        "tibia": [("tibia", part("tibia"), part_mass("tibia")),
                  ("foot_tip", part("foot_tip"), part_mass("foot_tip")),
                  ("knee horn", H["knee"] * horn(STS3032), SC_HORN)],
        "shoulder": [("shoulder", part("shoulder"), part_mass("shoulder")),
                     ("shoulder yaw horn", H["shyaw"] * horn(GS4308), GS_HORN),
                     ("shoulder pitch servo", P.L_PITCH * case(GS4308), GS_MASS)],
        "upper": [("upper_arm", part("upper_arm"), part_mass("upper_arm")),
                  ("pitch horn", H["pitch"] * horn(GS4308), GS_HORN),
                  ("claw servo", P.L_CLAW * case(GS4308), GS_MASS),
                  ("jaw pad", part("jaw_pad"), part_mass("jaw_pad"))],
        "dactyl": [("dactyl", part("dactyl_L"), part_mass("dactyl_L")),
                   ("claw horn", H["claw"] * horn(GS4308), GS_HORN),
                   ("jaw pad", P.to_dactyl(P.mirror_jaw(part("jaw_pad"))), part_mass("jaw_pad"))],
    }


@cache
def groups():
    """sim body name -> list of (label, solid mm in that body's frame, mass kg)."""
    left = _left_groups()
    out = {"body": left["body"]}

    def side(items, s, dactyl_r=False):
        res = []
        for label, solid, m in items:
            if dactyl_r and label == "dactyl":
                res.append((label, part("dactyl_R"), part_mass("dactyl_R")))
            elif s > 0:
                res.append((label, solid, m))
            else:
                res.append((label, mirror(solid, about=Plane.XZ), m))
        return res

    for lg in LEG_IDS:
        s = 1 if lg[1] == "L" else -1
        out[f"{lg}_coxa_link"] = side(left["coxa"], s)
        out[f"{lg}_femur_link"] = side(left["femur"], s)
        out[f"{lg}_tibia_link"] = side(left["tibia"], s)
    for a in ARM_IDS:
        s = 1 if a == "L" else -1
        out[f"{a}_shoulder"] = side(left["shoulder"], s)
        out[f"{a}_upper"] = side(left["upper"], s)
        out[f"{a}_dactyl"] = side(left["dactyl"], s, dactyl_r=(s < 0))
    return out


# ---------------------------------------------------------------- mass properties
def _inertia_about_com(solid, mass):
    """Inertia tensor (kg m^2) about the solid's centroid, uniform density, plus centroid (m)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid.wrapped, props)
    vol = props.Mass()                                   # mm^3
    c = props.CentreOfMass()
    m = props.MatrixOfInertia()                          # about centroid, for density 1 (mm^5)
    I = np.array([[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    return I * (mass / vol) * 1e-6, np.array([c.X(), c.Y(), c.Z()]) * 1e-3


def _box_inertia(mass, size_mm):
    a, b, c = (np.array(size_mm) * 1e-3) ** 2
    return mass / 12 * np.diag([b + c, a + c, a + b])


def mass_properties():
    """sim body -> dict(mass, com[3] m, inertia[6] fullinertia order xx yy zz xy xz yz)."""
    out = {}
    for body, items in groups().items():
        comps = [(m, *_inertia_about_com(s, m)) for _, s, m in items]
        if body == "body":
            comps += [(m, _box_inertia(m, size), np.array(c) * 1e-3) for _, m, c, size in ELECTRONICS]
        M = sum(c[0] for c in comps)
        com = sum(c[0] * c[2] for c in comps) / M
        I = np.zeros((3, 3))
        for m, Ic, ci in comps:                           # parallel-axis theorem
            d = ci - com
            I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        out[body] = dict(mass=float(M), com=com.tolist(),
                         inertia=[I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])
    return out


# ---------------------------------------------------------------- posing via MuJoCo
def body_poses(q):
    """Joint vector (20, JOINT_NAMES order) -> {body name: Location in the robot body frame (mm)}."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    from crab.params import JOINT_NAMES
    for i, n in enumerate(JOINT_NAMES):                 # by name: the model also has backlash joints
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = q[i]
    mujoco.mj_kinematics(m, d)
    out = {}
    for name in groups():
        bid = m.body(name).id
        R = d.xmat[bid].reshape(3, 3)
        p = d.xpos[bid] * 1000.0
        out[name] = Location(Plane(origin=tuple(p), x_dir=tuple(R[:, 0]), z_dir=tuple(R[:, 2])))
    return out


@cache
def group_solid(body):
    from build123d import Compound
    return Compound(children=[s for _, s, _ in groups()[body]])


def posed(q):
    poses = body_poses(q)
    return {b: poses[b] * group_solid(b) for b in groups()}


ADJACENT = set()
for _lg in LEG_IDS:
    ADJACENT |= {("body", f"{_lg}_coxa_link"), (f"{_lg}_coxa_link", f"{_lg}_femur_link"),
                 (f"{_lg}_femur_link", f"{_lg}_tibia_link")}
for _a in ARM_IDS:
    ADJACENT |= {("body", f"{_a}_shoulder"), (f"{_a}_shoulder", f"{_a}_upper"),
                 (f"{_a}_upper", f"{_a}_dactyl")}


def _apart(ba, bb, pad=0.0):
    return (ba.max.X + pad < bb.min.X or bb.max.X + pad < ba.min.X or ba.max.Y + pad < bb.min.Y or
            bb.max.Y + pad < ba.min.Y or ba.max.Z + pad < bb.min.Z or bb.max.Z + pad < ba.min.Z)


def interferences(q, min_volume=2.0, within_groups=False):
    """Overlapping components (> min_volume mm^3) in pose q.

    Checked component-by-component (OCC booleans on compounds whose children touch
    are unreliable). Returns [(body_a, label_a, body_b, label_b, volume_mm3)].
    """
    poses = body_poses(q)
    comps = []
    for body, items in groups().items():
        for label, solid, _ in items:
            s = poses[body] * solid
            comps.append((body, label, s, s.bounding_box()))
    hits = []
    for i, (ba, la, sa, xa) in enumerate(comps):
        for bb, lb, sb, xb in comps[i + 1:]:
            if ba == bb and not within_groups:
                continue
            if _apart(xa, xb):
                continue
            common = sa & sb                     # build123d may return None for an empty result
            v = common.volume if common is not None else 0.0
            if v > min_volume:
                hits.append((ba, la, bb, lb, v))
    return hits

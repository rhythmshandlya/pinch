"""Every cable of the v2 robot in the CAD, from configs/wiring_v2.toml (the checked wiring: crab/wiring.py).

    python cad/v2/wiring.py            # the cables, their lengths inside the body and their masses

Each power edge, signal link, I2C bus, the audio and touch-electrode leads and the eighteen servo cables becomes a
tube routed through the body: up from its part to a routing layer just under the cover's split, along x then y, and
down to the other part (a servo cable ends at its servo; the leg and arm cables leave the body at the joint openings
and their remaining length is carried by the links they run along). Tube diameter: the bundle of its conductors.
Masses per metre are estimates for silicone / PVC wire (marked so in MASS_G_M) until weighed; they replace the
layout's 60 g harness allowance. Units mm, body frame.
"""

import os
import sys
import tomllib
from functools import cache
from math import sqrt
from pathlib import Path

os.environ.setdefault("CRAB_DESIGN", "v2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from build123d import Cylinder, Location, Plane, Pos, Sphere  # noqa: E402

from crab.electronics import Layout  # noqa: E402

WIRING = ROOT / "configs" / "wiring_v2.toml"
Z_ROUTE = 23.0                 # routing layer: above the boards, under the cover's split (cad/v2/body.py SPLIT_Z 28)
# conductor outside diameter (mm) and mass (g/m) by AWG, silicone-insulated stranded wire: ESTIMATE (typical
# hobby-wire tables; weigh a metre of each when bought)
WIRE = {14: (3.4, 24.0), 16: (2.9, 16.0), 18: (2.4, 11.0), 20: (2.0, 7.0), 22: (1.7, 5.0), 24: (1.5, 3.4),
        26: (1.3, 2.4), 28: (1.1, 1.6)}
LINK = {"usb": (4.0, 18.0), "hdmi": (6.0, 9.0), "csi": (2.0, 2.0), "audio": (2.0, 4.8), "electrode": (1.1, 1.6),
        "ntc": (1.6, 3.2)}      # (bundle dia mm, g/m): ESTIMATE; hdmi = micro-HDMI flat cable (16 mm wide, drawn round)
SERVO_CABLE = (2.6, 10.2)       # 3 x 24 AWG flat servo lead: ESTIMATE


def _wiring():
    with open(WIRING, "rb") as f:
        return tomllib.load(f)


def _end(part):
    """Where a cable leaves a part: the middle of its top face (body frame, mm)."""
    p = Layout()[part] if isinstance(part, str) else part
    top = p.R @ np.array([0.0, 0.0, p.half[2]]) if hasattr(p, "half") else np.zeros(3)
    return np.array(p.pos, float) + top


def _servo_end(joint):
    """Where a servo cable leaves the body: at the hip (legs) or the shoulder (arms)."""
    from crab.params import PARAMS
    side = 1 if joint[1] == "L" or joint[0] == "L" else -1
    if joint[0] in "FR" and joint[1] in "LR":
        (x, y, _), _ = PARAMS.leg_mounts[joint[0]]
        return np.array([x * 1000, side * y * 1000, 0.0])
    (x, y, z), _ = PARAMS.arm_mount
    s = 1 if joint[0] == "L" else -1
    return np.array([x * 1000 - 10.0, s * y * 1000, z * 1000 - 20.0])


def route(a, b, lane=0.0):
    """Points of a cable from a to b: up to the routing layer (each cable on its own lane, 2.5 mm apart), along x,
    along y, down."""
    z = Z_ROUTE - lane
    pts = [a, np.array([a[0], a[1], z]), np.array([b[0], a[1], z]), np.array([b[0], b[1], z]), b]
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) > 0.5:
            out.append(p)
    return out


def tube(points, dia):
    """A tube through the points: a cylinder per segment and a ball at each bend."""
    r = dia / 2
    solid = None
    for p, q in zip(points[:-1], points[1:]):
        v = q - p
        n = float(np.linalg.norm(v))
        if n < 0.5:
            continue
        z = v / n
        x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0.0, 0.0])
        seg = Location(Plane(origin=tuple((p + q) / 2), x_dir=tuple(x / np.linalg.norm(x)), z_dir=tuple(z))) * Cylinder(r, n)
        solid = seg if solid is None else solid + seg
    for p in points[1:-1]:
        solid = solid + Pos(*p) * Sphere(r)
    return solid


def length(points):
    return float(sum(np.linalg.norm(q - p) for p, q in zip(points[:-1], points[1:])))


@cache
def cables():
    """[(name, kind ('power' | 'servo' | 'signal'), points, bundle dia mm, g/m, bought length mm or None)]."""
    w = _wiring()
    out = []
    lane = 0
    for e in w.get("edge", []):
        parts = e.get("parts", [])
        if e.get("element") or len(parts) != 2 and not e["id"].startswith("cable_"):
            continue                                            # a fuse / switch / shunt element, not a wire
        awg = int(e.get("awg", 22))
        dia1, gm1 = WIRE.get(awg, WIRE[22])
        if e["id"].startswith("cable_"):                        # a servo cable: hub header -> the servo
            joint = e["id"][len("cable_"):]
            pts = route(_end("hub"), _servo_end(joint), lane=0.0)
            dia, gm = SERVO_CABLE
            out.append((e["id"], "servo", pts, dia, gm, float(e["length_mm"]) if e.get("length_mm") != "auto" else None))
            continue
        a, b = parts
        pts = route(_end(a), _end(b), lane=2.5 * (lane % 3))
        lane += 1
        n = 2                                                   # supply and return
        if b.startswith("tof") or b in ("imu", "ina", "mpr121", "tof8") or b.startswith("mic"):
            n = 4 if not b.startswith("mic") else 2             # an I2C device: VCC, GND, SDA, SCL
        out.append((e["id"], "power" if awg <= 20 else "signal", pts, dia1 * sqrt(n), gm1 * n, None))
    for lk in w.get("link", []):
        dia, gm = LINK.get(lk["kind"], (1.5, 2.0))
        pts = route(_end(lk["from_part"]), _end(lk["to_part"]), lane=2.5 * (lane % 3))
        lane += 1
        out.append((lk["id"], "signal", pts, dia, gm, None))
    return out


@cache
def solids():
    """(name, kind, solid, mass inside the body kg) per cable."""
    out = []
    for name, kind, pts, dia, gm, _bought in cables():
        inside = length(pts)
        out.append((name, kind, tube(pts, dia), inside / 1000 * gm / 1000))
    return out


def rest_masses():
    """Servo cable length beyond the body (bought length minus the part inside), as point masses on the links it runs
    along: {simulator body: [(kg, (x, y, z) mm in that body's frame)]}."""
    from crab.params import PARAMS
    out = {}
    L = PARAMS.leg
    for name, kind, pts, _dia, gm, bought in cables():
        if kind != "servo" or not bought:
            continue
        rest = max(0.0, bought - length(pts)) / 1000 * gm / 1000
        joint = name[len("cable_"):]
        side, part = joint.split("_", 1)
        if part == "coxa":
            out.setdefault("body", []).append((rest, tuple(_servo_end(joint))))      # coiled at the hip servo
        elif part == "femur":
            out.setdefault(f"{side}_coxa_link", []).append((rest, (L.coxa_len * 500, 0.0, 0.0)))
        elif part == "tibia":
            out.setdefault(f"{side}_coxa_link", []).append((rest / 2, (L.coxa_len * 500, 0.0, 0.0)))
            out.setdefault(f"{side}_femur_link", []).append((rest / 2, (L.femur_len * 500, 0.0, 0.0)))
        elif part == "shoulder_yaw":
            out.setdefault("body", []).append((rest, tuple(_servo_end(joint))))
        elif part == "shoulder_pitch":
            out.setdefault(f"{side}_shoulder", []).append((rest, (PARAMS.arm.shoulder_offset * 500, 0.0, 0.0)))
        else:                                                                         # the claw servo
            out.setdefault(f"{side}_shoulder", []).append((rest / 2, (PARAMS.arm.shoulder_offset * 500, 0.0, 0.0)))
            out.setdefault(f"{side}_upper", []).append((rest / 2, (PARAMS.arm.upper_len * 500, 0.0, 0.0)))
    return out


if __name__ == "__main__":
    tot_in = 0.0
    for name, kind, pts, dia, gm, bought in cables():
        m = length(pts) / 1000 * gm
        tot_in += m
        print(f"{name:22s} {kind:6s} {length(pts):6.0f} mm inside  dia {dia:4.1f}  {m:5.1f} g" + (f"  (bought {bought:.0f} mm)" if bought else ""))
    rest = sum(m for v in rest_masses().values() for m, _ in v) * 1000
    print(f"cables inside the body {tot_in:.0f} g, servo cable beyond the body {rest:.0f} g, total {tot_in + rest:.0f} g "
          f"(was a 60 g allowance; masses per metre are estimates)")


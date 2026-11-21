"""Generate the MuJoCo model (MJCF) of the crab from crab/params.py (+ CAD outputs if present).

    python sim/build_model.py                 -> writes sim/models/crab.xml
    python sim/build_model.py --primitive     -> ignore CAD meshes, simple shapes only
    python sim/build_model.py --no-backlash
    python sim/build_model.py --design v2     -> writes sim/models/crab_v2.xml: the STS3215 robot in style B, from its
                                                 CAD when cad/v2/export.py has run (sim/models/v2_cad.json): the
                                                 parts' meshes, masses and inertias, and boxes round each leg part
                                                 for contact; the claw keeps its jaw and pad boxes (the CAD is built
                                                 from them) and the body its shell. --primitive: the hand-built shapes

The right side is a mirror of the left: y positions, yaw angles and z-axis hinge axes
are negated, so a given joint command means the same motion on both sides.
Every joint uses its own servo's model (crab/params.py JOINT_SERVO): stiffness,
torque limit, torque-speed slope, rotor inertia, friction and gear backlash.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if "--design" in sys.argv:                                  # the design is read once, by crab/params.py
    os.environ["CRAB_DESIGN"] = sys.argv[sys.argv.index("--design") + 1]

from crab.emotes import STAND  # noqa: E402
from crab.params import ARM_IDS, JOINT_NAMES, LEG_IDS, PARAMS, SERVOS, servo_for  # noqa: E402
from crab.shapes import Shell, superellipsoid, surface_y  # noqa: E402

P = PARAMS
OUT = ROOT / "sim" / "models" / P.model_file
S = P.scale                                   # size relative to the 100 mm-shin original
V2 = P.design == "v2"
K = P.arm_scale                               # claw / jaw geometry relative to v1
JAW_EXTRA = 24.0 if V2 else 0.0
# v2 claw pads: TPU (friction 1.2) and a stiff contact that wins over a light toy's soft default (priority), or a 6 g ball is squashed flat
PAD = ' priority="1" solref="0.004 1" friction="1.2 0.02 0.001"' if V2 else ""               # mm: v2's jaws are longer, so a 40 mm toy still fits between them
# style B colours (the owner's reference: an orange cartoon crab with dark joints and dark ball feet)
ORANGE, DARK, RUBBER = "0.93 0.40 0.14 1", "0.13 0.13 0.15 1", "0.22 0.22 0.24 1"

VIS_FILE = ROOT / "sim" / "models" / "visuals.json"
MP_FILE = ROOT / "sim" / "models" / "mass_props.json"
COL_FILE = ROOT / "sim" / "models" / "collision.json"
USE_CAD = VIS_FILE.exists() and MP_FILE.exists() and "--primitive" not in sys.argv and not V2
VIS = json.loads(VIS_FILE.read_text()) if USE_CAD else {}
MP = json.loads(MP_FILE.read_text()) if USE_CAD else {}
COL = json.loads(COL_FILE.read_text()) if (USE_CAD and COL_FILE.exists()) else None
V2_CAD_FILE = ROOT / "sim" / "models" / "v2_cad.json"
USE_V2_CAD = V2 and V2_CAD_FILE.exists() and "--primitive" not in sys.argv
V2C = json.loads(V2_CAD_FILE.read_text()) if USE_V2_CAD else {}
BACKLASH = "--no-backlash" not in sys.argv
USED = sorted({servo_for(j).name for j in JOINT_NAMES})


def f(*v):
    return " ".join(f"{x:.4g}" for x in v)


def fe(*v):
    return " ".join(f"{x:.6e}" for x in v)


def jnt(name, axis, rng):
    """Servo joint (+ its gear-play joint) with the servo's own class."""
    sv = servo_for(name).name
    out = f'<joint name="{name}" class="srv_{sv}" axis="{axis}" range="{f(*rng)}"/>'
    if BACKLASH:
        out += f'\n        <joint name="{name}_backlash" class="bl_{sv}" axis="{axis}"/>'
    return out


def qpos_tree_order(q_joints) -> list:
    out = []
    for i, _ in enumerate(JOINT_NAMES):
        out.append(q_joints[i])
        if BACKLASH:
            out.append(0.0)
    return out


def cad(body: str) -> str:
    if USE_V2_CAD:
        return v2_cad(body)
    if not USE_CAD or body not in MP:
        return ""
    mp = MP[body]
    out = [f'<inertial pos="{fe(*mp["com"])}" mass="{mp["mass"]:.6f}" fullinertia="{fe(*mp["inertia"])}"/>']
    for v in VIS.get(body, []):
        out.append(f'<geom class="visual" type="mesh" mesh="{v["mesh"]}" material="{v["material"]}"/>')
    return "\n        ".join(out)


def v2_cad(body: str) -> str:
    """The body's mass, centre of mass and inertia from the CAD, its meshes (looks) and the shapes round its parts
    (contact; legs only)."""
    b = V2C["bodies"][body]
    out = [f'<inertial pos="{fe(*b["com"])}" mass="{b["mass"]:.6f}" fullinertia="{fe(*b["inertia"])}"/>']
    out += [f'<geom class="visual" type="mesh" mesh="{v["mesh"]}" material="{v["material"]}"/>' for v in b["visuals"]]
    for c in b["collision"]:
        if c.get("type") == "capsule":
            out.append(f'<geom name="{body}_{c["name"]}" type="capsule" fromto="{fe(*c["fromto"])}" size="{c["r"]:.5f}" '
                       f'material="shell"/>')
        else:
            out.append(f'<geom name="{body}_{c["name"]}" type="box" pos="{fe(*c["pos"])}" size="{fe(*c["size"])}" '
                       f'material="shell"/>')
    return "\n        ".join(out)


def lookat_xyaxes(pos, target):
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return f(*x, *y)


def servo_box(joint):
    if USE_CAD:
        return ""
    L, W, H = (v / 2000 for v in servo_for(joint).size_mm)
    return f'<geom class="visual" type="box" size="{f(W, L, H)}" material="servo"/>'


def leg_xml(leg_id: str) -> str:
    s = 1 if leg_id[1] == "L" else -1
    (x, y, z), yaw = P.leg_mounts[leg_id[0]]
    L = P.leg
    return f"""
      <body name="{leg_id}_coxa_link" pos="{f(x, s*y, z)}" euler="0 0 {s*yaw:.4f}">
        {jnt(leg_id + "_coxa", f"0 0 {-s}", L.coxa_range_for(leg_id))}
        {cad(leg_id + "_coxa_link")}
        {servo_box(leg_id + "_coxa")}
        <geom type="capsule" fromto="0 0 0 {L.coxa_len} 0 0" size="{0.015*S:.4f}" mass="{L.coxa_mass}" material="shell"/>
        <body name="{leg_id}_femur_link" pos="{L.coxa_len} 0 0">
          {jnt(leg_id + "_femur", "0 -1 0", L.femur_range_for(leg_id))}
          {cad(leg_id + "_femur_link")}
          <geom type="capsule" fromto="0 0 0 {L.femur_len} 0 0" size="{0.013*S:.4f}" mass="{L.femur_mass}" material="shell"/>
          <body name="{leg_id}_tibia_link" pos="{L.femur_len} 0 0">
            {jnt(leg_id + "_tibia", "0 -1 0", L.tibia_range)}
            {cad(leg_id + "_tibia_link")}
            {servo_box(leg_id + "_tibia")}
            <geom type="capsule" fromto="0 0 0 {L.tibia_len} 0 0" size="{0.0085*S:.4f}" mass="{L.tibia_mass}" material="shell"/>
            <geom name="{leg_id}_foot" type="sphere" pos="{L.tibia_len} 0 0" size="{L.foot_radius}" mass="0.001" material="tip" friction="1.1 0.02 0.001"/>
            <site name="{leg_id}_foot_site" pos="{L.tibia_len} 0 0" size="{L.foot_radius*1.4:.4f}" rgba="0 0 0 0"/>
          </body>
        </body>
      </body>"""


def arm_xml(arm_id: str) -> str:
    """Arm: swing (yaw) -> lift (pitch) -> claw. The palm holds the claw motor; the moving finger
    (dactyl) hinges on that motor's shaft at the front of the palm."""
    s = 1 if arm_id == "L" else -1
    (x, y, z), yaw = P.arm_mount
    A = P.arm
    h, _fl = A.upper_len, A.finger_len
    JAW = jaw_geoms(s)
    yaw_servo = servo_box(arm_id + "_shoulder_yaw")
    sh_extra = up_extra = dac_extra = ""
    pads_p = "".join(f'\n          <geom name="{arm_id}_pollex_pad{i}" type="box" pos="{f(*(v / 1000 for v in c))}" '
                     f'quat="{f(*qq)}" size="{f(*(v / 1000 for v in hh))}" mass="0.00015" material="pad"{PAD}/>'
                     for i, (c, hh, qq) in enumerate(JAW["pads"]["pollex"])) if V2 else ""
    pads_d = "".join(f'\n            <geom name="{arm_id}_dactyl_pad{i}" type="box" pos="{f(*(v / 1000 for v in c))}" '
                     f'quat="{f(*qq)}" size="{f(*(v / 1000 for v in hh))}" mass="0.00015" material="pad"{PAD}/>'
                     for i, (c, hh, qq) in enumerate(JAW["pads"]["dactyl"])) if V2 else ""
    if USE_V2_CAD:              # the CAD draws the arm (cad(): meshes and masses); the jaws and pads stay as contact
        yaw_servo, sh_extra, up_extra, dac_extra = "", "", pads_p, pads_d
    elif V2:    # style B: the yaw servo stands in the body (body_v2); this arm carries the pitch servo and the claw servo
        sv = servo_for(arm_id + "_shoulder_pitch")
        lh, wh, hh = (v / 2000 for v in sv.size_mm)                  # half length, width, height (shaft along the height)
        yaw_servo = ""
        sh_extra = ("\n        " + f'<geom class="visual" type="cylinder" size="0.013 0.0138" material="servo"/>\n        '
                    f'<geom class="visual" type="box" size="{f(lh, hh, wh)}" pos="{A.shoulder_offset:.4f} 0 0" '
                    f'mass="{sv.mass:.5g}" material="servo"/>')
        up_extra = ("\n          " + f'<geom class="visual" type="cylinder" size="0.014 0.0155" euler="1.5708 0 0" material="servo"/>\n          '
                    f'<geom class="visual" type="capsule" fromto="0 0 0 {h - 0.010:.4f} 0 0" size="0.011" material="shell"/>\n          '
                    f'<geom class="visual" type="sphere" size="0.019" pos="{h - 0.006:.4f} 0 {-0.004 * K:.4g}" material="shell"/>\n          '
                    f'<geom class="visual" type="box" size="{f(lh, wh, hh)}" pos="{h - 0.004:.4f} 0 0.002" '
                    f'mass="{sv.mass:.5g}" material="servo"/>\n          '
                    f'<geom class="visual" type="cylinder" size="0.011 0.016" pos="{h:.4f} 0 0" material="servo"/>\n          '
                    f'<geom class="visual" type="mesh" mesh="v2_cover" pos="{f(JAW["strip"][0][0], JAW["strip"][0][1] - s * 0.007, JAW["strip"][0][2])}" '
                    f'euler="0 {-JAW["pitch"]:.4f} 0" material="claw"/>\n          '
                    + pads_p)
        dac_extra = ("\n            " + f'<geom class="visual" type="mesh" mesh="v2_cover" pos="{f(JAW["d_strip"][0][0], JAW["d_strip"][0][1] + s * 0.007, JAW["d_strip"][0][2])}" '
                     f'quat="{f(*JAW["d_strip"][2])}" material="claw"/>\n            '
                     + pads_d)
    return f"""
      <body name="{arm_id}_shoulder" pos="{f(x, s*y, z)}" euler="0 0 {s*yaw:.4f}">
        {jnt(arm_id + "_shoulder_yaw", f"0 0 {s}", A.shoulder_yaw_range)}
        {cad(arm_id + "_shoulder")}
        {yaw_servo}{sh_extra}
        <geom type="sphere" size="{0.007 * K:.4g}" mass="{A.shoulder_mass:.4g}" material="shell"/>
        <body name="{arm_id}_upper" pos="{A.shoulder_offset} 0 0">
          {jnt(arm_id + "_shoulder_pitch", "0 -1 0", A.shoulder_pitch_range)}
          {cad(arm_id + "_upper")}{up_extra}
          <geom type="capsule" fromto="0 0 0 {h - 0.010 * K:.4f} 0 0" size="{0.005 * K:.4g}" mass="{A.upper_mass * 0.3:.4f}" material="shell"/>
          <geom name="{arm_id}_palm" type="ellipsoid" pos="{h - 0.004 * K:.4f} 0 {-0.004 * K:.4g}" size="{f(0.011 * K, 0.0065 * K, 0.009 * K)}" mass="{A.upper_mass * 0.7:.4f}" material="shell"/>
          <!-- fixed finger: the gripping strip (blade + TPU pad, level with the desk at the grip pitch) and
               the blade above it, hanging below the palm (cad/parts.py jaw_side) -->
          <geom name="{arm_id}_pollex" type="box" pos="{f(*JAW['strip'][0])}" euler="0 {-JAW['pitch']:.4f} 0" size="{f(*JAW['strip'][1])}" mass="0.0008" material="claw"{PAD}/>
          <geom name="{arm_id}_pollex_blade" type="box" pos="{f(*JAW['blade'][0])}" size="{f(*JAW['blade'][1])}" mass="0.0006" material="claw"{PAD}/>
          <site name="{arm_id}_grip_site" pos="{f(*JAW['site'])}" euler="0 {-JAW['pitch']:.4f} 0" size="0.005" rgba="0 0 0 0"/>
          <body name="{arm_id}_dactyl" pos="{h:.4f} 0 0">
            {jnt(arm_id + "_claw", f"0 0 {s}", (V2C["claw_meet"], A.claw_range[1]) if USE_V2_CAD else A.claw_range)}
            {cad(arm_id + "_dactyl")}{dac_extra}
            <geom name="{arm_id}_dactyl_jaw" type="box" pos="{f(*JAW['d_strip'][0])}" quat="{f(*JAW['d_strip'][2])}" size="{f(*JAW['d_strip'][1])}" mass="{A.jaw_mass}" material="claw"{PAD}/>
            <geom name="{arm_id}_dactyl_blade" type="box" pos="{f(*JAW['d_blade'][0])}" quat="{f(*JAW['d_blade'][2])}" size="{f(*JAW['d_blade'][1])}" mass="0.0004" material="claw"{PAD}/>
            <geom name="{arm_id}_dactyl_drop" type="box" pos="{f(*JAW['d_drop'][0])}" size="{f(*JAW['d_drop'][1])}" mass="0.0004" material="claw"{PAD}/>
          </body>
        </body>
      </body>"""


def jaw_geoms(s):
    """Collision boxes for the claw's jaws, from the CAD numbers (cad/parts.py: GRIP_PITCH, JAW_X, JAW_LOW,
    STRIP_H, PAD_T, BLADE_T, D_DROP) - kept here as plain numbers so the model builds without the CAD.
    s = +1 left arm, -1 right (mirror). Returns {name: ((pos), (half sizes))} in metres, plus 'pitch'
    and the grip site."""
    pitch = np.radians(8.0 if V2 else 20.0)      # v2: the shoulder is low, the strip is level with the desk at an arm pitch of about -8 deg
    # (v2: the whole claw is K = 1.5 times bigger; v1: K = 1.0, so every number below is unchanged)
    x0, x1, low0, strip_h, pad_t, blade_t = (K * v for v in (40.3, 55.5, -24.5, 10.0, 1.2, 2.4))
    x1 += JAW_EXTRA
    if V2:
        strip_h = 22.0                                   # taller pads: a 36 mm ball is held above its middle, not by the pad's edge
    grip_w, q_par = 9.0 * K, np.radians(19.0)            # jaws parallel GRIP_W apart at CLAW_PARALLEL
    if V2:      # v2: parallel when shut, the backing plates 8 mm apart so the two 4 mm TPU pads meet (v2_pads)
        grip_w, q_par = 8.0, 0.0
    lu, dd, dd_y = P.arm.upper_len * 1000, tuple(K * v for v in (11.5, 15.0)), tuple(K * v for v in (4.6, 8.4))
    c, sn = np.cos(pitch), np.sin(pitch)
    ry = np.array([[c, 0, -sn], [0, 1, 0], [sn, 0, c]])       # rotation about y by -pitch (strip along x)

    def rz(a):
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    def quat(R):
        w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        return (w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))

    def to_dactyl(pos, R):
        """Arm frame with the claw at q_par -> the moving finger's frame (mirrored for the right arm)."""
        Rd = rz(-s * q_par)
        return Rd @ (np.asarray(pos) - [lu, 0, 0]), Rd @ R

    def low(x):
        return low0 + (x - x0) * np.tan(pitch)

    xm = (x0 + x1) / 2
    hx = (x1 - x0) / 2 / c - 0.2                          # along the strip
    hz = strip_h * c / 2                                  # across it
    hy = (pad_t + blade_t) / 2
    centre = (xm - hz * sn, low(xm) + hz * c)            # strip box centre (x, z): its bottom on the edge
    m = 1e-3
    mid = s * (grip_w - 1.0) / 2                          # plane midway between the pads when parallel
    strip_c = np.array([centre[0], -s * (0.5 + hy), centre[1]])
    blade_c = np.array([43.3 * K, -s * (0.5 + pad_t + blade_t / 2), -8.25 * K])
    d_strip = to_dactyl(strip_c * [1, -1, 1] + [0, 2 * mid, 0], ry)
    d_blade = to_dactyl(blade_c * [1, -1, 1] + [0, 2 * mid, 0], np.eye(3))
    out = {
        "pitch": float(pitch),
        "strip": (tuple(strip_c * m), (hx * m, hy * m, hz * m)),
        "blade": ((41.5 * K * m, -s * (0.5 + pad_t + blade_t / 2) * m, -11.0 * K * m),
                  (4.8 * K * m, blade_t / 2 * m, 8.0 * K * m)),
        # pen-sized things are held with their axis 3.5 mm above the strip's lower edge, mid-strip
        "site": ((xm - 3.5 * K * sn) * m, 0.0, (low(xm) + 3.5 * K * c) * m),
        "d_strip": (tuple(d_strip[0] * m), (hx * m, hy * m, hz * m), quat(d_strip[1])),
        "d_blade": (tuple(d_blade[0] * m), (3.0 * K * m, blade_t / 2 * m, 5.75 * K * m), quat(d_blade[1])),
        "d_drop": (((dd[0] + dd[1]) / 2 * m, s * (dd_y[0] + dd_y[1]) / 2 * m, 4.5 * K * m),
                   ((dd[1] - dd[0]) / 2 * m, (dd_y[1] - dd_y[0]) / 2 * m, 10.5 * K * m)),
    }
    if V2:
        out["pads"] = v2_pads(s, strip_c, ry, d_strip, hx, hy, hz, quat)
    if V2:      # tip hooks (a crab claw's curved tips): on the upper part of each pad so a ball or cube cannot roll or slip out
        #           forward, while a pen or an eraser lying low passes under them
        hook = (1.5, 4.0, 4.5)                                       # half sizes, mm
        off_f = np.array([hx - 1.5, s * (hy + hook[1]), hz - hook[2]])
        off_d = np.array([hx - 1.5, -s * (hy + hook[1]), hz - hook[2]])
        out["hook"] = (tuple((strip_c + ry @ off_f) * m), tuple(v * m for v in hook))
        out["d_hook"] = (tuple((d_strip[0] + d_strip[1] @ off_d) * m), tuple(v * m for v in hook), quat(d_strip[1]))
    return out


def v2_pads(s, strip_c, ry, d_strip, hx, hy, hz, quat):
    """The v2 claw's TPU pads (phase 2 redesign after docs/phase2/gripper_research.md: the SO-101-style single-hinge pincer
    kept, flat pads replaced by shaped ones). Built in each jaw's strip frame: x along the jaw toward the tip, n toward the
    other jaw (from the backing plate's inner face), z up the jaw (the desk is at z = -hz when the arm grips).
      groove: a 90 deg V along the proximal 60 % of the jaw at pen height (z = -hz + 4.5 mm), 3.5 mm deep in a 4 mm pad,
              so a 9 mm pen and a 5 mm stick centre themselves (the two grooves leave a 7 mm diamond when shut)
      rims:   flat 4 mm pad above and below the groove
      cup:    on the distal 40 %, at ball height (z = -hz + 15 mm), two faces leaning 35 deg about z form a pocket that
              stops a 30-36 mm ball sliding along the jaw
      lip:    at the tip, only above pen height, 1 mm proud of the pad: a ball cannot pass the tips, a pen slides under
    Returns {"pollex": [(pos, half, quat)], "dactyl": [...]} in mm, in the upper-arm and the dactyl frames."""
    PAD, DEPTH = 4.0, 3.5
    z_pen, z_ball = -hz + 4.5, min(-hz + 15.0, hz - 6.0)
    L = 2 * hx
    x_split = -hx + 0.6 * L
    boxes = []                                  # (centre (x, n, z), half (x, n, z), rotation about local x, about z)

    def plane(p0, p1, x0, x1, thick=1.6):
        """A thin box whose inner surface runs from p0 to p1 in the (n, z) plane, spanning x0..x1."""
        (n0, z0), (n1, z1) = p0, p1
        ln = np.hypot(n1 - n0, z1 - z0)
        ang = np.arctan2(z1 - z0, n1 - n0)       # direction in (n, z); the box's z half-axis runs along it
        nrm = np.array([np.sin(ang), -np.cos(ang)])
        if nrm[0] > 0:                          # keep the box behind the surface (away from the other jaw)
            nrm = -nrm
        c = np.array([(n0 + n1) / 2, (z0 + z1) / 2]) + nrm * thick / 2
        boxes.append(((0.5 * (x0 + x1), c[0], c[1]), ((x1 - x0) / 2, thick / 2, ln / 2), np.pi / 2 - ang, 0.0))

    def block(x0, x1, n0, n1, z0, z1, rz=0.0):
        boxes.append((((x0 + x1) / 2, (n0 + n1) / 2, (z0 + z1) / 2), ((x1 - x0) / 2, (n1 - n0) / 2, (z1 - z0) / 2), 0.0, rz))

    xg0, xg1 = -hx + 1.0, x_split
    g_lo, g_hi = z_pen - DEPTH, z_pen + DEPTH                         # 90 deg: as wide as twice the depth
    root = PAD - DEPTH
    plane((root, z_pen), (PAD, g_hi), xg0, xg1)                       # groove: upper face
    plane((root, z_pen), (PAD, g_lo), xg0, xg1)                       # groove: lower face
    block(xg0, xg1, 0.0, PAD, -hz, g_lo)                              # rim below
    block(xg0, xg1, 0.0, PAD, g_hi, hz)                               # rim above
    block(xg0, xg1, 0.0, root, g_lo, g_hi)                            # groove root (filled behind)
    # cup: two faces leaning 35 deg about z, meeting 3 mm deep in the middle of the distal part, around ball height
    xc, w = (x_split + hx) / 2, (hx - x_split) / 2 - 0.5
    lean = np.radians(35)
    for sgn in (1, -1):
        cx = xc + sgn * w / 2
        boxes.append(((cx, PAD - 1.5 - 0.8, z_ball), (w / 2 / np.cos(lean), 0.8, 7.0), 0.0, -sgn * lean))
    block(x_split, hx, 0.0, PAD - 3.0, z_ball - 7.0, z_ball + 7.0)    # cup backing
    block(x_split, hx, 0.0, PAD, -hz, z_ball - 7.0)                   # flat pad under the cup (pens pass here)
    block(hx - 2.5, hx, 0.0, PAD + 1.0, z_pen + 5.0, hz)              # tip lip, above pen height
    out = {"pollex": [], "dactyl": []}

    def rot(ax, a):
        c_, s_ = np.cos(a), np.sin(a)
        if ax == "x":
            return np.array([[1, 0, 0], [0, c_, -s_], [0, s_, c_]])
        return np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])

    for jaw, M, R0, c0 in (("pollex", np.diag([1.0, s, 1.0]), ry, np.asarray(strip_c)),
                           ("dactyl", np.diag([1.0, -s, 1.0]), d_strip[1], np.asarray(d_strip[0]))):
        face = np.array([0.0, hy, 0.0])                                    # local origin on the plate's inner face
        for (cx, cn, cz), half, rx, rz in boxes:
            loc = M @ (np.array([cx, cn, cz]) + face)
            Rl = M @ rot("x", rx) @ rot("z", rz) @ M
            out[jaw].append((tuple(c0 + R0 @ loc), tuple(half), quat(R0 @ Rl)))
    return out


def eyes_xml() -> str:
    if USE_CAD or V2:                  # v2 has the real face screen (face_xml), not eyes on stalks
        return ""
    S = 1.6 if V2 else globals()["S"]                    # v2: eyes sized with the 1.6x legs
    hx, hy, hz = P.body.half_size
    out = []
    for s in (1, -1):
        base, top = (0.67 * hx, s * 0.22 * hy, 0.7 * hz), (0.88 * hx, s * 0.27 * hy, 1.8 * hz)
        out.append(f"""
      <geom class="visual" type="capsule" fromto="{f(*base, *top)}" size="{0.0045*S:.4f}" material="shell"/>
      <geom class="visual" type="sphere" pos="{f(*top)}" size="{0.011*S:.4f}" material="eye"/>
      <geom class="visual" type="sphere" pos="{f(top[0] + 0.008*S, top[1], top[2] + 0.002*S)}" size="{0.0055*S:.4f}" material="pupil"/>""")
    return "".join(out)


def face_xml() -> str:
    """v2: the face lid. A lid hinged at the top-front of the shell (Layout chassis lid_hinge_mm), standing up `lid_tilt`
    rad back from vertical (0.35: facing the viewer; 1.50: folded flat on top; a friction hinge set by hand, so the
    lid is part of the body here). A dark rounded bezel (a boxy superellipsoid, 147 + 2 x 6 mm by 73.5 + 12 mm) and
    the lit area as sim/desk.py draws it: a thin visual box with material face_mat (emission 0.9) textured by
    face_tex (256 x 128, written by set_face), 2:1. Visual only: the layout's `visor` (the lid plate) and `oled`
    (the display module) carry the mass."""
    if not V2:
        return ""
    B = P.body
    H, t = np.asarray(B.lid_hinge), B.lid_tilt
    n, u = np.array([np.cos(t), 0.0, np.sin(t)]), np.array([-np.sin(t), 0.0, np.cos(t)])
    pw, ph, bz = LID_PANEL[0] / 2000.0, LID_PANEL[1] / 2000.0, LID_BEZEL / 1000.0
    ctr = H + u * (ph + bz) + n * 0.005
    if USE_V2_CAD:              # the CAD lid (cad/v2/lid.py) is drawn by cad("body"): only the lit area here
        fc = V2C["face"]
        lit = H + u * fc["up"] + n * fc["out"]
        return (f'<geom name="face_screen" class="visual" type="box" size="{f(0.0008, fc["size"][0] / 2, fc["size"][1] / 2)}" '
                f'pos="{f(*lit)}" quat="{f(*_quat_of(np.column_stack([n, [0.0, 1.0, 0.0], u])))}" material="face_mat" mass="0"/>')
    # the screen frame of sim/desk.py: x out of the screen, y across, z up the screen
    Rs = np.column_stack([n, [0.0, 1.0, 0.0], u])
    lit = ctr + n * 0.0055
    return "\n      ".join([
        f'<geom class="visual" type="mesh" mesh="v2_bezel" pos="{f(*ctr)}" euler="0 {-t:.5f} 0" material="servo" mass="0"/>',
        f'<geom name="face_screen" class="visual" type="box" size="{f(0.0008, 0.07209, 0.0333)}" pos="{f(*lit)}" '
        f'quat="{f(*_quat_of(Rs))}" material="face_mat" mass="0"/>',
        f'<geom class="visual" type="cylinder" size="0.007 {0.7 * pw:.4f}" pos="{f(*H)}" euler="1.5708 0 0" material="servo" mass="0"/>',
    ])


def servo_defaults() -> str:
    out = []
    for name in USED:
        sp = SERVOS[name]
        bl = np.radians(sp.backlash_deg)
        out.append(f"""      <!-- {name}: {sp.source} -->
      <default class="srv_{name}">
        <joint armature="{sp.armature:.6g}" damping="{sp.damping:.5g}" frictionloss="{sp.frictionloss:.5g}"/>
        <position kp="{sp.kp:.5g}" kv="{sp.kv}" forcerange="{-sp.force_limit:.5g} {sp.force_limit:.5g}"/>
      </default>
      <default class="bl_{name}">
        <joint damping="{0.01*S**3:.3g}" armature="{0.01*S**4:.3g}" frictionloss="0" range="{-bl:.5f} {bl:.5f}"/>
      </default>""")
    return "\n".join(out)


def leg_xml_v2(leg_id: str) -> str:
    """v2 leg, style B: an orange thigh cover (radius 19 mm), a tapered shin (radius 20 to 14 mm) and a dark 14 mm ball foot.
    The printed covers collide with the floor (capsules sized to match, the shin as two so it tapers) so the legs can push
    the body up and a fallen robot rests on its shell. Each STS3215 is a box with its real size and mass in the link that
    carries it: the thigh servo in the coxa link, the knee servo in the femur link (the hip servo is in the body).
    Printed mass = the link's mass minus its servo (and the foot ball). Dark cylinders show the joints."""
    s = 1 if leg_id[1] == "L" else -1
    (x, y, z), yaw = P.leg_mounts[leg_id[0]]
    L = P.leg
    sf, st = servo_for(leg_id + "_femur"), servo_for(leg_id + "_tibia")
    lf, wf, hf = (v / 2000 for v in sf.size_mm)          # half length, width, height (shaft along the height)
    lt, wt, ht = (v / 2000 for v in st.size_mm)
    FL, TL, r0, r1 = L.femur_len, L.tibia_len, L.shin_r0, L.shin_r1
    printed = L.tibia_mass - L.foot_mass                  # the shin cover
    cut = 0.45                                            # the shin's collision: a fat capsule to here, a slim one beyond
    r_mid = r0 + (r1 - r0) * cut

    def box(half, pos, mass):
        return (f'<geom class="visual" type="box" size="{f(*half)}" pos="{f(*pos)}" mass="{mass:.5g}" '
                f'material="servo"/>')

    def cap(r, half_len, pos, material):
        return (f'<geom class="visual" type="cylinder" size="{r:.4f} {half_len:.4f}" pos="{f(*pos)}" euler="1.5708 0 0" '
                f'material="{material}"/>')
    return f"""
      <body name="{leg_id}_coxa_link" pos="{f(x, s*y, z)}" euler="0 0 {s*yaw:.4f}">
        {jnt(leg_id + "_coxa", f"0 0 {-s}", L.coxa_range_for(leg_id))}
        <geom type="capsule" fromto="0 0 0 {L.coxa_len} 0 0" size="0.016" mass="{L.coxa_mass - sf.mass:.5g}" material="shell"/>
        {box((lf, hf, wf), (L.coxa_len - 0.3 * lf, 0, 0), sf.mass)}
        <geom class="visual" type="box" size="0.019 0.013 0.016" pos="0.012 0 0" material="servo"/>
        {cap(0.011, 0.015, (L.coxa_len, 0, 0), "shell")}
        <body name="{leg_id}_femur_link" pos="{L.coxa_len} 0 0">
          {jnt(leg_id + "_femur", "0 -1 0", L.femur_range_for(leg_id))}
          <geom type="capsule" fromto="0 0 0 {FL} 0 0" size="{L.thigh_r:.4f}" mass="{L.femur_mass - st.mass:.5g}" material="shell"/>
          <geom class="visual" type="mesh" mesh="v2_thigh" pos="{FL / 2:.4f} 0 0.004" material="shell"/>
          {cap(0.017, 0.019, (0, 0, 0), "servo")}
          {box((lt, ht, wt), (FL - 0.3 * lt, 0, 0), st.mass)}
          <body name="{leg_id}_tibia_link" pos="{FL} 0 0">
            {jnt(leg_id + "_tibia", "0 -1 0", L.tibia_range)}
            <geom type="capsule" fromto="0 0 0 {cut * TL:.4f} 0 0" size="{(r0 + r_mid) / 2 * 0.9:.4f}" mass="{printed * cut:.5g}" material="shell"/>
            <geom type="capsule" fromto="{cut * TL:.4f} 0 0 {TL:.4f} 0 0" size="{(r_mid + r1) / 2 * 0.9:.4f}" mass="{printed * (1 - cut):.5g}" material="shell"/>
            <geom class="visual" type="mesh" mesh="v2_shin" pos="{TL / 2:.4f} 0 0" material="shell"/>
            {cap(0.017, 0.019, (0, 0, 0), "servo")}
            <geom name="{leg_id}_foot" type="sphere" pos="{TL} 0 0" size="{L.foot_radius}" mass="{L.foot_mass:.5g}" material="tip" friction="1.1 0.02 0.001"/>
            <site name="{leg_id}_foot_site" pos="{TL} 0 0" size="{L.foot_radius*1.4:.4f}" rgba="0 0 0 0"/>
          </body>
        </body>
      </body>"""


def leg_xml_v2cad(leg_id: str) -> str:
    """v2 leg from the CAD (cad/v2/legs.py): each link's inertial, meshes and contact shapes, and the rubber tip (the
    same reach as tibia_len + foot_radius, so the stand height and the gaits' kinematics are unchanged)."""
    s = 1 if leg_id[1] == "L" else -1
    ft = V2C["foot"]
    (x, y, z), yaw = P.leg_mounts[leg_id[0]]
    L = P.leg
    return f"""
      <body name="{leg_id}_coxa_link" pos="{f(x, s*y, z)}" euler="0 0 {s*yaw:.4f}">
        {jnt(leg_id + "_coxa", f"0 0 {-s}", L.coxa_range_for(leg_id))}
        {cad(leg_id + "_coxa_link")}
        <body name="{leg_id}_femur_link" pos="{L.coxa_len} 0 0">
          {jnt(leg_id + "_femur", "0 -1 0", L.femur_range_for(leg_id))}
          {cad(leg_id + "_femur_link")}
          <body name="{leg_id}_tibia_link" pos="{L.femur_len} 0 0">
            {jnt(leg_id + "_tibia", "0 -1 0", L.tibia_range)}
            {cad(leg_id + "_tibia_link")}
            <geom name="{leg_id}_foot" type="sphere" pos="{ft['x']:.4f} 0 0" size="{ft['r']:.4f}" material="tip" friction="1.1 0.02 0.001"/>
            <site name="{leg_id}_foot_site" pos="{ft['x']:.4f} 0 0" size="{ft['r'] * 1.4:.4f}" rgba="0 0 0 0"/>
          </body>
        </body>
      </body>"""


def _quat_of(R):
    """Quaternion (w, x, y, z) of a rotation matrix."""
    t = np.trace(R)
    if t > 0:
        w = np.sqrt(1 + t) / 2
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    q = np.zeros(4)
    r = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) / 2
    q[1 + i] = r
    q[0] = (R[k, j] - R[j, k]) / (4 * r)
    q[1 + j] = (R[j, i] + R[i, j]) / (4 * r)
    q[1 + k] = (R[k, i] + R[i, k]) / (4 * r)
    return q


def layout_geoms() -> list:
    """Every electronic part of configs/layout_v2.toml as a box with its own mass (and so inertia) inside the body.
    They do not collide (visual class: no contact, group 2); the servos the model builds elsewhere are skipped."""
    from crab.electronics import Layout
    out = []
    for p in Layout().parts.values():
        if p.in_model:
            continue
        pos = p.pos / 1000.0
        if p.mass_only:
            out.append(f'<geom name="part_{p.name}" class="visual" type="sphere" size="0.003" pos="{f(*pos)}" '
                       f'mass="{p.mass_g / 1000:.5g}" material="pcb"/>')
            continue
        mat = {"battery": "servo", "servo": "servo"}.get(p.kind, "pcb")
        # the screen module and the lid plate are drawn by face_xml: here they only carry their mass
        look = 'rgba="0 0 0 0"' if p.d.get("face") or p.kind == "printed" else f'material="{mat}"'
        out.append(f'<geom name="part_{p.name}" class="visual" type="box" size="{f(*(p.half / 1000.0))}" '
                   f'pos="{f(*pos)}" quat="{f(*_quat_of(p.R))}" mass="{p.mass_g / 1000:.5g}" {look}/>')
    return out


LID_PANEL = (153.0, 70.0)             # mm: the lid panel inside the bezel (Layout chassis lid_panel_mm); lit area 144.18 x 66.6
LID_BEZEL = 6.0                       # mm: dark rounded bezel round it


def v2_meshes() -> str:
    """Inline meshes of the style B body: the pill shell (collides, mass on a thin surface), the lid bezel, the thigh and
    shin covers and the claw cover (visual)."""
    from crab.electronics import Layout
    L = P.leg
    sh = Shell.from_chassis(Layout().chassis)
    pw, ph, bz = LID_PANEL[0] / 2000.0, LID_PANEL[1] / 2000.0, LID_BEZEL / 1000.0
    items = [("v2_shell", sh.mesh(32), ' inertia="shell"'),
             ("v2_bezel", superellipsoid(0.005, pw + bz, ph + bz, 0.25, 0.25, n=28), ""),
             ("v2_thigh", superellipsoid(L.femur_len / 2 + 0.010, L.thigh_r * 0.85, L.thigh_r, 0.6, 0.6, n=24), ""),
             ("v2_shin", superellipsoid(L.tibia_len / 2, L.shin_r0, L.shin_r0 * 0.9, 0.7, 0.8, n=24,
                                        taper=lambda u: 1.0 - (1 - L.shin_r1 / L.shin_r0) * u), ""),
             ("v2_cover", superellipsoid(0.022, 0.004, 0.0105, 0.6, 0.6, n=14), "")]
    out = []
    for name, (v, fc), extra in items:
        out.append(f'    <mesh name="{name}" vertex="{" ".join(f"{c:.5g}" for c in v.ravel())}" '
                   f'face="{" ".join(str(int(i)) for i in fc.ravel())}"{extra}/>')
    return "\n".join(out)


def body_v2() -> str:
    """v2 body, style B: the pill shell (it collides: its mesh is the body's collision shape), the four hip servos
    (shaft vertical), the two arm yaw servos at the front of the shell, the dark camera window, the side vents and every
    electronic part of configs/layout_v2.toml where the layout puts it, with its own mass (the face lid's parts included)."""
    from crab.electronics import Layout
    B = P.body
    sv = servo_for("FL_coxa")
    lh, wh, hh = (v / 2000 for v in sv.size_mm)
    hx, hy = abs(P.leg_mounts["F"][0][0]), P.leg_mounts["F"][0][1]
    lay = Layout()
    sh = Shell.from_chassis(lay.chassis)
    out = [f'<geom name="shell" type="mesh" mesh="v2_shell" mass="{B.shell_mass:.5g}" material="shell"/>']
    if USE_V2_CAD:              # cad("body") draws the shell, lid, servos and electronics and carries their masses
        out += [g.replace('material="pcb"', 'rgba="0 0 0 0"').replace('material="servo"', 'rgba="0 0 0 0"')
                for g in layout_geoms()]
        return "\n      ".join(out)
    for sx in (1, -1):
        for sy in (1, -1):
            out.append(f'<geom class="visual" type="box" size="{f(lh, wh, hh)}" pos="{f(sx * hx, sy * hy, -0.15 * 2 * hh)}" '
                       f'mass="{sv.mass:.5g}" material="servo"/>')
    for name in ("shoulder_l", "shoulder_r"):                  # the arm yaw servos (placeholder STS3032)
        p = lay[name]
        out.append(f'<geom name="yaw_servo_{name}" class="visual" type="box" size="{f(*(p.half / 1000.0))}" pos="{f(*(p.pos / 1000.0))}" '
                   f'mass="{p.mass_g / 1000:.5g}" material="servo"/>')
    out += layout_geoms()
    # looks: the camera window (dark plate and a lens), the three small side vents a side
    pos, _d = lay["camera_lens"].window_pose()
    out.append(f'<geom class="visual" type="cylinder" size="0.011 0.0015" pos="{f(*((pos + [0.5, 0, 0]) / 1000.0))}" euler="0 1.5708 0" material="servo" mass="0"/>')
    out.append(f'<geom class="visual" type="sphere" size="0.0055" pos="{f(*((pos + [1.2, 0, 0]) / 1000.0))}" rgba="0.05 0.10 0.25 1" mass="0"/>')
    for sy in (1, -1):
        for k in range(3):
            x, z = sh.c[0] + 0.25 * sh.h[0] + k * 9.0, sh.c[2] - 0.2 * sh.h[2] - 3.0
            ys = surface_y(sh, x, z)
            out.append(f'<geom class="visual" type="box" size="0.0012 0.0008 0.0065" pos="{f(x / 1000.0, sy * ys / 1000.0, z / 1000.0)}" material="servo" mass="0"/>')
    return "\n      ".join(out)


def carapace() -> str:
    B = P.body
    if V2:
        return body_v2()
    if COL:
        return "\n      ".join(
            f'<geom name="{g["name"]}" type="{g.get("type", "box")}" pos="{f(*g["pos"])}" size="{f(*g["size"])}"'
            + (f' euler="{f(*g["euler"])}"' if "euler" in g else "") + ' material="shell"/>'
            for g in COL["body"])
    return f'<geom name="carapace" type="ellipsoid" size="{f(*B.half_size)}" mass="{B.mass}" material="shell"/>'


FACE_ASSET = ('    <texture name="face_tex" type="2d" builtin="flat" rgb1="0.02 0.02 0.03" width="256" height="128"/>\n'
              '    <material name="face_mat" texture="face_tex" emission="0.9" specular="0.3"/>') if V2 else ""


def build() -> str:
    B, L = P.body, P.leg
    fem, tib = STAND[JOINT_NAMES.index("FL_femur")], STAND[JOINT_NAMES.index("FL_tibia")]
    z0 = -(L.femur_len * np.sin(fem) + L.tibia_len * np.sin(fem + tib)) + L.foot_radius + 0.002
    cam_pos = tuple(np.array([0.42, -0.50, 0.30]) * S)
    # v2 from its CAD: the claw's joint stops where the jaws meet, its servo can still be told to go to 0 (it squeezes)
    rng = {j: (f'ctrlrange="{f(*P.arm.claw_range)}"' if USE_V2_CAD and j.endswith("_claw") else 'inheritrange="1"')
           for j in JOINT_NAMES}
    actuators = "\n".join(f'    <position name="{j}" joint="{j}" class="srv_{servo_for(j).name}" {rng[j]}/>'
                          for j in JOINT_NAMES)
    foot_sensors = "\n".join(f'    <touch name="{lg}_foot_touch" site="{lg}_foot_site"/>' for lg in LEG_IDS)
    meshes = "\n".join(f'    <mesh name="{v["mesh"]}" file="{v["file"]}" scale="0.001 0.001 0.001"/>'
                       for items in VIS.values() for v in items) if USE_CAD else ""
    if V2:
        meshes = v2_meshes()
    if USE_V2_CAD:
        meshes += "\n" + "\n".join(f'    <mesh name="{v["mesh"]}" file="{v["file"]}" scale="0.001 0.001 0.001"/>'
                                    for b in V2C["bodies"].values() for v in b["visuals"])
    eye_cam = COL["eye_cam"] if COL else (0.95 * B.half_size[0], 0, 0.5 * B.half_size[2])
    imu_pose, eye_axes = 'pos="0 0 0"', "0 -1 0 0 0 1"
    if V2:                              # the IMU and the camera are where configs/layout_v2.toml mounts them
        from crab.electronics import Layout
        lay = Layout()
        im = lay["imu"]
        imu_pose = f'pos="{f(*(im.pos / 1000))}" quat="{f(*_quat_of(im.R))}"'
        cpos, cdir = lay["camera_lens"].window_pose()
        right = np.cross(cdir, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        eye_cam, eye_axes = tuple(cpos / 1000.0), f(*right, *np.cross(right, cdir))
    prim_group = 3 if (USE_CAD or USE_V2_CAD) else 0
    return f"""<!-- AUTO-GENERATED by sim/build_model.py from crab/params.py{(' (design v2 from its CAD, cad/v2/export.py)' if USE_V2_CAD else ' (design v2, primitives only)') if V2 else ''}{' + CAD (cad/build.py)' if USE_CAD else ''}. Do not edit by hand. -->
<mujoco model="robotic_crab{'_v2' if V2 else ''}">
  <compiler angle="radian" autolimits="true" meshdir="meshes"/>
  <option timestep="0.002" integrator="implicitfast"/>

  <visual>
    <global offwidth="1280" offheight="720" azimuth="120" elevation="-20"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
    <map znear="0.005"/>
  </visual>

  <default>
    <default class="crab">
      <geom contype="1" conaffinity="0" condim="3" friction="0.8 0.02 0.001" group="{prim_group}"/>
      <default class="visual">
        <geom contype="0" conaffinity="0" group="2" density="0"/>
      </default>
{servo_defaults()}
    </default>
  </default>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.75 0.95" rgb2="0.95 0.97 1" width="256" height="256"/>
    <texture name="sand" type="2d" builtin="checker" rgb1="0.86 0.80 0.66" rgb2="0.80 0.74 0.60" width="512" height="512" mark="edge" markrgb="0.7 0.64 0.5"/>
    <material name="floor" texture="sand" texrepeat="{8/S:.3g} {8/S:.3g}" reflectance="0.05"/>
    <material name="shell" rgba="{ORANGE if V2 else '0.90 0.33 0.20 1'}" specular="{0.5 if V2 else 0.3}" shininess="{0.6 if V2 else 0.4}"/>
    <material name="claw" rgba="{ORANGE if V2 else '0.75 0.22 0.15 1'}" specular="0.3"/>
    <material name="pad" rgba="0.18 0.18 0.2 1" specular="0.1"/>
    <material name="tip" rgba="{RUBBER if V2 else '0.25 0.18 0.15 1'}"/>
    <material name="servo" rgba="{DARK if V2 else '0.12 0.12 0.14 1'}"/>
    <material name="eye" rgba="0.97 0.97 0.95 1"/>
    <material name="pupil" rgba="0.05 0.05 0.08 1"/>
    <material name="printed" rgba="0.92 0.40 0.22 1" specular="0.25" shininess="0.3"/>
    <material name="horn" rgba="0.75 0.76 0.78 1" specular="0.6" shininess="0.8"/>
    <material name="pcb" rgba="0.10 0.45 0.25 1"/>
{FACE_ASSET}
{meshes}
  </asset>

  <worldbody>
    <light pos="0.3 -0.3 1.5" dir="-0.2 0.2 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor" contype="0" conaffinity="1"/>

    <body name="body" pos="0 0 {z0:.4f}" childclass="crab">
      <freejoint name="root"/>
      <camera name="track" mode="trackcom" pos="{f(*cam_pos)}" xyaxes="{lookat_xyaxes(cam_pos, (0, 0, 0))}"/>
      <camera name="eye_cam" pos="{f(*eye_cam)}" xyaxes="{eye_axes}" fovy="{COL.get('eye_fovy', 102) if COL else 102}"/>
      <site name="imu" {imu_pose} size="{0.01*S:.4f}" rgba="0 0 0 0"/>
      {cad("body")}
      {carapace()}
      {eyes_xml()}
      {face_xml()}
{"".join((leg_xml_v2cad if USE_V2_CAD else leg_xml_v2 if V2 else leg_xml)(lg) for lg in LEG_IDS)}
{"".join(arm_xml(a) for a in ARM_IDS)}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>

  <sensor>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_accel" site="imu"/>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
{foot_sensors}
  </sensor>

  <keyframe>
    <key name="stand" qpos="0 0 {z0:.4f} 1 0 0 0 {f(*qpos_tree_order(STAND))}" ctrl="{f(*STAND)}"/>
  </keyframe>
</mujoco>
"""


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(OUT))
    print(f"wrote {OUT.relative_to(ROOT)}  |  bodies={m.nbody} joints={m.njnt} actuators={m.nu}  "
          f"total mass={sum(m.body_mass):.3f} kg  servos: "
          + ", ".join(f"{n} x{sum(servo_for(j).name == n for j in JOINT_NAMES)}" for n in USED))

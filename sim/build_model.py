"""Generate the MuJoCo model (MJCF) of the crab from crab/params.py (+ CAD outputs if present).

    python sim/build_model.py                 -> writes sim/models/crab.xml
    python sim/build_model.py --primitive     -> ignore CAD meshes, simple shapes only
    python sim/build_model.py --no-backlash
    python sim/build_model.py --design v2     -> writes sim/models/crab_v2.xml: the resized STS3215 robot
                                                 (primitives only, no CAD exists for it; docs/dev/V2_MODEL.md)

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

P = PARAMS
OUT = ROOT / "sim" / "models" / P.model_file
S = P.scale                                   # size relative to the 100 mm-shin original
V2 = P.design == "v2"
K = P.arm_scale                               # claw / jaw geometry relative to v1

VIS_FILE = ROOT / "sim" / "models" / "visuals.json"
MP_FILE = ROOT / "sim" / "models" / "mass_props.json"
COL_FILE = ROOT / "sim" / "models" / "collision.json"
USE_CAD = VIS_FILE.exists() and MP_FILE.exists() and "--primitive" not in sys.argv and not V2
VIS = json.loads(VIS_FILE.read_text()) if USE_CAD else {}
MP = json.loads(MP_FILE.read_text()) if USE_CAD else {}
COL = json.loads(COL_FILE.read_text()) if (USE_CAD and COL_FILE.exists()) else None
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
    if not USE_CAD or body not in MP:
        return ""
    mp = MP[body]
    out = [f'<inertial pos="{fe(*mp["com"])}" mass="{mp["mass"]:.6f}" fullinertia="{fe(*mp["inertia"])}"/>']
    for v in VIS.get(body, []):
        out.append(f'<geom class="visual" type="mesh" mesh="{v["mesh"]}" material="{v["material"]}"/>')
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
          {jnt(leg_id + "_femur", "0 -1 0", L.femur_range)}
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
    return f"""
      <body name="{arm_id}_shoulder" pos="{f(x, s*y, z)}" euler="0 0 {s*yaw:.4f}">
        {jnt(arm_id + "_shoulder_yaw", f"0 0 {s}", A.shoulder_yaw_range)}
        {cad(arm_id + "_shoulder")}
        {servo_box(arm_id + "_shoulder_yaw")}
        <geom type="sphere" size="{0.007 * K:.4g}" mass="{A.shoulder_mass:.4g}" material="shell"/>
        <body name="{arm_id}_upper" pos="{A.shoulder_offset} 0 0">
          {jnt(arm_id + "_shoulder_pitch", "0 -1 0", A.shoulder_pitch_range)}
          {cad(arm_id + "_upper")}
          <geom type="capsule" fromto="0 0 0 {h - 0.010 * K:.4f} 0 0" size="{0.005 * K:.4g}" mass="{A.upper_mass * 0.3:.4f}" material="shell"/>
          <geom name="{arm_id}_palm" type="ellipsoid" pos="{h - 0.004 * K:.4f} 0 {-0.004 * K:.4g}" size="{f(0.011 * K, 0.0065 * K, 0.009 * K)}" mass="{A.upper_mass * 0.7:.4f}" material="shell"/>
          <!-- fixed finger: the gripping strip (blade + TPU pad, level with the desk at the grip pitch) and
               the blade above it, hanging below the palm (cad/parts.py jaw_side) -->
          <geom name="{arm_id}_pollex" type="box" pos="{f(*JAW['strip'][0])}" euler="0 {-JAW['pitch']:.4f} 0" size="{f(*JAW['strip'][1])}" mass="0.0008" material="claw"/>
          <geom name="{arm_id}_pollex_blade" type="box" pos="{f(*JAW['blade'][0])}" size="{f(*JAW['blade'][1])}" mass="0.0006" material="claw"/>
          <site name="{arm_id}_grip_site" pos="{f(*JAW['site'])}" euler="0 {-JAW['pitch']:.4f} 0" size="0.005" rgba="0 0 0 0"/>
          <body name="{arm_id}_dactyl" pos="{h:.4f} 0 0">
            {jnt(arm_id + "_claw", f"0 0 {s}", A.claw_range)}
            {cad(arm_id + "_dactyl")}
            <geom name="{arm_id}_dactyl_jaw" type="box" pos="{f(*JAW['d_strip'][0])}" quat="{f(*JAW['d_strip'][2])}" size="{f(*JAW['d_strip'][1])}" mass="{A.jaw_mass}" material="claw"/>
            <geom name="{arm_id}_dactyl_blade" type="box" pos="{f(*JAW['d_blade'][0])}" quat="{f(*JAW['d_blade'][2])}" size="{f(*JAW['d_blade'][1])}" mass="0.0004" material="claw"/>
            <geom name="{arm_id}_dactyl_drop" type="box" pos="{f(*JAW['d_drop'][0])}" size="{f(*JAW['d_drop'][1])}" mass="0.0004" material="claw"/>
          </body>
        </body>
      </body>"""


def jaw_geoms(s):
    """Collision boxes for the claw's jaws, from the CAD numbers (cad/parts.py: GRIP_PITCH, JAW_X, JAW_LOW,
    STRIP_H, PAD_T, BLADE_T, D_DROP) - kept here as plain numbers so the model builds without the CAD.
    s = +1 left arm, -1 right (mirror). Returns {name: ((pos), (half sizes))} in metres, plus 'pitch'
    and the grip site."""
    pitch = np.radians(20.0)
    # (v2: the whole claw is K = 1.5 times bigger; v1: K = 1.0, so every number below is unchanged)
    x0, x1, low0, strip_h, pad_t, blade_t = (K * v for v in (40.3, 55.5, -24.5, 10.0, 1.2, 2.4))
    grip_w, q_par = 9.0 * K, np.radians(19.0)            # jaws parallel GRIP_W apart at CLAW_PARALLEL
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
    """v2: the same face as today's robot (cad/parts.py: visor 67.5 x 45 mm, module 61.5 x 39.5, lit area 55 x 27.5, leaning
    back FACE_TILT), set into the front of the body where configs/layout_v2.toml puts the OLED. An orange bezel in
    the shell's material (four bars round the module's opening, 6.5 mm deep like v1's), a dark module face, and the
    lit area as sim/desk.py draws it: a thin visual box with material face_mat (emission 0.9) textured by face_tex
    (256 x 128, written by set_face). Visual only: no mass (the layout's oled and visor parts carry it)."""
    if not V2:
        return ""
    from crab.electronics import Layout
    ol = Layout()["oled"]
    n = ol.R @ np.array([0.0, 0.0, 1.0])
    centre = ol.pos + n * 1.7                                 # FACE_CENTRE: the module's glass is 1.8 mm ahead of it
    # the screen frame of cad/parts.py FACE_LOC / sim/desk.py: x out of the screen, y across, z up the screen
    Rs = ol.R @ np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    q = f(*_quat_of(Rs))

    def box(p, half, mat):
        w = (centre + Rs @ np.asarray(p, float)) / 1000.0
        h = np.asarray(half, float) / 1000.0
        return f'<geom class="visual" type="box" size="{f(*h)}" pos="{f(*w)}" quat="{q}" material="{mat}" mass="0"/>'
    mw, mh, hw, hh = 61.5, 39.5, 67.5, 45.0
    ow, oh = mw + 0.6, mh + 0.6                              # the module pocket
    bz = (hh - oh) / 4.0, (hw - ow) / 4.0
    return "\n      ".join([
        box((0.25, 0, oh / 2 + bz[0]), (3.25, hw / 2, bz[0]), "shell"),             # bezel: top, bottom, left, right
        box((0.25, 0, -(oh / 2 + bz[0])), (3.25, hw / 2, bz[0]), "shell"),
        box((0.25, ow / 2 + bz[1], 0), (3.25, bz[1], hh / 2), "shell"),
        box((0.25, -(ow / 2 + bz[1]), 0), (3.25, bz[1], hh / 2), "shell"),
        box((1.1, 0, 0), (0.6, mw / 2, mh / 2), "servo"),                           # the module's dark face
        box((0.6, 0, 0), (1.2, 27.5, 13.75), "face_mat"),                           # lit area, glass front at x = 1.8
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
    """v2 leg: printed capsules (they collide, so the legs can push the body up from the floor) and a real servo
    box with the servo's real mass in the link that carries it: the thigh servo in the coxa link, the knee servo in
    the femur link (the hip servo is in the body). Printed mass = the link's mass minus its servo."""
    s = 1 if leg_id[1] == "L" else -1
    (x, y, z), yaw = P.leg_mounts[leg_id[0]]
    L = P.leg
    sf, st = servo_for(leg_id + "_femur"), servo_for(leg_id + "_tibia")
    lf, wf, hf = (v / 2000 for v in sf.size_mm)          # half length, width, height (shaft along the height)
    lt, wt, ht = (v / 2000 for v in st.size_mm)

    def box(half, pos, mass):
        return (f'<geom class="visual" type="box" size="{f(*half)}" pos="{f(*pos)}" mass="{mass:.5g}" '
                f'material="servo"/>')
    return f"""
      <body name="{leg_id}_coxa_link" pos="{f(x, s*y, z)}" euler="0 0 {s*yaw:.4f}">
        {jnt(leg_id + "_coxa", f"0 0 {-s}", L.coxa_range_for(leg_id))}
        <geom type="capsule" fromto="0 0 0 {L.coxa_len} 0 0" size="0.016" mass="{L.coxa_mass - sf.mass:.5g}" material="shell"/>
        {box((lf, hf, wf), (L.coxa_len - 0.3 * lf, 0, 0), sf.mass)}
        <body name="{leg_id}_femur_link" pos="{L.coxa_len} 0 0">
          {jnt(leg_id + "_femur", "0 -1 0", L.femur_range)}
          <geom type="capsule" fromto="0 0 0 {L.femur_len} 0 0" size="0.015" mass="{L.femur_mass - st.mass:.5g}" material="shell"/>
          {box((lt, ht, wt), (L.femur_len - 0.3 * lt, 0, 0), st.mass)}
          <body name="{leg_id}_tibia_link" pos="{L.femur_len} 0 0">
            {jnt(leg_id + "_tibia", "0 -1 0", L.tibia_range)}
            <geom type="capsule" fromto="0 0 0 {L.tibia_len} 0 0" size="0.010" mass="{L.tibia_mass - 0.001:.5g}" material="shell"/>
            <geom name="{leg_id}_foot" type="sphere" pos="{L.tibia_len} 0 0" size="{L.foot_radius}" mass="0.001" material="tip" friction="1.1 0.02 0.001"/>
            <site name="{leg_id}_foot_site" pos="{L.tibia_len} 0 0" size="{L.foot_radius*1.4:.4f}" rgba="0 0 0 0"/>
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
        # the screen module and the visor are drawn by face_xml: here they only carry their mass
        look = 'rgba="0 0 0 0"' if p.d.get("face") or p.kind == "printed" else f'material="{mat}"'
        out.append(f'<geom name="part_{p.name}" class="visual" type="box" size="{f(*(p.half / 1000.0))}" '
                   f'pos="{f(*pos)}" quat="{f(*_quat_of(p.R))}" mass="{p.mass_g / 1000:.5g}" {look}/>')
    return out


def body_v2() -> str:
    """v2 body from primitives (sizing study, sim/sizing/model.py): chassis box and dome shell (they collide), the
    four yaw servos at the hips (shaft vertical), and every electronic part of configs/layout_v2.toml where the
    layout puts it, with its own mass (this replaced the lumped pack and electronics blocks)."""
    B = P.body
    ch, cp = B.half_size, B.chassis_pos
    sv = servo_for("FL_coxa")
    lh, wh, hh = (v / 2000 for v in sv.size_mm)
    hx, hy = abs(P.leg_mounts["F"][0][0]), P.leg_mounts["F"][0][1]
    out = [f'<geom name="chassis" type="box" pos="{f(*cp)}" size="{f(*ch)}" mass="{B.shell_mass * 0.75:.5g}" material="shell"/>',
           f'<geom name="dome" type="ellipsoid" pos="{f(*B.dome_pos)}" size="{f(*B.dome_half)}" mass="{B.shell_mass * 0.25:.5g}" material="shell"/>']
    for sx in (1, -1):
        for sy in (1, -1):
            out.append(f'<geom class="visual" type="box" size="{f(lh, wh, hh)}" pos="{f(sx * hx, sy * hy, -0.15 * 2 * hh)}" '
                       f'mass="{sv.mass:.5g}" material="servo"/>')
    out += layout_geoms()
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
    actuators = "\n".join(f'    <position name="{j}" joint="{j}" class="srv_{servo_for(j).name}" inheritrange="1"/>'
                          for j in JOINT_NAMES)
    foot_sensors = "\n".join(f'    <touch name="{lg}_foot_touch" site="{lg}_foot_site"/>' for lg in LEG_IDS)
    meshes = "\n".join(f'    <mesh name="{v["mesh"]}" file="{v["file"]}" scale="0.001 0.001 0.001"/>'
                       for items in VIS.values() for v in items) if USE_CAD else ""
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
    prim_group = 3 if USE_CAD else 0
    return f"""<!-- AUTO-GENERATED by sim/build_model.py from crab/params.py{' (design v2, primitives only)' if V2 else ''}{' + CAD (cad/build.py)' if USE_CAD else ''}. Do not edit by hand. -->
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
    <material name="shell" rgba="0.90 0.33 0.20 1" specular="0.3" shininess="0.4"/>
    <material name="claw" rgba="0.75 0.22 0.15 1" specular="0.3"/>
    <material name="tip" rgba="0.25 0.18 0.15 1"/>
    <material name="servo" rgba="0.12 0.12 0.14 1"/>
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
{"".join((leg_xml_v2 if V2 else leg_xml)(lg) for lg in LEG_IDS)}
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

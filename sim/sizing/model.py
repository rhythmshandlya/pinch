"""Parametric crab for the sizing study, built with MjSpec.

    Design(name, s, servo)  ->  geometry + itemised mass
    build_spec(design)      ->  mujoco.MjSpec  (4 legs x 3 joints + 2 arms x 3 joints, position actuators)

What is scaled and what is not (every choice is stated, nothing is hidden):
* Leg links, hip positions, foot radius: today's values x s (coxa 22, femur 30, tibia 52 mm at s = 1).
* Servos: real envelopes and masses (STS3215 45.2 x 24.7 x 35 mm, 55 g; STS3032 23 x 12 x 27.5 mm, 20 g).
  Servo placement follows today's CAD: the coxa (yaw) servo sits in the body, the femur servo in the coxa
  link, the knee servo in the femur link (CAD mass_props: link centres of mass lie at the servo end).
* Printed parts: today's printed mass x s^print_exp (default 3 = by volume, the upper bound; a hollow shell of
  constant wall thickness would scale nearer s^2).
* Body: today's chassis box + dome footprint x max(s, what the servos need); the height grows only as far as
  servo height + battery thickness need (today: 27.5 mm servo + 18 mm pack).
* Battery: mass from energy at 137 Wh/kg (the BOM's Bonka 850 mAh 7.4 V 46 g pack = 6.29 Wh).
* Arms: exactly today's (GS4308 PWM servos, same links, same masses), mounted at the front edge of the body.
No gear-backlash joints (BAM has none) and no CAD meshes: boxes and capsules, masses from the itemised list.
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import copy  # noqa: E402

from servos import ServoModel, repo_servo  # noqa: E402

from crab.params import ARM_IDS, JOINT_NAMES, LEG_IDS, PARAMS  # noqa: E402

MOUNTS = copy.deepcopy(PARAMS.leg_mounts)       # today's hip mounts; crab.gait is later patched to the scaled size

G = 9.81
WH_PER_KG = 137.0                 # BOM pack: 7.4 V x 0.85 Ah = 6.29 Wh in 46 g
PACK_G_PER_CM3 = 46.0 / (5.7 * 3.0 * 1.8)     # BOM pack 57 x 30 x 18 mm
# Today's masses (mass_props.json, total 605 g), split by us: printed = what is left after servos, pack, electronics
TODAY = dict(
    leg_printed=(0.0252 - 0.020) + (0.0287 - 0.020) + 0.0074,   # coxa + femur + tibia link minus their 20 g servo, per leg
    arm_total=2 * (0.0074 + 0.0115 + 0.0022),
    body_total=0.31846,
    battery=0.046,
    electronics=0.080,        # ESTIMATE: Radxa ZERO 3W, camera, OLED, IMU, 2 regulators, hub, mics, speaker, ToF, wire
)
ARM_SERVO = repo_servo("GS4308")
TODAY["body_shell"] = TODAY["body_total"] - 4 * 0.020 - TODAY["battery"] - TODAY["electronics"]   # ~112 g printed
# Today's body (sim/models/collision.json): chassis box and dome, metres, body frame (hip axis at z = 0)
CHASSIS = dict(half=(0.04231, 0.05606, 0.01965), pos=(0.00432, 0.0, -0.01015))
DOME = dict(half=(0.04599, 0.05769, 0.0190), pos=(0.00601, 0.0, 0.0095))
TODAY_BODY_H = 27.5 + 18.0         # mm: servo + battery stack in today's chassis


@dataclass(frozen=True)
class Design:
    name: str
    s: float                       # leg scale vs today (1.0 = today's 22/30/52 mm links)
    servo: ServoModel              # the 12 leg servos
    batt_mah: float = 850.0
    batt_v: float = 7.4
    print_exp: float = 3.0
    extra_electronics: float = 0.0     # kg, e.g. a 7.4 V servo rail regulator
    arm_servo: ServoModel = ARM_SERVO

    # ---- geometry
    @property
    def coxa(self): return 0.022 * self.s
    @property
    def femur(self): return 0.030 * self.s
    @property
    def tibia(self): return 0.052 * self.s
    @property
    def foot_r(self): return 0.005 * self.s
    @property
    def hip(self): return (0.022 * self.s, 0.046 * self.s)          # |x|, |y| of the hip yaw axes

    @property
    def battery_mass(self): return self.batt_mah / 1000 * self.batt_v / WH_PER_KG

    @property
    def body_xy_scale(self):
        """Today's body scaled by s, but never so small that the hip servos poke out of it."""
        L, W, _H = (v / 1000 for v in self.servo.size_mm)
        hx, hy = self.hip
        need_x = (hx + L / 2 + 0.002) / CHASSIS["half"][0]
        need_y = (hy + W / 2 + 0.002) / CHASSIS["half"][1]
        return max(self.s, need_x, need_y)

    @property
    def chassis_half(self):
        k = self.body_xy_scale
        return (CHASSIS["half"][0] * k, CHASSIS["half"][1] * k, self.chassis_hz)

    @property
    def pack_thickness(self):
        """Pack thickness: 18 mm as today, thicker if the pack does not fit flat in half of the chassis floor."""
        k = self.body_xy_scale
        floor = (2 * CHASSIS["half"][0] * k) * (2 * CHASSIS["half"][1] * k)
        vol = self.battery_mass / (PACK_G_PER_CM3 * 1e3)          # m^3
        return max(0.018, vol / (0.5 * floor))

    @property
    def body_height_mm(self):
        return max(TODAY_BODY_H, self.servo.size_mm[2] + self.pack_thickness * 1000)

    @property
    def chassis_hz(self):
        return CHASSIS["half"][2] * self.body_height_mm / TODAY_BODY_H

    @property
    def body_volume_ratio(self):
        k = self.body_xy_scale
        return k * k * self.body_height_mm / TODAY_BODY_H

    # ---- itemised mass (kg)
    def mass_items(self) -> dict:
        n = self.servo.mass
        leg_print = 4 * TODAY["leg_printed"] * self.s ** self.print_exp
        return {
            "leg servos (12)": 12 * n,
            "arm servos (6)": 6 * self.arm_servo.mass,
            "printed legs": leg_print,
            "printed arms": TODAY["arm_total"] - 6 * self.arm_servo.mass,
            "printed body": TODAY["body_shell"] * self.body_volume_ratio,
            "electronics": TODAY["electronics"] + self.extra_electronics,
            "battery": self.battery_mass,
        }

    @property
    def mass(self): return sum(self.mass_items().values())

    def with_battery(self, mah): return replace(self, batt_mah=mah)


def _f(*v): return [float(x) for x in v]


def stand_height(d: Design, fem=0.35, tib=-1.92) -> float:
    """Hip axis height above the floor in the STAND pose (femur +0.35, tibia -1.92 rad: shins near vertical)."""
    drop = d.femur * np.sin(fem) + d.tibia * np.sin(fem + tib)
    return float(-drop + d.foot_r + 0.002)


def build_spec(d: Design) -> mujoco.MjSpec:
    from crab.params import PARAMS as P
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.modelname = f"crab_{d.name}"
    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=0, conaffinity=1,
                rgba=_f(0.86, 0.80, 0.66, 1))

    body = wb.add_body(name="body", pos=_f(0, 0, stand_height(d)))
    body.add_freejoint()

    def solid(b, shape, size, pos, mass, rgba=(0.9, 0.33, 0.2, 1), collide=False, **kw):
        return b.add_geom(type=shape, size=_f(*size), pos=_f(*pos), mass=mass, rgba=_f(*rgba),
                          contype=1 if collide else 0, conaffinity=0, **kw)

    items = d.mass_items()
    sv = d.servo
    L, W, H = (v / 1000 for v in sv.size_mm)
    BOX, CAP, SPH, ELL = (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE,
                          mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    # --- body: shell (collides), 4 yaw servos, pack low, electronics high. Shell mass split chassis/dome by volume.
    ch, k = d.chassis_half, d.body_xy_scale
    cpos = (CHASSIS["pos"][0] * k, 0.0, CHASSIS["pos"][2] * d.body_height_mm / TODAY_BODY_H)
    shell = items["printed body"]
    solid(body, BOX, ch, cpos, shell * 0.75, collide=True)
    dh = (DOME["half"][0] * k, DOME["half"][1] * k, DOME["half"][2])
    solid(body, ELL, dh, (DOME["pos"][0] * k, 0, cpos[2] + ch[2] - DOME["half"][2] * 0.5 + 0.0), shell * 0.25, collide=True)
    hx, hy = d.hip
    for sx in (1, -1):
        for sy in (1, -1):
            solid(body, BOX, (L / 2, W / 2, H / 2), (sx * hx, sy * hy, -0.15 * H), sv.mass, rgba=(0.12, 0.12, 0.14, 1))
    pack_t = d.pack_thickness
    pack_len = 2 * ch[0] * 0.85
    pack_w = d.battery_mass / (PACK_G_PER_CM3 * 1e3) / (pack_len * pack_t)
    solid(body, BOX, (pack_len / 2, pack_w / 2, pack_t / 2), (cpos[0], 0, cpos[2] - ch[2] + pack_t / 2 + 0.002),
          d.battery_mass, rgba=(0.2, 0.2, 0.8, 1))
    solid(body, BOX, (0.030, 0.035, 0.008), (0.005, 0, 0.012), items["electronics"], rgba=(0.1, 0.45, 0.25, 1))

    # --- legs
    rng = {"coxa": (np.radians(-55), np.radians(49)), "front_coxa": (np.radians(-55), np.radians(28)),
           "femur": P.leg.femur_range, "tibia": P.leg.tibia_range}
    pm = d.mass_items()["printed legs"] / 4
    f_c = TODAY["leg_printed"]
    shares = ((0.0252 - 0.020) / f_c, (0.0287 - 0.020) / f_c, 0.0074 / f_c)       # coxa, femur, tibia link printed split
    acts = []
    for lg in LEG_IDS:
        sgn = 1 if lg[1] == "L" else -1
        (mx, my, mz), yaw = MOUNTS[lg[0]]
        mx, my = mx * d.s, my * d.s
        cb = body.add_body(name=f"{lg}_coxa_link", pos=_f(mx, sgn * my, mz), euler=_f(0, 0, sgn * yaw))
        cb.add_joint(name=f"{lg}_coxa", type=mujoco.mjtJoint.mjJNT_HINGE, axis=_f(0, 0, -sgn),
                     range=_f(*(rng["front_coxa"] if lg[0] == "F" else rng["coxa"])))
        cb.add_geom(type=CAP, fromto=_f(0, 0, 0, d.coxa, 0, 0), size=_f(0.015 * d.s, 0, 0), mass=pm * shares[0],
                    contype=0, conaffinity=0, rgba=_f(0.9, 0.33, 0.2, 1))
        solid(cb, BOX, (L / 2, H / 2, W / 2), (d.coxa - 0.15 * L, 0, 0), sv.mass, rgba=(0.12, 0.12, 0.14, 1))
        fb = cb.add_body(name=f"{lg}_femur_link", pos=_f(d.coxa, 0, 0))
        fb.add_joint(name=f"{lg}_femur", type=mujoco.mjtJoint.mjJNT_HINGE, axis=_f(0, -1, 0), range=_f(*rng["femur"]))
        fb.add_geom(type=CAP, fromto=_f(0, 0, 0, d.femur, 0, 0), size=_f(0.013 * d.s, 0, 0), mass=pm * shares[1],
                    contype=0, conaffinity=0, rgba=_f(0.9, 0.33, 0.2, 1))
        solid(fb, BOX, (L / 2, H / 2, W / 2), (d.femur - 0.15 * L, 0, 0), sv.mass, rgba=(0.12, 0.12, 0.14, 1))
        tb = fb.add_body(name=f"{lg}_tibia_link", pos=_f(d.femur, 0, 0))
        tb.add_joint(name=f"{lg}_tibia", type=mujoco.mjtJoint.mjJNT_HINGE, axis=_f(0, -1, 0), range=_f(*rng["tibia"]))
        tb.add_geom(type=CAP, fromto=_f(0, 0, 0, d.tibia, 0, 0), size=_f(0.0085 * d.s, 0, 0), mass=pm * shares[2] - 0.001,
                    contype=0, conaffinity=0, rgba=_f(0.9, 0.33, 0.2, 1))
        tb.add_geom(name=f"{lg}_foot", type=SPH, size=_f(d.foot_r, 0, 0), pos=_f(d.tibia, 0, 0), mass=0.001,
                    contype=1, conaffinity=0, friction=_f(1.1, 0.02, 0.001), rgba=_f(0.25, 0.18, 0.15, 1))
        acts += [f"{lg}_coxa", f"{lg}_femur", f"{lg}_tibia"]

    # --- arms (today's, GS4308), at the front edge of the body
    A = PARAMS.arm
    ax, ay, az = PARAMS.arm_mount[0]
    ax = ch[0] * 0.945 + cpos[0]          # today: x = 0.040 on a chassis reaching 0.0466
    for aid in ARM_IDS:
        sgn = 1 if aid == "L" else -1
        sb = body.add_body(name=f"{aid}_shoulder", pos=_f(ax, sgn * ay, az), euler=_f(0, 0, sgn * PARAMS.arm_mount[1]))
        sb.add_joint(name=f"{aid}_shoulder_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=_f(0, 0, sgn),
                     range=_f(*A.shoulder_yaw_range))
        solid(sb, SPH, (0.007,), (0, 0, 0), 0.0074)
        ub = sb.add_body(name=f"{aid}_upper", pos=_f(A.shoulder_offset, 0, 0))
        ub.add_joint(name=f"{aid}_shoulder_pitch", type=mujoco.mjtJoint.mjJNT_HINGE, axis=_f(0, -1, 0),
                     range=_f(*A.shoulder_pitch_range))
        solid(ub, CAP, (0.005, 0.010), (0.010, 0, 0), 0.0115 * 0.3, euler=_f(0, np.pi / 2, 0))
        solid(ub, ELL, (0.011, 0.0065, 0.009), (A.upper_len - 0.004, 0, -0.004), 0.0115 * 0.7)
        db = ub.add_body(name=f"{aid}_dactyl", pos=_f(A.upper_len, 0, 0))
        db.add_joint(name=f"{aid}_claw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=_f(0, 0, sgn), range=_f(*A.claw_range))
        solid(db, BOX, (0.010, 0.002, 0.008), (0.012, 0, -0.004), 0.0022)
        acts += [f"{aid}_shoulder_yaw", f"{aid}_shoulder_pitch", f"{aid}_claw"]
    assert acts == JOINT_NAMES, "joint order must match crab.params.JOINT_NAMES"

    # --- servo model on every joint: damping/friction/armature on the joint, position actuator on top
    for j in spec.joints:
        if j.name == "root" or not j.name:
            continue
        sm = servo_for_joint(d, j.name)
        j.damping = [sm.damping, 0.0, 0.0]
        j.frictionloss, j.armature = sm.frictionloss, sm.armature
        a = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = sm.kp
        a.biasprm[1] = -sm.kp
        a.forcelimited = True
        a.forcerange = _f(-sm.forcerange, sm.forcerange)
        a.ctrllimited = True
        a.ctrlrange = _f(*j.range)
    return spec


def servo_for_joint(d: Design, joint: str) -> ServoModel:
    return d.arm_servo if joint[0] in "LR" and joint[1] == "_" else d.servo


def compile_stand(d: Design, frictionless=False):
    """Compile and put the crab into its STAND pose on the floor. Returns (model, data)."""
    from crab.emotes import STAND
    m = build_spec(d).compile()
    if frictionless:
        m.dof_frictionloss[:] = 0
    data = mujoco.MjData(m)
    data.qpos[:7] = [0, 0, stand_height(d), 1, 0, 0, 0]
    for i, jn in enumerate(JOINT_NAMES):
        data.qpos[m.jnt_qposadr[m.joint(jn).id]] = STAND[i]
    data.ctrl[:] = STAND
    mujoco.mj_forward(m, data)
    return m, data


def span_and_height(m, data) -> tuple:
    """Axis-aligned extent (m) of every geom of the standing crab: (length x, width y, height z)."""
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        R = data.geom_xmat[g].reshape(3, 3)
        sz = m.geom_size[g]
        t = m.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = np.full(3, sz[0])
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half = np.abs(R[:, 2]) * sz[1] + sz[0]
        else:
            half = np.abs(R) @ sz
        lo, hi = np.minimum(lo, data.geom_xpos[g] - half), np.maximum(hi, data.geom_xpos[g] + half)
    return tuple(hi - lo)


if __name__ == "__main__":
    from servos import sts3215
    for d in (Design("today", 1.0, repo_servo("STS3032")), Design("s1.5", 1.5, sts3215("m1"), 1500, 7.4)):
        m, data = compile_stand(d)
        for _ in range(500):
            mujoco.mj_step(m, data)
        print(d.name, f"mass spec {sum(m.body_mass)*1000:.0f} g vs itemised {d.mass*1000:.0f} g",
              "span", np.round(np.array(span_and_height(m, data)) * 100, 1), "cm", "z", round(data.qpos[2] * 100, 1),
              "tilt ok", round(data.xmat[1].reshape(3, 3)[2, 2], 3))

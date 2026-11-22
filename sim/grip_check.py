"""Grip test of design v2's claw (R-35): can it pick up, lift, shake and hold Pinch's toys?

    python sim/grip_check.py                       # every object, 50 trials each, left claw
    python sim/grip_check.py --objects pen,ball30 --trials 20 --video media/review/phase-02/grip.gif

The body is held at its standing height (legs in the stand pose) so only the gripper is tested. Each trial: the arm
lowers until the open claw's lower edge is 2 mm above the desk, around the object (placed on the desk with a random offset of +-4 mm and a random yaw of
+-20 deg), the claw closes (0.5 s), the arm lifts 0.6 rad (0.8 s), shakes its yaw +-0.15 rad at 2 Hz for 2 s, then holds
3 s. Success: the object is still within 30 mm of the grip point and off the desk. The arm servos are the v2 model's
(STS3032 at its 40 % torque limit). Contacts use the elliptic friction cone and impratio 10 in this test (better for
grasping; the walking physics is unchanged). Rates are printed with their 95 % ranges (sim/benches/stats.py).
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CRAB_DESIGN", "v2")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.emotes import STAND  # noqa: E402
from crab.params import JOINT_NAMES, PARAMS  # noqa: E402
from sim.benches.stats import rate  # noqa: E402

# name: (MuJoCo geom type, half sizes in m, mass in kg, how it rests: z of its centre)
OBJECTS = {
    "pen": ("capsule", (0.0045, 0.065), 0.010),           # 9 mm round pen, 140 mm long (lies along the claw)
    "rod5": ("capsule", (0.0025, 0.040), 0.003),          # a 5 mm stick
    "eraser": ("box", (0.020, 0.010, 0.005), 0.012),      # 40 x 20 x 10 mm
    "cube25": ("box", (0.0125, 0.0125, 0.0125), 0.008),
    "ball30": ("sphere", (0.015,), 0.006),
    "ball36": ("sphere", (0.018,), 0.010),
    "box40x20": ("box", (0.020, 0.010, 0.010), 0.015),
}
ARM = "L"
MODEL_PATH = os.environ.get("CRAB_MODEL_PATH") or str(ROOT / "sim" / "models" / PARAMS.model_file)   # or a candidate
PLACE = os.environ.get("GRIP_PLACE", "deep")       # where the toy is put: "deep" in the claw's mouth, "tips", "centroid"
DESK_GAP = float(os.environ.get("GRIP_DESK_GAP", "0.002"))   # m: the jaw bottom this far above the desk when it closes (negative: pressing on it)


def build(obj):
    spec = mujoco.MjSpec.from_file(MODEL_PATH)
    body = spec.body("body")
    for j in list(body.joints):                       # hold the body at its standing height
        spec.delete(j)
    key = spec.keys[0] if spec.keys else None
    kind, size, mass = OBJECTS[obj]
    ob = spec.worldbody.add_body(name="toy", pos=[0.3, 0.3, 0.05])
    ob.add_freejoint(name="toy_free")
    gtype = {"capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "box": mujoco.mjtGeom.mjGEOM_BOX,
             "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}[kind]
    ob.add_geom(name="toy", type=gtype, size=list(size) + [0] * (3 - len(size)), mass=mass,
                friction=[0.7, 0.01, 0.001], rgba=[0.2, 0.5, 0.9, 1], condim=4)
    if key is not None:
        spec.delete(key)
    m = spec.compile()
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    m.opt.impratio = 10.0
    return m


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def joint_ids(m):
    q = {j: m.jnt_qposadr[m.joint(j).id] for j in JOINT_NAMES}
    a = {j: m.actuator(j).id for j in JOINT_NAMES}
    return q, a


def toy_rest_z(obj):
    kind, size, _ = OBJECTS[obj]
    return size[0] if kind in ("capsule", "sphere") else size[2]


def pick_pose():
    """The legs while picking: the stand, or crouched by GRIP_CROUCH m (a crab lowers its body to pick things up; the
    walking clearance is only needed while walking). Returns (18-joint pose, how much lower the body is, m)."""
    from crab.emotes import crouch, pose
    from crab.kinematics import leg_fk
    want = float(os.environ.get("GRIP_CROUCH", "0.03"))       # v2.1 picks from a 30 mm crouch (belly 35 mm standing)
    if want <= 0:
        return STAND, 0.0
    idx = [JOINT_NAMES.index("FL_" + k) for k in ("coxa", "femur", "tibia")]
    z0 = -leg_fk("FL", STAND[idx])[2]
    for a in np.linspace(0.0, 1.2, 241):                 # the crouch (thighs up, shins stay upright) that lowers it
        p = pose(crouch(a))
        if -leg_fk("FL", p[idx])[2] <= z0 - want:
            return p, z0 + leg_fk("FL", p[idx])[2]
    return pose(crouch(1.2)), z0 + leg_fk("FL", pose(crouch(1.2))[idx])[2]


def setup(m, d, obj, rng, stand_z):
    q, a = joint_ids(m)
    mujoco.mj_resetData(m, d)
    base = m.body("body").id
    legs, drop = pick_pose()
    m.body_pos[base][2] = stand_z - drop
    for i, j in enumerate(JOINT_NAMES):
        d.qpos[q[j]] = legs[i]
        d.ctrl[a[j]] = legs[i]
    return q, a


def jaw_bottom(m, d):
    """The lowest point of the fixed jaw's parts (backing plate and pads), world z."""
    low = 9.0
    for g in range(m.ngeom):
        nm = m.geom(g).name or ""
        if nm.startswith(f"{ARM}_pollex") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            R = d.geom_xmat[g].reshape(3, 3)
            ext = np.abs(R) @ m.geom_size[g]
            low = min(low, d.geom_xpos[g][2] - ext[2])
    return low


def target_for(m, d, q, a, obj, opening):
    """Arm pitch that brings the claw's lower edge 2 mm above the desk (the pads' shapes then fit each toy's height),
    the claw open by `opening` rad."""
    site = m.site(f"{ARM}_grip_site").id
    best, pick = 9.0, 0.0
    d.qpos[q[f"{ARM}_claw"]] = opening
    for p in np.linspace(m.jnt_range[m.joint(f"{ARM}_shoulder_pitch").id][0], 0.6, 160):
        d.qpos[q[f"{ARM}_shoulder_pitch"]] = p
        mujoco.mj_kinematics(m, d)
        err = abs(jaw_bottom(m, d) - DESK_GAP)
        if err < best:
            best, pick = err, p
    d.qpos[q[f"{ARM}_shoulder_pitch"]] = pick
    mujoco.mj_kinematics(m, d)
    return pick, d.site_xpos[site].copy()


def place(m, d, tq, obj, quat):
    """The desk spot (x, y) deepest in the claw's mouth: among the spots where the open claw clears the toy by 1.5 mm
    and the shut moving jaw would pass through it (so closing grips it), the one nearest the claw's hinge (a deep
    grasp; out at the tips a toy slips forward). 2 mm grid; the roomiest spot if none clears 1.5 mm."""
    toy = m.geom("toy").id
    claw = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith((f"{ARM}_pollex", f"{ARM}_dactyl", f"{ARM}_palm"))]
    moving = [g for g in claw if (m.geom(g).name or "").startswith(f"{ARM}_dactyl")]
    qc = m.jnt_qposadr[m.joint(f"{ARM}_claw").id]
    open_q, shut_q = d.qpos[qc], m.jnt_range[m.joint(f"{ARM}_claw").id][0]
    jaw = d.geom_xpos[m.geom(f"{ARM}_pollex").id]
    hinge = d.xpos[m.body(f"{ARM}_dactyl").id][:2].copy()
    best, room, deep = (float(jaw[0]), float(jaw[1])), -1.0, None
    d.qpos[tq + 3:tq + 7] = quat
    for x in np.arange(jaw[0] - 0.05, jaw[0] + 0.07, 0.002):
        for y in np.arange(jaw[1] - 0.05, jaw[1] + 0.05, 0.002):
            d.qpos[tq:tq + 3] = [x, y, toy_rest_z(obj) + 0.0005]
            d.qpos[qc] = open_q
            mujoco.mj_kinematics(m, d)
            gap = min(mujoco.mj_geomDistance(m, d, toy, g, 0.05, None) for g in claw)
            if gap <= 0.0005:
                continue
            d.qpos[qc] = shut_q
            mujoco.mj_kinematics(m, d)
            if min(mujoco.mj_geomDistance(m, d, toy, g, 0.05, None) for g in moving) >= -0.001:
                continue
            r = float(np.hypot(x - hinge[0], y - hinge[1]))
            key = -r if PLACE == "tips" else r                        # "tips": the flat pads under the ball cup
            if gap >= 0.0015 and (deep is None or key < deep[0]):
                deep = (key, (float(x), float(y)))
            if gap > room:
                best, room = (float(x), float(y)), gap
    d.qpos[qc] = open_q
    return deep[1] if deep else best


def trial(m, obj, rng, stand_z, frames=None, renderer=None, cam=None):
    d = mujoco.MjData(m)
    q, a = setup(m, d, obj, rng, stand_z)
    op = m.jnt_range[m.joint(f"{ARM}_claw").id][1] * 0.75
    pitch, grip = target_for(m, d, q, a, obj, op)
    d.ctrl[a[f"{ARM}_shoulder_pitch"]] = pitch
    d.qpos[q[f"{ARM}_claw"]] = op
    d.ctrl[a[f"{ARM}_claw"]] = op
    tq = m.jnt_qposadr[m.joint("toy_free").id]
    # the toy goes where the open claw has the most room round it (searched over the desk under the claw), then the
    # usual imperfect placement: +-4 mm and +-20 deg
    yaw = rng.uniform(-np.radians(20), np.radians(20))
    qyaw = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    if OBJECTS[obj][0] == "capsule":                     # a capsule's axis is its z: lay it down along the claw (x)
        qlay = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
        qyaw = _qmul(qyaw, qlay)
    if PLACE == "centroid":                                # midway between the two jaws' centres
        mid = (d.geom_xpos[m.geom(f"{ARM}_pollex").id] + d.geom_xpos[m.geom(f"{ARM}_dactyl_jaw").id]) / 2
        best = (float(mid[0]), float(mid[1]))
    else:
        best = place(m, d, tq, obj, qyaw)
    off = rng.uniform(-0.004, 0.004, 2)
    d.qpos[tq:tq + 3] = [best[0] + off[0], best[1] + off[1], toy_rest_z(obj) + 0.0005]
    d.qpos[tq + 3:tq + 7] = qyaw
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    site = m.site(f"{ARM}_grip_site").id
    if cam is not None:                                   # the camera looks at the toy from the front-left
        cam.lookat[:] = d.xpos[m.body("toy").id] + [0.0, 0.0, 0.02]
        cam.distance, cam.azimuth, cam.elevation = 0.26, 210, -18
    every = int(round(1 / 25 / dt))

    def run(secs, fn):
        for k in range(int(secs / dt)):
            fn(k * dt)
            mujoco.mj_step(m, d)
            if frames is not None and k % every == 0:
                renderer.update_scene(d, cam)
                frames.append(renderer.render().copy())

    run(0.4, lambda t: None)                                                   # settle on the desk
    run(0.5, lambda t: d.ctrl.__setitem__(a[f"{ARM}_claw"], op * (1 - min(1, t / 0.4))))
    run(0.8, lambda t: d.ctrl.__setitem__(a[f"{ARM}_shoulder_pitch"], pitch + 0.6 * min(1, t / 0.7)))
    y0 = d.ctrl[a[f"{ARM}_shoulder_yaw"]]
    run(2.0, lambda t: d.ctrl.__setitem__(a[f"{ARM}_shoulder_yaw"], y0 + 0.15 * np.sin(2 * np.pi * 2 * t)))
    run(3.0, lambda t: d.ctrl.__setitem__(a[f"{ARM}_shoulder_yaw"], y0))
    toy = d.xpos[m.body("toy").id]
    held = between_jaws(m, d, obj, site) and toy[2] > toy_rest_z(obj) + 0.010
    return bool(held)


def between_jaws(m, d, obj, site, tol=0.015):
    """Part of the toy is at the grip point: for a long toy (pen, stick) its axis passes within `tol` of it (a pen is
    held near one end, its middle far from the claw); for the others its centre is within 30 mm."""
    kind, size, _ = OBJECTS[obj]
    g = m.geom("toy").id
    c, p = d.geom_xpos[g], d.site_xpos[site]
    if kind != "capsule":
        return np.linalg.norm(c - p) < 0.030
    axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
    t = np.clip(np.dot(p - c, axis), -size[1], size[1])
    return np.linalg.norm(c + t * axis - p) < tol + size[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=",".join(OBJECTS))
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None, help="a GIF of the first trial of each object")
    a = ap.parse_args()
    if PARAMS.design != "v2":
        sys.exit("design v2 only")
    m0 = mujoco.MjModel.from_xml_path(MODEL_PATH)
    stand_z = float(m0.key("stand").qpos[2])
    clips = []
    worst = 1.0
    for obj in a.objects.split(","):
        m = build(obj)
        rng = np.random.default_rng(a.seed)
        frames = renderer = cam = None
        if a.video:
            frames, renderer = [], mujoco.Renderer(m, 300, 420)
            cam = mujoco.MjvCamera()
            cam.type, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 0.30, 200, -20
            cam.lookat[:] = [0.19, 0.04, 0.05]
        ok = 0
        for k in range(a.trials):
            ok += trial(m, obj, rng, stand_z, frames if k == 0 else None, renderer, cam)
        worst = min(worst, ok / a.trials)
        print(f"{obj:9s} {rate(ok, a.trials)}")
        if frames:
            clips += frames
    if a.video and clips:
        from PIL import Image
        ims = [Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=96) for f in clips[::2]]
        ims[0].save(a.video, save_all=True, append_images=ims[1:], duration=80, loop=0)
        print("video:", a.video)
    print(f"lowest success {100 * worst:.0f} % (target >= 90 %)")
    return 0 if worst >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())

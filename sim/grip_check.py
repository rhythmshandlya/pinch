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
DESK_GAP = float(os.environ.get("GRIP_DESK_GAP", "0.002"))   # m: the jaw bottom this far above the desk when it closes (negative: pressing on it)


def build(obj):
    spec = mujoco.MjSpec.from_file(str(ROOT / "sim" / "models" / PARAMS.model_file))
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


def setup(m, d, obj, rng, stand_z):
    q, a = joint_ids(m)
    mujoco.mj_resetData(m, d)
    base = m.body("body").id
    m.body_pos[base][2] = stand_z
    for i, j in enumerate(JOINT_NAMES):
        d.qpos[q[j]] = STAND[i]
        d.ctrl[a[j]] = STAND[i]
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


def trial(m, obj, rng, stand_z, frames=None, renderer=None, cam=None):
    d = mujoco.MjData(m)
    q, a = setup(m, d, obj, rng, stand_z)
    op = m.jnt_range[m.joint(f"{ARM}_claw").id][1] * 0.75
    pitch, grip = target_for(m, d, q, a, obj, op)
    d.ctrl[a[f"{ARM}_shoulder_pitch"]] = pitch
    d.qpos[q[f"{ARM}_claw"]] = op
    d.ctrl[a[f"{ARM}_claw"]] = op
    tq = m.jnt_qposadr[m.joint("toy_free").id]
    # midway between the open jaws (the grip point is between the CLOSED jaws, inside the fixed one when open)
    mid = (d.geom_xpos[m.geom(f"{ARM}_pollex").id] + d.geom_xpos[m.geom(f"{ARM}_dactyl_jaw").id]) / 2
    off = rng.uniform(-0.004, 0.004, 2)
    yaw = rng.uniform(-np.radians(20), np.radians(20))
    d.qpos[tq:tq + 3] = [mid[0] + off[0], mid[1] + off[1], toy_rest_z(obj) + 0.0005]
    qyaw = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    if OBJECTS[obj][0] == "capsule":                     # a capsule's axis is its z: lay it down along the claw (x)
        qlay = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
        qyaw = _qmul(qyaw, qlay)
    d.qpos[tq + 3:tq + 7] = qyaw
    mujoco.mj_forward(m, d)
    # a long toy lying along the claw would start inside the palm: slide it forward out of the claw's mouth until it
    # touches nothing (a pen is picked up near one end)
    toy, fwd = m.geom("toy").id, d.xmat[m.body(f"{ARM}_upper").id].reshape(3, 3)[:, 0].copy()
    fwd[2] = 0.0
    fwd /= np.linalg.norm(fwd)
    for _ in range(30 if OBJECTS[obj][0] == "capsule" else 0):
        if not any(toy in (d.contact[i].geom1, d.contact[i].geom2) and d.contact[i].dist < -0.0005
                   and 0 not in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2])
                   for i in range(d.ncon)):
            break
        d.qpos[tq:tq + 2] += fwd[:2] * 0.004
        mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    site = m.site(f"{ARM}_grip_site").id
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
    m0 = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "models" / PARAMS.model_file))
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

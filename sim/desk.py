"""Pinch's desk: the digital twin's world.

A 70 x 45 cm desk top (z = 0) a metre above the room floor, so walking off the edge is a real fall;
a low desk tray (Pinch's tidy target, printed with the robot); desk things Pinch can pick up or push
(a pen, an eraser, a bottle cap, a ball); two downward distance sensors on Pinch's front corners
(the VL53L0X cliff sensors); and the face screen on Pinch's forehead, a texture the app redraws
with the live eyes.

    m = build_desk_model()           # mujoco.MjModel
    set_face(m, rgb)                 # (H, W, 3) uint8 face image -> the screen texture
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinch.world import Desk  # noqa: E402

CRAB_XML = ROOT / "sim" / "models" / "crab.xml"
FACE_TEX = (256, 128)                 # 2x the OLED's 128 x 64

# name: (shape, size, mass kg, rgba, graspable)
OBJECTS = {
    "pen":    ("capsule", (0.0045, 0.065), 0.009, (0.15, 0.35, 0.85, 1), True),
    "eraser": ("box", (0.010, 0.006, 0.0045), 0.006, (0.95, 0.55, 0.65, 1), True),   # 20 x 12 x 9 mm
    "cap":    ("cylinder", (0.014, 0.006), 0.003, (0.85, 0.15, 0.12, 1), False),
    "ball":   ("sphere", (0.015,), 0.006, (0.98, 0.82, 0.10, 1), False),
}
START = {"pen": (-0.18, -0.02, 0.6), "eraser": (0.10, -0.12, 0.3), "cap": (-0.12, 0.13, 0.0), "ball": (0.05, 0.15, 0.0)}


def build_desk_model(desk: Desk = None, objects=OBJECTS, start=START, face=True) -> mujoco.MjModel:
    desk = desk or Desk()
    spec = mujoco.MjSpec.from_file(str(CRAB_XML))
    spec.option.timestep = 0.002
    for g in spec.geoms:                                   # the old infinite floor becomes the room floor
        if g.name == "floor":
            g.pos = [0, 0, -0.75]
    wb = spec.worldbody
    top = wb.add_body(name="desk")
    top.add_geom(name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[desk.half_x, desk.half_y, 0.015],
                 pos=[0, 0, -0.015], rgba=[0.62, 0.47, 0.33, 1], contype=0, conaffinity=1,
                 friction=[0.8, 0.02, 0.001])
    for sx in (-1, 1):
        for sy in (-1, 1):
            top.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.36],
                         pos=[sx * (desk.half_x - 0.03), sy * (desk.half_y - 0.03), -0.39],
                         rgba=[0.45, 0.33, 0.22, 1], contype=0, conaffinity=0)
    # the desk tray: a 2 mm base, walls on three sides, open toward Pinch's side (front)
    tx, ty = desk.tray_xy
    hx, hy = desk.tray_half
    tray = wb.add_body(name="tray", pos=[tx, ty, 0])
    kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[0.30, 0.75, 0.70, 1], contype=0, conaffinity=1)
    tray.add_geom(name="tray_base", size=[hx, hy, 0.0005], pos=[0, 0, 0.0005], **kw)      # 1 mm floor
    tray.add_geom(name="tray_back", size=[hx, 0.002, 0.006], pos=[0, hy, 0.006], **kw)
    for s in (-1, 1):
        tray.add_geom(size=[0.002, hy, 0.006], pos=[s * hx, 0, 0.006], **kw)
    # chamfered front lip: a 2 cm ramp from the desk up to the floor (printed as one piece)
    # an upright tab at the back of the tray carrying an ArUco marker: Pinch's camera finds the tray
    # (and so where it is on the desk) from it
    spec.add_texture(name="marker_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[1, 1, 1], width=MARKER_PX, height=MARKER_PX)
    mmat = spec.add_material(name="marker_mat")
    mmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "marker_tex"
    tray.add_geom(name="tray_tab", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0015, MARKER_M / 2 + 0.004, MARKER_M / 2 + 0.004],
                  pos=[0, hy + 0.003, MARKER_M / 2 + 0.006], euler=[0, 0, -np.pi / 2], rgba=[1, 1, 1, 1],
                  contype=0, conaffinity=1)
    tray.add_geom(name="tray_marker", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0002, MARKER_M / 2, MARKER_M / 2],
                  pos=[0, hy + 0.0013, MARKER_M / 2 + 0.006], euler=[0, 0, -np.pi / 2], material="marker_mat",
                  contype=0, conaffinity=0, group=2)
    tray.add_geom(name="tray_ramp", size=[hx, 0.010, 0.0005], pos=[0, -hy - 0.0098, 0.0001], euler=[0.05, 0, 0], **kw)
    # desk things
    for name, (shape, size, mass, rgba, _) in objects.items():
        x, y, yaw = start.get(name, (0, 0, 0))
        b = wb.add_body(name=name, pos=[x, y, size[0] + 0.0005], euler=[0, 0, yaw])
        b.add_freejoint(name=f"{name}_free")
        kw = dict(name=f"{name}_geom", mass=mass, rgba=list(rgba), contype=1, conaffinity=1,
                  friction=[0.6, 0.005, 0.0002], condim=4 if shape == "sphere" else 3)
        if shape == "capsule":
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[size[0], size[1], 0], euler=[0, np.pi / 2, 0], **kw)
        elif shape == "box":
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), **kw)
        elif shape == "cylinder":
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[size[0], size[1], 0], **kw)
        else:
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[size[0], 0, 0], **kw)
    # claw fingertips: TPU pads (printed like the foot tips) -> grippy, and they resist twisting
    for g in spec.geoms:
        parent = g.parent.name if g.parent is not None else ""
        if g.name.endswith("_pollex") or parent.endswith("_dactyl") and g.type == mujoco.mjtGeom.mjGEOM_BOX:
            g.friction = [1.2, 0.02, 0.001]
            g.condim = 6
    body = next(b for b in spec.bodies if b.name == "body")
    # cliff sensors: VL53L0X behind the side windows, looking forward-down
    for side, y in (("left", 0.045), ("right", -0.045)):
        s = body.add_site(name=f"cliff_{side}", pos=[0.045, y, -0.010], size=[0.002, 0, 0],
                          euler=[0, np.radians(-130), 0])
        sen = spec.add_sensor(name=f"cliff_{side}", type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
                              objtype=mujoco.mjtObj.mjOBJ_SITE, objname=s.name)
        sen.cutoff = 0.5
        sen.intprm[0] = 1                                 # report the distance
    if face:
        spec.add_texture(name="face_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                         rgb1=[0.02, 0.02, 0.03], width=FACE_TEX[0], height=FACE_TEX[1])
        mat = spec.add_material(name="face_mat", emission=0.9, specular=0.3)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "face_tex"
        # the 2.42" OLED: 55 x 27.5 mm active area, on the forehead, tilted back 25 deg
        body.add_geom(name="face_screen", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0012, 0.0275, 0.01375],
                      pos=[0.047, 0, 0.026], euler=[0, np.radians(-25), 0], material="face_mat",
                      contype=0, conaffinity=0, group=2, mass=0.0)
        body.add_geom(name="face_bezel", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0018, 0.0310, 0.0180],
                      pos=[0.0455, 0, 0.0255], euler=[0, np.radians(-25), 0], rgba=[0.08, 0.08, 0.1, 1],
                      contype=0, conaffinity=0, group=2, mass=0.0)
    for c in spec.cameras:                                  # the wide-angle face camera (decision 16)
        if c.name == "eye_cam":
            c.fovy = EYE_FOVY
    m = spec.compile()
    set_marker(m)
    return m


MARKER_ID, MARKER_M, MARKER_PX = 7, 0.040, 240     # ArUco 4x4 #7, 40 mm (30 mm black), on the tray tab
EYE_FOVY = 80.0


def set_marker(m):
    import cv2
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), MARKER_ID,
                                        MARKER_PX * 6 // 8)
    img = np.pad(img, MARKER_PX // 8, constant_values=255)                     # white quiet zone
    tid = m.texture("marker_tex").id
    adr = m.tex_adr[tid]
    m.tex_data[adr:adr + MARKER_PX * MARKER_PX * 3] = np.repeat(img[:, :, None], 3, axis=2).ravel()


def set_face(m, rgb):
    """Write a face image (H, W, 3 uint8, any size) into the screen texture."""
    tid = m.texture("face_tex").id
    w, h = m.tex_width[tid], m.tex_height[tid]
    img = np.asarray(rgb, np.uint8)
    if img.shape[:2] != (h, w):
        ys = (np.arange(h) * img.shape[0] / h).astype(int)
        xs = (np.arange(w) * img.shape[1] / w).astype(int)
        img = img[ys][:, xs]
    adr = m.tex_adr[tid]
    m.tex_data[adr:adr + w * h * 3] = img.reshape(-1)
    return tid


def object_pose(m, d, name):
    """(x, y, yaw) of a desk thing, and its height."""
    b = m.body(name).id
    x, y, z = d.xpos[b]
    R = d.xmat[b].reshape(3, 3)
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))             # body x axis (the pen's capsule lies along it)
    return (float(x), float(y), yaw), float(z)


if __name__ == "__main__":
    from PIL import Image
    m = build_desk_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id) if m.nkey else None
    from pinch.face import Face
    f = Face()
    for _ in range(20):
        f.update(0.05, "happy")
    set_face(m, f.rgb(scale=2))
    for _ in range(300):
        mujoco.mj_step(m, d)
    r = mujoco.Renderer(m, 480, 720)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.02, 0, 0.02]
    cam.distance, cam.azimuth, cam.elevation = 0.55, 200, -28
    r.update_scene(d, cam)
    Image.fromarray(r.render()).save(ROOT / "docs" / "images" / "pinch_desk.png")
    print("nq", m.nq, "| objects", list(OBJECTS), "| cliff sensors", [d.sensor(f"cliff_{s}").data[0] for s in ("left", "right")])

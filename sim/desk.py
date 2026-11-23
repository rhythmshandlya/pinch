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

from crab.params import (
    CLIFF_BEAMS,  # noqa: E402
    PARAMS,  # noqa: E402
)
from pinch.world import Desk  # noqa: E402

CRAB_XML = ROOT / "sim" / "models" / PARAMS.model_file
FACE_TEX = (256, 128)                 # 2x the OLED's 128 x 64

# name: (shape, size, mass kg, rgba, graspable)
OBJECTS = {
    "pen":    ("capsule", (0.0045, 0.065), 0.009, (0.15, 0.35, 0.85, 1), True),
    "eraser": ("box", (0.010, 0.006, 0.0045), 0.006, (0.95, 0.55, 0.65, 1), True),   # 20 x 12 x 9 mm
    "cap":    ("cylinder", (0.014, 0.006), 0.003, (0.85, 0.15, 0.12, 1), False),
    "ball":   ("sphere", (0.015,), 0.006, (0.98, 0.82, 0.10, 1), False),
}
# things Pinch has never seen (no names, not in OBJECTS): PinchSim(props=[...]) adds them, parked off the desk until
# shown (sim.show_prop / move_prop). name: (shape, size, rgba)
PROPS = {
    "box":  ("box", (0.02, 0.02, 0.02), (0.55, 0.45, 0.30, 1)),       # a 4 cm cardboard cube
    "tube": ("cylinder", (0.015, 0.035), (0.20, 0.65, 0.35, 1)),     # a 3 cm wide, 7 cm tall tube standing up
    "toy":  ("sphere", (0.018,), (0.70, 0.30, 0.80, 1)),             # a 36 mm toy (moves along paths)
}
PROP_PARK = (3.0, 3.0, -0.5)          # where a hidden prop waits (far off the desk, above the room floor)
START = {"pen": (-0.22, -0.16, 0.6), "eraser": (0.25, -0.18, 0.3), "cap": (-0.18, 0.20, 0.0), "ball": (0.12, 0.21, 0.0)}   # clear of the v2.1 legs


def build_desk_model(desk: Desk = None, objects=OBJECTS, start=START, face=True, extras=()) -> mujoco.MjModel:
    """extras: fixed things on the desk for tests, e.g. dict(name="cable", type="cylinder", size=[r, half_len],
    pos=[x, y, r], euler=[...]) - geoms of the desk top."""
    desk = desk or Desk()
    spec = mujoco.MjSpec.from_file(str(CRAB_XML))
    spec.option.timestep = 0.002
    # no friction creep: with MuJoCo's default (pyramidal cones, no noslip pass) a pen squeezed in the claw slid
    # out under its own weight in 3.6 s, standing still - real TPU pads don't do that. 3 noslip iterations hold
    # it walking and turning (an elliptic cone at impratio 10 did too, but picked worse). Decision 54.
    spec.option.noslip_iterations = 3
    for g in spec.geoms:                                   # the old infinite floor becomes the room floor
        if g.name == "floor":
            g.pos = [0, 0, -0.75]
    wb = spec.worldbody
    top = wb.add_body(name="desk")
    top.add_geom(name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[desk.half_x, desk.half_y, 0.015],
                 pos=[0, 0, -0.015], rgba=[0.62, 0.47, 0.33, 1], contype=0, conaffinity=1,
                 friction=[0.8, 0.02, 0.001])
    for e in extras:
        if e.get("mocap"):                                 # a thing moved by script (PinchSim props): a mocap body
            b = wb.add_body(name=e["name"], mocap=True, pos=e["pos"])
            b.add_geom(name=f"{e['name']}_geom", type=getattr(mujoco.mjtGeom, "mjGEOM_" + e["type"].upper()),
                       size=list(e["size"]) + [0] * (3 - len(e["size"])), rgba=list(e.get("rgba", (0.5, 0.5, 0.55, 1))),
                       friction=[0.8, 0.02, 0.001])
            continue
        top.add_geom(name=e["name"], type=getattr(mujoco.mjtGeom, "mjGEOM_" + e["type"].upper()),
                     size=list(e["size"]) + [0] * (3 - len(e["size"])), pos=e["pos"], euler=e.get("euler", [0, 0, 0]),
                     rgba=[0.15, 0.15, 0.15, 1], friction=[0.8, 0.02, 0.001])
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
        # condim 6: rolling resistance. A pen is hexagonal - it barely rolls (a smooth capsule nudged once
        # rolled steadily off the desk); the ball is a soft toy ball (the desk's 1 mm)
        kw = dict(name=f"{name}_geom", mass=mass, rgba=list(rgba), contype=1, conaffinity=1,
                  friction=[0.6, 0.005, 0.003 if shape == "capsule" else 0.0002],
                  condim=6 if shape in ("sphere", "capsule") else 3)
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
        if g.name.endswith("_pollex") or g.name.endswith("_pollex_blade") or parent.endswith("_dactyl") or                 g.name.endswith("_geom") and g.name[:-5] in objects:
            # small rigid things: MuJoCo's default contact is a 20 ms soft spring, so a 9 g pen landing at
            # 0.7 m/s sinks ~10 mm - past its own radius, the contact flips and it drops through the desk.
            # 4 ms (2 timesteps) keeps it under 3 mm. priority=1: these settings win against the desk and
            # the robot, while the feet-on-desk contacts the gaits were tuned on stay as they were.
            g.solref, g.solimp = [0.004, 1.0], [0.95, 0.99, 0.001, 0.5, 2.0]
            g.priority = 1
        elif any(k in parent for k in ("_coxa", "_femur", "_tibia")) or parent == "body":
            g.priority = 2       # legs and shell keep MuJoCo's default (soft) contact, even against small things:
                                 # a stiff foot-pen contact squeezes a pen against the desk and shoots it off
    body = next(b for b in spec.bodies if b.name == "body")
    # the five VL53L0X cliff beams, exactly where cad/parts.py CLIFF_BEAMS puts their windows (decision 30).
    # The front pair keep their old names (cliff_left / cliff_right); the beam is each site's +z axis.
    names = {"front_left": "left", "front_right": "right", "rear": "rear",
             "side_left": "side_left", "side_right": "side_right"}
    for key, wx, wy, wz, d in CLIFF_BEAMS:
        side = names[key]
        z = np.asarray(d, float) / np.linalg.norm(d)
        axis = np.cross([0.0, 0.0, 1.0], z)
        ang = np.arccos(np.clip(z[2], -1, 1))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        quat = [np.cos(ang / 2), *(np.sin(ang / 2) * axis)]
        s = body.add_site(name=f"cliff_{side}", pos=[wx / 1000, wy / 1000, wz / 1000], size=[0.002, 0, 0],
                          quat=quat)
        sen = spec.add_sensor(name=f"cliff_{side}", type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
                              objtype=mujoco.mjtObj.mjOBJ_SITE, objname=s.name)
        sen.cutoff = 1.2                                 # the VL53L0X reaches ~1.2 m (also: how high it is lifted)
        sen.intprm[0] = 1                                 # report the distance
    if face and not any(t.name == "face_tex" for t in spec.textures):    # v2's model already has its own screen
        spec.add_texture(name="face_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                         rgb1=[0.02, 0.02, 0.03], width=FACE_TEX[0], height=FACE_TEX[1])
        mat = spec.add_material(name="face_mat", emission=0.9, specular=0.3)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "face_tex"
        # the 2.42" OLED: 55 x 27.5 mm active area, in the forehead visor (cad/parts.py FACE_CENTRE, FACE_TILT),
        # glass 1.8 mm in front of the module's centre, behind the visor's window (the visor is in the shell mesh)
        n = np.array([np.cos(np.radians(25)), 0, np.sin(np.radians(25))])
        centre = np.array([0.036, 0.0, 0.036]) + (0.0018 - 0.0012) * n
        body.add_geom(name="face_screen", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0012, 0.0275, 0.01375],
                      pos=list(centre), euler=[0, np.radians(-25), 0], material="face_mat",
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

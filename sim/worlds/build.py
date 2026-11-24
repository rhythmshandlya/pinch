"""A World (sim/worlds/generate.py) into MuJoCo: rooms, walls with doorways, furniture, the surface's material and
friction, a rug or desk mat, everyday things (Google Scanned Objects, or boxes of the same size), lights, people and
the fisheye camera's cube faces. Called by sim/desk.py build_desk_model(world=...) on the twin's MjSpec, before it
compiles; returns a function to run on the compiled model (it fills the face textures).

Frame: Pinch's surface top is z = 0 and centred on the origin (the twin's desk frame); the room floor is at
z = -height. Furniture and walls collide with Pinch (they matter on the floor), people collide with nothing.
"""

from pathlib import Path

import mujoco
import numpy as np

from sim.worlds import assets, textures
from sim.worlds.catalogue import THINGS
from sim.worlds.generate import WALL_H, WALL_T
from sim.worlds.people import add_person

G = mujoco.mjtGeom
FACES_DIR = Path(__file__).resolve().parent / "assets" / "faces"
STATIC = dict(contype=0, conaffinity=1)
VISUAL = dict(contype=0, conaffinity=0)


def face_images():
    """The synthetic face photos (sim/worlds/assets/faces, SFHQ, CC0/MIT) as RGB arrays."""
    import cv2
    return [cv2.imread(str(p))[:, :, ::-1] for p in sorted(FACES_DIR.glob("face_*.jpg"))]


class _Mats:
    def __init__(self, spec, seed):
        self.spec, self.seed, self.n = spec, seed, 0

    def __call__(self, pattern, rgb, repeat=(1, 1), uniform=True, reflect=0.0):
        self.n += 1
        name = f"wtex{self.n}"
        img = textures.pattern(pattern, rgb, self.seed + self.n)
        t = self.spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=img.shape[1],
                                  height=img.shape[0], nchannel=3)
        t.data = bytes(img)
        m = self.spec.add_material(name=f"wmat{self.n}", texrepeat=list(repeat), texuniform=uniform, reflectance=reflect)
        m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
        return m.name


def _box(body, pos, half, rgba=None, material=None, euler=(0, 0, 0), friction=None, kw=STATIC, name=None):
    a = dict(type=G.mjGEOM_BOX, size=list(half), pos=list(pos), euler=list(euler), **kw)
    if name:
        a["name"] = name
    if material:
        a["material"] = material
    else:
        rgba = rgba if rgba is not None else (0.5, 0.5, 0.5)
        a["rgba"] = list(rgba) + [1] if len(rgba) == 3 else list(rgba)
    if friction is not None:
        a["friction"] = [friction, 0.02, 0.001]
    return body.add_geom(**a)


def _rot(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return c * x - s * y, s * x + c * y


def _seed(*xs):
    return int(sum(abs(int(round(x * 1000))) * (k + 7) for k, x in enumerate(xs))) % (2**31)


def furniture(spec, f, H, mats, models=True):
    """A piece of furniture (its own frame: x along its width, its back toward +y): a Poly Haven model scaled to
    its size when fetched (sim/worlds/furniture.py), with simple boxes for contacts; boxes alone otherwise."""
    b = spec.worldbody.add_body(pos=[f["x"], f["y"], -H], euler=[0, 0, f["yaw"]])
    w, d, h, k = f["w"], f["d"], f["h"], f["kind"]
    if models:
        from sim.worlds import furniture as F
        entry = F.choose(k, w, d, h, _seed(f["x"], f["y"]))
        if entry is not None:
            F.add(spec, b, entry, w, d, h, f"furn{len(spec.bodies)}")
            n0 = len(b.geoms)
            _furniture_boxes(b, f, w, d, h, k, None, None)
            for g in list(b.geoms)[n0:]:                     # the boxes stay for contacts, out of sight
                g.group, g.rgba = 3, [0.5, 0.5, 0.5, 0.0]
            if k == "tv_stand":
                tv = F.choose("tv", min(w * 0.7, 1.2), 0.1, 0.6, 1)
                if tv is not None:
                    tb = spec.worldbody.add_body(pos=[f["x"], f["y"], -H + h], euler=[0, 0, f["yaw"]])
                    F.add(spec, tb, tv, min(w * 0.7, 1.2), 0.12, min(w * 0.7, 1.2) * 0.6, f"tv{len(spec.bodies)}")
            return b
    _furniture_boxes(b, f, w, d, h, k, mats, None)
    return b


def _furniture_boxes(b, f, w, d, h, k, mats, _):
    if mats is None:
        wood = fabric = None
    else:
        wood = mats("wood", f["rgb"], (2, 2))
        fabric = mats("noise", tuple(np.clip(np.array(f["rgb"]) * [0.6, 0.7, 0.9], 0, 1)), (3, 3))
    if k in ("desk", "table", "coffee"):
        _box(b, (0, 0, h - 0.015), (w / 2, d / 2, 0.015), material=wood)
        for sx in (-1, 1):
            for sy in (-1, 1):
                _box(b, (sx * (w / 2 - 0.04), sy * (d / 2 - 0.04), (h - 0.03) / 2), (0.02, 0.02, (h - 0.03) / 2), material=wood)
    elif k == "chair":
        _box(b, (0, 0, 0.45), (w / 2, d / 2, 0.02), material=wood)
        _box(b, (0, d / 2 - 0.02, (0.47 + h) / 2), (w / 2, 0.02, (h - 0.47) / 2), material=wood)
        for sx in (-1, 1):
            for sy in (-1, 1):
                _box(b, (sx * (w / 2 - 0.03), sy * (d / 2 - 0.03), 0.215), (0.015, 0.015, 0.215), material=wood)
    elif k == "sofa":
        _box(b, (0, -0.05, 0.22), (w / 2, d / 2 - 0.05, 0.22), material=fabric)
        _box(b, (0, d / 2 - 0.1, h / 2), (w / 2, 0.1, h / 2), material=fabric)
        for sx in (-1, 1):
            _box(b, (sx * (w / 2 - 0.1), -0.05, 0.32), (0.1, d / 2 - 0.05, 0.32), material=fabric)
    elif k == "bed":
        _box(b, (0, 0, 0.15), (w / 2, d / 2, 0.15), material=wood)
        _box(b, (0, -0.03, (h + 0.3) / 2), (w / 2 - 0.02, d / 2 - 0.05, (h - 0.3) / 2), rgba=(0.92, 0.92, 0.95), material=None)
        _box(b, (0, d / 2 - 0.03, 0.5), (w / 2, 0.03, 0.5), material=wood)
    elif k == "shelf":
        _box(b, (0, d / 2 - 0.01, h / 2), (w / 2, 0.01, h / 2), material=wood)
        for sx in (-1, 1):
            _box(b, (sx * (w / 2 - 0.01), 0, h / 2), (0.01, d / 2, h / 2), material=wood)
        n = max(2, int(h / 0.35))
        for i in range(n + 1):
            _box(b, (0, 0, i * (h - 0.02) / n + 0.01), (w / 2, d / 2, 0.01), material=wood)
    elif k == "tv_stand":
        _box(b, (0, 0, h / 2), (w / 2, d / 2, h / 2), material=wood)
        _box(b, (0, 0.05, h + 0.32), (min(w / 2 - 0.05, 0.6), 0.02, 0.32), rgba=(0.05, 0.05, 0.06), material=None)
    else:                                                # wardrobe, counter, nightstand: a block
        _box(b, (0, 0, h / 2), (w / 2, d / 2, h / 2), material=wood)


def _walls(spec, w, H, mat):
    for wl in w.walls:
        p0, p1 = np.array([wl["x0"], wl["y0"]]), np.array([wl["x1"], wl["y1"]])
        L = float(np.linalg.norm(p1 - p0))
        u = (p1 - p0) / L
        yaw = float(np.arctan2(u[1], u[0]))
        cuts = sorted([(c - dw / 2, c + dw / 2, 0.0, 2.05) for c, dw in wl["doors"]] +
                      [(c - dw / 2, c + dw / 2, z0, min(z1, WALL_H - 0.05)) for c, dw, z0, z1 in wl.get("windows", [])])
        a, pieces = 0.0, []
        for lo, hi, z0, z1 in cuts:                       # doorways (open to the floor) and windows
            lo, hi = max(lo, a), min(hi, L)
            if hi <= lo:
                continue
            pieces.append((a, lo, 0.0, WALL_H))
            if z0 > 0.01:
                pieces.append((lo, hi, 0.0, z0))          # under the window
            pieces.append((lo, hi, z1, WALL_H))           # above the doorway or window
            a = hi
        pieces.append((a, L, 0.0, WALL_H))
        t = wl.get("t", WALL_T)
        for s0, s1, z0, z1 in pieces:
            if s1 - s0 < 1e-3:
                continue
            c = p0 + u * (s0 + s1) / 2
            _box(spec.worldbody, (c[0], c[1], -H + (z0 + z1) / 2), ((s1 - s0) / 2 + t / 2, t / 2, (z1 - z0) / 2),
                 material=mat, euler=(0, 0, yaw))


def _thing(spec, k, t, H_surface_top, visual=True):
    gso, (tw, td, th), mass, _, shape, rgba = THINGS[t["name"]]
    pos = [t["x"], t["y"], H_surface_top + 0.002]
    quat = [np.cos(t["yaw"] / 2), 0, 0, np.sin(t["yaw"] / 2)]
    if assets.available(gso):
        child = mujoco.MjSpec.from_file(str(assets.folder(gso) / "model.xml"))
        if not visual:                                     # headless runs: drop the big textured mesh
            for g in list(child.geoms):
                if g.group == 2:
                    child.delete(g)
        for g in child.geoms:
            if g.group == 3:
                g.friction = [0.6, 0.005, 0.0001]
        frame = spec.worldbody.add_frame(pos=pos, quat=quat)
        body = frame.attach_body(child.body("model"), f"thing{k}_", "")
    else:
        body = spec.worldbody.add_body(pos=pos, quat=quat)
        if shape == "cylinder":
            body.add_geom(type=G.mjGEOM_CYLINDER, size=[min(tw, td) / 2, th / 2, 0], pos=[0, 0, th / 2], rgba=list(rgba))
        else:
            body.add_geom(type=G.mjGEOM_BOX, size=[tw / 2, td / 2, th / 2], pos=[0, 0, th / 2], rgba=list(rgba))
    body.name = f"thing{k}"
    body.add_freejoint(name=f"thing{k}_free")
    body.explicitinertial = True
    body.mass = mass
    body.ipos = [0, 0, th / 2]
    body.inertia = [mass / 12 * (td ** 2 + th ** 2), mass / 12 * (tw ** 2 + th ** 2), mass / 12 * (tw ** 2 + td ** 2)]
    return body


def _cube_cameras(spec):
    """Five 90-degree cameras (front, left, right, up, down) at the eye camera, for sim/worlds/fisheye.py."""
    cam = next((c for c in spec.cameras if c.name == "eye_cam"), None)
    if cam is None:
        return
    if cam.alt.type == mujoco.mjtOrientation.mjORIENTATION_XYAXES:
        x, y = np.array(cam.alt.xyaxes[:3], float), np.array(cam.alt.xyaxes[3:], float)
        x /= np.linalg.norm(x)
        y = y - x * (x @ y)
        y /= np.linalg.norm(y)
        q0 = np.zeros(4)
        mujoco.mju_mat2Quat(q0, np.column_stack([x, y, np.cross(x, y)]).ravel())
    else:
        q0 = np.array(cam.quat, float)
    for name, axis, ang in (("front", (0, 1, 0), 0.0), ("left", (0, 1, 0), 90.0), ("right", (0, 1, 0), -90.0),
                            ("up", (1, 0, 0), 90.0), ("down", (1, 0, 0), -90.0)):
        qr, q = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(qr, np.array(axis, float), np.radians(ang))
        mujoco.mju_mulQuat(q, q0, qr)
        cam.parent.add_camera(name=f"eye_{name}", pos=list(cam.pos), quat=list(q), fovy=90.0)


NEAR, FAR = 0.002, 40.0             # m: the camera sees from 2 mm (its own arms are 4-5 cm away) to across a house


def _hide_desk_top(spec):
    """The desk top box stays for contacts and the cliff beams, but is not drawn (a model is drawn instead): it goes
    to geom group 3, which renderers leave out. Not by making it transparent - MuJoCo's rangefinders skip
    geoms with alpha 0, so the cliff beams saw the floor through it and the edge guard held Pinch still."""
    for g in spec.geoms:
        if g.name == "desk_top":
            g.group = 3


def decorate(spec, w, things=True, people=True, visual=True, fisheye=True, models=None):
    """Add world w to the twin's spec (sim/desk.py has already made the desk top, the tray and Pinch's things)."""
    H = w.surface["height"]
    if models is None:                                     # furniture meshes only where they are seen
        models = visual
    mats = _Mats(spec, w.tex_seed)
    for li in [li for li in spec.lights if li.parent.name == "world"]:      # crab.xml's demo light: ours instead
        spec.delete(li)
    for g in spec.geoms:
        if g.name == "floor":
            r0 = w.rooms[0]["floor"]
            g.pos = [0, 0, -H - 0.02]
            g.material = mats(r0["pattern"], r0["rgb"], (8, 8))
            g.friction = [r0["friction"], 0.02, 0.001]
        elif g.name == "desk_top":
            g.material = mats(w.surface["pattern"], w.surface["rgb"], (1, 1) if w.kind != "floor" else (6, 6))
            g.rgba = [1, 1, 1, 1]
            g.friction = [w.surface["friction"], 0.02, 0.001]
    for r in w.rooms:                                      # each room's floor (Pinch's own floor is the desk top)
        fl = r["floor"]
        mat = mats(fl["pattern"], fl["rgb"], (4, 4))
        if "cells" in r:                                   # a ready-made room: its outline in rectangles, a touch
            low = 0.002 if w.kind == "floor" else 0.0      # below Pinch's patch of floor when it is on the floor
            for x0, y0, x1, y1 in r["cells"]:
                _box(spec.worldbody, ((x0 + x1) / 2, (y0 + y1) / 2, -H - 0.01 - low), ((x1 - x0) / 2, (y1 - y0) / 2, 0.01),
                     material=mat, friction=fl["friction"])
            continue
        if w.kind == "floor" and r["x0"] < 0 < r["x1"] and r["y0"] < 0 < r["y1"]:
            continue
        _box(spec.worldbody, ((r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2, -H - 0.01),
             ((r["x1"] - r["x0"]) / 2, (r["y1"] - r["y0"]) / 2, 0.01), material=mat, friction=fl["friction"])
    _walls(spec, w, H, mats("noise", w.wall_rgb, (2, 2)))
    xs = [v for r in w.rooms for v in (r["x0"], r["x1"])]
    ys = [v for r in w.rooms for v in (r["y0"], r["y1"])]
    _box(spec.worldbody, ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, -H + WALL_H + 0.02),     # the ceiling
         ((max(xs) - min(xs)) / 2 + WALL_T, (max(ys) - min(ys)) / 2 + WALL_T, 0.02),
         rgba=tuple(min(1.0, c * 1.05) for c in w.wall_rgb), kw=VISUAL, name="ceiling")
    own = None
    if w.kind != "floor" and "model" in w.surface:         # the house's own desk/table: drawn, its top is the desk top
        from sim.worlds import procthor
        mo = w.surface["model"]
        if visual:
            procthor.add(spec, dict(asset=mo["asset"], x=0.0, y=0.0, z=mo["z"], off=mo["off"], yaw=mo["yaw"]), H,
                         collide=False)
            _hide_desk_top(spec)
        own = mo
    elif w.kind != "floor" and models:                     # Pinch's desk/table as a real model with a flat top
        from sim.worlds import furniture as F
        hx, hy = w.surface["half_x"], w.surface["half_y"]
        own = F.choose(w.surface["furniture"], 2 * hx, 2 * hy, H, _seed(hx, hy, H), surface=True)
        if own is not None:
            F.add(spec, spec.worldbody.add_body(pos=[0, 0, -H]), own, 2 * hx, 2 * hy, H, "own")
            _hide_desk_top(spec)                       # contacts as before, the model's top is what is seen
    if w.kind != "floor" and own is None:                  # Pinch's desk/table: legs under the desk top
        hx, hy = w.surface["half_x"], w.surface["half_y"]
        leg = mats("wood", tuple(np.array(w.surface["rgb"]) * 0.8), (1, 1))
        for sx in (-1, 1):
            for sy in (-1, 1):
                _box(spec.worldbody, (sx * (hx - 0.04), sy * (hy - 0.04), (-H - 0.03) / 2), (0.02, 0.02, (H - 0.03) / 2),
                     material=leg, kw=VISUAL)
        if w.surface["furniture"] == "desk":
            _box(spec.worldbody, (0, hy - 0.02, (-H - 0.03) / 2 + 0.15), (hx - 0.06, 0.01, (H - 0.33) / 2), material=leg,
                 kw=VISUAL)
    for f in w.furniture:
        if "asset" in f:                                   # ready-made (ProcTHOR): contacts only near Pinch
            from sim.worlds import procthor
            procthor.add(spec, f, H, collide=np.hypot(f["x"], f["y"]) < 3.0 + max(f["w"], f["d"]) / 2, visual=visual)
        else:
            furniture(spec, f, H, mats, models)
    if w.mat is not None:
        mt = w.mat
        _box(spec.worldbody, (mt["xy"][0], mt["xy"][1], mt["thick"] / 2), (mt["half"][0], mt["half"][1], mt["thick"] / 2),
             material=mats(mt["pattern"], mt["rgb"], (2, 2)), euler=(0, 0, mt["yaw"]), friction=mt["friction"],
             name="world_mat")
    if things:
        for k, t in enumerate(w.things):
            _thing(spec, k, t, 0.0, visual=visual)
    # lights: the world's lamps and daylight; the headlight only as a little fill
    for li in w.lights:
        kw = dict(pos=list(li["pos"]), dir=list(li["dir"]), diffuse=[min(1.0, 0.9 * c * li["level"]) for c in li["rgb"]],
                  specular=[0.1] * 3, castshadow=bool(li["shadow"]) and not li.get("directional"))
        light = spec.worldbody.add_light(**kw)
        if li.get("directional"):
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        elif li["spot"]:
            light.type = mujoco.mjtLightType.mjLIGHT_SPOT
            light.cutoff = 60
        else:
            light.type = mujoco.mjtLightType.mjLIGHT_POINT if hasattr(mujoco.mjtLightType, "mjLIGHT_POINT") \
                else mujoco.mjtLightType.mjLIGHT_SPOT
            light.attenuation = [1.0, 0.0, 0.0]
    lvl = w.extra.get("level", 1.0)
    spec.visual.headlight.ambient = [min(0.6, 0.2 + w.ambient)] * 3             # room light bouncing off the walls
    spec.visual.headlight.diffuse = [min(0.5, 0.25 * lvl)] * 3
    if people:
        from sim.worlds import humans
        from sim.worlds.people import seat_of
        imgs = face_images()
        if w.kind != "floor" and any(p["role"] == "sitter" for p in w.people):    # something to sit on
            (sx, sy), seat, what = seat_of(w)
            dims = (0.46, 0.5, 0.9) if what == "chair" else (1.8, 0.85, 0.8)
            furniture(spec, dict(kind=what, x=sx, y=sy - (0.08 if what == "chair" else 0.15), yaw=np.pi, w=dims[0],
                                 d=dims[1], h=dims[2], rgb=(0.4, 0.3, 0.25)), H, mats, models)
        for i, p in enumerate(w.people):
            if humans.ready():                           # real bodies (Rocketbox / MakeHuman) when fetched
                humans.add(spec, i, p, w.people[:i])
                continue
            img = imgs[p["face"] % len(imgs)] if imgs else np.full((64, 64, 3), [200, 160, 130], np.uint8)
            tname = f"face_tex{i}"
            t = spec.add_texture(name=tname, type=mujoco.mjtTexture.mjTEXTURE_2D, width=img.shape[1],
                                 height=img.shape[0], nchannel=3)
            t.data = bytes(np.ascontiguousarray(img))
            mat = spec.add_material(name=f"face_mat{i}", emission=0.35, specular=0.1)   # (lit by the whole room)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tname
            add_person(spec, i, p, mat.name)
    if fisheye:
        _cube_cameras(spec)

    def post(m):
        # clipping in metres: MuJoCo's znear and zfar are fractions of the model's extent, which in a house (or with
        # people parked far off) is over 100 m - a fixed fraction there clipped everything within 6.6 cm of the lens,
        # Pinch's own arms included (found by the arm-occlusion session), and before that the desk under the camera
        m.vis.map.znear = NEAR / m.stat.extent
        m.vis.map.zfar = FAR / m.stat.extent
    return post

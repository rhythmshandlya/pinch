"""Ready-made homes: ProcTHOR-10K houses furnished with AI2-THOR's own models, as the worlds' rooms and furniture.

    python -m sim.worlds.procthor fetch        # 12,000 houses (61 MB) + the ~1,500 models they use (~60 MB of glb),
                                               # converted once -> models/procthor/ (not in git)

ProcTHOR-10K (allenai/procthor-10k, Apache-2.0; Deitke et al. 2022, "ProcTHOR: Large-Scale Embodied AI Using
Procedural Generation", arXiv 2206.06994, NeurIPS outstanding paper) is 10,000 training, 1,000 validation and 1,000
test houses made by ProcTHOR's generator: 1-10 rooms (bedrooms, bathrooms, living rooms, kitchens) with doors and
windows, and furniture and everyday things placed by its rules. The models are AI2-THOR's (Kolve et al. 2017,
arXiv 1712.05474; github.com/allenai/ai2thor, Apache-2.0), read as glb from the Habitat conversion that ManiSkill
mirrors on Hugging Face (haosulab/AI2THOR, declared CC-BY-4.0), one range request per model from its 15.6 GB zip.

Each model is converted once: its vertices into MuJoCo's frame, its parts by colour (the models are coloured, not
textured), its bounds, how flat its top is, and boxes for contacts (the mesh voxelised at 5 cm and merged into
boxes: a convex hull would make a table a block that Pinch could not walk under).

home(rng, split, kind) gives generate.py the house part of a world - rooms, walls with doors and windows, furniture,
Pinch's surface - in the same form as its own generator, so everything else (Pinch's things, lights, people,
scenarios, benches) is unchanged. Our training worlds use ProcTHOR's training houses and our held-out test worlds its
test houses, so the held-out worlds stay unseen places. Pinch's desk or table is one of the house's own (a top
0.35-1.1 m high, flat, at least 40 x 30 cm), turned so the side someone sits at faces free floor; the floor is the
largest clear rectangle of a room. Without the download, worlds use generate.py's own homes.
"""

import gzip
import json
import math
import sys
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from sim.worlds.catalogue import SURFACE_MIN

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "models" / "procthor"
HOUSES = ("allenai/procthor-10k", "439193522244720b86d8c81cde2e51e3a4d150cf")
MODELS = ("haosulab/AI2THOR", "1a173d5de042aaad8f1af09d4d2bc2ce4004b28a")
ZIP = f"https://huggingface.co/datasets/{MODELS[0]}/resolve/{MODELS[1]}/ai2thor.zip"
GLB = "ai2thor/ai2thor-hab/assets/objects/{}.glb"
B = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])          # glb (Habitat, y up) -> MuJoCo (z up)
SURFACE = {"desk": ("Desk",), "table": ("DiningTable", "CoffeeTable", "SideTable")}
ROOMS = {"Bedroom": "bedroom", "LivingRoom": "living", "Kitchen": "kitchen", "Bathroom": "bathroom"}
SEATS = ("Chair", "Stool", "ArmChair", "Ottoman")                  # moved out of where Pinch's person sits
PITCH = 0.05                                                       # contact boxes' voxel size


def available():
    return (DIR / "READY").exists()                                # (written when a fetch has finished)


# ----------------------------------------------------------------------------------------------- fetch + convert
def _compact(h):
    """A ProcTHOR house -> the parts we use, in MuJoCo's frame (x, y on the floor, z up; Unity's (x, y, z) ->
    (x, z, y), yaw = -rotation.y)."""
    def obj(o):
        p, r = o["position"], o["rotation"]
        return dict(type=o["id"].split("|")[0], asset=o["assetId"], x=round(p["x"], 4), y=round(p["z"], 4),
                    z=round(p["y"], 4), yaw=round(-math.radians(r["y"]), 5),
                    children=[obj(c) for c in o.get("children") or [] if c.get("assetId")])
    holes = {}
    for d in h["doors"] + h["windows"]:
        a, b = d["holePolygon"]
        holes.setdefault(d["wall0"], []).append((a["x"], b["x"], a["y"], b["y"]))
    walls = []
    for w in h["walls"]:
        p = w["polygon"]
        walls.append(dict(id=w["id"], a=(p[0]["x"], p[0]["z"]), b=(p[1]["x"], p[1]["z"]), holes=holes.get(w["id"], [])))
    rooms = [dict(type=r["roomType"], poly=[(q["x"], q["z"]) for q in r["floorPolygon"]]) for r in h["rooms"]]
    return dict(rooms=rooms, walls=walls, objects=[obj(o) for o in h["objects"] if o.get("assetId")])


def _assets(houses):
    used, top = set(), set()

    def walk(o, level):
        used.add(o["asset"])
        if level == 0:
            top.add(o["asset"])
        for c in o["children"]:
            walk(c, level + 1)
    for h in houses:
        for o in h["objects"]:
            walk(o, 0)
    return used, top


def convert(glb, boxes=True):
    """One glb -> parts (vertices, faces, colour) in MuJoCo's frame, bounds, top flatness, contact boxes."""
    import trimesh
    s = trimesh.load(str(glb), force="scene")
    parts, V_all, F_all, n = {}, [], [], 0
    for node in s.graph.nodes_geometry:
        T, gn = s.graph[node]
        g = s.geometry[gn].copy()
        g.apply_transform(T)
        try:
            c = tuple(int(x) for x in g.visual.material.main_color)
        except Exception:
            c = (150, 150, 150, 255)
        V = np.asarray(g.vertices) @ B.T
        P = parts.setdefault(c, [[], [], 0])
        P[0].append(V)
        P[1].append(np.asarray(g.faces) + P[2])
        P[2] += len(V)
        V_all.append(V)
        F_all.append(np.asarray(g.faces) + n)
        n += len(V)
    V, F = np.vstack(V_all), np.vstack(F_all)
    lo, hi = V.min(0), V.max(0)
    m = trimesh.Trimesh(V, F, process=False)
    up = m.face_normals[:, 2] > 0.95                              # the top: upward faces within 1 cm of the highest
    zc = V[F].mean(1)[:, 2]
    area = m.area_faces[up & (zc > hi[2] - 0.01)].sum()
    flat = float(area / max((hi[0] - lo[0]) * (hi[1] - lo[1]), 1e-9))
    out = dict(lo=lo.round(4).tolist(), hi=hi.round(4).tolist(), flat=round(min(flat, 1.0), 3))
    arrays = {}
    for k, (c, (vs, fs, _)) in enumerate(parts.items()):
        arrays[f"V{k}"], arrays[f"F{k}"] = np.vstack(vs).astype(np.float32), np.vstack(fs).astype(np.int32)
        arrays[f"C{k}"] = np.array(c, np.float32) / 255.0
    if boxes:
        vg = m.voxelized(PITCH).fill()
        M = vg.matrix.astype(bool)
        org = np.asarray(vg.transform)[:3, 3]
        bx = _merge(M)
        lo_b = org + bx[:, :3] * PITCH - PITCH / 2
        hi_b = org + bx[:, 3:] * PITCH - PITCH / 2
        arrays["box_c"], arrays["box_h"] = ((lo_b + hi_b) / 2).astype(np.float32), ((hi_b - lo_b) / 2).astype(np.float32)
    return out, arrays


def _merge(M):
    """Greedy merge of filled voxels into boxes: rows of (i0, j0, k0, i1, j1, k1), end exclusive."""
    seen = np.zeros_like(M)
    nx, ny, nz = M.shape
    out = []
    for i in range(nx):
        for j in range(ny):
            k = 0
            while k < nz:
                if M[i, j, k] and not seen[i, j, k]:
                    k2 = k
                    while k2 + 1 < nz and M[i, j, k2 + 1] and not seen[i, j, k2 + 1]:
                        k2 += 1
                    j2 = j
                    while j2 + 1 < ny and M[i, j2 + 1, k:k2 + 1].all() and not seen[i, j2 + 1, k:k2 + 1].any():
                        j2 += 1
                    i2 = i
                    while i2 + 1 < nx and M[i2 + 1, j:j2 + 1, k:k2 + 1].all() and not seen[i2 + 1, j:j2 + 1, k:k2 + 1].any():
                        i2 += 1
                    seen[i:i2 + 1, j:j2 + 1, k:k2 + 1] = True
                    out.append((i, j, k, i2 + 1, j2 + 1, k2 + 1))
                    k = k2 + 1
                else:
                    k += 1
    return np.array(out, float).reshape(-1, 6)


def _convert_one(job):
    name, glb, boxes = job
    try:
        info, arrays = convert(glb, boxes)
    except Exception as e:                                         # (a broken model: left out, its houses still work)
        return name, dict(error=str(e)[:200])
    np.savez_compressed(DIR / "models" / f"{name}.npz", **arrays)
    return name, info


def fetch(workers=4, cpu=4, local=None):
    """Download and convert (workers: download threads, cpu: converting processes). local: a folder that already
    has some of the files (<split>.jsonl.gz, <model>.glb) - only what is missing is downloaded."""
    from remotezip import RemoteZip
    local = Path(local) if local else None
    (DIR / "models").mkdir(parents=True, exist_ok=True)
    (DIR / "READY").unlink(missing_ok=True)
    houses = {}
    for split in ("train", "val", "test"):
        dst = DIR / f"houses_{split}.json.gz"
        if not dst.exists():
            tmp = local / f"{split}.jsonl.gz" if local and (local / f"{split}.jsonl.gz").exists() else None
            if tmp is None:
                tmp = Path(tempfile.mkdtemp()) / f"{split}.jsonl.gz"
                urllib.request.urlretrieve(f"https://raw.githubusercontent.com/{HOUSES[0]}/{HOUSES[1]}/{split}.jsonl.gz", tmp)
            hs = [_compact(json.loads(line)) for line in gzip.open(tmp, "rt", encoding="utf-8")]
            with gzip.open(dst, "wt", encoding="utf-8") as f:
                json.dump(hs, f)
        houses[split] = _load(split)
        print(split, len(houses[split]), "houses", flush=True)
    used, top = set(), set()
    for hs in houses.values():
        u, t = _assets(hs)
        used |= u
        top |= t
    part = DIR / "models.partial.json"                             # (progress; models.json only when done)
    man = json.loads(part.read_text()) if part.exists() else         json.loads((DIR / "models.json").read_text()) if (DIR / "models.json").exists() else {}
    raw = local or Path(tempfile.mkdtemp(prefix="thor_"))
    todo = [a for a in sorted(used) if a not in man]
    print(len(used), "models used,", len(todo), "to fetch", flush=True)

    import threading
    tls = threading.local()

    def get(a):
        if (raw / f"{a}.glb").exists():
            return a
        if not hasattr(tls, "z"):
            tls.z = RemoteZip(ZIP)
        try:
            (raw / f"{a}.glb").write_bytes(tls.z.read(GLB.format(a)))
            return a
        except KeyError:
            return None
    with ThreadPoolExecutor(workers) as ex:
        got = [a for a in ex.map(get, todo) if a]
    with ProcessPoolExecutor(cpu) as ex:
        for k, (name, info) in enumerate(ex.map(_convert_one, [(a, raw / f"{a}.glb", a in top) for a in got], chunksize=4)):
            man[name] = info
            if k % 100 == 0:
                print(k, "converted", flush=True)
                part.write_text(json.dumps(man))
    (DIR / "models.json").write_text(json.dumps(man))
    part.unlink(missing_ok=True)
    (DIR / "READY").write_text(f"{HOUSES[1]} {MODELS[1]}\n")
    (DIR / "SOURCE.txt").write_text(
        f"houses: https://github.com/{HOUSES[0]} at {HOUSES[1]} (Apache-2.0)\n"
        f"models: https://huggingface.co/datasets/{MODELS[0]} at {MODELS[1]} (AI2-THOR's models, Apache-2.0; "
        "the mirror declares CC-BY-4.0), converted by sim/worlds/procthor.py\n")
    print("models:", sum("error" not in v for v in man.values()), "converted,",
          sum("error" in v for v in man.values()), "failed; missing from the zip:", len(todo) - len(got))


# --------------------------------------------------------------------------------------------------- reading
@lru_cache(maxsize=3)
def _load(split):
    with gzip.open(DIR / f"houses_{split}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def models():
    return {k: v for k, v in json.loads((DIR / "models.json").read_text()).items() if "error" not in v}


@lru_cache(maxsize=512)
def model(name):
    z = np.load(DIR / "models" / f"{name}.npz")
    parts = []
    k = 0
    while f"V{k}" in z:
        V, F = z[f"V{k}"], z[f"F{k}"]
        area = 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()
        if area > 1e-6:                                            # (MuJoCo refuses parts with no area)
            parts.append((V, F, z[f"C{k}"]))
        k += 1
    boxes = (z["box_c"], z["box_h"]) if "box_c" in z else None
    return parts, boxes


def _rot(yaw, x, y):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


def _item(o):
    """A placed model -> a furniture dict: x, y the centre of its footprint, w, d its size along its own axes,
    h its top above the floor, z where the model's origin sits, off the origin relative to the centre. (ProcTHOR's
    position is the centre of the model's bounding box; the models' origins are at their bottom.)"""
    info = models()[o["asset"]]
    lo, hi = np.array(info["lo"]), np.array(info["hi"])
    c = (lo + hi) / 2
    return dict(kind=o["type"].lower(), x=o["x"], y=o["y"], yaw=o["yaw"], w=float(hi[0] - lo[0]),
                d=float(hi[1] - lo[1]), h=float(o["z"] + (hi[2] - lo[2]) / 2), z=float(o["z"] - c[2]),
                asset=o["asset"], off=(-float(c[0]), -float(c[1])), base=float(o["z"] - (hi[2] - lo[2]) / 2),
                flat=info["flat"])


def _surface_ok(it):
    lo, sh = max(it["w"], it["d"]), min(it["w"], it["d"])
    return (0.35 <= it["h"] <= 1.1 and it["flat"] >= 0.55 and lo >= SURFACE_MIN[0] and sh >= SURFACE_MIN[1]
            and it["base"] < 0.1)


@lru_cache(maxsize=3)
def _candidates(split):
    """For each kind, the (house, object) pairs that can be Pinch's surface; for the floor, every house."""
    out = {"desk": [], "table": [], "floor": []}
    ms = models()
    for hi_, h in enumerate(_load(split)):
        if all(o["asset"] in ms for o in h["objects"]):
            out["floor"].append((hi_, None))
            for kind, types in SURFACE.items():
                for oi, o in enumerate(h["objects"]):
                    if o["type"] in types and _surface_ok(_item(o)):
                        out[kind].append((hi_, oi))
    return out


def _inside(poly, p):
    """Whether point p is inside the polygon (even-odd rule; rooms are simple polygons, points never on an edge here)."""
    x, y, n, inside = p[0], p[1], len(poly), False
    for k in range(n):
        (x0, y0), (x1, y1) = poly[k], poly[k - 1]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _cells(poly):
    xs = sorted({round(p[0], 3) for p in poly})
    ys = sorted({round(p[1], 3) for p in poly})
    grid = [[_inside(poly, ((xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2)) for j in range(len(ys) - 1)]
            for i in range(len(xs) - 1)]
    cells = [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i in range(len(xs) - 1) for j in range(len(ys) - 1) if grid[i][j]]
    best, rect = 0.0, None                                         # the largest rectangle made of whole cells
    for i0 in range(len(xs) - 1):
        for i1 in range(i0, len(xs) - 1):
            for j0 in range(len(ys) - 1):
                for j1 in range(j0, len(ys) - 1):
                    if all(grid[i][j] for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)):
                        a = (xs[i1 + 1] - xs[i0]) * (ys[j1 + 1] - ys[j0])
                        if a > best:
                            best, rect = a, (xs[i0], ys[j0], xs[i1 + 1], ys[j1 + 1])
    return cells, rect


def _walls(h):
    """Each wall once (rooms list shared walls from both sides), with its door and window holes along it."""
    seen, out = {}, []
    for w in h["walls"]:
        a, b = np.array(w["a"], float), np.array(w["b"], float)
        key = tuple(sorted([tuple(np.round(a, 2)), tuple(np.round(b, 2))]))
        L = float(np.linalg.norm(b - a))
        if L < 1e-3:
            continue
        u = (b - a) / L
        if key not in seen:
            seen[key] = dict(x0=float(a[0]), y0=float(a[1]), x1=float(b[0]), y1=float(b[1]), doors=[], windows=[],
                             t=0.02)                               # (furniture stands flush with the wall line)
            out.append(seen[key])
        wl = seen[key]
        a0 = np.array([wl["x0"], wl["y0"]])
        u0 = np.array([wl["x1"] - wl["x0"], wl["y1"] - wl["y0"]]) / L
        for s0, s1, z0, z1 in w["holes"]:                          # along this side's wall -> along the kept one
            c = float((a + u * (s0 + s1) / 2 - a0) @ u0)
            width = abs(s1 - s0)
            if z0 < 0.05:
                if not any(abs(c - c2) < 0.05 for c2, _ in wl["doors"]):
                    wl["doors"].append((c, width))
            elif not any(abs(c - c2) < 0.05 for c2, *_ in wl["windows"]):
                wl["windows"].append((c, width, float(z0), float(z1)))
    return out


def _front_zone(it, k, depth=0.9):
    """The floor in front of side k of a surface item (k = 0..3: its -y, +x, +y, -x side) as (x, y, yaw, hw, hd)."""
    yaw = it["yaw"] + k * math.pi / 2
    hw, hd = (it["w"], it["d"]) if k % 2 == 0 else (it["d"], it["w"])
    dx, dy = _rot(yaw, 0.0, -(hd / 2 + depth / 2))
    return (it["x"] + dx, it["y"] + dy, yaw, hw / 2, depth / 2), yaw, hw, hd


def home(rng, split, kind):
    """A ProcTHOR house for one world, in generate.py's house form: (rooms, walls, furniture, surface item or None,
    Pinch's room, extra), or None when this draw doesn't work (the caller draws again)."""

    from sim.worlds.generate import _overlap
    cands = _candidates(split)[kind]
    hi_, oi = cands[int(rng.integers(len(cands)))]
    h = _load(split)[hi_]
    rooms = []
    for r in h["rooms"]:
        cells, rect = _cells(r["poly"])
        xs, ys = [p[0] for p in r["poly"]], [p[1] for p in r["poly"]]
        rooms.append(dict(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys), kind=ROOMS.get(r["type"], "living"),
                          cells=cells, rect=rect, poly=[tuple(p) for p in r["poly"]]))
    walls = _walls(h)
    furniture, surf = [], None
    for k, o in enumerate(h["objects"]):
        it = _item(o)
        if k == oi:
            surf = it                                              # (its things are cleared: Pinch's things go there)
            continue
        furniture.append(it)
        for c in o["children"]:
            if c["asset"] not in models():                         # (one small model is missing from the zip)
                continue
            ci = _item(c)
            ci["small"] = True
            furniture.append(ci)
    if surf is None:                                               # the floor: the largest clear rectangle of a room
        big = [r for r in rooms if r["rect"] and (r["rect"][2] - r["rect"][0]) >= 1.6 and (r["rect"][3] - r["rect"][1]) >= 1.6]
        if not big:
            return None
        home_room = big[int(rng.integers(len(big)))]
        x0, y0, x1, y1 = home_room["rect"]
        home_room = dict(home_room, x0=x0, y0=y0, x1=x1, y1=y1)
        rooms = [home_room if r["poly"] == home_room["poly"] else r for r in rooms]
        return rooms, walls, furniture, None, home_room, dict(house=f"{split}-{hi_}", source="procthor")
    room = next((r for r in rooms if _inside(r["poly"], (surf["x"], surf["y"]))), rooms[0])
    best = None                                                    # the side someone sits at: free floor in front
    for k in rng.permutation(4):
        zone, yaw, hw, hd = _front_zone(surf, int(k))
        zx, zy = zone[0], zone[1]
        corners = [(zx + a, zy + b) for a, b in [_rot(yaw, sx * zone[3], sy * zone[4]) for sx in (-1, 1) for sy in (-1, 1)]]
        inside = sum(_inside(room["poly"], c) for c in corners) + _inside(room["poly"], (zx, zy))
        blocked = sum(1 for f in furniture if not f.get("small") and f["kind"] not in [s.lower() for s in SEATS]
                      and _overlap(zone, (f["x"], f["y"], f["yaw"], f["w"] / 2, f["d"] / 2), 0.0))
        score = inside - 3 * blocked
        if best is None or score > best[0]:
            best = (score, int(k), zone, yaw, hw, hd)
    score, k, zone, yaw, hw, hd = best
    if score < 5:
        return None
    furniture = [f for f in furniture if not (_overlap(zone, (f["x"], f["y"], f["yaw"], f["w"] / 2, f["d"] / 2), 0.05)
                                              and (f["kind"] in [s.lower() for s in SEATS] or f.get("small")))]
    surf_item = dict(surf, kind={"Desk": "desk", "DiningTable": "table", "CoffeeTable": "coffee",
                                 "SideTable": "table"}[h["objects"][oi]["type"]],
                     yaw=float(yaw), w=float(hw), d=float(hd), model_yaw=float(surf["yaw"]))
    return rooms, walls, furniture, surf_item, room, dict(house=f"{split}-{hi_}", source="procthor")


# --------------------------------------------------------------------------------------------------- building
def add(spec, f, H, collide=True, visual=True, name=None):
    """A placed model (a furniture dict from home(), surface frame) into the spec: its coloured parts (visual), and
    boxes for contacts (furniture) or its own convex hull (small things) when collide. None if nothing to add."""
    import mujoco
    parts, boxes = model(f["asset"])
    allv = np.vstack([V for V, _, _ in parts])
    hull = collide and f.get("small") and (np.ptp(allv, 0) > 0.005).all()    # (a flat poster: no contacts)
    use_boxes = collide and not f.get("small") and boxes is not None
    if not (visual or hull or use_boxes):
        return None
    yaw = f.get("model_yaw", f["yaw"])
    ox, oy = _rot(yaw, *f["off"])
    body = spec.worldbody.add_body(pos=[f["x"] + ox, f["y"] + oy, -H + f["z"]], euler=[0, 0, yaw])
    if name:
        body.name = name
    meshes, mats = {m.name for m in spec.meshes}, {m.name for m in spec.materials}
    for k, (V, F, C) in enumerate(parts):
        mn = f"thor_{f['asset']}_{k}"
        if visual and mn not in meshes:
            m = spec.add_mesh(name=mn)
            m.uservert = V.ravel()
            m.userface = F.ravel()
            m.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            meshes.add(mn)
        if visual:
            mat = f"thorc_{int(C[0] * 255)}_{int(C[1] * 255)}_{int(C[2] * 255)}"
            if mat not in mats:
                spec.add_material(name=mat, rgba=[float(C[0]), float(C[1]), float(C[2]), 1.0], specular=0.15,
                                  shininess=0.2)
                mats.add(mat)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn, material=mat, contype=0, conaffinity=0,
                          group=1)
    if hull:                                                       # one hull round all its parts
        hn = f"thor_{f['asset']}_hull"
        if hn not in meshes:
            spec.add_mesh(name=hn, uservert=allv.ravel().tolist()).inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=hn, rgba=[0.5, 0.5, 0.5, 0], contype=0, conaffinity=1,
                      group=3)
    if use_boxes:
        for c, hh in zip(*boxes):
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=c.tolist(), size=hh.tolist(), rgba=[0.5, 0.5, 0.5, 0],
                          contype=0, conaffinity=1, group=3)
    return body


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--workers", type=int, default=4, help="download threads")
        ap.add_argument("--cpu", type=int, default=4, help="converting processes")
        ap.add_argument("--local", help="a folder with some files already downloaded")
        a = ap.parse_args()
        fetch(a.workers, a.cpu, a.local)
    else:
        print(f"{len(models())} models" if available() else "not fetched: python -m sim.worlds.procthor fetch")

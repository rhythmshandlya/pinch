"""Real furniture for the worlds: Poly Haven's CC0 models (desks, tables, sofas, beds, shelves, chairs ...).

    python -m sim.worlds.furniture fetch       # ~45 models -> models/furniture/ (glTF converted to OBJ + PNG, ~60 MB)

Poly Haven (polyhaven.com, all assets CC0) publishes 85 furniture models through its public API
(api.polyhaven.com/files/<id>). Each is fetched at 1k texture resolution, converted once with trimesh (glTF y-up ->
MuJoCo z-up; each primitive to an OBJ with its UVs and its base-colour texture as PNG) and described in a manifest
(size, and for tables how much of the top is flat). A world's furniture keeps the kind and size the generator drew
(sim/worlds/generate.py: worlds are the same with or without this download); the builder picks the model of that
kind whose proportions fit best and scales it to the size. Contacts stay on simple boxes; the meshes are for looks
(and for Pinch's camera). Pinch's own desk or table uses only models with a flat top at full height.
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "models" / "furniture"
API = "https://api.polyhaven.com/files/{}"
UA = {"User-Agent": "pinch-worlds-asset-fetch"}

MODELS = {
    "desk":       ["metal_office_desk", "SchoolDesk_01", "painted_wooden_table", "wooden_table_02", "WoodenTable_01"],
    "table":      ["dining_table", "WoodenTable_02", "WoodenTable_03", "gallinera_table", "round_wooden_table_01",
                   "painted_wooden_table", "wooden_table_02"],
    "coffee":     ["CoffeeTable_01", "coffee_table_round_01", "gothic_coffee_table", "industrial_coffee_table",
                   "modern_coffee_table_01", "modern_coffee_table_02", "chinese_tea_table"],
    "sofa":       ["Sofa_01", "sofa_02", "sofa_03", "chinese_sofa", "painted_wooden_sofa"],
    "chair":      ["WoodenChair_01", "dining_chair_02", "painted_wooden_chair_01", "painted_wooden_chair_02",
                   "SchoolChair_01", "gallinera_chair", "plastic_monobloc_chair_01"],
    "bed":        ["GothicBed_01", "old_bed_frame", "vintage_day_bed"],
    "wardrobe":   ["GothicCabinet_01", "chinese_cabinet", "painted_wooden_cabinet", "painted_wooden_cabinet_02",
                   "vintage_cabinet_01", "modern_wooden_cabinet"],
    "shelf":      ["Shelf_01", "painted_wooden_shelves", "steel_frame_shelves_01", "wooden_bookshelf_worn",
                   "wooden_display_shelves_01"],
    "tv_stand":   ["ClassicConsole_01", "chinese_console_table", "chinese_commode", "GothicCommode_01"],
    "nightstand": ["ClassicNightstand_01", "painted_wooden_nightstand", "side_table_01", "vintage_wooden_drawer_01"],
    "tv":         ["Television_01"],
}
Y2Z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def _get(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            out.write_bytes(r.read())


def _convert(mid):
    """glTF -> parts (OBJ + PNG) in the model's folder; returns its manifest entry."""
    import trimesh
    src = DIR / mid / "src"
    scene = trimesh.load(str(src / f"{mid}.gltf"), force="scene")
    parts, verts = [], []
    for k, node in enumerate(scene.graph.nodes_geometry):
        T, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(Y2Z @ T)
        name = f"part{k}"
        uv = getattr(m.visual, "uv", None)
        mat = getattr(m.visual, "material", None)
        img = getattr(mat, "baseColorTexture", None)
        rgba = None
        if uv is not None and img is not None:
            m.visual = trimesh.visual.TextureVisuals(uv=uv)
            m.export(DIR / mid / f"{name}.obj", include_texture=False)
            img.convert("RGB").resize((min(512, img.size[0]), min(512, img.size[1]))).save(DIR / mid / f"{name}.png")
        else:
            m.export(DIR / mid / f"{name}.obj")
            c = getattr(mat, "baseColorFactor", None)
            rgba = (np.array(c, float) / (255.0 if c is not None and max(c) > 1 else 1.0)).tolist() if c is not None else [0.7] * 3 + [1]
        parts.append(dict(name=name, texture=rgba is None, rgba=rgba))
        verts.append(np.asarray(m.vertices))
    V = np.concatenate(verts)
    lo, hi = V.min(0), V.max(0)
    flat = _top_flatness(scene, lo, hi)
    return dict(id=mid, parts=parts, lo=lo.tolist(), hi=hi.tolist(), top_flat=flat)


def _top_flatness(scene, lo, hi, n=12):
    """Share of a grid over the footprint (inner 80 %) whose highest surface is within 1.5 % of the top
    (from points sampled on the surface: no ray-casting library needed)."""
    import trimesh
    mesh = trimesh.util.concatenate([scene.geometry[scene.graph[nd][1]].copy().apply_transform(Y2Z @ scene.graph[nd][0])
                                     for nd in scene.graph.nodes_geometry])
    pts = trimesh.sample.sample_surface(mesh, 40000, seed=0)[0]
    ix = np.floor((pts[:, 0] - lo[0]) / (hi[0] - lo[0]) * n).astype(int)
    iy = np.floor((pts[:, 1] - lo[1]) / (hi[1] - lo[1]) * n).astype(int)
    top = np.full((n, n), -np.inf)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    np.maximum.at(top, (ix[ok], iy[ok]), pts[ok, 2])
    k = max(1, n // 10)
    inner = top[k:n - k, k:n - k]
    return float(np.mean(inner >= hi[2] - 0.015 * (hi[2] - lo[2])))


def fetch():
    import concurrent.futures as cf
    ids = sorted({m for v in MODELS.values() for m in v})

    def one(mid):
        if (DIR / mid / "manifest.json").exists():
            return mid, True
        try:
            f = json.loads(urllib.request.urlopen(urllib.request.Request(API.format(mid), headers=UA), timeout=60).read())
            g = f["gltf"]["1k"]["gltf"]
            _get(g["url"], DIR / mid / "src" / f"{mid}.gltf")
            for rel, inc in g["include"].items():
                _get(inc["url"], DIR / mid / "src" / rel)
            entry = _convert(mid)
            (DIR / mid / "manifest.json").write_text(json.dumps(entry))
            return mid, True
        except Exception as e:                            # noqa: BLE001 - one missing model is not fatal
            print("  skipped", mid, type(e).__name__, e)
            return mid, False
    with cf.ThreadPoolExecutor(6) as ex:
        ok = [m for m, good in ex.map(one, ids) if good]
    print(f"{len(ok)}/{len(ids)} furniture models in {DIR}")


def manifest():
    out = {}
    for p in DIR.glob("*/manifest.json"):
        e = json.loads(p.read_text())
        out[e["id"]] = e
    return out


_MAN = None


def choose(kind, w, d, h, seed, surface=False):
    """The fetched model of this kind whose proportions fit (w, d, h) best (ties broken by seed), or None."""
    global _MAN
    if _MAN is None:
        _MAN = manifest()
    cands = [_MAN[m] for m in MODELS.get(kind, []) if m in _MAN]
    if surface:
        cands = [c for c in cands if c["top_flat"] >= 0.75]
    if not cands:
        return None

    def misfit(c):
        s = np.array(c["hi"]) - np.array(c["lo"])
        return float(np.sum(np.abs(np.log(np.array([w, d, h]) / np.maximum(s, 1e-3)))))
    rng = np.random.default_rng(seed)
    scored = sorted(cands, key=lambda c: misfit(c) + 0.25 * rng.random())
    return scored[0]


def add(spec, body, entry, w, d, h, prefix):
    """The model's parts as visual mesh geoms in body, scaled to w x d x h with its base at z = 0, centred."""
    import mujoco
    lo, hi = np.array(entry["lo"]), np.array(entry["hi"])
    s = np.array([w, d, h]) / np.maximum(hi - lo, 1e-4)
    off = -np.array([(lo[0] + hi[0]) / 2 * s[0], (lo[1] + hi[1]) / 2 * s[1], lo[2] * s[2]])
    folder = DIR / entry["id"]
    for p in entry["parts"]:
        name = f"{prefix}_{p['name']}"
        mesh = spec.add_mesh(name=name, file=str(folder / f"{p['name']}.obj"), scale=list(s))
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL          # (thin parts; they are static anyway)
        if p["texture"]:
            if not any(t.name == f"{entry['id']}_{p['name']}" for t in spec.textures):
                spec.add_texture(name=f"{entry['id']}_{p['name']}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                 file=str(folder / f"{p['name']}.png"))
                m = spec.add_material(name=f"{entry['id']}_{p['name']}")
                m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{entry['id']}_{p['name']}"
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, material=f"{entry['id']}_{p['name']}",
                          pos=list(off), contype=0, conaffinity=0, group=0, mass=0)
        else:
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=list(p["rgba"])[:3] + [1.0],
                          pos=list(off), contype=0, conaffinity=0, group=0, mass=0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        for k, e in sorted(manifest().items()):
            print(f"{k:28s} size {np.round(np.array(e['hi']) - np.array(e['lo']), 2)}  flat top {e['top_flat']:.2f}")

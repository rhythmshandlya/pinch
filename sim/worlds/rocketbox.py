"""Ready-made people: Microsoft Rocketbox avatars (MIT), skinned in MuJoCo and posed like the MakeHuman bodies.

    python -m sim.worlds.rocketbox fetch        # 38 adults + 4 children, converted once -> models/humans/rocketbox/
                                                # (~1.4 GB downloaded and discarded, ~15 MB kept; not in git)

Rocketbox (microsoft/Microsoft-Rocketbox, MIT, 0.7k stars; Gonzalez-Franco et al. 2020, "The Rocketbox Library
and the Utility of Freely Available Rigged Avatars", Frontiers in Virtual Reality) is 115 rigged, photo-textured
people made for research: varied ancestry, ages, clothes and hair. Each avatar is a 3ds Max Biped FBX (81 bones)
with 2048 px body, head and hair/eyelash textures. Converting once (assimp) folds the skeleton into the 20 body
bones the people's motion drives - fingers into the hand, toes into the foot, the face's bones into the head -
named like MakeHuman's (pelvis, spine_01.., upperarm_l..) so humans.pose()/park() pose both kinds of body, and
keeps the meshes, UVs, weights and textures (1024 px JPEG/PNG) as plain arrays.

A world person gets an avatar from its seed: a child picks among the 4 children (about 10-12 years, 1.43 m, scaled
to the person's 1.0-1.3 m), an adult among the 38 adults, scaled to the person's height. Without the download
people fall back to the MakeHuman bodies (humans.py), then to capsules (people.py).
"""

import json
import shutil
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "models" / "humans" / "rocketbox"
REPO, COMMIT = "microsoft/Microsoft-Rocketbox", "0943055db6ec570bcef9f2c8b41c9e5467c808f9"
URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/Assets/Avatars"
ADULTS = [f"Female_Adult_{k:02d}" for k in range(1, 18)] + [f"Male_Adult_{k:02d}" for k in range(1, 22)]
CHILDREN = ["Female_Child_01", "Female_Child_02", "Male_Child_01", "Male_Child_02"]
TEX = 1024                                                     # px kept per texture (faces need the detail)

# Biped bone -> the body bone it folds into, named as MakeHuman's game-engine rig (humans.pose drives these)
NAMES = {"Pelvis": "pelvis", "Spine": "spine_01", "Spine1": "spine_02", "Spine2": "spine_03", "Neck": "neck_01",
         "Head": "head", "L Clavicle": "clavicle_l", "L UpperArm": "upperarm_l", "L Forearm": "lowerarm_l",
         "L Hand": "hand_l", "R Clavicle": "clavicle_r", "R UpperArm": "upperarm_r", "R Forearm": "lowerarm_r",
         "R Hand": "hand_r", "L Thigh": "thigh_l", "L Calf": "calf_l", "L Foot": "foot_l", "R Thigh": "thigh_r",
         "R Calf": "calf_r", "R Foot": "foot_r"}
PARENT = {"pelvis": None, "spine_01": "pelvis", "spine_02": "spine_01", "spine_03": "spine_02", "neck_01": "spine_03",
          "head": "neck_01", "clavicle_l": "spine_03", "upperarm_l": "clavicle_l", "lowerarm_l": "upperarm_l",
          "hand_l": "lowerarm_l", "clavicle_r": "spine_03", "upperarm_r": "clavicle_r", "lowerarm_r": "upperarm_r",
          "hand_r": "lowerarm_r", "thigh_l": "pelvis", "calf_l": "thigh_l", "foot_l": "calf_l", "thigh_r": "pelvis",
          "calf_r": "thigh_r", "foot_r": "calf_r"}
BONES = list(PARENT)


def _fold(name):
    """A Biped bone's name -> the body bone it belongs to (None: not part of the body)."""
    n = name.split(" ", 1)[1] if name.startswith("Bip0") and " " in name else name
    if n in NAMES:
        return NAMES[n]
    if "Toe" in n:
        return NAMES[n[0] + " Foot"]
    if "Finger" in n:
        return NAMES[n[0] + " Hand"]
    if n == "Footsteps" or (name.startswith("Bip0") and " " not in name):
        return None
    return "head"                                              # the face's bones: jaw, lips, brows, eyes


def available():
    return (DIR / "manifest.json").exists()


def avatars():
    return json.loads((DIR / "manifest.json").read_text())


# ----------------------------------------------------------------------------------------------- fetch + convert
def convert(fbx, texdir, out, name):
    """One avatar's FBX and textures -> out/<name>.npz (rest-pose meshes in metres, z up, facing -y, feet on z = 0)
    and its textures as 1024 px images."""
    import assimp_py as ai
    from PIL import Image
    sc = ai.import_file(str(fbx), ai.Process_Triangulate | ai.Process_JoinIdenticalVertices)
    head = {}
    for b in sc.meshes[0].bones:                               # rest joints: the inverse of each bone's bind matrix
        n = b.name.split(" ", 1)[1] if " " in b.name else b.name
        if n in NAMES:
            head[NAMES[n]] = np.linalg.inv(np.array(b.offset_matrix).reshape(4, 4))[:3, 3] * 0.01
    arrays, meshes = {}, []
    for k, m in enumerate(sc.meshes):
        mat = sc.materials[m.material_index]
        tex = mat["TEXTURES"][1][0].replace("\\", "/").split("/")[-1]
        alpha = "opacity" in tex
        img = Image.open(Path(texdir) / tex).convert("RGBA" if alpha else "RGB")
        img = img.resize((TEX, TEX), Image.LANCZOS)
        fn = f"{name}_{k}.png" if alpha else f"{name}_{k}.jpg"
        img.save(Path(out) / fn, **({} if alpha else dict(quality=90)))
        V = np.array(m.vertices, np.float64).reshape(-1, 3) * 0.01
        W = np.zeros((len(V), len(BONES)), np.float32)
        for b in m.bones:
            c = _fold(b.name)
            if c is not None and b.weights is not None:
                np.add.at(W[:, BONES.index(c)], np.array(b.weight_vertex_ids, int), np.array(b.weights, np.float32))
        W[W.sum(1) == 0, BONES.index("head")] = 1.0
        W /= W.sum(1, keepdims=True)
        uv = np.array(m.texcoords[0], np.float32).reshape(-1, m.num_uv_components[0])[:, :2].copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        arrays[f"V{k}"], arrays[f"F{k}"], arrays[f"UV{k}"], arrays[f"W{k}"] = (
            V.astype(np.float32), np.array(m.indices, np.int32).reshape(-1, 3), uv, W.astype(np.float16))
        meshes.append(dict(tex=fn, alpha=alpha, part=("hair" if alpha else "head" if "head" in tex else "body")))
    allv = np.vstack([arrays[f"V{k}"] for k in range(len(meshes))])
    arrays["head"] = np.stack([head[b] for b in BONES]).astype(np.float32)
    np.savez_compressed(Path(out) / f"{name}.npz", **arrays)
    return dict(meshes=meshes, height=float(allv[:, 2].max() - allv[:, 2].min()))


def _fetch_one(name):
    import assimp_py as ai
    grp = "Children" if "Child" in name else "Adults"
    if (DIR / f"{name}.npz").exists():
        return name, None
    tmp = Path(tempfile.mkdtemp(prefix="rocketbox_"))
    try:
        fbx = tmp / f"{name}.fbx"
        urllib.request.urlretrieve(f"{URL}/{grp}/{name}/Export/{name}.fbx", fbx)
        for mt in ai.import_file(str(fbx), 0).materials:
            fn = mt["TEXTURES"][1][0].replace("\\", "/").split("/")[-1]
            urllib.request.urlretrieve(f"{URL}/{grp}/{name}/Textures/{fn}", tmp / fn)
        return name, convert(fbx, tmp, DIR, name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch(workers=4):
    DIR.mkdir(parents=True, exist_ok=True)
    man = avatars() if available() else {}
    with ThreadPoolExecutor(workers) as ex:
        for name, info in ex.map(_fetch_one, ADULTS + CHILDREN):
            if info is not None:
                man[name] = info
                print(name, f"{info['height']:.2f} m", [x["part"] for x in info["meshes"]])
    (DIR / "manifest.json").write_text(json.dumps(man, indent=1))
    (DIR / "SOURCE.txt").write_text(f"https://github.com/{REPO} at {COMMIT} (MIT licence), converted by "
                                    "sim/worlds/rocketbox.py\n")


# --------------------------------------------------------------------------------------------------- building
def choose(p, others=()):
    """The avatar for a world person: deterministic from its seed (children among the children), never one that
    `others` (the people before it in its world) already have."""
    man = avatars()
    pool = [a for a in (CHILDREN if p["child"] else ADULTS) if a in man]
    taken, k = {choose(q, others[:j]) for j, q in enumerate(others)}, int(np.random.default_rng(p["seed"] + 101).integers(len(pool)))
    for _ in range(len(pool)):
        if pool[k] not in taken:
            break
        k = (k + 1) % len(pool)
    return pool[k]


@lru_cache(maxsize=48)
def geometry(name, height):
    """An avatar scaled to a height, pelvis over the origin: meshes, bones with rest joints and parents (the keys
    humans.pose / humans.lengths use)."""
    info = avatars()[name]
    z = np.load(DIR / f"{name}.npz")
    s = height / info["height"]
    hd = z["head"].astype(np.float64) * s
    off = np.array([hd[0, 0], hd[0, 1], 0.0])
    meshes = []
    for k, mi in enumerate(info["meshes"]):
        meshes.append(dict(V=z[f"V{k}"].astype(np.float64) * s - off, F=z[f"F{k}"], UV=z[f"UV{k}"],
                           W=z[f"W{k}"].astype(np.float32), **mi))
    head = {b: hd[i] - off for i, b in enumerate(BONES)}
    hv = np.vstack([ms["V"] for ms in meshes])
    hv = hv[(hv[:, 2] > head["head"][2]) & (np.abs(hv[:, 0]) < 0.04 * s)]
    nose = hv[hv[:, 1].argmin()]                               # the face's front-most point (the avatars face -y)
    return dict(name=name, meshes=meshes, bones=BONES, head=head, parent=PARENT, bi={b: i for i, b in enumerate(BONES)},
                height=height, face_up=float(nose[2] - head["head"][2]))


@lru_cache(maxsize=96)
def _image(fn):
    from PIL import Image
    return np.ascontiguousarray(np.array(Image.open(DIR / fn)))


def add(spec, i, p, others=()):
    """Person i as a skinned Rocketbox avatar on mocap bones (person{i}_<bone>). Returns the geometry."""
    import mujoco
    g = geometry(choose(p, others), round(p["height"], 2))
    name = f"person{i}"
    for b in g["bones"]:
        spec.worldbody.add_body(name=f"{name}_{b}", mocap=True, pos=list(g["head"][b] + [60.0 + 3 * i, 60.0, -10.0]))
    for k, ms in enumerate(g["meshes"]):
        if ms["alpha"]:      # eyelash/hair cards: MuJoCo draws skins opaque (texture alpha ignored), so they'd be
            continue         # black rings round the eyes; the head texture has the hair painted on (judged on renders)
        img = _image(ms["tex"])
        tn = f"{name}_tex{k}"
        t = spec.add_texture(name=tn, type=mujoco.mjtTexture.mjTEXTURE_2D, width=img.shape[1], height=img.shape[0],
                             nchannel=img.shape[2])
        t.data = img.tobytes()
        mat = spec.add_material(name=tn, specular=0.05, shininess=0.1, emission=0.05)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tn
        sk = spec.add_skin(name=f"{name}_{ms['part']}{k}")
        sk.vert = ms["V"].flatten()
        sk.face = ms["F"].flatten()
        sk.texcoord = ms["UV"].flatten()
        sk.material = tn
        W = ms["W"]
        used = [b for b in g["bones"] if (W[:, g["bi"][b]] > 1e-4).any()]
        sk.bodyname = [f"{name}_{b}" for b in used]
        sk.bindpos = np.concatenate([g["head"][b] for b in used])
        sk.bindquat = np.tile([1.0, 0, 0, 0], len(used))
        sk.vertid = [np.where(W[:, g["bi"][b]] > 1e-4)[0].astype(np.int32) for b in used]
        sk.vertweight = [W[:, g["bi"][b]][W[:, g["bi"][b]] > 1e-4].astype(np.float32) for b in used]
    return g


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        print(f"{len(avatars())} avatars" if available() else "not fetched: python -m sim.worlds.rocketbox fetch")

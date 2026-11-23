"""Real-looking people: MakeHuman bodies (CC0), skinned in MuJoCo and posed by the people's motion (people.py).

    python -m sim.worlds.humans fetch          # the base mesh, skeleton, weights and age/gender/ethnicity targets
                                               # -> models/humans/ (~16 MB, not in git)

MakeHuman (makehumancommunity/makehuman, 1.7k stars; MPFB2, 0.7k stars, active) publishes its base mesh, its
"game engine" skeleton (53 bones) with skinning weights, and its morph targets as CC0 (LICENSE.ASSETS.md in both
repositories). A person's age, sex and ancestry blend the targets (MakeHuman's macro modifiers), the mesh is scaled
to their height, split into skin, shirt, trousers and shoes by the dominant bone of each face, and becomes MuJoCo
<skin>s bound to one mocap body per bone - MuJoCo then deforms the mesh with linear blend skinning as the bones
move. Posing is forward kinematics from the motion model's targets (where the pelvis, feet and hands go, where the
head looks) with two-bone IK for arms and legs, so a hand that reaches Pinch's shell is really there.

Without the download (CI, a fresh checkout) people are built from capsules (people.py add_person).
"""

import json
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "models" / "humans"
MPFB = ("makehumancommunity/mpfb2", "650c5c53bd6989a8ac264f31fa1d11d4a0c7bd8a", "src/mpfb/data")
MH = ("makehumancommunity/makehuman", "a8bc2d54ff0ac92e78ff71431b1023eda42bf482", "makehuman/data/targets/macrodetails")
FILES = {"base.obj": "3dobjs/base.obj", "rig.json": "rigs/standard/rig.game_engine.json",
         "weights.json": "rigs/standard/weights.game_engine.json"}
AGES = {"baby": 0.0, "child": 0.1875, "young": 0.5, "old": 1.0}          # MakeHuman: 1, 11, 25 and 90 years
ETHNIC = ("african", "asian", "caucasian")
N_BODY = 13380                                                             # body vertices (the rest are helpers)


def _targets():
    out = [f"universal-{g}-{a}-averagemuscle-averageweight.target" for g in ("female", "male") for a in AGES]
    return out + [f"{e}-{g}-{a}.target" for e in ETHNIC for g in ("female", "male") for a in AGES]


def available():
    return all((DIR / f).exists() for f in FILES) and all((DIR / "targets" / t).exists() for t in _targets())


def fetch():
    import concurrent.futures as cf
    jobs = [(f"https://raw.githubusercontent.com/{MPFB[0]}/{MPFB[1]}/{MPFB[2]}/{src}", DIR / dst) for dst, src in FILES.items()]
    jobs += [(f"https://raw.githubusercontent.com/{MH[0]}/{MH[1]}/{MH[2]}/{t}", DIR / "targets" / t) for t in _targets()]

    def get(job):
        url, out = job
        if out.exists():
            return
        out.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            out.write_bytes(r.read())
    with cf.ThreadPoolExecutor(12) as ex:
        list(ex.map(get, jobs))
    print("MakeHuman data in", DIR, "ok" if available() else "INCOMPLETE")


# ------------------------------------------------------------------------------------------------------- the mesh
@lru_cache(maxsize=1)
def _base():
    V, VT, G, g = [], [], {}, None
    for line in (DIR / "base.obj").read_text().splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(x) for x in t[1:4]])
        elif t[0] == "vt":
            VT.append([float(x) for x in t[1:3]])
        elif t[0] == "g":
            g = t[1]
            G[g] = []
        elif t[0] == "f":
            G[g].append([tuple(int(i) - 1 for i in x.split("/")[:2]) for x in t[1:]])
    rig = json.loads((DIR / "rig.json").read_text())
    W = json.loads((DIR / "weights.json").read_text())["weights"]
    return np.array(V), np.array(VT), G, rig, W


@lru_cache(maxsize=64)
def _target(name):
    idx, d = [], []
    for line in (DIR / "targets" / name).read_text().splitlines():
        if line and line[0] not in "#":
            t = line.split()
            idx.append(int(t[0]))
            d.append([float(x) for x in t[1:4]])
    return np.array(idx, dtype=int), np.array(d, float).reshape(-1, 3)


def _morph(n, age, male, eth):
    """MakeHuman's macro modifiers: piecewise-linear age weights x sex x ancestry over the targets."""
    ks = list(AGES)
    wa = dict.fromkeys(ks, 0.0)
    for a, b in zip(ks, ks[1:]):
        if AGES[a] <= age <= AGES[b]:
            t = (age - AGES[a]) / (AGES[b] - AGES[a])
            wa[a], wa[b] = max(wa[a], 1 - t), max(wa[b], t)
    out = np.zeros((n, 3))
    for gn, wg in (("female", 1 - male), ("male", male)):
        for an, w in wa.items():
            if w * wg == 0:
                continue
            i, d = _target(f"universal-{gn}-{an}-averagemuscle-averageweight.target")
            out[i] += w * wg * d
            for en, we in zip(ETHNIC, eth):
                if we:
                    i, d = _target(f"{en}-{gn}-{an}.target")
                    out[i] += w * wg * we * d
    return out


def params(p):
    """A world person -> MakeHuman age (0..1), maleness (0..1) and ancestry weights; deterministic from its seed."""
    rng = np.random.default_rng(p["seed"] + 101)
    age = rng.uniform(0.07, 0.17) if p["child"] else rng.uniform(0.32, 0.95)          # ~4-10 y or ~20-80 y
    male = float(rng.random() < 0.5)
    tone = min(range(5), key=lambda i: abs(sum(p["skin"]) - [2.54, 2.2, 1.77, 1.32, 0.9][i]))  # SKIN light -> dark
    prior = [(0.0, 0.3, 0.7), (0.05, 0.5, 0.45), (0.2, 0.6, 0.2), (0.6, 0.3, 0.1), (0.9, 0.05, 0.05)][tone]
    eth = tuple(float(x) for x in rng.dirichlet(np.array(prior) * 8 + 0.2))
    return round(float(age), 2), male, tuple(round(x, 2) for x in eth)


@lru_cache(maxsize=32)
def geometry(age, male, eth, height):
    """Rest-pose body (metres, z up, facing -y, feet on z = 0, pelvis over the origin): vertices, faces by zone,
    bones with their joint positions and parents, and skinning weights."""
    V, VT, G, rig, W = _base()
    V = V + _morph(len(V), age, male, eth)
    P = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1) * 0.1                  # MakeHuman decimetres, y up -> metres, z up
    body = P[:N_BODY]
    P = P - [0, 0, body[:, 2].min()]
    P *= height / (body[:, 2].max() - body[:, 2].min())

    def vset(grp):
        return np.unique([v for f in G[grp] for v, _ in f])
    bones = [b for b in rig if b != "Root" and rig[b]["head"].get("cube_name")]
    head = {b: P[vset(rig[b]["head"]["cube_name"])].mean(0) for b in bones}
    P = P - [head["pelvis"][0], head["pelvis"][1], 0]
    head = {b: h - [head["pelvis"][0], head["pelvis"][1], 0] for b, h in head.items()}
    parent = {b: (rig[b]["parent"] if rig[b]["parent"] in head else None) for b in bones}
    bi = {b: i for i, b in enumerate(bones)}
    Wm = np.zeros((len(V), len(bones)))
    for b, lst in W.items():
        if b in bi:
            for v, w in lst:
                Wm[v, bi[b]] = w
    Wm[Wm.sum(1) == 0, bi["head"]] = 1.0                                   # eyes and other helpers follow the head
    Wm /= Wm.sum(1, keepdims=True)
    dom = Wm.argmax(1)

    def zone(f):
        n = bones[np.bincount([dom[v] for v, _ in f], minlength=len(bones)).argmax()]
        if n.startswith(("spine", "clavicle", "upperarm")):
            return "shirt"
        if n.startswith(("pelvis", "thigh", "calf")):
            return "trousers"
        if n.startswith(("foot", "ball")):
            return "shoes"
        return "skin"
    zones = {z: [] for z in ("skin", "shirt", "trousers", "shoes", "hair", "brows", "lips")}
    eL, eR = P[vset("helper-l-eye")].mean(0), P[vset("helper-r-eye")].mean(0)
    hv = P[np.where(dom[:N_BODY] == bi["head"])[0]]
    nose = hv[hv[:, 1].argmin()]
    sc = height / 1.75
    mz = (eL[2] + eR[2]) / 2 - 0.063 * sc
    for f in G["body"]:                                                    # painted features (the agent's recipe)
        c = P[[v for v, _ in f]].mean(0)
        front = not (c[1] > nose[1] + 0.045 * sc and c[2] < eL[2] + 0.08 * sc)
        if c[2] > eL[2] + 0.075 * sc and c[2] > head["head"][2] + 0.05 * sc and (c[1] > eL[1] - 0.03 * sc or c[2] > eL[2] + 0.1 * sc):
            zones["hair"].append(f)
        elif front and abs(c[0]) < 0.026 * sc and abs(c[2] - mz) < 0.0045 * sc:
            zones["lips"].append(f)
        elif front and any(abs(c[0] - e[0]) < 0.03 * sc and abs(c[2] - (e[2] + 0.027 * sc)) < 0.0035 * sc for e in (eL, eR)):
            zones["brows"].append(f)
        else:
            zones[zone(f)].append(f)
    zones["eyes"] = G["helper-l-eye"] + G["helper-r-eye"]
    # the face: the front of the head between brows and chin, where a photo face goes (add(photo=...))
    xm, ez = (eL[0] + eR[0]) / 2, (eL[2] + eR[2]) / 2
    face = []
    for z in ("skin", "lips", "brows"):
        keep = []
        for f in zones[z]:
            vs = [v for v, _ in f]
            c = P[vs].mean(0)
            front = c[1] < eL[1] + 0.012 * sc and np.all(dom[vs] == bi["head"])
            if front and ((c[0] - xm) / (0.068 * sc)) ** 2 + ((c[2] - (ez - 0.02 * sc)) / (0.085 * sc)) ** 2 < 1.0:
                face.append(f)
            else:
                keep.append(f)
        zones[z] = keep
    zones["face"] = face
    marks = dict(eyes=(eL, eR), mouth_z=mz)
    return dict(P=P, VT=VT, zones=zones, bones=bones, head=head, parent=parent, W=Wm, bi=bi, height=height, marks=marks)


FACES = Path(__file__).resolve().parent / "assets" / "faces"


def face_photo(p):
    """A synthetic face photo for person p (a child's for a child): (RGB image, eye centres, mouth centre) in px."""
    import cv2
    info = json.loads((FACES / "faces.json").read_text())["faces"]
    names = [k for k, v in sorted(info.items()) if v["child"] == bool(p["child"])] or sorted(info)
    k = names[p["face"] % len(names)]
    img = cv2.imread(str(FACES / k))[:, :, ::-1]
    return img, np.array(info[k]["eyes"], float), np.array(info[k]["mouth"], float)


def _face_uv(g, P_face, eyes_px, mouth_px, size):
    """Texture coordinates for face vertices: the photo placed so its eyes and mouth land on the mesh's."""
    (eL, eR), mz = g["marks"]["eyes"], g["marks"]["mouth_z"]
    ex_m, ez_m = (eL[0] + eR[0]) / 2, (eL[2] + eR[2]) / 2
    sx = (eyes_px[1, 0] - eyes_px[0, 0]) / abs(eL[0] - eR[0])            # mesh +x = the photo's right (viewer's right)
    ey_px = eyes_px[:, 1].mean()
    sz = (mouth_px[1] - ey_px) / max(ez_m - mz, 1e-4)
    px = eyes_px[:, 0].mean() + sx * (P_face[:, 0] - ex_m)
    py = ey_px - sz * (P_face[:, 2] - ez_m)
    return np.stack([px / size[1], py / size[0]], 1)                     # (MuJoCo skins: row 0 at v = 0)


def lengths(g):
    h = g["head"]

    def d(a, b):
        return float(np.linalg.norm(h[a] - h[b]))
    return dict(thigh=d("thigh_l", "calf_l"), shin=d("calf_l", "foot_l"), uarm=d("upperarm_l", "lowerarm_l"),
                farm=d("lowerarm_l", "hand_l"), hip=float(h["pelvis"][2]), shoulder=float(abs(h["upperarm_l"][0])),
                torso=float(h["upperarm_l"][2] - h["pelvis"][2]), head_up=float(h["head"][2] - h["upperarm_l"][2]))


def add(spec, i, p):
    """Person i as skinned MakeHuman meshes on mocap bones (person{i}_<bone>), with a photo face. Returns the
    geometry."""
    import mujoco
    age, male, eth = params(p)
    g = geometry(age, male, eth, round(p["height"], 2))
    P, W = g["P"], g["W"]
    skin = tuple(p["skin"])
    photo = None
    if (FACES / "faces.json").exists():
        photo = face_photo(p)
        img = photo[0]
        cheek = img[int(img.shape[0] * 0.55):int(img.shape[0] * 0.62), int(img.shape[1] * 0.22):int(img.shape[1] * 0.3)]
        skin = tuple(float(c) / 255 for c in cheek.reshape(-1, 3).mean(0))     # the body's skin matches the face
        tname = f"person{i}_face"
        t = spec.add_texture(name=tname, type=mujoco.mjtTexture.mjTEXTURE_2D, width=img.shape[1], height=img.shape[0],
                             nchannel=3)
        t.data = bytes(np.ascontiguousarray(img))
        mat = spec.add_material(name=tname, specular=0.1, emission=0.15)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tname
    rng = np.random.default_rng(p["seed"] + 7)
    hair = tuple(float(x) for x in [(0.08, 0.06, 0.05), (0.25, 0.15, 0.08), (0.55, 0.4, 0.2), (0.7, 0.7, 0.7)][
        int(rng.choice(4, p=[0.5, 0.25, 0.15, 0.1]))])
    rgba = dict(skin=skin, shirt=tuple(p["shirt"]), trousers=tuple(p["trousers"]), shoes=(0.12, 0.12, 0.14),
                hair=hair, brows=tuple(c * 0.8 for c in hair), lips=(skin[0] * 0.8, skin[1] * 0.45, skin[2] * 0.5),
                eyes=(0.14, 0.09, 0.06))
    name = f"person{i}"
    for b in g["bones"]:
        spec.worldbody.add_body(name=f"{name}_{b}", mocap=True, pos=list(g["head"][b] + [60.0 + 3 * i, 60.0, -10.0]))
    for z, faces in g["zones"].items():
        pair, tri = {}, []
        for f in faces:
            for k in range(1, len(f) - 1):
                tri.append([pair.setdefault(x, len(pair)) for x in (f[0], f[k], f[k + 1])])
        if not tri or (photo is not None and z == "eyes"):
            continue
        order = sorted(pair, key=pair.get)
        vid = np.array([v for v, _ in order])
        sk = spec.add_skin(name=f"{name}_{z}")
        sk.vert = P[vid].flatten()
        sk.face = np.array(tri, np.int32).flatten()
        if z == "face" and photo is not None:
            sk.texcoord = _face_uv(g, P[vid], photo[1], photo[2], photo[0].shape).astype(np.float32).flatten()
            sk.material = f"person{i}_face"
        else:
            sk.rgba = list(rgba.get(z, skin)) + [1.0]
        sk.inflate = {"shirt": 0.004, "trousers": 0.004, "shoes": 0.004}.get(z, 0.0008 if z in ("lips", "brows", "hair") else 0.0)
        used = [b for b in g["bones"] if (W[vid, g["bi"][b]] > 1e-4).any()]
        sk.bodyname = [f"{name}_{b}" for b in used]
        sk.bindpos = np.concatenate([g["head"][b] for b in used])
        sk.bindquat = np.tile([1.0, 0, 0, 0], len(used))
        sk.vertid = [np.where(W[vid, g["bi"][b]] > 1e-4)[0].astype(np.int32) for b in used]
        sk.vertweight = [W[vid, g["bi"][b]][W[vid, g["bi"][b]] > 1e-4].astype(np.float32) for b in used]
    return g


# ------------------------------------------------------------------------------------------------------- posing
def _q_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    c = np.cross(a, b)
    w = 1.0 + float(a @ b)
    if w < 1e-8:                                                           # opposite: any perpendicular axis
        ax = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        return np.array([0.0, *(ax / np.linalg.norm(ax))])
    q = np.array([w, *c])
    return q / np.linalg.norm(q)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _rot(q, v):
    u = np.array(q[1:])
    return v + 2 * np.cross(u, np.cross(u, v) + q[0] * v)


def _qz(a):
    return np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])


def _two_bone(root, target, l1, l2, pole):
    """Elbow/knee position for a two-bone chain from root toward target, bending toward the pole direction."""
    d = target - root
    L = float(np.linalg.norm(d))
    u = d / (L + 1e-9)
    L = min(L, (l1 + l2) * 0.999)
    a = (l1 * l1 - l2 * l2 + L * L) / (2 * L)
    h = np.sqrt(max(l1 * l1 - a * a, 0.0))
    pv = pole - u * (pole @ u)
    pv = pv / (np.linalg.norm(pv) + 1e-9)
    return root + u * a + pv * h, root + u * L


def pose(m, d, i, g, T):
    """Pose person i. T: pelvis (world point), yaw (facing direction), feet/hands targets per side ("l", "r"),
    knee/elbow pole directions, look (yaw, pitch)."""
    h, par = g["head"], g["parent"]
    R0 = _qz(T["yaw"] + np.pi / 2)                                         # rest faces -y
    lean = T.get("lean", 0.0)
    fwd = np.array([np.cos(T["yaw"]), np.sin(T["yaw"]), 0.0])
    left_axis = np.array([-np.sin(T["yaw"]), np.cos(T["yaw"]), 0.0])
    Rspine = _qmul(np.array([np.cos(lean / 2), *(np.sin(lean / 2) * left_axis)]), R0) if lean else R0   # + leans forward
    rot, pos, over = {}, {}, {}

    def world(b):
        if b in pos:
            return
        p = par[b]
        if p is None:
            rot[b], pos[b] = R0, np.asarray(T["pelvis"], float)
            return
        world(p)
        pos[b] = pos[p] + _rot(rot[p], h[b] - h[p])
        if b in over:
            rot[b] = over[b]
        elif b.startswith("spine"):
            rot[b] = Rspine
        elif b == "head" or b == "neck_01":
            ly, lp = T["look"]
            q = _qmul(_qz(ly + np.pi / 2), np.array([np.cos(lp / 2), np.sin(lp / 2), 0, 0]))   # + looks down
            rot[b] = q if b == "head" else Rspine
        else:
            rot[b] = rot[p]
    for side in ("l", "r"):                                                # legs: thigh -> calf -> foot
        world(f"thigh_{side}")
        hip = pos[f"thigh_{side}"]
        l1, l2 = np.linalg.norm(h[f"calf_{side}"] - h[f"thigh_{side}"]), np.linalg.norm(h[f"foot_{side}"] - h[f"calf_{side}"])
        knee, foot = _two_bone(hip, np.asarray(T["feet"][side], float), l1, l2, np.asarray(T.get("knee", fwd), float))
        over[f"thigh_{side}"] = _qmul(_q_between(_rot(R0, h[f"calf_{side}"] - h[f"thigh_{side}"]), knee - hip), R0)
        rot[f"thigh_{side}"] = over[f"thigh_{side}"]
        over[f"calf_{side}"] = _qmul(_q_between(_rot(R0, h[f"foot_{side}"] - h[f"calf_{side}"]), foot - knee), R0)
        over[f"foot_{side}"] = R0 if T.get("feet_flat", True) else over[f"calf_{side}"]
    for side in ("l", "r"):                                                # arms: upperarm -> lowerarm -> hand
        world(f"upperarm_{side}")
        sh = pos[f"upperarm_{side}"]
        l1 = np.linalg.norm(h[f"lowerarm_{side}"] - h[f"upperarm_{side}"])
        l2 = np.linalg.norm(h[f"hand_{side}"] - h[f"lowerarm_{side}"])
        out = left_axis if side == "l" else -left_axis
        elbow, wrist = _two_bone(sh, np.asarray(T["hands"][side], float), l1, l2,
                                 np.asarray(T.get("elbow", 0.5 * out + np.array([0, 0, -0.85])), float))
        over[f"upperarm_{side}"] = _qmul(_q_between(_rot(R0, h[f"lowerarm_{side}"] - h[f"upperarm_{side}"]), elbow - sh), R0)
        rot[f"upperarm_{side}"] = over[f"upperarm_{side}"]
        over[f"lowerarm_{side}"] = _qmul(_q_between(_rot(R0, h[f"hand_{side}"] - h[f"lowerarm_{side}"]), wrist - elbow), R0)
    for b in g["bones"]:
        world(b)
    for b in g["bones"]:
        mid = m.body(f"person{i}_{b}").mocapid[0]
        d.mocap_pos[mid], d.mocap_quat[mid] = pos[b], rot[b]
    return pos


def park(m, d, i, g):
    for b in g["bones"]:
        mid = m.body(f"person{i}_{b}").mocapid[0]
        d.mocap_pos[mid] = g["head"][b] + [60.0 + 3 * i, 60.0, -10.0]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch()
    else:
        print("available" if available() else "not fetched: python -m sim.worlds.humans fetch")

"""The world generator: (split, index) -> one home with Pinch in it, as plain data. Same input, same world, always.

Splits are disjoint by construction (Procgen's train-on-N-levels protocol, Cobbe et al. 2020 arXiv 1912.01588, but
enforced, as ProcTHOR-10K's stored train/val/test houses are, arXiv 2206.06994): each world's random stream is
numpy's SeedSequence([SALT, split id, index]), an index outside its split is refused, and tests/test_worlds.py
checks that no test world's layout hash equals a training world's.

    world("train", i)   i in 0..999    what skills are trained in
    world("val", i)     i in 0..99     for choosing between training runs
    world("test", i)    i in 0..199    held out: only for judging; never trained or tuned on

How a home is made (the ProcTHOR recipe, arXiv 2206.06994, in plain numpy): a house rectangle is cut into 1-3 rooms
(guillotine cuts, each room at least 2.2 m a side), every shared wall gets a doorway, each room gets the furniture
its kind has (study, living room, bedroom, kitchen) by rejection sampling of footprints that keep doorways clear.
Pinch's surface is one of them - a desk against a wall, a dining or coffee table, or the floor of a room - and the
whole home is moved so that surface is centred on the origin with its top at z = 0, the front (where you sit) at
-y: the frame the twin and the mind use (pinch/world.py Desk). Then the surface gets its things: the tray, Pinch's
four desk things, 0-10 everyday things (Google Scanned Objects), a rug or desk mat, and the lights and people.
Randomisation ranges follow common indoor domain-randomisation practice (Tobin et al. 2017, arXiv 1703.06907).
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from sim.worlds.catalogue import (
    CLOTHES,
    FLOORS,
    FURNITURE,
    ROOM_FURNITURE,
    RUGS,
    SKIN,
    SURFACES,
    THINGS,
    TROUSERS,
    WALLS,
)

SALT = 0x5EED5                       # phase 5: change it and every world changes
SPLITS = {"train": (1, 1000), "val": (2, 100), "test": (3, 200)}     # name: (id, how many)
KINDS = {"desk": 0.5, "table": 0.25, "floor": 0.25}
STYLES = {"indifferent": 0.4, "curious": 0.35, "fearful": 0.15, "playful": 0.10}   # people toward a small pet
WALL_T, WALL_H = 0.10, 2.4
TRAY_HALF = (0.08, 0.035)
NAMED = {"pen": 0.07, "eraser": 0.012, "cap": 0.015, "ball": 0.016}  # Pinch's own desk things: half-size to keep clear
CRAB_CLEAR = 0.14                    # free radius around Pinch's start


@dataclass
class World:
    split: str
    index: int
    kind: str                        # desk | table | floor
    surface: dict                    # half_x, half_y, height (above the room floor), material, rgb, friction, furniture
    tray_xy: tuple
    home_xy: tuple
    owner_xy: tuple
    crab: tuple                      # x, y, yaw where Pinch starts
    named: dict                      # Pinch's desk things: name -> (x, y, yaw)
    things: list                     # everyday things on its surface: dicts (name, x, y, yaw, ...)
    mat: dict | None                 # a rug (floor) or desk mat under/near Pinch, or None
    rooms: list                      # dicts: x0, y0, x1, y1, kind, floor material
    walls: list                      # dicts: x0, y0, x1, y1, doors [(centre along the wall, width)]
    furniture: list                  # dicts: kind, x, y, yaw, w, d, h, rgb
    lights: list                     # dicts: pos, dir, rgb, spot, shadow
    ambient: float
    wall_rgb: tuple
    people: list                     # dicts: role, style, seed, skin, shirt, trousers, height, child, face
    tex_seed: int
    extra: dict = field(default_factory=dict)

    @property
    def name(self):
        return f"{self.split}-{self.index:04d}"

    def desk(self):
        """The twin's and the mind's view of the surface (pinch/world.py Desk)."""
        from pinch.world import Desk
        s = self.surface
        return Desk(half_x=s["half_x"], half_y=s["half_y"], owner_xy=tuple(self.owner_xy), tray_xy=tuple(self.tray_xy),
                    tray_half=TRAY_HALF, home_xy=tuple(self.home_xy))

    def to_dict(self):
        return asdict(self)

    def layout_hash(self):
        """A hash of the layout (rounded to cm and degrees): two worlds with the same hash are the same place."""
        d = self.to_dict()
        for k in ("split", "index", "tex_seed", "extra"):
            d.pop(k)
        return hashlib.sha1(json.dumps(_round(d), sort_keys=True).encode()).hexdigest()[:16]


def _round(x):
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v) for v in x]
    if isinstance(x, float):
        return round(x, 2)
    return x


def split_of(name):
    split, idx = name.rsplit("-", 1)
    return split, int(idx)


def worlds(split):
    """Every world of a split, in order (a generator: 1,000 worlds take ~2 s)."""
    return (world(split, i) for i in range(SPLITS[split][1]))


def rng_for(split, index, stream=0):
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r} (one of {list(SPLITS)})")
    sid, n = SPLITS[split]
    if not 0 <= index < n:
        raise ValueError(f"{split} has worlds 0..{n - 1}, not {index}")
    return np.random.default_rng(np.random.SeedSequence([SALT, sid, index, stream]))


def world(split, index):
    rng = rng_for(split, index)
    kind = str(rng.choice(list(KINDS), p=list(KINDS.values())))
    for _ in range(50):                                # a home that doesn't fit is drawn again (Scenic's require)
        w = _try_world(rng, split, index, kind)
        if w is not None:
            return w
    raise RuntimeError(f"no world fits for {split} {index}")


# ----------------------------------------------------------------------------------------------------- geometry
def _corners(x, y, yaw, hw, hd):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([(x + c * a - s * b, y + s * a + c * b) for a, b in ((hw, hd), (-hw, hd), (-hw, -hd), (hw, -hd))])


def _pts(x, y, yaw, hw, hd):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * a - s * b, y + s * a + c * b) for a, b in ((hw, hd), (-hw, hd), (-hw, -hd), (hw, -hd))]


def _overlap(a, b, gap=0.0):
    """Do two oriented rectangles (x, y, yaw, half_w, half_d) come closer than gap? (separating axes, after a
    bounding-circle check; plain floats: this runs ~50 times per world)"""
    if math.hypot(a[0] - b[0], a[1] - b[1]) > math.hypot(a[3], a[4]) + math.hypot(b[3], b[4]) + gap:
        return False
    pa, pb = _pts(*a[:3], a[3] + gap / 2, a[4] + gap / 2), _pts(*b[:3], b[3] + gap / 2, b[4] + gap / 2)
    for p in (pa, pb):
        for i in range(2):
            nx, ny = -(p[i + 1][1] - p[i][1]), p[i + 1][0] - p[i][0]
            ra = [q[0] * nx + q[1] * ny for q in pa]
            rb = [q[0] * nx + q[1] * ny for q in pb]
            if max(ra) < min(rb) or max(rb) < min(ra):
                return False
    return True


def _inside(r, half_x, half_y, margin=0.0):
    p = _corners(*r)
    return bool(np.all(np.abs(p[:, 0]) <= half_x - margin) and np.all(np.abs(p[:, 1]) <= half_y - margin))


def _spot(rng, taken, half_x, half_y, hw, hd, margin=0.01, gap=0.02, yaw=None, region=None, tries=60):
    """A free place for a hw x hd (half sizes) footprint on the surface, or None. taken grows."""
    for _ in range(tries):
        a = rng.uniform(-np.pi, np.pi) if yaw is None else yaw
        if region is None:
            x, y = rng.uniform(-half_x, half_x), rng.uniform(-half_y, half_y)
        else:
            x, y = rng.uniform(region[0], region[1]), rng.uniform(region[2], region[3])
        r = (float(x), float(y), float(a), hw, hd)
        if _inside(r, half_x, half_y, margin) and not any(_overlap(r, t, gap) for t in taken):
            taken.append(r)
            return r
    return None


# ----------------------------------------------------------------------------------------------------- the home
def _rooms(rng):
    W, D = rng.uniform(3.2, 7.5), rng.uniform(3.0, 6.0)
    n = int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
    rooms, cuts = [(0.0, 0.0, W, D)], []
    for _ in range(n - 1):
        rooms.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        x0, y0, x1, y1 = rooms.pop()
        horiz = (x1 - x0) < (y1 - y0)                  # cut across the longer side
        lo, hi = (y0, y1) if horiz else (x0, x1)
        if hi - lo < 4.4:
            rooms.append((x0, y0, x1, y1))
            break
        c = rng.uniform(max(lo + 2.2, lo + 0.35 * (hi - lo)), min(hi - 2.2, lo + 0.65 * (hi - lo)))
        if horiz:
            rooms += [(x0, y0, x1, c), (x0, c, x1, y1)]
            cuts.append((x0, c, x1, c))
        else:
            rooms += [(x0, y0, c, y1), (c, y0, x1, y1)]
            cuts.append((c, y0, c, y1))
    walls = [dict(x0=0.0, y0=0.0, x1=W, y1=0.0, doors=[]), dict(x0=W, y0=0.0, x1=W, y1=D, doors=[]),
             dict(x0=0.0, y0=D, x1=W, y1=D, doors=[]), dict(x0=0.0, y0=0.0, x1=0.0, y1=D, doors=[])]
    for x0, y0, x1, y1 in cuts:                        # a doorway in every inner wall (so every room is reachable)
        L = abs(x1 - x0) + abs(y1 - y0)
        w = rng.uniform(0.8, 0.95)
        walls.append(dict(x0=x0, y0=y0, x1=x1, y1=y1, doors=[(float(rng.uniform(0.3 + w / 2, L - 0.3 - w / 2)), w)]))
    if rng.random() < 0.6:                             # a door out of the home
        k = int(rng.integers(4))
        L = W if k % 2 == 0 else D
        w = rng.uniform(0.8, 0.95)
        walls[k]["doors"].append((float(rng.uniform(0.3 + w / 2, L - 0.3 - w / 2)), w))
    kinds = list(rng.permutation(list(ROOM_FURNITURE)))
    out = [dict(x0=r[0], y0=r[1], x1=r[2], y1=r[3], kind=str(kinds[i % 4])) for i, r in enumerate(rooms)]
    return W, D, out, walls


def _door_zones(walls):
    """Keep-clear rectangles (house frame, x, y, yaw, hw, hd) on both sides of each doorway."""
    out = []
    for wl in walls:
        L = np.hypot(wl["x1"] - wl["x0"], wl["y1"] - wl["y0"])
        ux, uy = (wl["x1"] - wl["x0"]) / L, (wl["y1"] - wl["y0"]) / L
        for c, w in wl["doors"]:
            out.append((wl["x0"] + ux * c, wl["y0"] + uy * c, float(np.arctan2(uy, ux)), w / 2 + 0.1, 0.9))
    return out


def _place_furniture(rng, room, kind, taken, wall=None):
    (wr, dr, hr), against = FURNITURE[kind]
    w, d, h = (rng.uniform(*wr), rng.uniform(*dr), rng.uniform(*hr))
    x0, y0, x1, y1 = room["x0"] + WALL_T / 2, room["y0"] + WALL_T / 2, room["x1"] - WALL_T / 2, room["y1"] - WALL_T / 2
    for _ in range(60):
        if against:
            side = int(rng.integers(4)) if wall is None else wall
            yaw = (0.0, np.pi / 2, np.pi, -np.pi / 2)[side]       # its back to the north, west, south, east wall
            along_x = side % 2 == 0
            span = (x1 - x0 - w) / 2 if along_x else (y1 - y0 - w) / 2
            if span < 0:
                return None
            t = rng.uniform(-span, span)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if side == 0:
                x, y = cx + t, y1 - d / 2 - 0.01
            elif side == 2:
                x, y = cx + t, y0 + d / 2 + 0.01
            elif side == 1:
                x, y = x0 + d / 2 + 0.01, cy + t
            else:
                x, y = x1 - d / 2 - 0.01, cy + t
        else:
            yaw = float(rng.integers(4)) * np.pi / 2
            hw, hd = (w / 2, d / 2) if yaw % np.pi == 0 else (d / 2, w / 2)
            if x1 - x0 < 2 * hw + 0.6 or y1 - y0 < 2 * hd + 0.6:
                return None
            x, y = rng.uniform(x0 + hw + 0.3, x1 - hw - 0.3), rng.uniform(y0 + hd + 0.3, y1 - hd - 0.3)
        reach = 0.9 if kind in ("desk", "table", "coffee") else 0.0     # room in front to sit or walk
        fp = (x, y, yaw, w / 2, d / 2)
        fp_use = (x - np.sin(yaw) * -reach / 2, y + np.cos(yaw) * -reach / 2, yaw, w / 2, d / 2 + reach / 2) \
            if reach else fp
        hw_ok = all(abs(c[0] - (x0 + x1) / 2) <= (x1 - x0) / 2 + 1e-6 and abs(c[1] - (y0 + y1) / 2) <= (y1 - y0) / 2 + 1e-6
                    for c in np.vstack([_corners(*fp), _corners(*fp_use)]))       # (the seat in front too)
        if hw_ok and not any(_overlap(fp_use, t, 0.05) for t in taken):
            taken.append(fp_use)
            return dict(kind=kind, x=float(x), y=float(y), yaw=float(yaw), w=float(w), d=float(d), h=float(h))
    return None


def _transform(items, c, yaw):
    """House frame -> surface frame (surface centre c, its front facing -y)."""
    R = np.array([[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]])
    out = []
    for it in items:
        it = dict(it)
        for kx, ky in (("x", "y"), ("x0", "y0"), ("x1", "y1")):
            if kx in it:
                it[kx], it[ky] = (float(v) for v in R @ (np.array([it[kx], it[ky]]) - c))
        if "yaw" in it:
            it["yaw"] = float((it["yaw"] - yaw + np.pi) % (2 * np.pi) - np.pi)
        if "poly" in it:                                  # (ready-made rooms: their outline and floor rectangles)
            it["poly"] = [tuple(float(v) for v in R @ (np.array(p) - c)) for p in it["poly"]]
        for k in ("cells", "rect"):
            rects = it.get(k)
            if rects is None:
                continue
            many = k == "cells"
            new = []
            for x0, y0, x1, y1 in (rects if many else [rects]):
                a, b = R @ (np.array([x0, y0]) - c), R @ (np.array([x1, y1]) - c)
                new.append((float(min(a[0], b[0])), float(min(a[1], b[1])), float(max(a[0], b[0])), float(max(a[1], b[1]))))
            it[k] = new if many else new[0]
        out.append(it)
    return out


def _colour(rng, rgb, jitter=0.12):
    return tuple(float(np.clip(v * rng.uniform(1 - jitter, 1 + jitter), 0, 1)) for v in rgb)


def _kelvin(t):
    """Approximate RGB of a light of colour temperature t K (Tanner Helland's fit), normalised to max 1."""
    t /= 100
    r = 1.0 if t <= 66 else 1.292936 * (t - 60) ** -0.1332047
    g = 0.3900816 * np.log(t) - 0.6318414 if t <= 66 else 1.129891 * (t - 60) ** -0.0755148
    b = 1.0 if t >= 66 else 0.0 if t <= 19 else 0.5432068 * np.log(t - 10) - 1.1962541
    v = np.clip([r, g, b], 0, 1)
    return tuple(float(x) for x in v / v.max())


def _floors(rng, rooms):
    for r in rooms:
        fk = str(rng.choice(list(FLOORS)))
        pat, cols, fr = FLOORS[fk]
        r["floor"] = dict(material=fk, pattern=pat, rgb=_colour(rng, cols[rng.integers(len(cols))]),
                          friction=float(rng.uniform(*fr)))


def use_procthor():
    """Ready-made ProcTHOR homes when fetched (sim/worlds/procthor.py), unless PINCH_WORLDS=own."""
    import os

    from sim.worlds import procthor
    return os.environ.get("PINCH_WORLDS", "") != "own" and procthor.available()


def _own_home(rng, kind):
    """generate.py's own home (the fallback): rooms, walls, furniture, Pinch's surface item (None on the floor),
    Pinch's room."""
    W, D, rooms, walls = _rooms(rng)
    _floors(rng, rooms)
    taken = list(_door_zones(walls))
    home_room = rooms[int(rng.integers(len(rooms)))]
    furniture, surf_item = [], None
    if kind == "desk":
        home_room["kind"] = "study" if rng.random() < 0.6 else home_room["kind"]
        surf_item = _place_furniture(rng, home_room, "desk", taken)
    elif kind == "table":
        surf_item = _place_furniture(rng, home_room, "table" if rng.random() < 0.5 else "coffee", taken)
    if kind != "floor" and surf_item is None:
        return None
    if surf_item is not None:
        furniture.append(surf_item)
    for r in rooms:
        for fk in ROOM_FURNITURE[r["kind"]]:
            if surf_item is not None and r is home_room and fk == surf_item["kind"]:
                continue
            if rng.random() < 0.8:
                f = _place_furniture(rng, r, fk, taken)
                if f is not None:
                    furniture.append(f)
    for f in furniture:
        f["rgb"] = _colour(rng, SURFACES[str(rng.choice(["oak", "walnut", "white_lam", "grey_lam"]))][1][0], 0.15)
    return rooms, walls, furniture, surf_item, home_room, dict(house=(float(W), float(D)))


def _try_world(rng, split, index, kind):
    if use_procthor():
        from sim.worlds import procthor
        home = procthor.home(rng, split, kind)
        if home is not None:
            _floors(rng, home[0])
    else:
        home = _own_home(rng, kind)
    if home is None:
        return None
    rooms, walls, furniture, surf_item, home_room, extra = home

    # -- Pinch's surface and the frame centred on it
    if kind == "floor":
        r = home_room
        c, yaw = np.array([(r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2]), 0.0
        half_x, half_y = (r["x1"] - r["x0"]) / 2 - WALL_T / 2 - 0.01, (r["y1"] - r["y0"]) / 2 - WALL_T / 2 - 0.01
        height, fl = 0.0, r["floor"]
        surface = dict(half_x=float(half_x), half_y=float(half_y), height=0.0, material=fl["material"],
                       pattern=FLOORS[fl["material"]][0], rgb=fl["rgb"], friction=fl["friction"], furniture=None)
    else:
        c, yaw = np.array([surf_item["x"], surf_item["y"]]), surf_item["yaw"]
        mk = str(rng.choice(list(SURFACES)))
        pat, cols, fr = SURFACES[mk]
        surf_item["rgb"] = _colour(rng, cols[rng.integers(len(cols))], 0.1)
        surface = dict(half_x=surf_item["w"] / 2, half_y=surf_item["d"] / 2, height=surf_item["h"], material=mk,
                       pattern=pat, rgb=surf_item["rgb"], friction=float(rng.uniform(*fr)),
                       furniture=surf_item["kind"])
        if "asset" in surf_item:                         # a ready-made model: drawn as it is, the top as contacts
            mo = surf_item
            surface["model"] = dict(asset=mo["asset"], z=mo["z"], off=list(mo["off"]),
                                    yaw=float((mo["model_yaw"] - yaw + np.pi) % (2 * np.pi) - np.pi))
        height = surf_item["h"]
    rooms_f = _transform(rooms, c, yaw)
    for r in rooms_f:                                    # a rotation by 90 degrees swaps corners: keep x0 < x1
        r["x0"], r["x1"] = sorted((r["x0"], r["x1"]))
        r["y0"], r["y1"] = sorted((r["y0"], r["y1"]))
    walls_f = _transform(walls, c, yaw)
    furn_f = _transform(furniture, c, yaw)
    hx, hy = surface["half_x"], surface["half_y"]

    # -- what's on the surface (surface frame)
    on = []                                             # footprints taken on the surface
    if kind == "floor":                                  # furniture standing in Pinch's room is in the way
        for f in furn_f:
            on.append((f["x"], f["y"], f["yaw"], f["w"] / 2, f["d"] / 2))
    if kind == "floor":
        tray = _spot(rng, on, hx, hy, TRAY_HALF[0] + 0.03, TRAY_HALF[1] + 0.05, margin=0.05, yaw=0.0)
    else:
        tray = _spot(rng, on, hx, hy, TRAY_HALF[0] + 0.03, TRAY_HALF[1] + 0.05, margin=0.02, yaw=0.0,
                     region=(-hx + 0.12, hx - 0.12, max(0.0, hy * 0.15), hy - 0.06))
    if tray is None:
        return None
    crab = _spot(rng, on, hx, hy, CRAB_CLEAR, CRAB_CLEAR, margin=0.05, gap=0.0)
    if crab is None:
        return None
    crab_yaw = float(rng.uniform(-np.pi, np.pi))
    home = _spot(rng, on, hx, hy, 0.10, 0.10, margin=0.04, yaw=0.0)
    if home is None:
        return None
    if kind == "floor":
        o = _spot(rng, list(on), hx, hy, 0.2, 0.2, margin=0.0, yaw=0.0)
        owner = (o[0], o[1]) if o else (0.0, -hy + 0.3)
    else:
        owner = (float(rng.uniform(-0.5, 0.5) * min(hx, 0.4)), -hy - 0.075)
    named = {}
    for nm, half in NAMED.items():
        r = _spot(rng, on, hx, hy, half, min(half, 0.015), margin=0.03)
        if r is None:
            return None
        named[nm] = (r[0], r[1], r[2])
    mat = None
    if kind == "floor" and rng.random() < 0.6:          # a rug, Pinch on it or near it
        rk = str(rng.choice(list(RUGS)))
        pat, cols, fr = RUGS[rk]
        mat = dict(kind="rug", material=rk, pattern=pat, rgb=_colour(rng, cols[rng.integers(len(cols))]),
                   friction=float(rng.uniform(*fr)), half=(float(rng.uniform(0.5, 1.2)), float(rng.uniform(0.4, 0.9))),
                   thick=float(rng.uniform(0.003, 0.010)),
                   xy=(float(crab[0] + rng.uniform(-0.6, 0.6)), float(crab[1] + rng.uniform(-0.5, 0.5))),
                   yaw=float(rng.uniform(-0.4, 0.4)))
    elif kind != "floor" and rng.random() < 0.3:        # a desk mat
        mat = dict(kind="mat", material="mat", pattern="noise",
                   rgb=_colour(rng, [(0.15, 0.15, 0.17), (0.25, 0.35, 0.3), (0.5, 0.4, 0.3)][rng.integers(3)]),
                   friction=float(rng.uniform(0.8, 1.1)), half=(float(min(hx - 0.02, rng.uniform(0.2, 0.4))),
                                                              float(min(hy - 0.02, rng.uniform(0.15, 0.25)))),
                   thick=float(rng.uniform(0.002, 0.004)), xy=(float(rng.uniform(-0.1, 0.1)), float(-hy * 0.3)), yaw=0.0)
    things = []
    where = "floor" if kind == "floor" else "desk"
    pool = [k for k, v in THINGS.items() if v[3] in (where, "any")]
    for _ in range(int(rng.integers(0, 11))):
        nm = str(rng.choice(pool))
        gso, (tw, td, th), mass, _, shape, rgba = THINGS[nm]
        r = _spot(rng, on, hx, hy, tw / 2, td / 2, margin=0.005, gap=0.03)
        if r is not None:
            things.append(dict(name=nm, x=r[0], y=r[1], yaw=r[2]))

    # -- lights: a ceiling lamp or two, maybe daylight through a window; dim evenings to bright days
    lights = []
    level = float(np.exp(rng.uniform(np.log(0.3), np.log(1.4))))
    ceiling = WALL_H - 0.05 - height
    for _ in range(int(rng.integers(1, 4))):
        r = rooms_f[int(rng.integers(len(rooms_f)))]
        pos = (float(rng.uniform(r["x0"] + 0.3, r["x1"] - 0.3)), float(rng.uniform(r["y0"] + 0.3, r["y1"] - 0.3)), ceiling)
        lights.append(dict(pos=pos, dir=(0.0, 0.0, -1.0), rgb=_kelvin(rng.uniform(2700, 6500)),
                           level=float(level * rng.uniform(0.5, 1.0)), spot=bool(rng.random() < 0.3), shadow=False))
    if rng.random() < 0.5:                               # daylight: a low, directional sun
        az, el = rng.uniform(-np.pi, np.pi), rng.uniform(0.3, 1.1)
        d = (float(-np.cos(az) * np.cos(el)), float(-np.sin(az) * np.cos(el)), float(-np.sin(el)))
        lights.append(dict(pos=(float(-d[0] * 3), float(-d[1] * 3), float(-d[2] * 3)), dir=d, rgb=_kelvin(rng.uniform(5000, 6500)),
                           level=float(rng.uniform(0.3, 0.9)), spot=False, shadow=True, directional=True))
    near = min(lights, key=lambda li: np.hypot(li["pos"][0], li["pos"][1]))
    near["shadow"] = True

    # -- people: someone sitting with Pinch (often), people walking through (sometimes)
    people = []

    def person(role):
        child = bool(rng.random() < 0.2)
        return dict(role=role, style=str(rng.choice(list(STYLES), p=list(STYLES.values()))),
                    seed=int(rng.integers(2**31)), skin=SKIN[int(rng.integers(len(SKIN)))],
                    shirt=_colour(rng, CLOTHES[int(rng.integers(len(CLOTHES)))]),
                    trousers=_colour(rng, TROUSERS[int(rng.integers(len(TROUSERS)))]),
                    height=float(rng.uniform(1.0, 1.3) if child else rng.uniform(1.5, 1.9)), child=child,
                    face=int(rng.integers(1 << 16)))
    if rng.random() < 0.8:
        people.append(person("sitter"))                  # at the desk / table, or on the floor near Pinch
    for _ in range(int(rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))):
        people.append(person("walker"))

    return World(split=split, index=index, kind=kind, surface=surface, tray_xy=(tray[0], tray[1]), home_xy=(home[0], home[1]),
                 owner_xy=owner, crab=(crab[0], crab[1], crab_yaw), named=named, things=things, mat=mat, rooms=rooms_f,
                 walls=walls_f, furniture=[f for f in furn_f if f is not None and f is not _skip(furn_f, surf_item, c, yaw)],
                 lights=lights, ambient=float(rng.uniform(0.15, 0.4) * min(1.0, level)),
                 wall_rgb=_colour(rng, WALLS[int(rng.integers(len(WALLS)))], 0.05), people=people,
                 tex_seed=int(rng.integers(2**31)), extra=dict(level=level, **extra))


def _skip(furn_f, surf_item, c, yaw):
    """Pinch's own desk/table among the furniture (the builder makes it from the surface instead)."""
    if surf_item is None:
        return None
    for f in furn_f:
        if f["kind"] == surf_item["kind"] and abs(f["x"]) < 1e-6 and abs(f["y"]) < 1e-6:
            return f
    return None

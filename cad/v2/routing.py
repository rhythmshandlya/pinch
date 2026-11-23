"""Cable routing inside the v2 body: every cable path-found through the free space between the parts, along cable
channels, never through a part, a servo or a joint's swing, and clear of the cover's seam.

The body's inside is a 2 mm voxel grid: free where it is inside the shell (less the wall), between the leg bays'
walls, outside the arm bays, below the seam, and more than 1 mm from every part's box, servo case and hip bracket
plate. A* finds each cable's path between its two connectors (the middle of a connector's cable room on the part's
named face). Costs steer it:
- channels: within 7 mm of a bay wall or of the floor, steps cost 0.4 (cables run along the walls, where they are tied);
- power keeps to the right half and the floor, signal to the left and the upper band, so the two stay apart;
- within 6 mm of the seam, steps cost 4 (no wire under the cover's lip);
- a cable already routed makes its cells cheaper for the next cable of its kind, so they bundle.
The path is then pulled tight (line of sight through free cells), so runs are straight and turn at few points.

Rules from small-robot practice (docs/DECISIONS.md, the cable management entry): servo cables bundle per leg to that
leg's hip port; a tie point within 20 mm of each connector and every 40 mm along a channel (tie_points()); bends of
at least 10 x the cable's diameter where it flexes (at the hip ports: the bundle loops inside the leg bay).
"""

import heapq
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from crab.electronics import Layout  # noqa: E402
from crab.shapes import Shell  # noqa: E402

STEP = 2.0
LO = np.array([-82.0, -58.0, -54.0])
HI = np.array([96.0, 58.0, 60.0])
SEAM_BAND = 6.0
CHANNEL = 7.0


def _B():
    from cad.v2 import body  # lazy: body.py imports the wiring, which imports this
    return body


@cache
def grid():
    """(free bool grid, cell centres (nx, ny, nz, 3), base cost grid, channel mask)."""
    B = _B()
    lay = Layout()
    shell = Shell.from_chassis(lay.chassis)
    xs, ys, zs = (np.arange(LO[i] + STEP / 2, HI[i], STEP) for i in range(3))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X, Y, Z], -1)
    free = shell.f(P, shell.wall + 1.0) <= 1.0                                         # inside the shell, 1 mm off it
    free &= B.plan_contains(P, B.WALL + 1.0)                                           # inside the body's walls
    boxes = []
    for n, p in lay.parts.items():
        if p.mass_only or n in ("oled", "visor") or n.startswith("pad_") or p.d.get("in_model"):
            continue
        boxes.append((p.pos, p.half + 1.0, p.R))
    for leg in B.LEGS:
        lo, hi = B.hip_case_aabb(leg)
        boxes.append(((lo + hi) / 2, (hi - lo) / 2 + 1.0, np.eye(3)))
    for arm in B.ARMS:
        lo, hi = B.shoulder_case_aabb(arm, ears=True)
        boxes.append(((lo + hi) / 2, (hi - lo) / 2 + 1.0, np.eye(3)))
    for c, h, R in boxes:
        q = np.abs((P - np.asarray(c, float)) @ np.asarray(R, float))
        free &= ~np.all(q <= np.asarray(h, float), axis=-1)
    for leg in B.LEGS:                                                                 # the hip bracket plates' swing
        p = B.joints()[leg]["pos"]
        r = np.hypot(X - p[0], Y - p[1])
        for z0, z1 in B.HIP_PLATES:
            free &= ~((r < 22.0) & (Z > z0 - 3.0) & (Z < z1 + 3.0))
    # channels: near a bay wall or the floor (the lowest free cell of each column, and the cells just above it)
    wall = np.abs(Y) >= B.WAIST_Y - 1.0 - CHANNEL
    lowest = np.argmax(free, axis=2)
    has = free.any(axis=2)
    k = np.arange(free.shape[2])[None, None, :]
    floor = has[..., None] & (k >= lowest[..., None]) & (k < lowest[..., None] + int(CHANNEL / STEP))
    channel = free & (wall | floor)
    cost = np.where(channel, 0.4, 1.0)
    cost = np.where(np.abs(Z - B.SPLIT_Z) < SEAM_BAND, cost * 4.0, cost)      # cross the seam only where needed
    return free, P, cost, channel


def _cell(p):
    return tuple(np.clip(((np.asarray(p, float) - LO) / STEP).astype(int), 0, np.array(grid()[0].shape) - 1))


def _nearest_free(c):
    free = grid()[0]
    if free[c]:
        return c
    best = None
    r = 1
    while best is None and r < 12:
        sl = tuple(slice(max(0, c[i] - r), min(free.shape[i], c[i] + r + 1)) for i in range(3))
        idx = np.argwhere(free[sl])
        if len(idx):
            idx = idx + np.array([s.start for s in sl])
            d = np.abs(idx - np.array(c)).sum(1)
            j = int(np.argmin(d))
            best = tuple(idx[j])
        r += 1
    return best


TURN = 3.0                 # a bend costs as much as 3 cells (6 mm) of run: straight runs, few corners
MOVES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def astar(a, b, cost):
    """Cheapest 6-connected path between the free cells nearest a and b; a change of direction costs TURN."""
    free = grid()[0]
    s, g = _nearest_free(_cell(a)), _nearest_free(_cell(b))
    if s is None or g is None:
        return None
    shape = free.shape
    h = lambda c: 0.4 * (abs(c[0] - g[0]) + abs(c[1] - g[1]) + abs(c[2] - g[2]))   # noqa: E731
    start = (s, -1)
    gs, prev = {start: 0.0}, {}
    pq = [(h(s), start)]
    end = None
    while pq:
        f, st = heapq.heappop(pq)
        c, dprev = st
        gc = gs[st]
        if f - h(c) > gc + 1e-9:
            continue
        if c == g:
            end = st
            break
        for k, d in enumerate(MOVES):
            n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if not (0 <= n[0] < shape[0] and 0 <= n[1] < shape[1] and 0 <= n[2] < shape[2]) or not free[n]:
                continue
            ng = gc + cost[n] + (TURN if dprev not in (-1, k) else 0.0)
            ns = (n, k)
            if ng < gs.get(ns, 1e18):
                gs[ns] = ng
                prev[ns] = st
                heapq.heappush(pq, (ng + h(n), ns))
    if end is None:
        return None
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return [c for c, _ in path[::-1]]


def corners(path):
    """The voxel path's corners (straight runs merged)."""
    out = [path[0]]
    for i in range(1, len(path) - 1):
        d0 = tuple(path[i][k] - path[i - 1][k] for k in range(3))
        d1 = tuple(path[i + 1][k] - path[i][k] for k in range(3))
        if d0 != d1:
            out.append(path[i])
    out.append(path[-1])
    return out


def _clear_line(c0, c1):
    free = grid()[0]
    n = int(max(abs(c1[i] - c0[i]) for i in range(3))) * 2 + 1
    for t in np.linspace(0, 1, n):
        c = tuple(int(round(c0[i] + (c1[i] - c0[i]) * t)) for i in range(3))
        if not free[c]:
            return False
    return True


def pull_tight(path, channel):
    """Shortcut the voxel path where the straight line stays in free cells and keeps to the same channel state."""
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not (_clear_line(path[i], path[j]) and channel[path[i]] == channel[path[j]]
                                 and all(channel[path[k]] == channel[path[i]] for k in range(i, j + 1, max(1, (j - i) // 8)))):
            j -= 1
        out.append(path[j])
        i = j
    return out


def side_cost(kind):
    free, P, cost, channel = grid()
    Y, Z = P[..., 1], P[..., 2]
    c = cost.copy()
    if kind == "power":                                          # the right half and the floor
        c = np.where((Y > 10.0) & ~channel, c * 1.6, c)
    elif kind == "signal":                                       # the left half and above the floor
        c = np.where((Y < -10.0) & ~channel, c * 1.6, c)
        c = np.where(channel & (Z < -30.0), c * 1.4, c)
    return c


@cache
def routes(spec):
    """spec: tuple of (name, kind, a (x,y,z), b (x,y,z)). Returns {name: [points mm]} (straight segments)."""
    free, P, _, channel = grid()
    costs = {k: side_cost(k) for k in ("power", "signal", "servo")}
    out = {}
    for name, kind, a, b in spec:
        c = costs[kind]
        path = astar(a, b, c)
        if path is None:
            out[name] = None
            continue
        for cell in path:                                         # the next cable of this kind bundles with it
            c[cell] *= 0.8
        tight = corners(path)
        pts = [np.asarray(a, float)] + [P[cell] for cell in tight] + [np.asarray(b, float)]
        out[name] = pts
    return out


def tie_points(paths, every=40.0, near=20.0):
    """Where a cable is tied: within `near` mm of each end and every `every` mm along its run in a channel. Returns
    [(point mm, run direction unit vector)], merged within 12 mm."""
    channel = grid()[3]
    pts = []
    for p in paths:
        if p is None:
            continue
        p = np.asarray(p, float)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        total = seg.sum()
        if total < 2 * near:
            continue
        marks = [near] + list(np.arange(near + every, total - near, every)) + [total - near]
        acc = np.concatenate([[0.0], np.cumsum(seg)])
        for m in marks:
            k = min(int(np.searchsorted(acc, m)) - 1, len(seg) - 1)
            k = max(k, 0)
            t = (m - acc[k]) / max(seg[k], 1e-9)
            q = p[k] + (p[k + 1] - p[k]) * t
            if not channel[_cell(q)]:
                continue
            d = (p[k + 1] - p[k]) / max(seg[k], 1e-9)
            if all(np.linalg.norm(q - e) > 12.0 for e, _ in pts):
                pts.append((q, d))
    return pts

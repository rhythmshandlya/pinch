"""Does everything fit? The fit check of the v2 body (configs/layout_v2.toml).

    python sim/fit_check.py            # prints the table and every failure, exit code 1 if anything fails
    python sim/fit_check.py --top      # also writes docs/images/layout_v2_top.png (a top view; needs matplotlib)

It fails on:
  outside     a part (or its cable room) is not inside the chassis interior (walls and the dome's inner surface)
  overlap     two parts touch or overlap (a part mounted `on` another one may touch it, never go through it)
  clearance   two parts are closer than rules.clearance_mm
  cable room  a connector has no free slab of reach_mm in front of it (a part, or the wall, is in the way)
  window      a sensor window is not at the shell, or something stands in the cone it looks through

Boxes are exact (oriented boxes, separating axis test). The clearance test grows one box by the clearance, which is a
little stricter than a round clearance at box edges and never less strict. Pure numpy: it runs in the plain
environment and in tests."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.electronics import Layout  # noqa: E402

# -- geometry ------------------------------------------------------------------------------------------------------

def _axes(R):
    return [R[:, i] for i in range(3)]


def obb_gap(c1, h1, R1, c2, h2, R2):
    """A lower bound on the distance between two oriented boxes by the separating axis test (negative: they
    overlap by at least that much). Exact for the faces' normals, a bound for edge-edge cases."""
    A, B = _axes(R1), _axes(R2)
    t = np.asarray(c2, float) - np.asarray(c1, float)
    best = -1e9
    axes = list(A) + list(B)
    for a in A:
        for b in B:
            n = np.cross(a, b)
            if np.linalg.norm(n) > 1e-9:
                axes.append(n / np.linalg.norm(n))
    for n in axes:
        r1 = sum(h1[i] * abs(n @ A[i]) for i in range(3))
        r2 = sum(h2[i] * abs(n @ B[i]) for i in range(3))
        best = max(best, abs(t @ n) - r1 - r2)
    return float(best)


def ray_box(o, d, c, h, R):
    """Distance along the ray o + s d (unit d) to an oriented box, or None."""
    lo = R.T @ (np.asarray(o, float) - c)
    ld = R.T @ d
    s0, s1 = 0.0, 1e9
    for i in range(3):
        if abs(ld[i]) < 1e-12:
            if abs(lo[i]) > h[i]:
                return None
            continue
        a, b = (-h[i] - lo[i]) / ld[i], (h[i] - lo[i]) / ld[i]
        if a > b:
            a, b = b, a
        s0, s1 = max(s0, a), min(s1, b)
        if s0 > s1:
            return None
    return s0 if s0 >= 0 else (s1 if s1 >= 0 else None)


class Interior:
    """The inside of the chassis: the box and the dome, each shrunk by the wall (and by a margin)."""

    def __init__(self, ch):
        self.bc = np.asarray(ch["box_center_mm"], float)
        self.bh = np.asarray(ch["box_half_mm"], float)
        self.dc = np.asarray(ch["dome_center_mm"], float)
        self.dh = np.asarray(ch["dome_half_mm"], float)
        self.wall = float(ch["wall_mm"])

    def inside(self, p, margin=0.0):
        p = np.asarray(p, float)
        m = self.wall + margin
        if np.all(np.abs(p - self.bc) <= self.bh - m):
            return True
        q = (p - self.dc) / (self.dh - m)
        return bool(q @ q <= 1.0)

    def depth(self, p):
        """How far a point is below the outer surface (mm; negative outside). Union of box and dome."""
        p = np.asarray(p, float)
        db = float(np.min(self.bh - np.abs(p - self.bc)))
        q = float(np.sqrt((((p - self.dc) / self.dh) ** 2).sum()))
        dd = (1.0 - q) * float(self.dh.min()) if q <= 1.5 else -1e3
        return max(db, dd)

    def box_inside(self, c, h, R, margin=0.0, n=4):
        """All of an oriented box (its corners and a grid of n points a side) is inside; returns the worst point."""
        g = np.linspace(-1, 1, n)
        pts = np.array([[a, b, d] for a in g for b in g for d in g]) * h
        w = np.asarray(c, float) + pts @ R.T
        for p in w:
            if not self.inside(p, margin):
                return False, p
        return True, None


# -- the check -----------------------------------------------------------------------------------------------------

class Report:
    def __init__(self):
        self.failures = []
        self.rows = []

    def fail(self, kind, text):
        self.failures.append((kind, text))

    @property
    def ok(self):
        return not self.failures

    def kinds(self):
        return {k for k, _ in self.failures}


def _family(layout, a, b):
    """a and b are the same assembly: one is mounted on the other (or both on the same part)."""
    return a.on == b.name or b.on == a.name or (a.on is not None and a.on == b.on)


def check(layout=None, report=None):
    L = layout if layout is not None else Layout()
    rep = report if report is not None else Report()
    inter = Interior(L.chassis)
    clr = float(L.rules["clearance_mm"])
    wall_tol = float(L.rules.get("window_depth_tol_mm", 2.5))
    parts = [p for p in L.solid()]

    # 1. inside the chassis
    for p in parts:
        if p.d.get("outside_ok"):
            continue                                         # by design partly outside (the arm pivots)
        m = -inter.wall - float(p.d.get("proud_mm", 0.3)) if p.d.get("wall_mounted") else (0.0 if p.on else clr)
        ok, bad = inter.box_inside(p.pos, p.half, p.R, margin=m)
        if not ok:
            rep.fail("outside", f"{p.name} sticks out of the chassis interior near {np.round(bad, 1).tolist()} mm")

    # 2. parts against parts
    min_gap = (1e9, None)
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            if a.in_model and b.in_model:
                continue                                    # both are servos the model builds itself
            fam = _family(L, a, b)
            g = obb_gap(a.pos, a.half, a.R, b.pos, b.half, b.R)
            if g < -1e-6:
                rep.fail("overlap", f"{a.name} and {b.name} overlap by {-g:.1f} mm")
            elif g < (0.0 if fam else clr) - 1e-6:
                rep.fail("clearance", f"{a.name} and {b.name} are {g:.1f} mm apart (need {clr:.1f} mm)")
            if not fam and g < min_gap[0]:
                min_gap = (g, f"{a.name} / {b.name}")
    rep.min_gap = min_gap

    # 3. cable room in front of every connector
    for p in parts:
        for c in p.connectors:
            cc, ch, cR = p.connector_box(c)
            label = f"{p.name}.{c['name']}"
            rm = -inter.wall if p.d.get("wall_mounted") else 0.0         # a wall-mounted part may reach into its pocket
            ok, bad = inter.box_inside(cc, ch, cR, margin=rm)
            if not ok:
                rep.fail("cable room", f"{label}: the {c['reach_mm']:.0f} mm of cable room leaves the chassis near "
                                       f"{np.round(bad, 1).tolist()} mm")
            for q in parts:
                if q is p or _family(L, p, q) and q.on == p.name:
                    continue
                g = obb_gap(cc, ch, cR, q.pos, q.half, q.R)
                if g < -1e-6 and not (q.on == p.name):
                    rep.fail("cable room", f"{label}: {q.name} is inside its {c['reach_mm']:.0f} mm of cable room "
                                           f"({-g:.1f} mm)")

    # 4. windows: at the shell, and a clear cone
    for p in parts:
        if not p.window:
            continue
        w = p.window
        pos, d = p.window_pose()
        label = f"{p.name} window"
        dep = inter.depth(pos)
        deep = float(w.get("max_depth_mm", inter.wall + wall_tol))
        if not (-wall_tol <= dep <= deep):
            rep.fail("window", f"{label} is {dep:.1f} mm below the outer surface (must be within "
                               f"{-wall_tol:.1f}..{inter.wall + wall_tol:.1f} mm)")
        half = math.radians(float(w.get("cone_deg", 12.5)))
        up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0, 0])
        u = np.cross(d, up)
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        rays = [d]
        for k in range(16):
            ang = 2 * math.pi * k / 16
            rays.append(math.cos(half) * d + math.sin(half) * (math.cos(ang) * u + math.sin(ang) * v))
        for r in rays:
            r = r / np.linalg.norm(r)
            o = pos + 0.05 * r
            for q in parts:
                if q is p or _family(L, p, q):
                    continue
                s = ray_box(o, r, q.pos, q.half, q.R)
                if s is not None and s < float(w.get("clear_mm", 400.0)):
                    rep.fail("window", f"{label}: {q.name} is in the way of its view, {s:.0f} mm out")
                    break
            else:
                continue
            break
    return rep


def table(layout=None):
    L = layout if layout is not None else Layout()
    rows = [f"{'part':22s} {'size mm':>17s} {'g':>6s} {'position x, y, z mm':>22s}  {'rpy deg':>14s}  size/mass source"]
    for n, p in L.parts.items():
        sz = "-" if p.mass_only else "x".join(f"{v:g}" for v in p.size)
        rows.append(f"{n:22s} {sz:>17s} {p.mass_g:6.1f} {p.pos[0]:7.1f},{p.pos[1]:7.1f},{p.pos[2]:7.1f}  "
                    f"{str(tuple(round(a) for a in p.rpy)):>14s}  {p.d.get('size_source', '-')}/{p.d.get('mass_source')}")
    ch = L.chassis
    rows.append(f"chassis box {2 * np.asarray(ch['box_half_mm'])} mm, dome {2 * np.asarray(ch['dome_half_mm'])} mm, "
                f"wall {ch['wall_mm']} mm; parts {L.total_mass_g():.0f} g")
    return "\n".join(rows)


def top_view(layout, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    ch = layout.chassis
    bc, bh = np.asarray(ch["box_center_mm"]), np.asarray(ch["box_half_mm"])
    for ax, (ia, ib, title) in zip(axs, ((0, 1, "top (x right, y up)"), (0, 2, "side (x right, z up)"))):
        ax.add_patch(Rectangle((bc[ia] - bh[ia], bc[ib] - bh[ib]), 2 * bh[ia], 2 * bh[ib], fill=False, ec="k", lw=2))
        for n, p in layout.parts.items():
            if p.mass_only:
                continue
            c = p.corners()
            pts = c[:, [ia, ib]]
            from scipy.spatial import ConvexHull  # noqa: PLC0415
            hull = ConvexHull(pts)
            ax.add_patch(Polygon(pts[hull.vertices], alpha=0.35, ec="b"))
            ax.text(p.pos[ia], p.pos[ib], n, fontsize=6, ha="center")
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_title(title)
    fig.savefig(out, dpi=110)


def main():
    L = Layout()
    print(table(L))
    rep = check(L)
    print(f"\nclosest pair (not mounted on each other): {rep.min_gap[1]} at {rep.min_gap[0]:.1f} mm "
          f"(rule {L.rules['clearance_mm']} mm)")
    if rep.failures:
        print(f"\nFIT CHECK FAILED ({len(rep.failures)}):")
        for k, t in rep.failures:
            print(f"  [{k}] {t}")
    else:
        print("\nFIT CHECK PASSED: every part is inside, nothing overlaps, every connector has its cable room, "
              "every window sees out.")
    if "--top" in sys.argv:
        top_view(L, ROOT / "docs" / "images" / "layout_v2_top.png")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())

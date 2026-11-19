"""Minimum wall thickness check for printed parts (robu.in rule: walls >= 1 mm, we use >= 1.6 mm).

From sample points on every face, a ray is cast straight into the material along
the inward normal; the distance to where it leaves the solid is the local wall
thickness there. This also catches thin webs between a hole and an edge.

    python cad/walls.py                # all parts
    python cad/walls.py coxa_link      # one part
"""

import sys
from pathlib import Path

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Dir, gp_Lin, gp_Pnt, gp_Pnt2d
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_REVERSED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_WALL = 1.6     # mm


def thickness_samples(solid, n=7):
    """[(thickness_mm, (x, y, z), face_index)] sampled over all faces."""
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(solid.wrapped, 1e-6)
    out = []
    for fi, face in enumerate(solid.faces()):
        f = face.wrapped
        umin, umax, vmin, vmax = BRepTools.UVBounds_s(f)
        surf = BRepAdaptor_Surface(f)
        for i in range(n):
            for j in range(n):
                u = umin + (i + 0.5) / n * (umax - umin)
                v = vmin + (j + 0.5) / n * (vmax - vmin)
                st = BRepClass_FaceClassifier(f, gp_Pnt2d(u, v), 1e-7).State()
                if st not in (TopAbs_IN, TopAbs_ON):
                    continue
                props = BRepLProp_SLProps(surf, u, v, 1, 1e-7)
                if not props.IsNormalDefined():
                    continue
                p, nrm = props.Value(), props.Normal()
                if f.Orientation() == TopAbs_REVERSED:
                    nrm.Reverse()
                d = gp_Dir(-nrm.X(), -nrm.Y(), -nrm.Z())           # into the material
                start = gp_Pnt(p.X() + d.X() * 1e-4, p.Y() + d.Y() * 1e-4, p.Z() + d.Z() * 1e-4)
                inter.Perform(gp_Lin(start, d), 0.0, 1e4)
                hits = [inter.WParameter(k) for k in range(1, inter.NbPnt() + 1) if inter.WParameter(k) > 1e-3]
                if hits:
                    out.append((min(hits), (p.X(), p.Y(), p.Z()), fi))
    return out


def check_part(name, solid, verbose=True):
    """A face fails if its *median* thickness is under MIN_WALL: a genuinely thin web is thin
    across its whole face, while a few low readings right at a knife edge (a 2 mm rim meeting a
    sloped dome) are a sampling artefact, not a thin wall."""
    s = thickness_samples(solid)
    t = np.array([x[0] for x in s])
    by_face = {}
    for th, pt, fi in s:
        by_face.setdefault(fi, []).append((th, pt))
    thin = []
    for fi, items in by_face.items():
        med = float(np.median([th for th, _ in items]))
        if med < MIN_WALL:
            thin += [(med, min(items)[1])]
    ok = not thin
    thinnest = float(min(np.median([th for th, _ in v]) for v in by_face.values()))
    if verbose:
        print(f"  {name:11s} {'ok ' if ok else 'THIN'} thinnest face (median) "
              f"{min(np.median([th for th, _ in v]) for v in by_face.values()):5.2f} mm   "
              f"raw min {t.min():5.2f} mm   ({len(t)} samples, {len(by_face)} faces, {len(thin)} thin faces)")
        seen = []
        for th, pt in sorted(thin)[:40]:
            if all(np.linalg.norm(np.subtract(pt, q)) > 3 for q in seen):
                seen.append(pt)
                print(f"      {th:4.2f} mm at ({pt[0]:6.1f}, {pt[1]:6.1f}, {pt[2]:6.1f})")
            if len(seen) >= 6:
                break
    return ok, thinnest


def main():
    from cad import assembly as A
    names = sys.argv[1:] or list(A.PRINTED)
    print(f"== wall thickness (>= {MIN_WALL} mm) ==")
    ok = True
    for n in names:
        ok &= check_part(n, A.part(n))[0]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

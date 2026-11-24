"""Export the v2 CAD to the simulator: meshes, masses, centres of mass and inertias per simulator body.

    python cad/v2/export.py                  # then: python sim/build_model.py

  sim/models/meshes/v2/<body>__<part>.stl   every part in its simulator body's frame (mm), for looks
  sim/models/v2_cad.json                    per body: mass, com, full inertia, the meshes with their materials, and
                                            collision boxes (each part's box in the body frame); the face screen's pose

Masses: printed parts from their solids (PETG/TPU density x fill, cad/v2/assembly.py), servos their real masses,
electronics the layout's (configs/layout_v2.toml), the face module's mass on the panel solid, the wiring harness at
the layout's point. Inertias assume each part is uniform (the STS3215 by a solid of its outside sizes: its STEP does
not give clean volumes). This is the Microduck / Open Duck Mini pattern (onshape-to-robot): the robot is drawn first,
the simulator model is generated from the drawing, never edited by hand.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from build123d import Box, Pos, export_stl  # noqa: E402

from cad.v2 import assembly as A  # noqa: E402
from cad.v2 import legs, lid  # noqa: E402

MESH_DIR = ROOT / "sim" / "models" / "meshes" / "v2"
OUT = ROOT / "sim" / "models" / "v2_cad.json"
MATERIAL = {"shell": "shell", "printed": "printed", "servo": "servo", "pcb": "pcb", "dark": "servo", "screen": "servo",
            "wire_power": "wire_power", "wire_servo": "wire_servo", "wire_signal": "wire_signal",
            "tip": "tip", "metal": "horn", "battery": "battery", "white": "connector"}


def inertia_about_com(solid, mass):
    """Inertia tensor (kg m^2) about the solid's centroid for a uniform part, and the centroid (m)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid.wrapped, props)
    vol = props.Mass()
    c = props.CentreOfMass()
    m = props.MatrixOfInertia()
    I = np.array([[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])
    return I * (mass / vol) * 1e-6, np.array([c.X(), c.Y(), c.Z()]) * 1e-3


def extra_masses():
    """Masses the solids do not carry: the face module (on the panel) and the wiring harness (a point)."""
    from crab.electronics import Layout
    lay = Layout()
    out = {"panel": lay["oled"].mass_g / 1000.0}
    # the harness allowance is replaced by the modelled cables (cad/v2/wiring.py); servo cable beyond the body sits on
    # the links it runs along
    point = {"body": [(p.name, p.mass_g / 1000.0, np.array(p.pos) / 1000.0) for p in lay.parts.values()
                      if p.mass_only and p.name != "harness"]}
    from cad.v2 import wiring
    for body, items in wiring.rest_masses().items():
        point.setdefault(body, []).extend(("cable_rest", m, np.array(c) / 1000.0) for m, c in items)
    return out, point


def combine(comps):
    """[(mass, I about own com, com)] -> mass, com, I about the common com."""
    M = sum(c[0] for c in comps)
    com = sum(c[0] * c[2] for c in comps) / M
    I = np.zeros((3, 3))
    for m, Ic, ci in comps:
        d = ci - com
        I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return M, com, I


PART_MASSES = []          # (simulator body, part, kg): every mass in the model, for the weight report


def mass_properties():
    on_solid, points = extra_masses()
    out = {}
    PART_MASSES.clear()
    for body, items in A.groups(proxy=True).items():
        comps = []
        for label, solid, mass, _ in items:
            mass = mass + on_solid.get(label, 0.0)
            if mass > 0:
                comps.append((mass, *inertia_about_com(solid, mass)))
                PART_MASSES.append((body, label, float(mass)))
        PART_MASSES.extend((body, name, float(m)) for name, m, _ in points.get(body, []))
        comps += [(m, np.zeros((3, 3)), c) for _, m, c in points.get(body, [])]
        M, com, I = combine(comps)
        out[body] = dict(mass=float(M), com=com.tolist(), inertia=[I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])
    return out


def collision_boxes():
    """Each part's box (centre, half sizes; m) in its body's frame, for floor and object contact. The arms keep the
    simulator's own jaw and pad boxes (the claw CAD is built from them); the body collides with its shell."""
    out = {}
    for body, items in A.groups(proxy=True).items():
        if body == "body" or "_" in body and body.split("_", 1)[1] in ("upper", "dactyl"):
            continue
        boxes = []
        for label, solid, _, _ in items:
            if label == "foot":
                continue                                        # the foot is a sphere in the model
            if label == "tibia":                                # the hub and yoke as a box, the shin as a capsule
                x1 = legs.YOKE_X + 3.0
                solid = solid & (Pos(x1 / 2 - 50, 0, 0) * Box(x1 + 100, 200, 200))
                y0 = sum(legs.PLATE_Y) / 2 * (-1 if body[1] == "R" else 1)      # right legs are mirror images
                S = legs.SHIN_SECTIONS
                for k, (sa, sb) in enumerate(((S[0], S[2]), (S[2], S[4]))):      # two capsules follow the taper
                    r = (min(sa[1], sa[2]) + min(sb[1], sb[2])) / 2
                    boxes.append(dict(name=f"shin{k}", type="capsule", r=r / 1000,
                                      fromto=[max(sa[0], x1) / 1000, y0 / 1000, sa[3] / 1000, sb[0] / 1000, y0 / 1000, sb[3] / 1000]))
            bb = solid.bounding_box()
            lo, hi = np.array([bb.min.X, bb.min.Y, bb.min.Z]), np.array([bb.max.X, bb.max.Y, bb.max.Z])
            boxes.append(dict(name=label, pos=((lo + hi) / 2000).tolist(), size=((hi - lo) / 2000).tolist()))
        out[body] = boxes
    return out


def claw_meet(arm="L", tol=0.004):
    """The claw angle (rad) at which the two jaws' pads meet: the claw's mechanical stop (bisection on the CAD)."""
    from cad.v2.sweep_check import Poser
    poser = Poser()
    g = A.groups(proxy=True)
    up, dac = f"{arm}_upper", f"{arm}_dactyl"

    def touching(q):
        P = poser.poses(f"{arm}_claw", q)
        d = P[dac] * g[dac][0][1]
        for _, s, _, _ in g[up]:
            r = (P[up] * s) & d                          # (None when the two don't meet)
            if r is not None and r.volume >= 1.0:
                return True
        return False
    lo, hi = 0.0, 0.3
    if not touching(lo):
        return 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if touching(mid) else (lo, mid)
    return hi


def face_pose():
    """The lit area of the face screen in the lid frame (m): up the lid from the hinge line, out of the lid's back."""
    return dict(up=lid.PANEL_Z / 1000.0, out=(lid.LID_T + 0.5) / 1000.0, size=[lid.LIT_W / 1000.0, lid.LIT_H / 1000.0])


def dump_features(out=ROOT / ".ops" / "cad_parts"):
    """Save the body's built features (BRep + manifest.json) so cad/v2/bisect_shell.py can find a failing one."""
    from build123d import export_brep

    from cad.v2 import body as B
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, f in enumerate(B.features()):
        if f.solid is None or getattr(f.solid, "_wrapped", None) is None:
            continue
        fn = f"f{k}.brep"
        export_brep(f.solid, str(out / fn))
        rows.append((f.name, f.owner, f.half, f.op, f.clip, f.trimmed, fn))
    (out / "manifest.json").write_text(json.dumps({"features": rows}))
    print(f"features saved to {out} ({len(rows)})", file=sys.stderr)


def export_meshes():
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    vis, written = {}, set()
    for body, items in A.groups().items():
        vis[body] = []
        for label, solid, _, mat in items:
            name = re.sub(r"[^a-zA-Z0-9]+", "_", f"{body}__{label}").strip("_")
            fn = MESH_DIR / f"{name}.stl"
            tol = 0.25 if mat == "servo" else 0.15
            if solid is None or getattr(solid, "_wrapped", None) is None:
                if label in ("tray", "cover"):
                    dump_features()
                raise ValueError(f"{body}/{label}: the CAD produced an empty solid (a boolean failed upstream); "
                                 f"its features are in .ops/cad_parts: python cad/v2/bisect_shell.py --half {label}")
            write_stl(solid, fn, tol)
            written.add(fn.name)
            vis[body].append(dict(mesh=f"v2_{name}", file=f"v2/{name}.stl", material=MATERIAL[mat]))
    return vis, written


MAX_STL_BYTES = 4_000_000     # the repository refuses files over 5 MB; a maker's board model can mesh finer than that


def write_stl(solid, fn, tol, angular=0.35):
    """Mesh at tol; a very detailed part (a maker's STEP with every SMD part) is re-meshed coarser until it fits."""
    export_stl(solid, str(fn), tolerance=tol, angular_tolerance=angular)
    while fn.stat().st_size > MAX_STL_BYTES and tol < 2.0:
        tol, angular = tol * 2, min(angular * 1.6, 1.0)
        export_stl(solid, str(fn), tolerance=tol, angular_tolerance=angular)


def main():
    meet = claw_meet()                      # (reads the current model: before its meshes change)
    vis, written = export_meshes()
    mp = mass_properties()
    col = collision_boxes()
    out = {"bodies": {b: dict(**mp[b], visuals=vis[b], collision=col.get(b, [])) for b in mp}, "face": face_pose(),
           "lid_h": lid.LID_H / 1000.0, "lid_w": lid.LID_W / 1000.0, "claw_meet": meet,
           "foot": dict(x=legs.TIP_X / 1000.0, r=legs.TIP_R / 1000.0), "parts": PART_MASSES}
    print(f"the claw's jaws meet at {meet:.3f} rad")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for f in MESH_DIR.glob("*.stl"):         # meshes of parts that no longer exist
        if f.name not in written:
            f.unlink()
    total = sum(v["mass"] for v in mp.values())
    size = sum(f.stat().st_size for f in MESH_DIR.glob("*.stl")) / 1e6
    print(f"meshes -> {MESH_DIR.relative_to(ROOT)} ({sum(len(v) for v in vis.values())} files, {size:.1f} MB)")
    print(f"masses -> {OUT.relative_to(ROOT)}: robot {total * 1000:.0f} g")
    for b, v in mp.items():
        print(f"  {b:16s} {v['mass'] * 1000:7.1f} g  com {np.round(np.array(v['com']) * 1000, 1)} mm")


if __name__ == "__main__":
    main()

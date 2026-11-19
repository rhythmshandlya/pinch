"""Export everything the CAD produces.

    python cad/build.py

  cad/out/print/<part>.stl + .step   files for the robu.in print service (one per unique part)
  cad/out/print_sheet.csv            part, qty, material, infill, size, est. mass
  sim/models/meshes/*.stl            per-component meshes in each simulator body's frame (mm)
  sim/models/mass_props.json         mass / centre of mass / inertia per simulator body
  sim/models/visuals.json            which mesh goes on which body, with a material
  sim/models/collision.json          simple boxes for the body's floor contact + the eye camera pose

Then run `python sim/build_model.py` (it picks these up automatically).
"""

import csv
import json
import re
import sys
from pathlib import Path

from build123d import export_step, export_stl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cad import assembly as A  # noqa: E402

OUT = ROOT / "cad" / "out"
PRINT = OUT / "print"
MESH_DIR = ROOT / "sim" / "models" / "meshes"

INFILL = {"chassis": "40% gyroid, 3 walls", "top_shell": "solid (2 mm wall), 3 walls",
          "foot_tip": "TPU 95A, 50% gyroid", "jaw_pad": "TPU 95A, solid", "default": "50% gyroid, 4 walls"}
ORIENT = {"chassis": "floor down (supports inside the camera pocket)",
          "top_shell": "rim down (supports under the eye stalks and the speaker ring)",
          "belly_door": "flat", "pi_tray": "flat, stand-offs up",
          "coxa_link": "cradle floor down", "femur_link": "thigh plate down",
          "tibia": "horn plate down", "foot_tip": "sphere up", "shoulder": "spine side down",
          "upper_arm": "arm plate down (supports under the palm)",
          "dactyl_L": "outer face down (supports under the hub)", "dactyl_R": "outer face down (supports under the hub)",
          "jaw_pad": "flat"}


def material_of(label):
    if "servo" in label:
        return "servo"
    if "horn" in label:
        return "horn"
    if "foot" in label or "pad" in label:
        return "tip"
    if label in ("top_shell",):
        return "shell"
    if label in ("pi_tray",):
        return "pcb"
    if "dactyl" in label or "upper_arm" in label:
        return "claw"
    return "printed"


def main():
    PRINT.mkdir(parents=True, exist_ok=True)
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    # 1. print files + sheet
    rows = []
    for name, (_, rho, fill, mat, qty) in A.PRINTED.items():
        p = A.part(name)
        export_stl(p, str(PRINT / f"{name}.stl"), tolerance=0.02, angular_tolerance=0.1)
        export_step(p, str(PRINT / f"{name}.step"))
        bb = p.bounding_box()
        rows.append(dict(part=name, qty=qty, material=mat, infill=INFILL.get(name, INFILL["default"]),
                         orientation=ORIENT.get(name, ""),
                         size_mm=f"{bb.size.X:.0f} x {bb.size.Y:.0f} x {bb.size.Z:.0f}",
                         est_mass_g=round(A.part_mass(name) * 1000, 1),
                         volume_cm3=round(p.volume / 1000, 1)))
    with open(OUT / "print_sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"print files -> {PRINT.relative_to(ROOT)} ({len(rows)} unique parts, "
          f"{sum(r['qty'] for r in rows)} pieces)")

    # 2. simulator meshes (coarser tessellation is plenty for visuals)
    visuals = {}
    for body, items in A.groups().items():
        visuals[body] = []
        for label, solid, _ in items:
            fname = re.sub(r"[^a-zA-Z0-9]+", "_", f"{body}__{label}").strip("_") + ".stl"
            export_stl(solid, str(MESH_DIR / fname), tolerance=0.15, angular_tolerance=0.3)
            visuals[body].append(dict(mesh=fname[:-4], file=fname, material=material_of(label)))
    (ROOT / "sim" / "models" / "visuals.json").write_text(json.dumps(visuals, indent=1))
    print(f"sim meshes  -> {MESH_DIR.relative_to(ROOT)} ({sum(len(v) for v in visuals.values())} files)")

    # 3. collision boxes for the body (floor contact only; body geoms don't collide with the legs)
    from cad import parts as P
    o = P.body_outline().buffer(-2.0)
    x0, y0, x1, y1 = o.bounds
    zf, zr = P.Z_FLOOR[0], P.Z_RIM
    boxes = [dict(name="chassis_box", pos=[(x0 + x1) / 2e3 - 0.004, 0, (zf + zr) / 2e3],
                  size=[(x1 - x0) / 2e3 - 0.004, 0.030, (zr - zf) / 2e3]),
             dict(name="dome_box", pos=[(x0 + x1) / 2e3 - 0.004, 0, (zr + 24.0) / 2e3],
                  size=[(x1 - x0) / 2e3 - 0.010, 0.040, (24.0 - zr) / 2e3])]
    xf = P.body_outline().bounds[2]                    # camera sits in the chassis front wall
    # lens flush with the front face; OV5647 standard lens: 53.5 x 41.4 deg field of view
    col = dict(body=boxes, eye_cam=[(xf + 0.3) / 1e3, 0.0, P.CAMERA_Z / 1e3], eye_fovy=41.4)
    (ROOT / "sim" / "models" / "collision.json").write_text(json.dumps(col, indent=1))
    print("collision   -> sim/models/collision.json")

    # 4. mass properties
    mp = A.mass_properties()
    (ROOT / "sim" / "models" / "mass_props.json").write_text(json.dumps(mp, indent=1))
    print(f"mass props  -> sim/models/mass_props.json (robot total {sum(v['mass'] for v in mp.values()):.3f} kg)")


if __name__ == "__main__":
    main()

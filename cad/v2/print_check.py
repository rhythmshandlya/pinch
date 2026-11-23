"""Printability of every printed part: for each, the flat orientation that needs the least support, how much area
still overhangs past 45 deg there, and whether it fits the bed. Reads the meshes cad/v2/export.py wrote.

    python cad/v2/print_check.py            # table; exit 1 if a part does not fit the bed

Overhang: a downward-facing triangle steeper than 45 deg from vertical that is not on the bed needs support (Hubs and
Prusa design rules: overhangs <= 45 deg print cleanly; bridges < 5 mm are fine, so small faces are ignored).
"""

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MESHES = ROOT / "sim" / "models" / "meshes" / "v2"
BED_MM = 220.0
# mesh file stem -> (material, infill %, walls). Load-carrying links print denser: SPINDER (Maoudj et al., Sci. Rep.
# 16:26147, 2026) measured 0.3-0.5 mm tip deflection on 40 % infill printed femurs (1-2 mm foot error) and 40 % less
# at 60 % for +3 g a leg. Gyroid infill; 0.4 mm nozzle, so 4 walls = 1.6-1.8 mm of solid shell.
PRINTED = {
    "body_tray": ("PETG", 30, 3), "body_cover": ("PETG", 20, 3), "body_battery_door": ("PETG", 30, 3),
    "body_lid": ("PETG", 20, 3), "body_lid_back": ("PETG", 20, 3), "body_deck": ("PETG", 30, 3),
    "FL_coxa_link_coxa": ("PETG", 60, 4), "FL_femur_link_femur": ("PETG", 60, 4),
    "FL_tibia_link_tibia": ("PETG", 60, 4), "FL_tibia_link_foot": ("TPU 95A", 40, 3),
    "L_shoulder_shoulder": ("PETG", 60, 4), "L_upper_upper": ("PETG", 60, 4), "L_dactyl_dactyl": ("PETG", 60, 4),
}
MIN_FACE_MM2 = 4.0           # a face smaller than this bridges


def read_stl(path):
    data = path.read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = [list(map(float, ln.split()[1:4])) for ln in data.decode(errors="ignore").splitlines() if ln.strip().startswith("vertex")]
        return np.array(v).reshape(-1, 3, 3)
    n = struct.unpack("<I", data[80:84])[0]
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].astype(float)


def assess(tri, up):
    """(overhang area mm2, height mm) with `up` as the print's +z."""
    up = np.asarray(up, float)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    n = np.cross(e1, e2)
    a = np.linalg.norm(n, axis=1) / 2
    n = n / np.maximum(2 * a[:, None], 1e-12)
    z = tri @ up
    zmin = z.min()
    on_bed = z.max(axis=1) < zmin + 0.3
    down = (n @ up) < -np.cos(np.radians(45.0))
    over = down & ~on_bed & (a > MIN_FACE_MM2 / 50.0)
    return float(a[over].sum()), float(z.max() - zmin)


def main():
    rows, bad = [], 0
    for stem, (mat, infill, walls) in PRINTED.items():
        f = MESHES / f"{stem}.stl"
        mat = f"{mat} {infill}%/{walls}w"
        if not f.exists():
            rows.append((stem, mat, "missing", "", "", ""))
            bad += 1
            continue
        tri = read_stl(f)
        size = tri.reshape(-1, 3).max(0) - tri.reshape(-1, 3).min(0)
        best = min(((assess(tri, u), u) for u in ([0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])),
                   key=lambda r: r[0][0])
        (over, h), u = best
        fits = bool(np.all(np.sort(size)[:2] <= BED_MM))
        bad += not fits
        label = stem[len("body_"):] if stem.startswith("body_") else stem.rsplit("_", 1)[-1]
        rows.append((label, mat, f"{'+-'[min(u) < 0]}{'xyz'[int(np.argmax(np.abs(u)))]} up",
                     f"{over / 100:.1f} cm2", f"{h:.0f} mm", "fits" if fits else "TOO BIG"))
    w = max(len(r[0]) for r in rows)
    print(f"{'part':{w}s}  material, infill/walls  orientation  support area  height  bed {BED_MM:.0f} mm")
    for r in rows:
        print(f"{r[0]:{w}s}  {r[1]:21s}  {r[2]:11s}  {r[3]:>12s}  {r[4]:>6s}  {r[5]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

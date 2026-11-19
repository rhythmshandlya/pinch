"""Mechanical checks on the CAD:

  1. print rules: every part is one valid solid and fits the robu.in build volume
  2. servo fit: no servo case or horn overlaps its own bracket (stand pose)
  3. self-collision across real poses: stand, every emote keyframe, scripted gait
     samples, and each joint at the ends of its range

    python cad/check.py            # full sweep (~10-20 min on a laptop)
    python cad/check.py --quick    # stand + emotes only
Writes cad/out/check_report.txt.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cad import assembly as A  # noqa: E402
from cad.walls import MIN_WALL, check_part as wall_check  # noqa: E402
THIN_BY_DESIGN = {"desk_tray": 0.45,   # 1 mm floor + a ramp that tapers to the desk: printed flat, fine
                  "jaw_pad": 1.0,      # a 1.2 mm TPU grip pad glued on a jaw blade (not a wall)
                  "top_shell": 1.05}   # the face visor's 4 spacer tubes: 1.1 mm walls around M2 screws (compressed)
from crab.emotes import EMOTES, STAND  # noqa: E402
from crab.gait import _S, _T, GAITS, Gait  # noqa: E402
from crab.params import JOINT_NAMES, PARAMS  # noqa: E402

BUILD_VOLUME = 350.0
OUT = ROOT / "cad" / "out"


def joint_ranges():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(A.MODEL))
    return {n: tuple(m.jnt_range[m.joint(n).id]) for n in JOINT_NAMES}


def pose_set(quick=False):
    poses = [("stand", STAND)]
    for name, emo in EMOTES.items():
        for k, (t, q) in enumerate(emo.keyframes):
            poses.append((f"emote {name} #{k}", q))
    if quick:
        return poses
    v, w = _S / _T, 1 / _T                                   # speeds scaled to this robot's size
    cmds = [("fwd", (0.08 * v, 0, 0)), ("side", (0, 0.1 * v, 0)), ("turn", (0, 0, 0.6 * w))]
    for gname in GAITS:
        for cname, cmd in cmds:
            g = Gait(GAITS[gname], com_offset=np.array([0.015 * _S, 0.0]))
            dt, T = 0.01, GAITS[gname].period
            qs = [g.step(dt, cmd) for _ in range(int((1.0 + 2 * T) / dt))]
            for k in np.linspace(int(1.0 / dt), len(qs) - 1, 6).astype(int):
                poses.append((f"gait {gname} {cname} t={k * dt:.2f}", qs[k]))
    for n, (lo, hi) in joint_ranges().items():
        if not (n.startswith("FL") or n.startswith("RL") or n.startswith("L_")):
            continue                                          # right side mirrors the left
        for label, v in (("min", lo), ("max", hi)):
            q = STAND.copy()
            q[JOINT_NAMES.index(n)] = v
            poses.append((f"range {n} {label} ({np.degrees(v):.0f} deg)", q))
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    say(f"== 1. print rules (FDM: one valid solid, <= {BUILD_VOLUME:.0f} mm, walls >= {MIN_WALL} mm) ==")
    ok_print = True
    for name in A.PRINTED:
        p = A.part(name)
        bb = p.bounding_box()
        big = max(bb.size.X, bb.size.Y, bb.size.Z)
        one = len(p.solids()) == 1
        walls_ok, thinnest = wall_check(name, p, verbose=False)
        if name in THIN_BY_DESIGN:                     # a flat floor / wedge that is thin on purpose
            walls_ok = thinnest >= THIN_BY_DESIGN[name]
        ok = p.is_valid and one and big <= BUILD_VOLUME and walls_ok
        ok_print &= ok
        say(f"  {name:11s} {'ok ' if ok else 'BAD'} solids={len(p.solids())} valid={p.is_valid} "
            f"size {bb.size.X:.0f}x{bb.size.Y:.0f}x{bb.size.Z:.0f} mm  walls {'ok' if walls_ok else 'THIN'} "
            f"(thinnest {thinnest:.2f})  {A.part_mass(name) * 1000:.1f} g")

    say("\n== 2. servo fit (stand pose, within each body) ==")
    t = time.time()
    fit = [h for h in A.interferences(STAND, within_groups=True) if h[0] == h[2]]
    for h in fit:
        say(f"  OVERLAP {h[0]}: {h[1]} x {h[3]}  {h[4]:.1f} mm3")
    say(f"  {'all servos fit their brackets' if not fit else f'{len(fit)} fit problems'}  ({time.time() - t:.0f}s)")

    say("\n== 3. self-collision across poses ==")
    poses = pose_set(a.quick)
    bad = 0
    t = time.time()
    for k, (label, q) in enumerate(poses):
        hits = A.interferences(q)
        if hits:
            bad += 1
            worst = sorted(hits, key=lambda h: -h[4])[:3]
            say(f"  COLLIDE {label}: " + "; ".join(f"{h[0]}/{h[1]} x {h[2]}/{h[3]} {h[4]:.0f}mm3" for h in worst))
        if (k + 1) % 10 == 0:
            print(f"    ... {k + 1}/{len(poses)} poses ({time.time() - t:.0f}s)", flush=True)
    say(f"  {len(poses) - bad}/{len(poses)} poses collision-free  ({time.time() - t:.0f}s)")

    (OUT / "check_report.txt").write_text("\n".join(lines), encoding="utf-8")
    sys.exit(0 if ok_print and not fit and bad == 0 else 1)


if __name__ == "__main__":
    main()

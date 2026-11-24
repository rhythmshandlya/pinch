"""Does Pinch's face detector find people in its own rendered fisheye view? (phase 5 exit check)

    python sim/benches/faces_rendered.py            # 60 held-out test worlds with someone at the desk
    python sim/benches/faces_rendered.py -n 200 --json out.json

For each held-out test world with a person sitting with Pinch: Pinch turns toward them, its camera's real view is
rendered (sim/worlds/fisheye.py: five cube faces remapped through the 160-degree lens), and the robot's own face
pipeline (pinch/faces.py: YuNet detection, SFace recognition) runs on the frame. Reported: how often a face is
found (with the 95 % range), by distance and by world kind; how often the face found is the right one (its box
contains where the person's face projects through the lens); and whether the same person is recognised again
from a second frame a little later (their head has moved). Faces are synthetic (SFHQ), people are kinematic.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.benches.common import run_parallel, save_json  # noqa: E402
from sim.benches.stats import wilson  # noqa: E402


def check_world(index, split="test", save=None):
    import mujoco

    from pinch.faces import FaceID
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    from sim.worlds import world
    from sim.worlds.live import Life
    w = world(split, index)
    if not any(p["role"] == "sitter" for p in w.people):
        return None
    sim = PinchSim(memory=Memory(path=None), world=w, world_opts=dict(things=True))
    life = Life(sim, camera=True)
    sim.autonomous = False                               # it stays where it is; we turn it toward the person
    life.run(0.6)
    sitter = next(p for p in life.people if p.p["role"] == "sitter")
    f = sitter.face_xyz()
    x, y, _ = sim.pose()
    yaw = float(np.arctan2(f[1] - y, f[0] - x))
    q = sim.d.qpos
    q[3:7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    mujoco.mj_forward(sim.m, sim.d)
    fid = FaceID()
    out = dict(world=w.name, kind=w.kind, dist=float(np.hypot(f[0] - x, f[1] - y)), child=sitter.p["child"])
    frames = []
    for k in range(2):
        if k:
            life.run(1.5)                                 # they move their head and hands
            mujoco.mj_forward(sim.m, sim.d)
        bgr = life.cam.render(sim.d)
        dets = fid.detect(bgr)
        # where the face should be: the face's centre through the lens
        cid = sim.m.camera("eye_cam").id
        R, p = sim.d.cam_xmat[cid].reshape(3, 3), sim.d.cam_xpos[cid]
        def near(dd, uv):
            u, v = uv
            return dd[0] - 0.3 * dd[2] <= u <= dd[0] + 1.3 * dd[2] and dd[1] - 0.3 * dd[3] <= v <= dd[1] + 1.3 * dd[3]
        uvs = {}
        for q in life.people:                             # where each person's face lands through the lens
            if q.pose:
                pc = R.T @ (q.face_xyz() - p)
                uvs[q.i] = (life.cam.project(pc), -pc[2] > 0)
        (u, v), ahead = uvs[sitter.i]
        hit = [dd for dd in dets if near(dd, (u, v))]
        others = [dd for dd in dets if not any(near(dd, uv) for uv, _ in uvs.values())]
        frames.append((bgr, hit, dets, (u, v), ahead and 0 <= u < 640 and 0 <= v < 480, others))
    out["found"] = bool(frames[0][1])
    out["in_view"] = bool(frames[0][4])
    out["false"] = len(frames[0][5])
    out["others_found"] = len(frames[0][2]) - len(frames[0][1]) - len(frames[0][5])
    out["size_px"] = float(frames[0][1][0][3]) if frames[0][1] else None
    out["score"] = float(frames[0][1][0][-1]) if frames[0][1] else None
    if frames[0][1] and frames[1][1]:
        a = fid.fingerprint(frames[0][0], frames[0][1][0])
        b = fid.fingerprint(frames[1][0], frames[1][1][0])
        out["same_person"] = float(a @ b)
    if save:
        import cv2
        img = frames[0][0].copy()
        for d in frames[0][2]:
            cv2.rectangle(img, (int(d[0]), int(d[1])), (int(d[0] + d[2]), int(d[1] + d[3])), (0, 255, 0), 2)
        cv2.imwrite(str(Path(save) / f"face_{w.name}.jpg"), img)
    life.cam.close()
    return out


def _job(i, save=None):
    return (check_world(i, save=save),)                # (None,): no one to look for; None: the worker was lost


def main():
    from pinch.faces import MATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=60, help="test worlds to try (those with a person at Pinch's surface)")
    ap.add_argument("--save", help="a folder for the frames")
    ap.add_argument("--json")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    from functools import partial
    todo, got = list(range(a.n)), {}
    for _ in range(2):                                # a world lost to its worker (out of memory) is tried again
        res = run_parallel(partial(_job, save=a.save), todo, workers=a.workers, fresh_every=8, keep_going=True)
        got.update({i: r[0] for i, r in zip(todo, res) if r is not None})
        todo = [i for i, r in zip(todo, res) if r is None]
        if not todo:
            break
    if todo:
        print(f"{len(todo)} worlds lost twice and left out: {todo}")
    rs_all = [got[i] for i in sorted(got) if got[i] is not None]
    rs = [r for r in rs_all if r["in_view"]]
    k, n = sum(r["found"] for r in rs), len(rs)
    lo, hi = wilson(k, n)
    print(f"faces found in Pinch's fisheye view: {k}/{n} = {k / n:.0%} (95 % range {lo:.0%}-{hi:.0%})")
    print(f"  (+ {len(rs_all) - n} worlds where the face was outside the lens's view from where Pinch stood, e.g. someone "
          f"sitting close on the floor; found anyway in {sum(r['found'] for r in rs_all if not r['in_view'])})")
    print(f"  other people's faces also found: {sum(r['others_found'] for r in rs_all)}")
    for lab, sel in (("< 0.7 m", lambda r: r["dist"] < 0.7), ("0.7-1.0 m", lambda r: 0.7 <= r["dist"] < 1.0),
                     (">= 1.0 m", lambda r: r["dist"] >= 1.0), ("children", lambda r: r["child"])):
        sub = [r for r in rs if sel(r)]
        if sub:
            print(f"  {lab:10s} {sum(r['found'] for r in sub)}/{len(sub)}")
    for kd in ("desk", "table", "floor"):
        sub = [r for r in rs if r["kind"] == kd]
        if sub:
            print(f"  {kd:10s} {sum(r['found'] for r in sub)}/{len(sub)}")
    sizes = [r["size_px"] for r in rs if r["size_px"]]
    if sizes:
        print(f"  face size {np.median(sizes):.0f} px (median), false detections (no face there) {sum(r['false'] for r in rs_all)}")
    same = [r["same_person"] for r in rs if "same_person" in r]
    if same:
        ks = sum(s >= MATCH for s in same)
        print(f"  recognised again 1.5 s later: {ks}/{len(same)} (SFace cosine >= {MATCH}; median {np.median(same):.2f})")
    save_json(a.json, dict(found=k, n=n, lo=lo, hi=hi, out_of_view=len(rs_all) - n, runs=rs_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())

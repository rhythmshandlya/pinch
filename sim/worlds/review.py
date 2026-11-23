"""The owner's review of the worlds: a montage of random worlds with people in them, as video.

    python -m sim.worlds.review --out media/review/phase-05/worlds_montage.webm      # 20 worlds, ~4 s each

Each clip: a camera in the room (placed where no wall is in the way) looking at Pinch and the person nearest it,
Pinch's own fisheye view in the corner (with the faces its detector finds boxed), and a caption saying which
world it is and who is there. Pinch runs its full mind and body; the people move by their style.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

W, H, FPS = 960, 540, 15


def pick_worlds(n, seed=5):
    from sim.worlds import world
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        split = "test" if rng.random() < 0.5 else "train"
        w = world(split, int(rng.integers(200 if split == "test" else 1000)))
        if w.people and w.name not in [x.name for x in out]:
            out.append(w)
    return out


def camera_for(sim, life, rng):
    """A free camera on Pinch and its nearest person, at an angle with no wall between."""
    import mujoco
    x, y, _ = sim.pose()
    crab = np.array([x, y, 0.05])
    near = [p for p in life.people if p.pose]
    if near:
        p = min(near, key=lambda q: np.hypot(*(q.base - crab[:2])))
        tgt = np.array([*p.base, max(0.1, p.targets["pelvis"][2] + 0.4)]) if p.targets else crab
    else:
        tgt = crab + [0.3, 0.0, 0.2]
    look = crab + 0.4 * (tgt - crab)
    look[2] = max(look[2], 0.08)
    base_az = np.degrees(np.arctan2(tgt[1] - crab[1], tgt[0] - crab[0]))   # look from behind Pinch toward them
    want, best = 1.5, None
    group = np.array([1, 0, 0, 0, 0, 0], np.uint8)
    for k, daz in enumerate([0, 35, -35, 70, -70, 110, -110, 150, -150, 180]):
        az, el = np.radians(base_az + daz), np.radians(22.0)
        v = np.array([-np.cos(az) * np.cos(el), -np.sin(az) * np.cos(el), np.sin(el)])   # lookat -> camera
        gid = np.array([-1], np.int32)
        hit = mujoco.mj_ray(sim.m, sim.d, look, v, group, 1, -1, gid)
        clear = want if hit < 0 else min(want, hit - 0.15)
        score = clear - 0.002 * abs(daz)
        if best is None or score > best[0]:
            best = (score, base_az + daz, clear)
        if clear >= want and k < 3:
            break
    cam = mujoco.MjvCamera()
    cam.lookat[:] = look
    cam.azimuth, cam.elevation, cam.distance = best[1], -22.0, max(0.45, best[2])
    return cam


def caption(w, life):
    who = ", ".join(f"{'child' if p.p['child'] else 'adult'} {p.p['style']} {'sitting' if p.p['role'] == 'sitter' else 'walking'}"
                    for p in life.people) or "nobody"
    where = {"desk": "on a desk", "table": "on a table", "floor": "on the floor"}[w.kind]
    return (f"{w.name}  {where} ({2 * w.surface['half_x'] * 100:.0f} x {2 * w.surface['half_y'] * 100:.0f} cm),"
            f" {len(w.rooms)} room{'s' if len(w.rooms) > 1 else ''}, {len(w.things)} things", f"people: {who}")


def clip(w, seconds, writer, rng):
    import cv2
    import mujoco

    from pinch.faces import FaceID
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    from sim.desk import set_face
    from sim.worlds.live import Life
    sim = PinchSim(memory=Memory(path=None), world=w, seed=int(rng.integers(1000)))
    life = Life(sim, camera=True)
    fid = FaceID() if FaceID.available() else None
    life.run(1.5)
    r = mujoco.Renderer(sim.m, H, W)
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    tid = sim.m.texture("face_tex").id
    cam = camera_for(sim, life, rng)
    top, bottom = caption(w, life)
    found = 0
    for k in range(int(seconds * FPS)):
        life.run(1.0 / FPS)
        if k % int(FPS * 1.5) == 0:
            cam = camera_for(sim, life, rng)
        mujoco.mj_forward(sim.m, sim.d)
        set_face(sim.m, sim.face.rgb(scale=2))
        r._gl_context.make_current()
        mujoco.mjr_uploadTexture(sim.m, r._mjr_context, tid)
        r.update_scene(sim.d, cam, scene_option=opt)
        img = np.ascontiguousarray(r.render()[:, :, ::-1])
        if k % 3 == 0 or life.last_frame is None:
            fish = life.cam.render(sim.d)
            if fid is not None:
                for dd in fid.detect(fish):
                    found += 1
                    cv2.rectangle(fish, (int(dd[0]), int(dd[1])), (int(dd[0] + dd[2]), int(dd[1] + dd[3])), (60, 255, 60), 3)
            life.last_frame = fish
        inset = cv2.resize(life.last_frame, (288, 216))
        img[8:224, W - 296:W - 8] = inset
        cv2.rectangle(img, (W - 297, 7), (W - 7, 225), (255, 255, 255), 1)
        cv2.putText(img, "Pinch's camera", (W - 290, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, H - 46), (W, H), (0, 0, 0), -1)
        cv2.putText(img, top, (10, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, bottom, (10, H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 255), 1, cv2.LINE_AA)
        writer.write(img)
    act = sim.mind.activity.name if sim.mind.activity else "-"
    out = dict(world=w.name, kind=w.kind, people=len(life.people), touches=len(life.touches), faces_boxed=found,
               activity=act, falls=sim.falls)
    life.cam.close()
    r.close()
    return out


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--out", default=str(ROOT.parent / "media" / "review" / "phase-05" / "worlds_montage.webm"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*("VP80" if out.suffix == ".webm" else "mp4v")), FPS, (W, H))
    rng = np.random.default_rng(11)
    for w in pick_worlds(a.n):
        info = clip(w, a.seconds, writer, rng)
        print(info, flush=True)
    writer.release()
    print("wrote", out)


if __name__ == "__main__":
    main()

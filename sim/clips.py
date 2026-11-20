"""Short video clips of Pinch in the twin (GIFs), for review and as evidence of behaviour.

    python sim/clips.py                      # all clips -> media/
    python sim/clips.py right_self hurdle    # some
Clips: right_self (on its back, gets up), hurdle (walks over a 6 mm bar with the trained walking corrections:
the hurdle bench's own episode), pick (picks a pen up and drops it in the tray), life (a minute of pet life, 4x speed).
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "media"
W, H, FPS = 400, 300, 12


class Recorder:
    def __init__(self, m, d, dist=0.38, elev=-28.0, azim=135.0):
        self.m, self.d = m, d
        self.r = mujoco.Renderer(m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation, self.cam.azimuth = dist, elev, azim
        self.frames = []
        self.body = m.body("body").id

    def grab(self, label=None):
        target = self.d.xpos[self.body].copy()
        target[2] = max(target[2], 0.02)
        self.cam.lookat[:] = target
        self.r.update_scene(self.d, self.cam)
        img = Image.fromarray(self.r.render())
        if label:
            from PIL import ImageDraw
            ImageDraw.Draw(img).text((8, 6), label, fill=(255, 255, 255))
        self.frames.append(img)

    def save(self, name):
        OUT.mkdir(exist_ok=True)
        p = OUT / f"{name}.gif"
        self.frames[0].save(p, save_all=True, append_images=self.frames[1:], duration=int(1000 / FPS), loop=0,
                            optimize=True)
        return p


def _sim(**kw):
    from pinch.memory import Memory
    from pinch.twin import PinchSim
    return PinchSim(memory=Memory(path=None), **kw)


def clip_right_self():
    sim = _sim(seed=0)
    q = sim.d.qpos
    q[2], q[3:7] = 0.07, [0.0, 1.0, 0.0, 0.0]
    mujoco.mj_forward(sim.m, sim.d)
    rec = Recorder(sim.m, sim.d)
    for _k in range(int(6.0 * FPS)):
        sim.run(1.0 / FPS)
        rec.grab(f"t={sim.t:4.1f}s  {sim.mind.activity.name if sim.mind.activity else ''} {sim.mind.phase}")
    return rec.save("right_self")


def clip_hurdle(height_mm=6, way="forward", policy="gait"):
    """The hurdle bench's own episode (sim/benches/hurdles.py), recorded, with its verdict on the last frames."""
    from crab import policy_io as pio
    from sim.benches.hurdles import case
    every = max(1, int(round(1.0 / FPS / pio.CTRL_DT)))
    st = {}

    def tick(m, d, t):
        if "rec" not in st:
            st["rec"] = Recorder(m, d, azim=60.0 if way == "sideways" else 135.0)
        if int(round(t / pio.CTRL_DT)) % every == 0:
            st["rec"].grab(f"{height_mm} mm bar, walking {way}  t={t:4.1f}s")

    res = case(("hurdle", height_mm, way, "policy", 0, policy), on_tick=tick)
    for _ in range(FPS):                                   # a second on the verdict
        st["rec"].grab(f"{height_mm} mm bar, walking {way}: {'crossed' if res['crossed'] else 'not crossed'}")
    return st["rec"].save(f"hurdle_{height_mm}mm_{way}"), res["crossed"]


def clip_pick(seed=1):
    sim = _sim(seed=seed)
    sim.body.grasp_policy = None
    sim.autonomous = False
    rng = np.random.default_rng(seed)
    sim.run(1.5)
    while True:
        xy = (rng.uniform(-0.25, 0.25), rng.uniform(-0.13, 0.13))
        if not sim.desk.in_tray(xy) and np.hypot(*xy) > 0.13 and sim.desk.edge_distance(xy) > 0.09:
            break
    sim.place("pen", xy, rng.uniform(-np.pi, np.pi))
    sim.run(0.5)
    rec = Recorder(sim.m, sim.d, dist=0.5, elev=-40.0)
    sim.body.start(("pick", "pen"))
    while sim.body.busy and sim.t < 60:
        sim.run(2.0 / FPS)                                       # (2x speed)
        rec.grab(f"pick the pen  t={sim.t:4.1f}s  (2x)")
    sim.body.start(("drop", sim.desk.tray_xy))
    while sim.body.busy and sim.t < 120:
        sim.run(2.0 / FPS)
        rec.grab(f"put it in the tray  t={sim.t:4.1f}s  (2x)")
    for _ in range(FPS):
        sim.run(2.0 / FPS)
        rec.grab(f"in the tray: {sim.in_tray('pen')}")
    return rec.save("pick_pen")


def clip_life(seconds=60.0, speed=4.0):
    sim = _sim(seed=5)
    rec = Recorder(sim.m, sim.d, dist=0.6, elev=-45.0)
    said = False
    while sim.t < seconds:
        if not said and sim.t > 10:
            sim.say("hi pinch, dance!")
            said = True
        sim.run(speed / FPS)
        a = sim.mind.activity
        rec.grab(f"{speed:.0f}x  t={sim.t:4.0f}s  {sim.mind.state}: {a.name if a else ''} {sim.mind.phase}  "
                 f"feeling {sim.mind.affect.feeling()}")
    return rec.save("pet_life")


CLIPS = dict(right_self=clip_right_self, hurdle=clip_hurdle, pick=clip_pick, life=clip_life)

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(CLIPS)):
        print(name, "->", CLIPS[name]())

"""How closed the body is: what inside it can be seen from outside, on the model generated from the CAD.

    CRAB_DESIGN=v2 python sim/closure_check.py [model.xml]      # exit 1 above the limit

Rays come from all round at three elevations (0, 25 and 50 deg) and many heights, aimed at the body's axis; legs and
arms are looked past (a leg moves); the first thing on the body hit is classified. The shell, the face, the servos
and the parts meant to be seen (camera, windows, switch, plug, pads, mics, speaker grille) are fine; a board, the
battery or a wire seen from outside is not. Pass: at most 2 % of the rays that hit the body (docs: design acceptance
A10)."""
import sys
from collections import Counter, defaultdict

import mujoco
import numpy as np

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
m = mujoco.MjModel.from_xml_path(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "sim" / "models" / "crab_v2.xml"))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
mujoco.mj_forward(m, d)
body = m.body("body").id
c = d.xpos[body].copy()
OK = ("tray", "cover", "lid", "battery_door", "panel", "hip_servo", "yaw_servo", "camera", "tof", "pad", "mic",
      "speaker", "switch", "pogo", "xt60", "face")
GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)          # what the viewer shows by default


def name_of(g):
    mid = m.geom_dataid[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and mid >= 0:
        return m.mesh(mid).name
    return m.geom(g).name or f"geom{g}"


def first_body_hit(p, v):
    gid = np.zeros(1, np.int32)
    for _ in range(12):
        dist = mujoco.mj_ray(m, d, p, v, GROUPS, 1, -1, gid)
        if dist < 0:
            return None
        g = int(gid[0])
        if m.geom_bodyid[g] == body:
            return g
        p = p + v * (dist + 1e-4)                    # a leg or arm: look past it
    return None


exposed, seen = defaultdict(list), Counter()
for el in (0.0, 25.0, 50.0):
    for az in np.arange(0, 360, 2.0):
        a, e = np.radians(az), np.radians(el)
        for h in np.arange(-0.05, 0.06, 0.004):
            v = -np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            target = c + [0, 0, h]
            p = target - v * 0.3
            g = first_body_hit(p, v.copy())
            if g is None:
                continue
            n = name_of(g)
            seen["all"] += 1
            if not any(k in n for k in OK):
                exposed[n].append((int(az), int(el), int(round(h * 1000))))
total = seen["all"]
nexp = sum(len(v) for v in exposed.values())
print(f"rays hitting the body: {total}; inside parts seen: {nexp} ({100 * nexp / max(total, 1):.1f} %)")
for n, hits in sorted(exposed.items(), key=lambda kv: -len(kv[1])):
    azs = Counter(a // 30 * 30 for a, _, _ in hits)
    print(f"  {n:28s} {len(hits):5d}  by azimuth {dict(sorted(azs.items()))}  heights {min(h for *_, h in hits)}..{max(h for *_, h in hits)} mm")
LIMIT = 2.0
pct = 100 * nexp / max(total, 1)
print(f"CLOSURE {'PASSED' if pct <= LIMIT else 'FAILED'}: {pct:.1f} % of views see inside (limit {LIMIT:.0f} %)")
sys.exit(0 if pct <= LIMIT else 1)

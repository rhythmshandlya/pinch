"""Render the v2 robot standing, with an expression on its face screen (the face the twin shows): a whole-robot view
like docs/images/pinch_desk.png above, a close-up of the face below.

    python sim/render_v2_face.py [expression] [out.png]       # default: media/review/v2/v2_face.png
"""

import os
import sys
from pathlib import Path

os.environ["CRAB_DESIGN"] = "v2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from crab.params import PARAMS  # noqa: E402
from pinch.face import Face  # noqa: E402
from sim.desk import set_face  # noqa: E402


def main():
    expr = sys.argv[1] if len(sys.argv) > 1 else "happy"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "media" / "review" / "v2" / "v2_face.png"
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim" / "models" / PARAMS.model_file))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    for _ in range(500):
        d.ctrl[:] = m.key("stand").ctrl
        mujoco.mj_step(m, d)
    face = Face()
    for _ in range(80):
        face.update(0.02, expr, (0.0, 0.0))
    set_face(m, face.rgb(scale=2))
    r = mujoco.Renderer(m, 480, 720)
    body = m.body("body").id
    shots = []
    for dist, az, el, look in ((0.62, 150, -22, (0.0, 0.0, 0.06)), (0.20, 180, -25, (0.065, 0.0, 0.032))):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = d.xpos[body] + np.array(look)
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        r.update_scene(d, cam)
        shots.append(r.render().copy())
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack(shots)).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""Where the open claw's gap is, for given joint angles (forward kinematics from the robot model).

Used by the body's grab and push skills on the robot and in the twin. Needs `mujoco` (pip install
mujoco works on the Radxa) and sim/models/crab.xml, but no simulation: kinematics only.
"""

from pathlib import Path

import mujoco
import numpy as np

from crab.params import JOINT_NAMES

MODEL = Path(__file__).resolve().parents[1] / "sim" / "models" / "crab.xml"
GAP = 0.006            # aim a thing this far off the fixed jaw, into the open gap (mid-jaw)


class ClawFK:
    def __init__(self, model=None):
        self.m = model if model is not None else mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINT_NAMES]

    def tip(self, side, q):
        """Centre of the open claw's gap in Pinch's body frame (x, y), for joint targets q."""
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qpos[0:3] = 0
        d.qpos[3:7] = (1, 0, 0, 0)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        s = "L" if side == "left" else "R"
        site = self.m.site(f"{s}_grip_site").id
        R = d.site_xmat[site].reshape(3, 3)
        p = d.site_xpos[site] + R @ np.array([0.0, (1 if side == "left" else -1) * GAP, 0.0])
        return float(p[0]), float(p[1])

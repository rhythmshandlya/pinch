"""Arm poses as robot geometry: posing one arm in a full joint vector, and the pushing paddle (decision 68).

Shared by the robot's body skills (pinch/body) and the training environments (training/), which must not
depend on the pet's brain package.
"""

import numpy as np

from .emotes import crouch, pose
from .params import JOINT_NAMES

J = {n: i for i, n in enumerate(JOINT_NAMES)}
PUSH_LOWER = 0.011              # walk 11 mm lower while pushing: claw flat on the desk
PADDLE_PITCH = -0.28            # pushing: jaw blades ~1 mm above the desk (the old claw: -0.62)
PADDLE_CLAW = 0.5               # ... half open: the two tips cradle a round thing


def arm_pose(base, side, pitch=None, yaw=None, claw=None):
    """A copy of joint vector `base` with one arm ("left"/"right") set where given."""
    q = np.array(base, float).copy()
    s = "L" if side == "left" else "R"
    if pitch is not None:
        q[J[f"{s}_shoulder_pitch"]] = pitch
    if yaw is not None:
        q[J[f"{s}_shoulder_yaw"]] = yaw
    if claw is not None:
        q[J[f"{s}_claw"]] = claw
    return q


def paddle_pose(side):
    """Pushing: the `side` claw flat and half open just above the desk, the other arm tucked up."""
    return arm_pose(arm_pose(pose(crouch(0.25)), side, pitch=PADDLE_PITCH, yaw=0.0, claw=PADDLE_CLAW),
                    "right" if side == "left" else "left", pitch=0.5)

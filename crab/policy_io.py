"""The contract between a walking policy and the robot.

Defines exactly what the policy observes and how its actions become servo
targets. The training environment (JAX) and the real robot / CPU sim (numpy)
both build observations through `observation()`, using whichever array module
they pass in, so the two can never drift apart.

Only signals the real robot has are used: IMU (gyro + gravity direction) and
servo encoders (position, velocity). No foot sensors, no true body velocity.
"""

import numpy as np

from .emotes import STAND
from .params import JOINT_NAMES, LEG_IDS, LEG_JOINTS

CTRL_DT = 0.02                       # 50 Hz policy
ACTION_SCALE = 0.30                  # rad per unit action
GAIT_PERIOD = 0.70                   # s, phase clock fed to the policy

LEG_JOINT_NAMES = [f"{lg}_{j}" for lg in LEG_IDS for j in LEG_JOINTS]
LEG_ACT_IDX = np.array([JOINT_NAMES.index(n) for n in LEG_JOINT_NAMES])   # into the 20 actuators
N_ACT = len(LEG_ACT_IDX)                                                    # 12
DEFAULT_LEG_POSE = STAND[LEG_ACT_IDX].copy()
FULL_DEFAULT = STAND.copy()          # arms hold this while walking

# observation layout (sizes) -- keep in sync with observation()
OBS_LAYOUT = [("gyro", 3), ("gravity", 3), ("command", 3), ("joint_pos", N_ACT),
              ("joint_vel", N_ACT), ("last_action", N_ACT), ("phase", 2)]
OBS_SIZE = sum(n for _, n in OBS_LAYOUT)
JOINT_VEL_SCALE = 0.05
GYRO_SCALE = 0.25


def observation(xp, gyro, gravity, command, q_legs, qd_legs, last_action, t):
    """Build the policy observation.

    xp:        numpy or jax.numpy
    gyro:      body-frame angular velocity (rad/s), 3
    gravity:   unit gravity vector in the body frame (0,0,-1 when level), 3
    command:   (vx, vy, wz), 3
    q_legs:    12 leg joint positions (rad), order LEG_JOINT_NAMES
    qd_legs:   12 leg joint velocities (rad/s)
    last_action: previous policy output, 12
    t:         time since start (s), for the phase clock
    """
    ph = 2 * xp.pi * t / GAIT_PERIOD
    return xp.concatenate([
        gyro * GYRO_SCALE,
        gravity,
        command,
        q_legs - DEFAULT_LEG_POSE,
        qd_legs * JOINT_VEL_SCALE,
        last_action,
        xp.array([xp.sin(ph), xp.cos(ph)]),
    ])


def action_to_targets(xp, action, arm_pose=None):
    """12 policy outputs -> 20 servo position targets (arms held still)."""
    full = xp.asarray(FULL_DEFAULT if arm_pose is None else arm_pose)
    legs = DEFAULT_LEG_POSE + ACTION_SCALE * action
    if xp is np:
        out = full.copy()
        out[LEG_ACT_IDX] = legs
        return out
    return full.at[LEG_ACT_IDX].set(legs)

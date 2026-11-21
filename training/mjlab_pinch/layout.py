"""The one observation / action / command layout every mjlab Pinch policy shares (pure numpy, no torch).

The trainer (mdp.py, env_cfg.py) and the sim-to-sim replay (sim_replay.py) read the sizes, scales and the
frame builder from here, so they cannot drift apart. The signals are only what the real robot has: gyro, gravity
direction (IMU), leg-servo position and velocity (encoders), the last action and the command. The two arms hold
a fixed pose and report nothing, so they are not observed.

Observation (actor), oldest to newest history frame, term by term (mjlab's term-major order):
    gyro (3) | gravity (3) | leg_pos (12) | leg_vel (12) | last_action (12), each over HISTORY frames, then
    command (3) once.
"""

import numpy as np

from crab.emotes import STAND
from crab.params import JOINT_NAMES, LEG_IDS, LEG_JOINTS

CTRL_HZ = 50
CTRL_DT = 1.0 / CTRL_HZ
HISTORY = 3                                    # frames per history term (60 ms)

LEG_JOINT_NAMES = [f"{lg}_{j}" for lg in LEG_IDS for j in LEG_JOINTS]          # FL FR RL RR x coxa femur tibia
ARM_JOINT_NAMES = [n for n in JOINT_NAMES if n not in LEG_JOINT_NAMES]
LEG_REGEX = r"^[FR][LR]_(coxa|femur|tibia)$"
ARM_REGEX = r"^[LR]_(shoulder_yaw|shoulder_pitch|claw)$"
N_LEG = len(LEG_JOINT_NAMES)                   # 12 actions
STAND_POSE = {n: float(STAND[JOINT_NAMES.index(n)]) for n in JOINT_NAMES}
DEFAULT_LEG_POS = np.array([STAND_POSE[n] for n in LEG_JOINT_NAMES])
ACTION_SCALE = 0.3                             # rad per unit action, around the stand pose

GYRO_SCALE = 0.25                              # the project's existing scalings (crab/policy_io.py)
JOINT_VEL_SCALE = 0.05
# command: vx, vy (m/s) and yaw rate (rad/s), sized for a 20 cm crab walking at most ~6 cm/s
CMD_MAX = (0.06, 0.06, 0.6)
CMD_SCALE = tuple(1.0 / m for m in CMD_MAX)    # the policy sees the command in units of its maximum

FRAME_TERMS = [("gyro", 3), ("gravity", 3), ("leg_pos", N_LEG), ("leg_vel", N_LEG), ("last_action", N_LEG)]
OBS_FRAME = sum(n for _, n in FRAME_TERMS)     # 45
OBS_SIZE = OBS_FRAME * HISTORY + 3             # 138 with the command
OBS_LAYOUT = [(name, n * HISTORY) for name, n in FRAME_TERMS] + [("command", 3)]


def frame(gyro, gravity, leg_pos, leg_vel, last_action):
    """One history frame (45), already scaled. leg_pos is relative to the stand pose; gyro in rad/s, leg_vel in
    rad/s, gravity a unit vector in the body frame (the IMU's)."""
    return np.concatenate([np.asarray(gyro) * GYRO_SCALE, gravity, leg_pos, np.asarray(leg_vel) * JOINT_VEL_SCALE,
                           last_action]).astype(np.float32)


class History:
    """mjlab's per-term history (oldest first), including the backfill: the first frame fills every slot."""

    def __init__(self):
        self.frames = None

    def reset(self):
        self.frames = None

    def push(self, fr):
        if self.frames is None:
            self.frames = [fr.copy() for _ in range(HISTORY)]
        else:
            self.frames = self.frames[1:] + [fr]

    def obs(self, command):
        """Term-major vector: every term's HISTORY frames oldest to newest, then the scaled command."""
        parts, off = [], 0
        for _, n in FRAME_TERMS:
            parts.extend(f[off:off + n] for f in self.frames)
            off += n
        parts.append(np.asarray(command, np.float32) * np.asarray(CMD_SCALE, np.float32))
        return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------- commands
# Buckets of the command distribution (probabilities), so rare but important cases are trained, not left to
# chance: standing still, turning on the spot, walking along one axis only, sideways as often as forward.
BUCKETS = dict(zero=0.10, turn=0.15, forward=0.15, sideways=0.15, mixed=0.45)
TURN_MIN = 0.35                                # a turn-in-place command is at least this fraction of the max yaw
MIN_LIN = 0.25                                 # a one-axis walking command is at least this fraction of the max


def _mag(x, lo, hi):
    """A signed magnitude in [lo, hi] (either sign) from one uniform draw x."""
    sign = 1.0 - 2.0 * (x < 0.5)
    return sign * (lo + (hi - lo) * abs(2 * x - 1))


def bucket_commands(u, cmd_max=CMD_MAX, buckets=BUCKETS):
    """(n, 3) commands (vx, vy, wz) from uniform randoms u (n, 4) in [0, 1): works on numpy arrays and torch
    tensors alike. u[:, 0] picks the bucket, u[:, 1:4] are the vx, vy, wz draws. Also returns the bucket index
    (0 zero, 1 turn, 2 forward, 3 sideways, 4 mixed) per row."""
    names = list(buckets)
    total = float(sum(buckets.values()))
    b = u[:, 0] * total
    masks, acc = {}, 0.0
    for name in names:
        masks[name] = ((b >= acc) & (b < acc + buckets[name])) * 1.0
        acc += buckets[name]
    vx_full, vy_full, wz_full = ((2 * u[:, 1 + i] - 1) * cmd_max[i] for i in range(3))
    vx = masks["forward"] * _mag(u[:, 1], MIN_LIN * cmd_max[0], cmd_max[0]) + masks["mixed"] * vx_full
    vy = masks["sideways"] * _mag(u[:, 2], MIN_LIN * cmd_max[1], cmd_max[1]) + masks["mixed"] * vy_full
    wz = masks["turn"] * _mag(u[:, 3], TURN_MIN * cmd_max[2], cmd_max[2]) + masks["mixed"] * wz_full
    idx = sum(i * masks[n] for i, n in enumerate(names))
    return vx, vy, wz, idx

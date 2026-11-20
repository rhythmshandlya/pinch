"""The contract between the learned policies and the robot.

Defines exactly what each policy observes and how its actions become servo targets: the residual walking
and emote balance policy (emote_observation / emote_targets) and the grasp policy (grasp_observation /
grasp_targets). The training environments (JAX) and the robot and twin (numpy) build them through these
same functions, passing in whichever array module they use, so the two can never drift apart.

Only signals the real robot has are used: IMU (gyro + gravity direction) and servo encoders (position,
velocity). No foot sensors, no true body velocity.
"""

import numpy as np

from .emotes import STAND
from .params import JOINT_NAMES, LEG_IDS, LEG_JOINTS

CTRL_DT = 0.02                       # 50 Hz policy

LEG_JOINT_NAMES = [f"{lg}_{j}" for lg in LEG_IDS for j in LEG_JOINTS]
LEG_ACT_IDX = np.array([JOINT_NAMES.index(n) for n in LEG_JOINT_NAMES])   # into the 18 actuators
N_ACT = len(LEG_ACT_IDX)                                                    # 12
DEFAULT_LEG_POSE = STAND[LEG_ACT_IDX].copy()
JOINT_VEL_SCALE = 0.05                # observation scaling (unchanged since the first policies)
GYRO_SCALE = 0.25
_FEMUR = np.array([LEG_JOINT_NAMES.index(f"{lg}_femur") for lg in LEG_IDS])
_TIBIA = np.array([LEG_JOINT_NAMES.index(f"{lg}_tibia") for lg in LEG_IDS])


def shin_clamp(xp, legs):
    """Keep thigh + shin >= shin_min (CAD: the shin's underside meets the hip link past it)."""
    from .params import PARAMS
    tib = xp.maximum(legs[_TIBIA], PARAMS.leg.shin_min - legs[_FEMUR])
    if xp is np:
        legs = legs.copy()
        legs[_TIBIA] = tib
        return legs
    return legs.at[_TIBIA].set(tib)


# ------------------------------------------------------------------ emote tracking policy
# One policy for every keyframed emote: the arms follow the keyframes, the policy adds small leg
# corrections to keep the emote's look (leg pose, body attitude, height) under real-world physics.
# It sees the reference itself (now and 0.1 s ahead) plus progress, not an emote id, so new
# emotes and tricks work without retraining.
from .emotes import EMOTES as EMOTE_LIB  # noqa: E402

EMOTE_NAMES = sorted(EMOTE_LIB)
JOINT_NAMES_ALL = list(JOINT_NAMES)
ARM_ACT_IDX = np.array([i for i in range(len(JOINT_NAMES)) if i not in set(LEG_ACT_IDX.tolist())])
EMOTE_ACTION_SCALE = 0.15                        # rad per unit action: corrections, not new motions
EMOTE_OBS_LAYOUT = [("gyro", 3), ("gravity", 3), ("leg_err", N_ACT), ("joint_vel", N_ACT),
                    ("last_action", N_ACT), ("ref_legs", N_ACT), ("ref_legs_ahead", N_ACT),
                    ("ref_arms", len(ARM_ACT_IDX)), ("progress", 2)]
EMOTE_OBS_SIZE = sum(n for _, n in EMOTE_OBS_LAYOUT)


def emote_observation(xp, gyro, gravity, q_legs, qd_legs, last_action, ref_now, ref_ahead, _onehot, progress):
    """ref_now / ref_ahead: full 18-joint reference poses. progress: 0..1 through the emote."""
    legs_now = ref_now[LEG_ACT_IDX]
    return xp.concatenate([
        gyro * GYRO_SCALE,
        gravity,
        q_legs - legs_now,
        qd_legs * JOINT_VEL_SCALE,
        last_action,
        legs_now - DEFAULT_LEG_POSE,
        ref_ahead[LEG_ACT_IDX] - DEFAULT_LEG_POSE,
        ref_now[ARM_ACT_IDX],
        xp.array([xp.sin(xp.pi * progress), xp.cos(xp.pi * progress)]),
    ])


def emote_targets(xp, ref, action):
    """Reference pose (18) + leg corrections (12 actions) -> 18 servo targets."""
    legs = shin_clamp(xp, ref[LEG_ACT_IDX] + EMOTE_ACTION_SCALE * action)
    if xp is np:
        out = np.array(ref, dtype=float).copy()
        out[LEG_ACT_IDX] = legs
        return out
    return ref.at[LEG_ACT_IDX].set(legs)


# ------------------------------------------------------------------ grasp skill policy (right claw)
# Pinch has walked up and roughly lined its right claw up with the thing; this policy does the last
# few centimetres: legs shift/lean the body (small corrections around a nose-down crouch) and the arm
# swings, lowers and closes. Arm actions are absolute (the arm servos have no feedback to correct).
from .emotes import crouch as _crouch  # noqa: E402
from .emotes import pose as _pose

_R_ARM = [JOINT_NAMES.index(f"R_{j}") for j in ("shoulder_yaw", "shoulder_pitch", "claw")]
GRASP_NOMINAL = _pose(_crouch(0.38, "F?"), _crouch(-0.20, "R?"), {"R_shoulder_yaw": 0.17,
                                                                     "R_shoulder_pitch": -0.1, "R_claw": 0.48})
GRASP_N_ACT = 15
GRASP_LEG_SCALE = 0.15
GRASP_KINDS = ("stick", "block", "ball")     # pens/pencils/markers, erasers/sharpeners, marbles: anything ~15 mm wide
GRASP_OBS_SIZE = 3 + 3 + N_ACT + N_ACT + GRASP_N_ACT + 3 + 3 + 2 + 3 + 3 + 1
ARM_RANGES = ((0.0, 0.34), (-0.69, 0.50), (0.0, 0.96))            # yaw, pitch, claw (CAD limits)
ARM_RATE = (2.0, 2.0, 4.0)                   # rad/s the arm's targets may move (yaw, pitch, claw): eased like a servo
GRASP_ARM_IDX = _R_ARM


def grasp_targets(xp, nominal, action, arm_prev=None, dt=CTRL_DT):
    """15 actions -> 18 servo targets: legs = nominal + small corrections; right arm = absolute positions,
    eased toward at most ARM_RATE from arm_prev (the previous arm targets) when given."""
    legs = shin_clamp(xp, xp.asarray(nominal)[LEG_ACT_IDX] + GRASP_LEG_SCALE * action[:12])
    arm = xp.stack([lo + (hi - lo) * (action[12 + i] + 1) / 2 for i, (lo, hi) in enumerate(ARM_RANGES)])
    if arm_prev is not None:
        rate = xp.asarray(ARM_RATE) * dt
        arm = arm_prev + xp.clip(arm - arm_prev, -rate, rate)
    if xp is np:
        out = np.array(nominal, float).copy()
        out[LEG_ACT_IDX] = legs
        out[_R_ARM] = arm
        return out
    out = xp.asarray(nominal).at[LEG_ACT_IDX].set(legs)
    return out.at[xp.array(_R_ARM)].set(arm)


def grasp_observation(xp, gyro, gravity, q_legs, qd_legs, last_action, arm, rel, ang, kind, size, tfrac):
    """arm: the right arm's current targets (yaw, pitch, claw); rel: the grab point on the thing (nearest
    point of its long axis) relative to the open claw's gap (body frame, m); ang: the thing's long-axis
    direction (rad, body frame); kind: index into GRASP_KINDS; size: its half-extents (long, wide, high, m);
    tfrac: time since the grasp started / 4 s. On the robot all of the thing's numbers come from the camera."""
    lo = xp.asarray([r[0] for r in ARM_RANGES])
    hi = xp.asarray([r[1] for r in ARM_RANGES])
    return xp.concatenate([
        gyro * GYRO_SCALE, gravity,
        q_legs - GRASP_NOMINAL[LEG_ACT_IDX],
        qd_legs * JOINT_VEL_SCALE,
        last_action,
        2 * (xp.asarray(arm) - lo) / (hi - lo) - 1,
        rel * 20.0,
        xp.array([xp.sin(2 * ang), xp.cos(2 * ang)]),          # a stick is the same either way round
        (xp.arange(3) == kind).astype(xp.float32) if xp is not np else (np.arange(3) == kind).astype(float),
        xp.asarray(size) * 50.0,
        xp.array([tfrac]),
    ])


# ------------------------------------------------------------------ self-righting policy (decision 76)
# All 18 joints, absolute targets over each joint's full range (no reference motion): on its back the
# whole body, shell and claws included, is what it rolls with.
RIGHTING_OBS_SIZE = 3 + 3 + 18 + 18 + 18
RIGHTING_VEL_SCALE = 0.1


def righting_observation(xp, gravity, gyro, q, qd, last_action):
    """gravity: unit vector in the body frame; gyro: body angular velocity (rad/s); q, qd: all 18 joints
    (JOINT_NAMES order); last_action: the previous output (18)."""
    from .emotes import STAND
    return xp.concatenate([gravity, gyro * GYRO_SCALE, q - xp.asarray(STAND), qd * RIGHTING_VEL_SCALE,
                           last_action])


def righting_targets(xp, action, lo, hi):
    """Actions in [-1, 1] -> joint targets over [lo, hi] (each servo's range, JOINT_NAMES order)."""
    return lo + (hi - lo) * (xp.clip(action, -1, 1) + 1) / 2

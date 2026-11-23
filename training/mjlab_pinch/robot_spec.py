"""The Pinch robot for mjlab, built from the project's MJCF (pure mujoco: importable without mjlab or torch).

The robot model file and the servo numbers are both swappable: MODEL names the MJCF (default
sim/models/crab.xml, today's 12 leg + 6 arm joints) and the servo numbers come from crab/params.py or, for the
measured STS3215, from a BAM file (env_cfg.py). A resized robot only has to keep the joint, foot and site names.

spec() strips what mjlab supplies itself or does not need: the floor and sky, lights, cameras, the CAD display
meshes, the model's own sensors and position actuators (the actuator configs add theirs) and its keyframe.
Collisions are feet only (like training/robot_model.py); mjlab's CollisionCfg zeroes the rest.
"""

import os
from pathlib import Path

import mujoco

from crab.params import (
    JOINT_NAMES,
    PARAMS,  # noqa: E402
    servo_for,
)
from training.mjlab_pinch.layout import ARM_JOINT_NAMES, LEG_JOINT_NAMES, STAND_POSE

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ.get("PINCH_MJLAB_MODEL", ROOT / "sim" / "models" / PARAMS.model_file))
FOOT_GEOMS = tuple(f"{lg}_foot" for lg in ("FL", "FR", "RL", "RR"))
FOOT_SITES = tuple(f"{lg}_foot_site" for lg in ("FL", "FR", "RL", "RR"))
BODY = "body"
START_HEIGHT = PARAMS.stand_body_z + 0.002   # stand height plus a little: it settles onto its feet


def spec() -> mujoco.MjSpec:
    s = mujoco.MjSpec.from_file(str(MODEL))
    for coll in (s.sensors, s.actuators, s.keys, s.cameras, s.lights):
        for x in list(coll):
            s.delete(x)
    for g in list(s.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            s.delete(g)                  # CAD visuals: looks only
        elif g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            s.delete(g)                  # the floor: the scene brings its own terrain
        else:
            g.material = ""
    for coll in (s.meshes, s.materials, s.textures):
        for x in list(coll):
            s.delete(x)
    return s


def compile_check() -> mujoco.MjModel:
    return spec().compile()


def servo_numbers(joint: str) -> dict:
    """Today's servo model for a joint, from crab/params.py (the numbers crab.xml was generated from)."""
    sv = servo_for(joint)
    return dict(stiffness=sv.kp, effort_limit=sv.force_limit, armature=sv.armature, frictionloss=sv.frictionloss,
                viscous_damping=sv.damping)


def leg_servo_numbers() -> dict:
    """The leg servos are all one model (a mix is not supported by one actuator config): check and return it."""
    nums = [servo_numbers(j) for j in LEG_JOINT_NAMES]
    assert all(n == nums[0] for n in nums), "legs with different servos need one actuator config per servo"
    return nums[0]


def arm_servo_numbers() -> dict:
    nums = [servo_numbers(j) for j in ARM_JOINT_NAMES]
    assert all(n == nums[0] for n in nums)
    return nums[0]


def init_joint_pos() -> dict:
    """Stand pose as the exact-name regexes mjlab's init state takes."""
    return {f"^{n}$": STAND_POSE[n] for n in JOINT_NAMES}

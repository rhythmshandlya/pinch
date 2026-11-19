"""Shared by the body's parts: joint layout, poses, speed limits, claw geometry constants and small helpers."""

import copy

import numpy as np

from crab import policy_io as pio
from crab.arm import PADDLE_CLAW, PADDLE_PITCH, PUSH_LOWER, arm_pose, paddle_pose  # noqa: F401 (robot geometry)
from crab.emotes import EMOTES, STAND, crouch, pose
from crab.gait import GAITS, Gait
from crab.params import JOINT_NAMES
from robot.crab_robot import DT
from robot.servos import SERVO_IDS
from pinch.world import Desk

J = {n: i for i, n in enumerate(JOINT_NAMES)}
SUB = 10                                               # gait integrates at 500 Hz inside each 20 ms tick
POSES = {"stand": STAND, "low": pose(crouch(0.25)), "nap": EMOTES["sleep"].keyframes[1][1]}
V_MAX, W_MAX = 0.06, 0.8
REACH_LOW = {"crouch": 0.38, "pitch": -0.13}           # jaws flat on the desk, pressing lightly (measured: -0.11)
CARRY_PITCH = 0.45                                      # carrying: the held part ~3 cm up, the short end clear of the legs
PEN_HALF = 0.069                                        # half length of a pen (65 mm + rounded ends)
PEN_GRAB = 0.055                                        # hold a pen this far from its middle: its short end stays clear of the front leg
# things the jaws must run along (their long side between the jaws): name -> grab point's offset from the middle
LONG_THINGS = {"pen": PEN_GRAB, "eraser": 0.0}
GRASP_ARM0 = (0.17, 0.08, 0.96)          # the arm when the learned grab takes over (training/grasp_env.py ARM0)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def extent(name, yaw, heading, default=0.015):
    """Half-size of a thing along the push direction (a pen pushed end-on is 7 cm long, sideways 4.5 mm)."""
    if name == "pen":
        a = abs(np.cos(yaw - heading))
        return PEN_HALF * a + 0.0045 * np.sqrt(max(0.0, 1 - a * a))
    return {"cap": 0.014, "ball": 0.015, "eraser": 0.011}.get(name, default)


GAIT_LOOK = 5             # control ticks (0.1 s): how far ahead the walking policy sees the scripted gait

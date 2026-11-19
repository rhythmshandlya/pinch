"""Shared by the twin's parts: imports and loop constants."""

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.params import JOINT_NAMES  # noqa: E402
from crab.policy_np import POLICY_DIR, NumpyPolicy  # noqa: E402
from pinch.body import Body  # noqa: E402
from pinch.face import Face  # noqa: E402
from pinch.hearing import Hearing  # noqa: E402
from pinch.memory import Memory  # noqa: E402
from pinch.mind import Mind  # noqa: E402
from pinch.touch import BodySense, CliffReflex, StuckSense, twin_sensor  # noqa: E402
from pinch.world import Desk  # noqa: E402
from robot.backends import SimBackend  # noqa: E402
from robot.crab_robot import DT, Crab  # noqa: E402
from robot.servos import Calibration  # noqa: E402
from sim.desk import OBJECTS, START, build_desk_model, object_pose, set_face  # noqa: E402

MIND_EVERY = 5                        # mind at 10 Hz
FACE_EVERY = 3                        # face texture at ~17 Hz


SIM_EPOCH = 1795059000.0          # the twin's calendar starts 2026-11-19 09:00 IST (03:30 UTC): runs replay exactly
TWIN_UTC_OFFSET_H = 5.5           # and its local time is India's on any machine (pinch/routine.py)

"""Shared by the firmware's parts: imports, the logger and loop constants."""

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.policy_np import load_policies  # noqa: E402
from pinch.body import Body  # noqa: E402
from pinch.face import Face  # noqa: E402
from pinch.hearing import AudioDirection, Hearing  # noqa: E402
from pinch.memory import Memory  # noqa: E402
from pinch.mind import Mind  # noqa: E402
from pinch.touch import BodySense, CliffReflex, StuckSense, TouchGestures, cliff_beams  # noqa: E402
from pinch.world import Desk  # noqa: E402
from robot.crab_robot import DT, Crab, SafetyStop  # noqa: E402
from robot.servos import Calibration  # noqa: E402

log = logging.getLogger("pinch")
DIR_N = 1024              # samples per mic for the voice direction (GCC-PHAT); a 0.1 s chunk has 1600
SAVE_EVERY = 60.0         # s between memory saves (plus one at shutdown)

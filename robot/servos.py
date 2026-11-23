"""Servo IDs, directions and the joint <-> servo-position conversion.

Joint angles use the simulator's conventions (crab/params.py). A servo reports
its own angle; the two differ by a direction sign (how the servo is mounted)
and an offset (which spline tooth the horn landed on, measured by calibration).

    servo_rad = sign * joint_rad + offset
    raw       = round((servo_rad + pi) * 4096 / (2 pi))      # 2048 = 0 rad (rustypot convention)
"""

import json
import os
from dataclasses import dataclass, field
from math import pi
from pathlib import Path

from crab.params import JOINT_NAMES

# Calibration belongs to this robot, not to the code: it lives outside the install folder (which
# deploy/install.sh overwrites on every update). PINCH_CALIBRATION overrides the place.
CAL_FILE = Path(os.environ.get("PINCH_CALIBRATION", "/var/lib/pinch/calibration.json"))


class NotCalibrated(RuntimeError):
    pass

# Every joint is a Feetech bus servo on one TTL bus, and every one reports its position, load, voltage and
# temperature: the 12 leg servos (STS3215, IDs 1..12) and the 6 arm servos (STS3032, IDs 13..18).
SERVO_IDS = {name: i + 1 for i, name in enumerate(JOINT_NAMES)}      # 1..18, see hardware/electronics.md

# Direction signs derived from the CAD mounting (servo horn normal vs simulator joint axis),
# assuming a servo's position increases counter-clockwise when you look at its horn.
# robot/calibrate.py --check-directions checks every one on the real robot and fixes any that are the other way round.
_SIGN_LEFT = {"coxa": -1, "femur": -1, "tibia": +1,
              "shoulder_yaw": +1, "shoulder_pitch": -1, "claw": +1}


def default_sign(joint: str) -> int:
    side, kind = joint.split("_", 1)
    left = side in ("FL", "RL", "L")
    s = _SIGN_LEFT[kind]
    return s if left else -s


def rad_to_raw(rad: float) -> int:
    return int(round((rad + pi) * 4096 / (2 * pi))) % 4096


def raw_to_rad(raw: float) -> float:
    return raw * 2 * pi / 4096 - pi


@dataclass
class Calibration:
    sign: dict = field(default_factory=lambda: {j: default_sign(j) for j in JOINT_NAMES})
    offset: dict = field(default_factory=lambda: {j: 0.0 for j in JOINT_NAMES})   # rad

    def joint_to_servo(self, joint: str, q: float) -> float:
        return self.sign[joint] * q + self.offset[joint]

    def servo_to_joint(self, joint: str, s: float) -> float:
        return (s - self.offset[joint]) * self.sign[joint]

    def save(self, path=None):
        p = Path(path or CAL_FILE)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps({"sign": self.sign, "offset": self.offset}, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        return p

    @classmethod
    def load(cls, path=None, required=False):
        """required=True (the firmware): no calibration file is an error, never a silent default: an
        uncalibrated crab has every horn up to 7 deg off and walks badly, or pushes joints into stops."""
        p = Path(path or CAL_FILE)
        if not p.exists():
            if required:
                raise NotCalibrated(f"no calibration at {p}: run robot/calibrate.py first (hardware/ASSEMBLY.md)")
            return cls()
        d = json.loads(p.read_text())
        return cls(sign={k: int(v) for k, v in d["sign"].items()},
                   offset={k: float(v) for k, v in d["offset"].items()})

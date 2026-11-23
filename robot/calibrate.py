"""Calibrate the crab after assembly (and after replacing a servo). Two short steps, for all 18 servos
(every one is a bus servo that reports its angle: legs STS3215, arms STS3032):

    python robot/calibrate.py                     # 1. measure each horn's offset
    python robot/calibrate.py --check-directions  # 2. check each servo turns the way we think
    python robot/calibrate.py --sim               # rehearse both in the simulator

1. Torque is switched OFF; you hold the crab in the calibration pose and press Enter. The pose is
   "everything straight": legs pointing straight out from their hip (along the arrow on the chassis),
   thigh and shin in one straight horizontal line; arms straight forward-out along their shoulder
   arrow, horizontal, claw tips just touching (no squeeze). A straight edge along each limb is enough
   (1 deg error ~ 2 mm at the foot).
2. Still torque OFF: for each joint you bend it a little by hand in the direction the prompt names;
   any servo mounted the other way round than the software expects is fixed.

Everything goes into /var/lib/pinch/calibration.json (PINCH_CALIBRATION overrides); each step keeps
what the other step saved. On the robot, run it with the firmware's own Python, as root (it writes
/var/lib/pinch and must stop the firmware's hold on the servo bus first):

    sudo systemctl stop pinch
    sudo /opt/pinch/venv/bin/python /opt/pinch/robot/calibrate.py   (then --check-directions)
    sudo systemctl start pinch

The firmware refuses to start without this file.
"""

import argparse
import copy
import sys
from math import pi
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.params import JOINT_NAMES  # noqa: E402
from robot.servos import SERVO_IDS, Calibration  # noqa: E402

CAL_POSE = np.zeros(len(JOINT_NAMES))     # "everything straight" = all joint angles 0

# what "+" means for each joint, in words you can see on the robot
DIRECTION = {"coxa": "swing the foot forward (toward the face)",
             "femur": "lift the thigh (knee goes up)",
             "tibia": "raise the foot tip (bend the knee upward)",
             "shoulder_yaw": "swing the arm outward (away from the face)",
             "shoulder_pitch": "lift the arm up",
             "claw": "open the claw"}


def _kind(j):
    return j.split("_", 1)[1]


def measure_offsets(backend, cal_pose=CAL_POSE, samples=10, base=None):
    """Every servo, robot held in cal_pose: offset = servo_angle - sign * joint_angle (averaged).
    The signs come from `base` (check_directions() fixes them)."""
    cal = copy.deepcopy(base) if base is not None else Calibration()
    acc = dict.fromkeys(JOINT_NAMES, 0.0)
    for _ in range(samples):
        st = backend.read_state()
        for j in JOINT_NAMES:
            acc[j] += st[SERVO_IDS[j]].position
        backend.wait(0.01)
    for j in JOINT_NAMES:
        cal.offset[j] = float(acc[j] / samples - cal.sign[j] * cal_pose[JOINT_NAMES.index(j)])
    return cal


def check_directions(backend, cal, ask=input, log=print, joints=None):
    """Torque off: you bend one joint by hand in its + direction and the servo's reading must
    change the way cal.sign predicts; wrong signs are flipped. (Offsets measured in the straight
    pose stay valid: at joint angle 0 the sign doesn't enter.) Returns the flipped joints."""
    flipped = []
    for j in joints or JOINT_NAMES:
        sid = SERVO_IDS[j]
        for _ in range(3):
            a0 = backend.read_state()[sid].position
            ask(f"{j}: from straight, {DIRECTION[_kind(j)]} by about 20 deg, hold it, press Enter ")
            a1 = backend.read_state()[sid].position
            d = (a1 - a0 + pi) % (2 * pi) - pi
            if abs(d) >= np.radians(5):
                break
            log("   that servo didn't move - right joint? try again")
        else:
            log(f"   {j}: no movement seen, skipped (check its cable and ID)")
            continue
        if np.sign(d) != cal.sign[j]:
            cal.sign[j] = -cal.sign[j]
            flipped.append(j)
            log(f"   {j}: this servo turns the other way round - fixed")
    return flipped


class SimPerson:
    """Stands in for you in rehearsals and tests: bends joints of the simulated robot by hand."""

    def __init__(self, sim):
        self.sim = sim

    def __call__(self, prompt):
        j = prompt.split(":")[0].strip()
        if "by about 20 deg" in prompt:                         # bend this joint, keep the rest straight
            pose = CAL_POSE.copy()
            pose[JOINT_NAMES.index(j)] = np.radians(20)
            self.sim.held = pose
            self.sim.wait(0.01)
        return ""


def rehearse():
    """Both steps against the full emulated stack (bus bytes, physics), with random horn offsets
    and two servos (a leg and an arm) the software has the wrong way round."""
    from robot.backends import emulated_bus_backend, random_horn_offsets
    truth = random_horn_offsets(seed=7)
    hw = emulated_bus_backend(mount_offsets=truth)
    hw.sim.held = CAL_POSE                                       # the "person" holds the robot straight
    hw.wait(0.01)
    cal = Calibration()
    wrong = ["FR_tibia", "R_claw"]
    for j in wrong:
        cal.sign[j] = -cal.sign[j]
    cal = measure_offsets(hw, base=cal)
    flipped = check_directions(hw, cal, ask=SimPerson(hw.sim), log=lambda *_: None)
    worst = max(abs(cal.offset[j] - truth[j]) for j in JOINT_NAMES)
    signs = cal.sign == Calibration().sign
    print(f"horn offsets recovered for all {len(JOINT_NAMES)} servos, worst error {np.degrees(worst):.2f} deg")
    print(f"direction check flipped {flipped} (expected {wrong}); all directions right: {signs}")
    return 0 if worst < np.radians(0.5) and signs and sorted(flipped) == sorted(wrong) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="rehearse in simulation with random horn offsets")
    ap.add_argument("--check-directions", action="store_true", help="step 2: servo directions")
    a = ap.parse_args()
    if a.sim:
        return rehearse()
    from robot.backends import RealBackend
    hw = RealBackend()
    hw.torque(False)
    cal = Calibration.load()
    if a.check_directions:
        input("Torque is OFF. Put the crab in the straight calibration pose, then press Enter... ")
        flipped = check_directions(hw, cal)
        print("all servos turn the expected way" if not flipped else f"fixed: {', '.join(flipped)}")
    else:
        input("Torque is OFF. Hold the crab in the calibration pose (see the top of this file), then press Enter... ")
        cal = measure_offsets(hw, base=cal)
        for j in JOINT_NAMES:
            print(f"  {j:18s} offset {np.degrees(cal.offset[j]):7.2f} deg")
            if abs(cal.offset[j]) > np.radians(20):
                print("    ! more than 20 deg: was that horn mounted a few spline teeth off? (see assembly guide)")
    print(f"saved {cal.save()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

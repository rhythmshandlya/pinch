"""Calibrate the crab after assembly (and after replacing a servo). Three short steps:

    python robot/calibrate.py                     # 1. legs: measure each horn's offset
    python robot/calibrate.py --check-directions  # 2. legs: check each servo turns the way we think
    python robot/calibrate.py --trim-arms         # 3. arms: check directions, line them up by eye
    python robot/calibrate.py --sim               # rehearse all three in the simulator

1. Legs (their servos report their angle). Torque is switched OFF; you hold the crab in the
   calibration pose and press Enter. The pose is "everything straight": legs pointing straight
   out from their hip (along the arrow on the chassis), thigh and shin in one straight
   horizontal line. A straight edge along each limb is enough (1 deg error ~ 2 mm at the foot).
2. Still torque OFF: for each leg joint you bend it a little by hand in the direction the
   prompt names; any servo mounted the other way round than the software expects is fixed.
3. Arms (the tiny PWM servos can't report anything). Rest the crab on its belly. Each arm servo
   moves a little and you say whether it went the right way; then you nudge it with + / -
   until it looks straight: arm straight forward-out along its shoulder arrow, horizontal,
   claw tips just touching (no squeeze). Once, the left arm lifts to about 30 deg and you say
   whether it went 15, 30 or 60 deg: that sets how far these servos turn per pulse.

Everything goes into /var/lib/pinch/calibration.json (PINCH_CALIBRATION overrides); each step keeps
what the other steps saved. On the robot, run it with the firmware's own Python, as root (it writes
/var/lib/pinch and must stop the firmware's hold on the servo bus first):

    sudo systemctl stop pinch
    sudo /opt/pinch/venv/bin/python /opt/pinch/robot/calibrate.py   (then --check-directions, --trim-arms)
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

from crab.params import JOINT_NAMES, PARAMS  # noqa: E402
from robot.servos import BUS_JOINTS, PWM_JOINTS, SERVO_IDS, Calibration  # noqa: E402

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


# arm joints whose straight (0) position is the end of their CAD-checked travel
AT_STOP = {k: abs(r[0]) < 1e-6 for k, r in (("shoulder_yaw", PARAMS.arm.shoulder_yaw_range),
                                             ("shoulder_pitch", PARAMS.arm.shoulder_pitch_range),
                                             ("claw", PARAMS.arm.claw_range))}


def measure_offsets(backend, cal_pose=CAL_POSE, samples=10, base=None):
    """Leg servos, robot held in cal_pose: offset = servo_angle - sign * joint_angle (averaged).
    Arm (PWM) servos keep their values from `base`: they are set with trim_arms()."""
    cal = copy.deepcopy(base) if base is not None else Calibration()
    acc = dict.fromkeys(BUS_JOINTS, 0.0)
    for _ in range(samples):
        st = backend.read_state()
        for j in BUS_JOINTS:
            acc[j] += st[SERVO_IDS[j]].position
        backend.wait(0.01)
    for j in BUS_JOINTS:
        cal.offset[j] = float(acc[j] / samples - cal.sign[j] * cal_pose[JOINT_NAMES.index(j)])
    return cal


def check_directions(backend, cal, ask=input, log=print, joints=None):
    """Legs, torque off: you bend one joint by hand in its + direction and the servo's reading must
    change the way cal.sign predicts; wrong signs are flipped. (Offsets measured in the straight
    pose stay valid: at joint angle 0 the sign doesn't enter.) Returns the flipped joints."""
    flipped = []
    for j in joints or BUS_JOINTS:
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


TRAVEL = {"a": 1.0, "b": 2.0, "c": 0.5}      # answer -> gain (b: it went half as far, send twice the pulse)


def trim_arms(backend, cal, ask=input, log=print, settle=0.4, max_nudges=100):
    """Arms (PWM servos): check each one's direction, measure their travel once, then nudge each
    until it looks straight. Lift joints go first, so the travel is known before the rest."""
    q = dict.fromkeys(PWM_JOINTS, 0.0)
    if hasattr(backend, "enable_arms"):              # pulses to the arms while the legs stay limp
        backend.enable_arms()

    def send():
        backend.write_positions({SERVO_IDS[j]: cal.joint_to_servo(j, q[j]) for j in PWM_JOINTS})
        backend.wait(settle)

    send()
    travel_known = False
    for j in sorted(PWM_JOINTS, key=lambda n: (n[0], _kind(n) != "shoulder_pitch")):
        q[j] = np.radians(30 if _kind(j) == "claw" else 15)            # a small test move
        send()
        if ask(f"{j}: did it just {DIRECTION[_kind(j)]}? [Y/n] ").strip().lower().startswith("n"):
            cal.sign[j] = -cal.sign[j]                                  # same position at 0, other way round
            log("   reversed it")
        if _kind(j) == "shoulder_pitch" and not travel_known:          # all six are the same model: ask once
            q[j] = np.radians(30)
            send()
            for _ in range(5):
                a = ask(f"{j}: look from the side - the arm should point about 30 deg up (a third of the way "
                        "to straight up). Is it [a] about there, [b] only about half that (15 deg), or "
                        "[c] about twice that (60 deg)? ").strip().lower()[:1]
                if a in TRAVEL:
                    break
            g = TRAVEL.get(a, 1.0)
            for k in PWM_JOINTS:
                cal.gain[k] = g
            travel_known = True
            if g != 1.0:
                log(f"   these servos turn {1 / g:g}x as far per pulse as assumed - corrected")
        q[j] = 0.0
        hint = "straight"
        if AT_STOP[_kind(j)]:
            # "straight" is also the end of this joint's travel (claw shut / arm against its stop).
            # A servo sent past it just pushes there and still *looks* right, so start a little on
            # the free side and come back until it just touches - never past it.
            cal.offset[j] += cal.sign[j] * cal.gain[j] * np.radians(8)
            hint = "the claw tips just touch" if _kind(j) == "claw" else "it lines up with its arrow"
            log(f"   {j} starts a little {'open' if _kind(j) == 'claw' else 'outward'}: press - until {hint}")
        send()
        for _ in range(max_nudges):
            k = ask(f"{j}: type + or - to nudge 1 deg (++ / -- for 5 deg), just Enter when {hint} ").strip()
            if not k:
                break
            step = {"+": 1, "-": -1, "++": 5, "--": -5}.get(k)
            if step is None:
                log("   type +, -, ++, -- or just press Enter")
                continue
            cal.offset[j] += cal.sign[j] * cal.gain[j] * np.radians(step)   # + always moves the arm its + way
            send()
        if abs(cal.offset[j] / cal.gain[j]) > np.radians(12):
            log(f"   {j} needed {np.degrees(abs(cal.offset[j] / cal.gain[j])):.0f} deg of trim: if you can, move its horn "
                "one spline tooth the other way and trim again (keeps the full range of travel)")
    return cal


class SimPerson:
    """Stands in for you in rehearsals and tests: bends joints of the simulated robot by hand and
    judges arm positions by eye (to about 0.8 deg)."""

    def __init__(self, sim, eye=np.radians(0.8)):  # noqa: B008 - an immutable float
        self.sim, self.eye = sim, eye

    def __call__(self, prompt):
        j = prompt.split(":")[0].strip()
        if "by about 20 deg" in prompt:                         # bend this joint, keep the rest straight
            pose = CAL_POSE.copy()
            pose[JOINT_NAMES.index(j)] = np.radians(20)
            self.sim.held = pose
            self.sim.wait(0.01)
            return ""
        a = self.sim.link_angle(j)
        if "[a] about there" in prompt:                      # travel check: 15, 30 or 60 deg?
            d = np.degrees(a)
            return "b" if d < 21 else "c" if d > 42 else "a"
        if "did it just" in prompt:
            return "y" if a > np.radians(5) else "n"
        if abs(a) <= self.eye:
            return ""
        big = abs(a) > np.radians(4)
        return ("--" if big else "-") if a > 0 else ("++" if big else "+")


def rehearse():
    """All three steps against the full emulated stack (bus bytes, PWM registers, physics),
    with random horn offsets, two servos the software has the wrong way round, and arm servos
    that turn only half as far per pulse as assumed."""
    from robot.backends import emulated_bus_backend, random_horn_offsets
    truth = random_horn_offsets(seed=7)
    hw = emulated_bus_backend(mount_offsets=truth, pwm_travel=0.5)
    hw.sim.held = CAL_POSE                                       # the "person" holds the robot straight
    hw.wait(0.01)
    cal = Calibration()
    wrong = ["FR_tibia", "R_claw"]
    for j in wrong:
        cal.sign[j] = -cal.sign[j]
    cal = measure_offsets(hw, base=cal)
    flipped = check_directions(hw, cal, ask=SimPerson(hw.sim), log=lambda *_: None)
    hw = emulated_bus_backend(mount_offsets=truth, pwm_travel=0.5)   # robot resting, arms free to move
    trim_arms(hw, cal, ask=SimPerson(hw.sim), log=lambda *_: None)
    leg = max(abs(cal.offset[j] - truth[j]) for j in BUS_JOINTS)
    arm = max(abs(cal.offset[j] - truth[j]) * 0.5 for j in PWM_JOINTS)    # in real arm degrees
    signs = cal.sign == Calibration().sign
    print(f"legs: horn offsets recovered, worst error {np.degrees(leg):.2f} deg")
    print(f"legs: direction check flipped {flipped}")
    print(f"arms: travel gain {cal.gain[PWM_JOINTS[0]]:g} (expected 2), trimmed by eye, worst error "
          f"{np.degrees(arm):.2f} deg; all directions right: {signs}")
    return 0 if leg < np.radians(0.5) and arm < np.radians(3) and signs and cal.gain[PWM_JOINTS[0]] == 2.0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="rehearse in simulation with random horn offsets")
    ap.add_argument("--check-directions", action="store_true", help="step 2: leg servo directions")
    ap.add_argument("--trim-arms", action="store_true", help="step 3: arm directions + trim by eye")
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
        print("all leg servos turn the expected way" if not flipped else f"fixed: {', '.join(flipped)}")
    elif a.trim_arms:
        if hw.pwm is None:
            print("no arm servo outputs - enable the six PWM overlays in rsetup (see ASSEMBLY.md stage 1)")
            return 1
        input("Rest the crab on its belly so the legs carry no weight, then press Enter... ")
        trim_arms(hw, cal)
        hw.torque(False)
    else:
        input("Torque is OFF. Hold the crab in the calibration pose (see the top of this file), then press Enter... ")
        cal = measure_offsets(hw, base=cal)
        for j in BUS_JOINTS:
            print(f"  {j:18s} offset {np.degrees(cal.offset[j]):7.2f} deg")
            if abs(cal.offset[j]) > np.radians(20):
                print("    ! more than 20 deg: was that horn mounted a few spline teeth off? (see assembly guide)")
    print(f"saved {cal.save()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

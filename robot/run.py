"""Run the crab: real robot (on the Radxa ZERO 3W) or the simulated stand-in.

    python robot/run.py --sim                    # demo in the MuJoCo viewer
    python robot/run.py --sim --emote happy
    python robot/run.py --check                  # list servos + sensors, torque stays off
    python robot/run.py                          # real robot (after setup_servo + calibrate)
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.gait import _S, _T  # noqa: E402
from crab.params import JOINT_NAMES, servo_for  # noqa: E402
from robot.crab_robot import Crab, SafetyStop  # noqa: E402
from robot.servos import SERVO_IDS  # noqa: E402

V, W = _S / _T, 1 / _T          # demo speeds scaled to this robot's size


def hardware_check(hw, out=print):
    """Ping all 18 bus servos and read the sensors. Torque is never switched on. Returns True if all is well."""
    ok = True
    bus = getattr(hw, "bus", None)
    states = None if bus is not None else hw.read_state()
    out(f"{'ID':>3s}  {'joint':18s} {'servo':8s} {'position':>9s} {'volts':>6s} {'temp':>5s}")
    for j in JOINT_NAMES:
        sid = SERVO_IDS[j]
        if bus is not None and not bus.ping(sid):
            out(f"{sid:3d}  {j:18s} {servo_for(j).name:8s}   MISSING - check its cable, ID and the hub")
            ok = False
            continue
        s = bus.state(sid) if bus is not None else vars(states[sid])
        out(f"{sid:3d}  {j:18s} {servo_for(j).name:8s} {s['position']:8.2f}r {s['voltage']:5.1f}V {s['temperature']:4.0f}C")
    gyro, grav = hw.read_imu()
    level = grav[2] < -0.9
    out(f"IMU gravity {grav.round(2)} -> {'level' if level else 'NOT level (or IMU missing / mounted wrong way)'}")
    ok &= bool(level)
    batt = hw.battery_voltage() if hasattr(hw, "battery_voltage") else None
    if batt is None:
        out("battery monitor: not found (INA219)")
    else:
        out(f"battery {batt:.2f} V ({batt / 2:.2f} V per cell)" + ("  <- charge it" if batt < 7.0 else ""))
    out("ALL OK" if ok else "PROBLEMS FOUND (see above)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="use the simulator instead of the real servos")
    ap.add_argument("--emote", help="play one emote and exit")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--check", action="store_true", help="list servos and sensors, then exit (torque stays off)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if a.sim:
        from robot.backends import SimBackend
        hw = SimBackend(render=True)
    else:
        from robot.backends import RealBackend
        hw = RealBackend(port=a.port)

    if a.check:
        sys.exit(0 if hardware_check(hw) else 1)

    crab = Crab(hw)
    try:
        crab.start()
        if a.emote:
            crab.play_emote(a.emote)
        else:
            for name in ("wave", "happy"):
                crab.play_emote(name)
            crab.walk((0.08 * V, 0, 0), 3)
            crab.walk((0, 0.10 * V, 0), 3)
            crab.walk((0, -0.10 * V, 0), 3)
            crab.walk((0, 0, 0.6 * W), 3)
            crab.play_emote("angry")
        crab.shutdown()
    except SafetyStop as e:
        logging.error("stopped for safety: %s", e)
    except KeyboardInterrupt:
        crab.shutdown()


if __name__ == "__main__":
    main()

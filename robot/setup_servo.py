"""Prepare ONE new servo at a time, before it goes into the robot.

    python robot/setup_servo.py --id 2                 # real servo on /dev/ttyUSB0 (FE-URT-1)
                                                       # (leg servos, IDs 1-12; arm servos need no setup)
    python robot/setup_servo.py --id 2 --rehearse      # same steps against an emulated new servo

1. Connect only this servo to the FE-URT-1 (new servos are all ID 1).
2. The script sets its ID, writes the settings below, and moves it to the centre
   position (0 rad). Keep it powered while you fit the horn and link in the "straight"
   calibration orientation, so offsets stay small.
3. Put a sticker with the ID on the servo.

Register values are explained in hardware/electronics.md.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.params import PARAMS  # noqa: E402
from robot.feetech_bus import MODELS, FeetechBus  # noqa: E402
from robot.servos import SERVO_IDS  # noqa: E402

SETTINGS = {           # register -> value
    "p_gain": 32, "i_gain": 0, "d_gain": 0,
    "return_delay": 0,
}
# max / min input voltage, 0.1 V units. The bus is a regulated 5 V; the STS3032 is rated 4.8-6 V, so a
# reading above 6.5 V means a regulator fault and the servo should refuse rather than cook (hardware audit)
VOLTAGE_LIMITS = {"sts3215": (65, 40), "scs0009": (65, 40)}


def setup(bus: FeetechBus, target_id: int, current_id: int = 1, log=print):
    if not bus.ping(current_id):
        raise SystemExit(f"no servo answers at ID {current_id} - is exactly one servo connected and powered?")
    if current_id != target_id:
        bus.set_id(current_id, target_id)
        time.sleep(0.02)
    sid = target_id
    if not bus.ping(sid):
        raise SystemExit("ID change failed")
    bus.write(sid, "lock", 0)
    for reg, val in SETTINGS.items():
        bus.write(sid, reg, val)
    vmax, vmin = VOLTAGE_LIMITS[bus.m.name]
    bus.write(sid, "max_voltage", vmax)
    bus.write(sid, "min_voltage", vmin)
    bus.write(sid, "lock", 1)
    bus.write(sid, "torque_enable", 1)
    bus.write(sid, "goal_position", bus.m.centre)
    names = {v: k for k, v in SERVO_IDS.items()}
    log(f"servo is now ID {sid} ({names.get(sid, '?')}), settings written, centred. "
        f"Fit the horn/link straight, then label it.")
    return sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True, help="target ID (see the ID map)")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--current-id", type=int, default=1, help="ID the servo has now (new = 1)")
    ap.add_argument("--rehearse", action="store_true", help="run against an emulated brand-new servo")
    a = ap.parse_args()
    from robot.backends import bus_models
    from robot.servos import PWM_CHANNEL
    joint = {v: k for k, v in SERVO_IDS.items()}.get(a.id)
    if joint in PWM_CHANNEL:
        raise SystemExit(f"ID {a.id} is {joint}: a tiny PWM arm servo. It has no ID to set - plug it into "
                         f"output {PWM_CHANNEL[joint]} of the PCA9685 board and use calibrate.py --trim-arms.")
    model = bus_models().get(a.id, MODELS[PARAMS.servo.bus_model])     # legs: STS3032 (STS protocol)
    print(f"servo ID {a.id}: {model.name} protocol")
    if a.rehearse:
        from robot.bus_emulator import EmulatedPort
        port = EmulatedPort(model, [a.current_id])
    else:
        import serial                                      # pip install pyserial
        port = serial.Serial(a.port, baudrate=1_000_000, timeout=0.05)
    setup(FeetechBus(port, model), a.id, a.current_id)


if __name__ == "__main__":
    main()

"""Pinch's firmware: the program that runs on the robot (Radxa ZERO 3W), started by systemd at boot.

    python robot/pinch_main.py                 # the real robot
    python robot/pinch_main.py --emulate 60    # the same firmware against emulated hardware (self-test)

One 50 Hz loop runs the body (robot.crab_robot.Crab: limits, speed, battery and bus checks), the mind at
10 Hz (pinch/mind.py - the same brain as the digital twin and the laptop app) and the reflexes. The senses
run in their own threads and only hand results to the loop, so nothing slow ever sits in the 20 ms tick:
  touch    MPR121 zones -> gestures (in the loop: one 2-byte read)
  cliff    5 x VL53L0X ring, own thread          ears   2 I2S mics (stereo) -> hearing, own thread
  eyes     wide camera, own thread: where am I (tray marker), how fast am I really moving (ground flow),
           my things, your face - the loop only receives the answers
  face     SSD1309 OLED, own I2C bus, own thread  sound  tones through the I2S amp (aplay)
A small HTTP API on port 8080 lets the laptop or a phone talk to Pinch (typed speech, status):
    POST /say {"text": "..."}   POST /touch {"gesture": "stroke"}   GET /status
Speech understanding uses the laptop's Ollama when it's reachable (PINCH_OLLAMA=http://laptop:11434),
and the instant keyword rules otherwise.

Hardware hiccups (a NACK on I2C, a garbled servo reply) are counted and ridden out (stats["errors"]);
only a fault that persists becomes a safety stop. `systemctl stop` (SIGTERM) sits Pinch down, turns the
servos' torque off and saves its memory; the memory is also saved every minute.

The code lives in robot/firmware/ (the firmware), robot/localizer.py and robot/http_api.py; this file is
the entry point systemd runs.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot.firmware import PinchFirmware  # noqa: E402,F401
from robot.firmware.common import log  # noqa: E402
from robot.http_api import serve  # noqa: E402,F401
from robot.localizer import Localizer  # noqa: E402,F401  (old import path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emulate", type=float, default=0, help="run against emulated hardware for N seconds")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PINCH_PORT", 8080)))
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--record", default=os.environ.get("PINCH_RECORD") or None, metavar="DIR",
                    help="record the session (timeline, samples, observer report) into DIR (pinch/recorder.py)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.emulate:
        from robot.hw_emulated import EmulatedHardware
        hw = EmulatedHardware()
        fw = PinchFirmware(hw, llm=False, memory_path=False, record_dir=a.record)
        serve(fw, a.port)
        fw.run(a.emulate, realtime=False)
        print(json.dumps(fw.status(), indent=1))
        return 0
    from robot.hw_real import RealHardware
    from robot.servos import NotCalibrated
    try:
        hw = RealHardware()
    except NotCalibrated as e:
        log.error("%s", e)
        return 2                         # systemd retries a few times, then gives up (StartLimitBurst)
    fw = PinchFirmware(hw, llm=not a.no_llm, record_dir=a.record)
    serve(fw, a.port)
    fw.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

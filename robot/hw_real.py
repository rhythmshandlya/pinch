"""The real robot's hardware for the firmware (Radxa ZERO 3W). See hardware/ASSEMBLY.md stage 1 for the
overlays this needs (I2C3-M0, I2C4-M0, the six PWMs, the camera, crab-audio).
"""

import logging
import shutil
import subprocess
import threading

import numpy as np

from pinch.world import Desk
from robot.backends import RealBackend
from robot.claw_fk import ClawFK
from robot.servos import Calibration

log = logging.getLogger("pinch")
SENSOR_BUS, FACE_BUS = 3, 4             # I2C3-M0 (pins 3/5), I2C4-M0 (pins 27/28)
XSHUT_LINES = (("gpiochip3", 1), ("gpiochip3", 2))    # pins 11 / 13 = GPIO3_A1 / GPIO3_A2


class RealHardware:
    threaded = True
    realtime = True

    def __init__(self):
        from smbus2 import SMBus
        self.desk = Desk()
        self.backend = RealBackend()
        self.calibration = Calibration.load()
        self.start_pose = (0.0, 0.0, 0.0)          # corrected as soon as the camera sees the tray marker
        self.fk = ClawFK()
        self.oled = self.touch = self.cliff = None
        try:
            from robot.drivers.i2c_devices import SSD1309
            self.oled = SSD1309(SMBus(FACE_BUS))
        except OSError as e:
            log.warning("face screen not found on I2C%d: %s", FACE_BUS, e)
        sensors = SMBus(SENSOR_BUS)
        try:
            from robot.drivers.i2c_devices import MPR121
            self.touch = MPR121(sensors)
        except OSError as e:
            log.warning("touch controller not found: %s", e)
        try:
            from robot.drivers.i2c_devices import cliff_pair
            self.cliff = cliff_pair(sensors, *[self._gpio_out(c, n) for c, n in XSHUT_LINES])
        except Exception as e:                       # noqa: BLE001 - the robot still runs without them
            log.warning("cliff sensors not available (%s): edges rely on the camera's desk map", e)
        self.camera_model = None
        self.camera = None
        try:
            import cv2
            from robot.vision import Camera
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera_model = Camera.load()
            self.camera = self._read_camera
        except Exception as e:                       # noqa: BLE001
            log.warning("camera not available: %s", e)
        self._arec = None
        self.audio = self._read_audio if shutil.which("arecord") else None
        self._play_lock = threading.Lock()

    @staticmethod
    def _gpio_out(chip, line):
        import gpiod                                   # apt install python3-libgpiod
        req = gpiod.request_lines(f"/dev/{chip}", consumer="pinch",
                                  config={line: gpiod.LineSettings(direction=gpiod.line.Direction.OUTPUT)})
        return lambda v: req.set_value(line, gpiod.line.Value.ACTIVE if v else gpiod.line.Value.INACTIVE)

    def tip(self, side, q):
        return self.fk.tip(side, q)

    def _read_camera(self):
        ok, frame = self._cap.read()
        return frame if ok else None

    def _read_audio(self):
        if self._arec is None:
            self._arec = subprocess.Popen(["arecord", "-q", "-D", "plughw:CARD=crabaudio", "-f", "S32_LE", "-c", "2",
                                           "-r", "16000", "-t", "raw"], stdout=subprocess.PIPE)
        raw = self._arec.stdout.read(1600 * 2 * 4)    # 0.1 s
        if not raw:
            return None
        a = np.frombuffer(raw, "<i4").astype(np.float32) / 2 ** 31
        return a.reshape(-1, 2)

    def play(self, name):
        from brain.sounds import render

        def go():
            with self._play_lock:
                subprocess.run(["aplay", "-q", "-"], input=render(name), check=False)
        threading.Thread(target=go, daemon=True).start()

    def battery_fraction(self):
        v = self.backend.battery_voltage()
        return 1.0 if v is None else float(np.clip((v - 6.4) / (8.4 - 6.4), 0, 1))

    def feet_down(self):
        """Feet carrying weight, from the thigh servos' load (a lifted crab's legs hang slack)."""
        st = self.backend.read_state()
        from robot.servos import SERVO_IDS
        loads = [abs(st[SERVO_IDS[f"{lg}_femur"]].load) for lg in ("FL", "FR", "RL", "RR")]
        return sum(load > 0.04 for load in loads)

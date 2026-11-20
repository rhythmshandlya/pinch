"""The real robot's hardware for the firmware (Radxa ZERO 3W). See hardware/ASSEMBLY.md stage 1 for the
overlays this needs (I2C3-M0, I2C4-M0, the six PWMs, the camera, crab-audio).

Each I2C device gets its own bus handle: smbus2 sets the device address and then talks to it in two
separate system calls, so two threads sharing one handle (the cliff thread and the touch reads) could
send one device's bytes to the other. Separate handles are safe: the kernel serialises whole transfers.
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
SENSOR_BUS, FACE_BUS, CLIFF_BUS = 3, 4, 5      # I2C3-M0 (pins 3/5), I2C4-M0 (27/28), I2C5-M0 (29/31)
# The cliff ring (hardware/electronics.md): beam -> its I2C address and XSHUT line. The one at 0x29 wakes
# last. GPIO3_xn is line 8 * (x - A) + n of gpiochip3 (check with `gpioinfo gpiochip3` on the first boot).
CLIFF_WIRING = {
    "front_left": (0x30, ("gpiochip3", 1)),       # pin 11, GPIO3_A1
    "front_right": (0x29, ("gpiochip3", 2)),      # pin 13, GPIO3_A2
    "rear": (0x31, ("gpiochip3", 8)),             # pin 15, GPIO3_B0
    "side_left": (0x32, ("gpiochip3", 17)),       # pin 22, GPIO3_C1
    "side_right": (0x33, ("gpiochip3", 7)),       # pin 36, GPIO3_A7
}
AUDIO_CARD = "plughw:CARD=crabaudio"     # hardware/radxa/crab-audio.dts (mics and amp)
AUDIO_CHUNK = 1600                       # frames per read: 0.1 s at 16 kHz


def gpio_output(chip, line, gpiod=None):
    """One GPIO line as an output, low to start. Returns set(level).

    libgpiod has two incompatible Python APIs: v2 (gpiod.request_lines, pip `gpiod>=2`) and v1 (gpiod.Chip /
    get_line / request), which is what Debian 12's python3-libgpiod 1.6 is. Both are handled."""
    if gpiod is None:
        import gpiod
    path = f"/dev/{chip}"
    if hasattr(gpiod, "request_lines"):                              # v2
        Value = gpiod.line.Value
        req = gpiod.request_lines(path, consumer="pinch", config={line: gpiod.LineSettings(
            direction=gpiod.line.Direction.OUTPUT, output_value=Value.INACTIVE)})
        setter = lambda v: req.set_value(line, Value.ACTIVE if v else Value.INACTIVE)  # noqa: E731
        setter.keep = req                                            # the request lives as long as the setter
        return setter
    c = gpiod.Chip(path)                                             # v1 (1.5 / 1.6)
    ln = c.get_line(line)
    ln.request(consumer="pinch", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    setter = lambda v: ln.set_value(1 if v else 0)                   # noqa: E731
    setter.keep = (c, ln)
    return setter


class RealHardware:
    threaded = True
    realtime = True

    def __init__(self, smbus=None, backend=None, gpiod=None, video=None):
        """The keyword arguments exist for the tests (fake smbus2 / backend / gpiod / camera)."""
        if smbus is None:
            from smbus2 import SMBus as smbus
        self.desk = Desk()
        self.calibration = Calibration.load(required=True)       # refuse to run an uncalibrated robot
        self.backend = backend if backend is not None else RealBackend()
        self.start_pose = (0.0, 0.0, 0.0)          # corrected as soon as the camera sees the tray marker
        self.fk = ClawFK()
        self.oled = self.touch = self.cliff = None
        self._gpio = []
        try:
            from robot.drivers.i2c_devices import SSD1309
            self.oled = SSD1309(smbus(FACE_BUS))
        except OSError as e:
            log.warning("face screen not found on I2C%d: %s", FACE_BUS, e)
        try:
            from robot.drivers.i2c_devices import MPR121
            self.touch = MPR121(smbus(SENSOR_BUS))                   # its own handle (see the module docstring)
        except OSError as e:
            log.warning("touch controller not found: %s", e)
        self.cliff = self._cliff_ring(smbus, gpiod)
        self.camera_model = None
        self.camera = None
        self._cap = None
        try:
            import cv2

            from robot.vision import Camera
            self._cap = (video or cv2.VideoCapture)(0)
            if not self._cap.isOpened():
                raise OSError("no camera at /dev/video0 (ribbon cable, or the camera overlay in rsetup)")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera_model = Camera.load()
            self.camera = self._read_camera
        except Exception as e:                       # noqa: BLE001
            log.warning("camera not available: %s", e)
            if self._cap is not None:
                self._cap.release()
            self._cap = None
        self._arec = None
        self.audio = self._read_audio if shutil.which("arecord") else None
        self._play_lock = threading.Lock()

    def _cliff_ring(self, smbus, gpiod):
        """{beam: VL53L0X or None}. Whatever fails, the beam stays in the dict as None, so the firmware
        knows it is missing and won't walk toward its side."""
        from robot.drivers.i2c_devices import cliff_ring
        ring = dict.fromkeys(CLIFF_WIRING)
        try:
            bus = smbus(CLIFF_BUS)                    # only the cliff thread uses this bus
        except OSError as e:
            log.warning("cliff sensor bus I2C%d not found (%s): enable I2C5-M0 in rsetup", CLIFF_BUS, e)
            return ring
        sensors = []
        for name, (addr, (chip, line)) in CLIFF_WIRING.items():
            try:
                set_level = gpio_output(chip, line, gpiod)
            except Exception as e:                   # noqa: BLE001 - one bad line costs one beam
                log.warning("cliff sensor %s: XSHUT line %s/%d not available (%s)", name, chip, line, e)
                continue
            self._gpio.append(set_level)
            sensors.append((name, addr, set_level))
        ring.update(cliff_ring(bus, sensors, log=log.warning))
        return ring

    def tip(self, side, q):
        return self.fk.tip(side, q)

    def _read_camera(self):
        ok, frame = self._cap.read()
        return frame if ok else None

    def _read_audio(self):
        if self._arec is None or self._arec.poll() is not None:     # (re)start arecord if it died
            self._arec = subprocess.Popen(["arecord", "-q", "-D", AUDIO_CARD, "-f", "S32_LE", "-c", "2",
                                           "-r", "16000", "-t", "raw"], stdout=subprocess.PIPE)
        raw = self._arec.stdout.read(AUDIO_CHUNK * 2 * 4)
        if not raw:
            return None
        a = np.frombuffer(raw[:len(raw) // 8 * 8], "<i4").astype(np.float32) / 2 ** 31
        return a.reshape(-1, 2)

    def play(self, name):
        from pinch.sounds import render

        def go():
            with self._play_lock:
                subprocess.run(["aplay", "-q", "-D", AUDIO_CARD, "-"], input=render(name), check=False)
        threading.Thread(target=go, daemon=True).start()

    def cpu_temp(self, path="/sys/class/thermal/thermal_zone0/temp"):
        """The RK3566's temperature, C (it sits in a closed shell with a stick-on heatsink), or None."""
        try:
            with open(path) as f:
                return int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            return None

    def poweroff(self):
        """The pack is empty: shut the computer down cleanly, so the Radxa and the screen stop draining it
        (the 2S protection board is the hard floor if this never happens)."""
        subprocess.run(["systemctl", "poweroff"], check=False)

    def battery_fraction(self):
        v = self.backend.battery_voltage()
        return 1.0 if v is None else float(np.clip((v - 6.4) / (8.4 - 6.4), 0, 1))

    def feet_down(self, state_fn=None):
        """Feet carrying weight, from the thigh servos' load (a lifted crab's legs hang slack). state_fn: the
        firmware's once-per-tick cached servo state (robot.crab_robot.Crab.state)."""
        st = state_fn() if state_fn is not None else self.backend.read_state()
        from robot.servos import SERVO_IDS
        loads = [abs(st[SERVO_IDS[f"{lg}_femur"]].load) for lg in ("FL", "FR", "RL", "RR")]
        return sum(load > 0.04 for load in loads)

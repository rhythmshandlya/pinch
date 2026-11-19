"""Emulated hardware for the firmware self-test: the desk digital twin stands in for the robot.

The firmware runs unchanged on top. The I2C drivers (OLED, touch, cliff sensors) are the real driver
code talking to fake chips; the camera is a render of the twin's eye camera; the microphones play
synthetic voices (quiet room, talk to Pinch, or a meeting).
"""

import threading

import mujoco
import numpy as np

from pinch.hearing import RATE, synth_speech
from pinch.world import Desk
from robot.backends import SimBackend
from robot.claw_fk import ClawFK
from robot.drivers.fakes import FakeBus, FakeOLED, FakeRanger, FakeTouch
from robot.drivers.i2c_devices import MPR121, SSD1309, VL53L0X
from robot.servos import Calibration
from robot.vision import Camera
from sim.desk import build_desk_model


class EmulatedHardware:
    threaded = False                     # deterministic: the firmware polls the senses in its own loop
    realtime = False

    def __init__(self, start=None):
        self.desk = Desk()
        self.m = build_desk_model(self.desk, start=start) if start else build_desk_model(self.desk)
        self.backend = SimBackend(model=self.m)
        self.d = self.backend.d
        self.calibration = Calibration()
        self.start_pose = (0.0, 0.0, 0.0)
        self.fk = ClawFK(self.m)
        self.oled_chip = FakeOLED()
        self.oled = SSD1309(FakeBus(self.oled_chip))
        self.touch_chip = FakeTouch()
        self.touch = MPR121(FakeBus(self.touch_chip))
        self.rangers = [FakeRanger(0x29), FakeRanger(0x29)]
        self.cliff = [VL53L0X(FakeBus(r)) for r in self.rangers]
        self.camera_model = Camera.from_mujoco(self.m)
        self._renderer = None
        self.audio_mode = "quiet"        # quiet | talk | meeting
        self._audio_buf = np.zeros((0, 2))
        self.played = []
        self.lock = threading.Lock()

    def tip(self, side, q):
        return self.fk.tip(side, q)

    def camera(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, 480, 640)
        self._renderer.update_scene(self.d, camera="eye_cam")
        return np.ascontiguousarray(self._renderer.render()[:, :, ::-1])     # RGB -> BGR like a real camera

    def audio(self):
        """0.1 s of stereo audio; the voice reaches the left mic a little earlier (you sit to Pinch's left)."""
        if len(self._audio_buf) < RATE // 10:
            if self.audio_mode == "quiet":
                mono = np.random.default_rng().normal(0, 0.002, RATE)
            else:
                mono = synth_speech(1.0, talk=0.55 if self.audio_mode == "meeting" else 0.35,
                                    seed=int(self.d.time * 10))
            right = np.concatenate([np.zeros(2), mono[:-2]])
            self._audio_buf = np.concatenate([self._audio_buf, np.stack([mono, right], axis=1)])
        out, self._audio_buf = self._audio_buf[:RATE // 10], self._audio_buf[RATE // 10:]
        return out

    def update_cliff(self):
        for r, name in zip(self.rangers, ("cliff_left", "cliff_right")):
            v = float(self.d.sensor(name).data[0])
            r.mm = 9999 if v < 0 else int(v * 1000)

    def play(self, name):
        self.played.append(name)

    def battery_fraction(self):
        return 0.8

    def feet_down(self):
        feet = {self.m.geom(f"{lg}_foot").id for lg in ("FL", "FR", "RL", "RR")}
        desk = self.m.geom("desk_top").id
        n = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if desk in (c.geom1, c.geom2):
                o = c.geom2 if c.geom1 == desk else c.geom1
                if o in feet:
                    n.add(o)
        return len(n)

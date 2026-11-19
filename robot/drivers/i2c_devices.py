"""I2C drivers for Pinch's face screen, touch pads and cliff sensors (plain registers over smbus2).

  SSD1309  2.42" 128x64 OLED (the face) - its own bus, I2C4-M0 (pins 27/28), address 0x3C
  MPR121   12-channel capacitive touch (copper-tape pads inside the shell) - I2C3, 0x5A
  VL53L0X  time-of-flight distance: the cliff ring, five beams looking out and down (front pair, rear,
           both sides; crab/params.py CLIFF_BEAMS) on their own bus, I2C5-M0 (pins 29/31). All start
           at 0x29; at boot cliff_ring() brings them up one at a time with their XSHUT pins and moves each
           to its own address.

Every class takes any object with the smbus2 SMBus methods it uses, so the self-test can drive the
exact same code against fake chips (robot/drivers/fakes.py).
"""

import time

import numpy as np


class SSD1309:
    """128x64 monochrome OLED. show(pages) takes the 1024-byte page layout from pinch.face.Face."""

    INIT = [0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x00, 0xA1, 0xC8,
            0xDA, 0x12, 0x81, 0xCF, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0xAF]

    def __init__(self, bus, address=0x3C):
        self.bus, self.addr = bus, address
        for c in self.INIT:
            self.bus.write_byte_data(self.addr, 0x00, c)          # 0x00 = command stream

    def show(self, pages: bytes):
        self.bus.write_i2c_block_data(self.addr, 0x00, [0x21, 0, 127, 0x22, 0, 7])   # whole screen
        for i in range(0, len(pages), 32):                                          # smbus blocks <= 32
            self.bus.write_i2c_block_data(self.addr, 0x40, list(pages[i:i + 32]))  # 0x40 = data stream

    def contrast(self, level):
        self.bus.write_i2c_block_data(self.addr, 0x00, [0x81, int(level) & 0xFF])

    def off(self):
        self.bus.write_byte_data(self.addr, 0x00, 0xAE)


ZONES = ["head", "back_left", "back_right", "side_left", "side_right", "face", "claw_left", "claw_right"]


class MPR121:
    """Touch zones: electrode k = ZONES[k] (copper tape inside the shell, wired to E0..E7)."""

    def __init__(self, bus, address=0x5A, touch=12, release=6):
        self.bus, self.addr = bus, address
        w = lambda r, v: self.bus.write_byte_data(self.addr, r, v)
        w(0x80, 0x63)                        # soft reset
        time.sleep(0.001)
        w(0x5E, 0x00)                        # stop mode to configure
        for e in range(12):
            w(0x41 + 2 * e, touch)           # touch threshold
            w(0x42 + 2 * e, release)         # release threshold
        for r, v in ((0x2B, 0x01), (0x2C, 0x01), (0x2D, 0x0E), (0x2E, 0x00), (0x2F, 0x01), (0x30, 0x05),
                     (0x31, 0x01), (0x32, 0x00), (0x5B, 0x00), (0x5C, 0x10), (0x5D, 0x20)):
            w(r, v)                          # baseline filters, debounce, charge current/time
        w(0x5E, 0x80 | len(ZONES))           # run: baseline tracking on, electrodes 0..7

    def touched(self) -> set:
        lo, hi = self.bus.read_i2c_block_data(self.addr, 0x00, 2)
        bits = lo | (hi << 8)
        return {z for k, z in enumerate(ZONES) if bits & (1 << k)}


class VL53L0X:
    """Minimal single-shot ranging for cliff detection: the default settings, no fine tuning. Plenty to
    tell 'desk 5 cm away' from 'nothing within 50 cm'. (Verify on the real sensor: README section.)"""

    DEFAULT = 0x29                  # every VL53L0X wakes at this address

    def __init__(self, bus, address=0x29):
        self.bus, self.addr = bus, address
        w = lambda r, v: self.bus.write_byte_data(self.addr, r, v)
        for r, v in ((0x88, 0x00), (0x80, 0x01), (0xFF, 0x01), (0x00, 0x00)):
            w(r, v)
        self.stop = self.bus.read_byte_data(self.addr, 0x91)
        for r, v in ((0x00, 0x01), (0xFF, 0x00), (0x80, 0x00)):
            w(r, v)

    def set_address(self, new):
        self.bus.write_byte_data(self.addr, 0x8A, new & 0x7F)
        self.addr = new

    FAR = 9999          # "nothing in range": no desk under the beam (a real reading, not a failure)

    def start(self):
        """Begin one single-shot measurement (~30 ms); several sensors can measure at the same time."""
        w = lambda r, v: self.bus.write_byte_data(self.addr, r, v)
        for r, v in ((0x80, 0x01), (0xFF, 0x01), (0x00, 0x00), (0x91, self.stop), (0x00, 0x01), (0xFF, 0x00),
                     (0x80, 0x00), (0x00, 0x01)):
            w(r, v)
        self._t0 = time.monotonic()

    def result(self, timeout=0.05):
        """The measurement start() began: mm; FAR when nothing is in range; None when the sensor didn't
        answer in time (a failure: the caller must read it as neither desk nor cliff)."""
        t0 = getattr(self, "_t0", time.monotonic())
        while self.bus.read_byte_data(self.addr, 0x00) & 0x01:          # start bit clears when it starts
            if time.monotonic() - t0 > timeout:
                return None
        while not (self.bus.read_byte_data(self.addr, 0x13) & 0x07):    # result ready
            if time.monotonic() - t0 > timeout:
                return None
        hi, lo = self.bus.read_i2c_block_data(self.addr, 0x14 + 10, 2)
        self.bus.write_byte_data(self.addr, 0x0B, 0x01)                  # clear interrupt
        mm = (hi << 8) | lo
        return self.FAR if mm >= 8000 else mm

    def range_mm(self, timeout=0.05):
        self.start()
        return self.result(timeout)


def cliff_ring(bus, sensors, log=None):
    """Bring up a ring of VL53L0X that all wake at 0x29. sensors: [(name, address, xshut)] where xshut is
    set(level) for that sensor's XSHUT pin. All are held in reset, then woken one at a time and moved to their
    address; the one that keeps 0x29 comes last. A sensor that doesn't come up is put back in reset (so it
    can't sit on 0x29 and answer for another) and returned as None: the firmware then treats its direction as
    "don't know" (no walking that way). Returns {name: VL53L0X or None}."""
    for _, _, xshut in sensors:
        xshut(0)
    time.sleep(0.01)
    out = {}
    for name, addr, xshut in sorted(sensors, key=lambda s: s[1] == VL53L0X.DEFAULT):
        xshut(1)
        time.sleep(0.01)                                         # boot: 1.2 ms typical
        try:
            dev = VL53L0X(bus)
            if addr != VL53L0X.DEFAULT:
                dev.set_address(addr)
            out[name] = dev
        except OSError as e:
            xshut(0)
            out[name] = None
            if log:
                log(f"cliff sensor {name} (0x{addr:02x}) not answering: {e}")
    return out


def read_ring(ring, timeout=0.05):
    """One reading from every sensor of a ring, all measuring at once (5 x 30 ms one after another would
    make the readings 150 ms old): {name: mm, FAR or None}. A sensor that raises is None."""
    started = {}
    for name, dev in ring.items():
        if dev is None:
            continue
        try:
            dev.start()
            started[name] = dev
        except OSError:
            pass
    out = {name: None for name in ring}
    for name, dev in started.items():
        try:
            out[name] = dev.result(timeout)
        except OSError:
            pass
    return out


def pages_to_image(pages: bytes) -> np.ndarray:
    """Inverse of the page layout (for tests and the companion app): 64 x 128 bool image."""
    a = np.frombuffer(bytes(pages), np.uint8).reshape(8, 128)
    return ((a[:, None, :] >> np.arange(8)[None, :, None]) & 1).reshape(64, 128).astype(bool)

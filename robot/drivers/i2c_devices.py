"""I2C drivers for Pinch's face screen, touch pads and cliff sensors (plain registers over smbus2).

  SSD1309  2.42" 128x64 OLED (the face) - its own bus, I2C4-M0 (pins 27/28), address 0x3C
  MPR121   12-channel capacitive touch (copper-tape pads inside the shell) - I2C3, 0x5A
  VL53L0X  time-of-flight distance, two of them looking forward-down (cliff + obstacle) - I2C3,
           both start at 0x29; at boot the XSHUT pins bring them up one at a time and the first is
           moved to 0x30.

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

    def range_mm(self, timeout=0.05):
        w = lambda r, v: self.bus.write_byte_data(self.addr, r, v)
        for r, v in ((0x80, 0x01), (0xFF, 0x01), (0x00, 0x00), (0x91, self.stop), (0x00, 0x01), (0xFF, 0x00),
                     (0x80, 0x00), (0x00, 0x01)):
            w(r, v)
        t0 = time.monotonic()
        while self.bus.read_byte_data(self.addr, 0x00) & 0x01:          # start bit clears when it starts
            if time.monotonic() - t0 > timeout:
                return None
        while not (self.bus.read_byte_data(self.addr, 0x13) & 0x07):    # result ready
            if time.monotonic() - t0 > timeout:
                return None
        hi, lo = self.bus.read_i2c_block_data(self.addr, 0x14 + 10, 2)
        w(0x0B, 0x01)                                                    # clear interrupt
        mm = (hi << 8) | lo
        return None if mm >= 8000 else mm


def cliff_pair(bus, xshut_left, xshut_right):
    """Bring up both VL53L0X: right held in reset while the left moves to 0x30. xshut_* are callables
    set(level) for the two GPIO lines (pins 11 / 13)."""
    xshut_left(0)
    xshut_right(0)
    time.sleep(0.01)
    xshut_left(1)
    time.sleep(0.01)
    left = VL53L0X(bus)
    left.set_address(0x30)
    xshut_right(1)
    time.sleep(0.01)
    right = VL53L0X(bus)
    return left, right


def pages_to_image(pages: bytes) -> np.ndarray:
    """Inverse of the page layout (for tests and the companion app): 64 x 128 bool image."""
    a = np.frombuffer(bytes(pages), np.uint8).reshape(8, 128)
    return ((a[:, None, :] >> np.arange(8)[None, :, None]) & 1).reshape(64, 128).astype(bool)

"""Fake I2C chips with the same register behaviour the drivers rely on - for the firmware self-test.

FakeOLED keeps the last frame written; FakeTouch reports whatever zones the test 'touches';
FakeRanger answers ranging with a distance the test (or the simulator's cliff sensors) sets.
"""


class FakeBus:
    """One I2C bus with several fake chips on it (address -> chip)."""

    def __init__(self, *chips):
        self.chips = {c.addr: c for c in chips}

    def _chip(self, addr):
        if addr not in self.chips:
            raise OSError(121, "Remote I/O error")
        return self.chips[addr]

    def write_byte_data(self, addr, reg, val):
        c = self._chip(addr)
        c.write(reg, [val])
        if c.addr != addr:                                   # address change
            self.chips[c.addr] = self.chips.pop(addr)

    def write_i2c_block_data(self, addr, reg, data):
        self._chip(addr).write(reg, list(data))

    def read_byte_data(self, addr, reg):
        return self._chip(addr).read(reg, 1)[0]

    def read_i2c_block_data(self, addr, reg, n):
        return self._chip(addr).read(reg, n)


class FakeOLED:
    def __init__(self, addr=0x3C):
        self.addr = addr
        self.on = False
        self.frame = bytearray(1024)
        self.ptr = 0
        self.frames = 0

    def write(self, reg, data):
        if reg == 0x00:
            if 0xAF in data:
                self.on = True
            if 0xAE in data:
                self.on = False
            if data[:1] == [0x21]:
                self.ptr = 0
        elif reg == 0x40:
            for b in data:
                self.frame[self.ptr % 1024] = b
                self.ptr += 1
            if self.ptr % 1024 == 0:
                self.frames += 1

    def read(self, reg, n):
        return [0] * n


class FakeTouch:
    def __init__(self, addr=0x5A):
        self.addr = addr
        self.touching = set()
        self.running = False

    def write(self, reg, data):
        if reg == 0x5E:
            self.running = bool(data[0] & 0x0F)

    def read(self, reg, n):
        from robot.drivers.i2c_devices import ZONES
        bits = sum(1 << k for k, z in enumerate(ZONES) if z in self.touching)
        return [bits & 0xFF, bits >> 8][:n]


class FakeRanger:
    def __init__(self, addr=0x29):
        self.addr = addr
        self.mm = 55
        self.regs = {0x91: 0x3C}

    def write(self, reg, data):
        if reg == 0x8A:
            self.addr = data[0]
        self.regs[reg] = data[0]

    def read(self, reg, n):
        if reg == 0x00:
            return [0]                                       # measurement started
        if reg == 0x13:
            return [0x04]                                    # result ready
        if reg == 0x14 + 10:
            return [(self.mm >> 8) & 0xFF, self.mm & 0xFF][:n]
        return [self.regs.get(reg, 0)] * n

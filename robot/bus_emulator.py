"""Byte-level emulation of a chain of Feetech bus servos, for testing without hardware.

EmulatedPort behaves like a serial port: the driver writes real packets, every addressed
servo parses them against its own register memory and answers with real status packets.
An optional `physics` object links the servos to the MuJoCo simulator:
    physics.set_goal(servo_id, rad)  /  physics.get_state(servo_id) -> dict(position, speed, ...)
"""

from robot.feetech_bus import BROADCAST, PING, READ, SYNC_READ, SYNC_WRITE, WRITE, Model, checksum


class EmulatedServo:
    def __init__(self, sid: int, model: Model):
        self.m = model
        self.mem = bytearray(256)
        self.mem[5] = sid
        self._put("model", 9 if model.name == "scs0009" else 0x0309)
        self._put("present_position", model.centre)
        self._put("goal_position", model.centre)
        self._put("present_voltage", 50)                    # 5.0 V bus
        self._put("present_temperature", 32)
        self.mem[self.m.reg["lock"][0]] = 1

    @property
    def id(self):
        return self.mem[5]

    def _put(self, name, value):
        a, s = self.m.reg[name]
        self.mem[a:a + s] = (value & ((1 << 8 * s) - 1)).to_bytes(s, "little" if self.m.little_endian else "big")

    def _get(self, name):
        a, s = self.m.reg[name]
        return int.from_bytes(self.mem[a:a + s], "little" if self.m.little_endian else "big")

    def write_mem(self, addr, data):
        eeprom = addr < 40                               # EEPROM area is write-protected while locked
        if eeprom and self.mem[self.m.reg["lock"][0]] == 1:
            return 0x08                                   # "instruction error" flag
        self.mem[addr:addr + len(data)] = data
        return 0


class EmulatedPort:
    def __init__(self, model, ids, physics=None, extra=None):
        """model: one Model, or {id: Model} for a mixed bus."""
        pick = (lambda i: model[i]) if isinstance(model, dict) else (lambda i: model)
        self.m = pick(next(iter(ids)))
        self.servos = {i: EmulatedServo(i, pick(i)) for i in ids}
        for s in (extra or []):                           # e.g. a brand-new servo still at ID 1
            self.servos[s.id] = s
        self.physics = physics
        self.rx = bytearray()
        self.log = []                                     # (instr, ids) for tests

    # serial-port interface
    def write(self, data: bytes):
        buf = bytes(data)
        while len(buf) >= 6 and buf[:2] == b"\xff\xff":
            sid, length, instr = buf[2], buf[3], buf[4]
            params, chk = buf[5:3 + length], buf[3 + length]      # LEN counts params + instr + chk
            if checksum(bytes([sid, length, instr]) + params) == chk:
                self._handle(sid, instr, params)
            buf = buf[4 + length:]
        return len(data)

    def read(self, n: int) -> bytes:
        out, self.rx[:] = bytes(self.rx[:n]), self.rx[n:]
        return out

    def reset_input_buffer(self):
        self.rx.clear()

    # protocol
    def _reply(self, servo, err=0, params=b""):
        body = bytes([servo.id, len(params) + 2, err]) + params
        self.rx += b"\xff\xff" + body + bytes([checksum(body)])

    def _refresh(self, servo):
        if self.physics is not None:
            m = servo.m
            st = self.physics.get_state(servo.id)
            servo._put("present_position", m.rad_to_raw(st["position"]))
            spd = int(round(st["speed"] * m.steps / m.span))
            servo._put("present_speed", (abs(spd) | (0x8000 if spd < 0 else 0)) & 0xFFFF)
            ld = int(round(st["load"] * 1000))
            servo._put("present_load", (min(abs(ld), 1000) | (0x400 if ld < 0 else 0)))
            servo._put("present_voltage", int(round(st["voltage"] * 10)))
            servo._put("present_temperature", int(st["temperature"]))

    def _after_write(self, servo, addr):
        if self.physics is None:
            return
        m = servo.m
        ga = m.reg["goal_position"][0]
        if addr <= ga < addr + 2 or addr == ga:
            self.physics.set_goal(servo.id, m.raw_to_rad(servo._get("goal_position")))
        ta = m.reg["torque_enable"][0]
        if addr == ta:
            self.physics.set_torque(servo.id, bool(servo.mem[ta]))

    def _handle(self, sid, instr, p):
        if instr == SYNC_WRITE and sid == BROADCAST:
            addr, size = p[0], p[1]
            ids = []
            for k in range(2, len(p), size + 1):
                s = self.servos.get(p[k])
                if s:
                    s.write_mem(addr, p[k + 1:k + 1 + size])
                    self._after_write(s, addr)
                    ids.append(p[k])
            self.log.append(("sync_write", tuple(ids)))
            return
        if instr == SYNC_READ and sid == BROADCAST:          # only STS-family servos answer SYNC_READ
            addr, size = p[0], p[1]
            for i in p[2:]:
                s = self.servos.get(i)
                if s and s.m.sync_read:
                    self._refresh(s)
                    self._reply(s, 0, bytes(s.mem[addr:addr + size]))
            self.log.append(("sync_read", tuple(p[2:])))
            return
        s = self.servos.get(sid)
        if s is None:
            return                                         # nobody answers: timeout
        if instr == PING:
            self._reply(s)
        elif instr == READ:
            self._refresh(s)
            self._reply(s, 0, bytes(s.mem[p[0]:p[0] + p[1]]))
        elif instr == WRITE:
            old_id = s.id
            err = s.write_mem(p[0], p[1:])
            if s.id != old_id:                             # ID change: re-register under the new ID
                self.servos[s.id] = self.servos.pop(old_id)
            self._reply(s, err)
            self._after_write(s, p[0])
        self.log.append((instr, (sid,)))


class EmulatedPCA9685:
    """Register-level PCA9685 behind a fake smbus: same calls as smbus2.SMBus. Models the parts a
    driver can get wrong: prescale is ignored unless the chip is asleep, block writes only
    auto-increment when MODE1.AI is set, and an asleep chip outputs nothing.
    physics.set_pulse(channel, us or None) receives every output change."""

    def __init__(self, address=0x41, osc_hz=25e6, physics=None):
        self.addr, self.osc, self.physics = address, osc_hz, physics
        self.mem = bytearray(256)
        self.mem[0x00], self.mem[0x01], self.mem[0xFE] = 0x11, 0x04, 0x1E    # power-on values
        for ch in range(17):                                # every output (and ALL_LED) starts full-off
            self.mem[0x09 + 4 * ch if ch < 16 else 0xFD] = 0x10
        self.writes = []

    def _check(self, addr):
        if addr != self.addr:
            raise OSError(121, "Remote I/O error (no device at this address)")

    def write_byte_data(self, addr, reg, val):
        self._check(addr)
        if reg == 0xFE and not self.mem[0x00] & 0x10:
            return                                          # prescale only writable while asleep
        self.mem[reg] = val & 0xFF
        self.writes.append((reg, val))
        self._outputs()

    def write_i2c_block_data(self, addr, reg, data):
        self._check(addr)
        ai = bool(self.mem[0x00] & 0x20)
        for k, v in enumerate(data):
            r = reg + k if ai else reg
            if r == 0xFE and not self.mem[0x00] & 0x10:
                continue
            self.mem[r] = v & 0xFF
            if 0xFA <= r <= 0xFD:                          # ALL_LED registers mirror into every channel
                for ch in range(16):
                    self.mem[0x06 + 4 * ch + (r - 0xFA)] = v & 0xFF
        self.writes.append((reg, tuple(data)))
        self._outputs()

    def read_byte_data(self, addr, reg):
        self._check(addr)
        return self.mem[reg]

    def read_i2c_block_data(self, addr, reg, n):
        self._check(addr)
        return list(self.mem[reg:reg + n])

    def frequency(self):
        return self.osc / (4096 * (self.mem[0xFE] + 1))

    def pulse_us(self, ch):
        """Output pulse width in microseconds, or None when the output is off."""
        r = 0x06 + 4 * ch
        on = self.mem[r] | (self.mem[r + 1] & 0x0F) << 8
        off = self.mem[r + 2] | (self.mem[r + 3] & 0x0F) << 8
        if self.mem[0x00] & 0x10 or self.mem[r + 3] & 0x10:   # asleep, or this output full-off
            return None
        return ((off - on) % 4096) * 1e6 / (self.frequency() * 4096)

    def _outputs(self):
        if self.physics is not None:
            for ch in range(16):
                self.physics.set_pulse(ch, self.pulse_us(ch))

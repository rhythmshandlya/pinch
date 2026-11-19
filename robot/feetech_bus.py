"""Minimal Feetech serial-bus servo driver (SCS / STS protocol), pure Python.

Works over anything with write(bytes) / read(n): a pyserial port on the robot
(through the Feetech FE-URT-1 USB adapter), or robot.bus_emulator.EmulatedPort in tests.

Packet:  FF FF ID LEN INSTR PARAM... CHK      LEN = n_params + 2
         CHK = ~(ID + LEN + INSTR + sum(PARAM)) & 0xFF
Status:  FF FF ID LEN ERR  PARAM... CHK

Register maps follow Feetech's own SDK (FTServo_Python: sms_sts.py for STS, scscl.py for SCS).
Two differences matter: STS stores 2-byte values little-endian, SCS big-endian; STS has
4096 steps / 360 deg (centre 2048), the SCS0009 1024 steps / 300 deg (centre 511).
"""

import time
from dataclasses import dataclass
from math import pi, radians

PING, READ, WRITE, REG_WRITE, ACTION, RESET, SYNC_READ, SYNC_WRITE = 0x01, 0x02, 0x03, 0x04, 0x05, 0x0A, 0x82, 0x83
BROADCAST = 0xFE


@dataclass(frozen=True)
class Model:
    name: str
    little_endian: bool
    steps: int            # position counts over `span`
    span: float           # rad covered by `steps`
    centre: int           # raw position for 0 rad
    reg: dict             # register name -> (address, size)
    sync_read: bool       # supports SYNC_READ (0x82)

    def rad_to_raw(self, rad: float) -> int:
        return int(round(self.centre + rad * self.steps / self.span))

    def raw_to_rad(self, raw: int) -> float:
        return (raw - self.centre) * self.span / self.steps


_COMMON = {
    "model": (3, 2), "id": (5, 1), "baud": (6, 1), "return_delay": (7, 1),
    "min_angle": (9, 2), "max_angle": (11, 2), "max_temp": (13, 1),
    "max_voltage": (14, 1), "min_voltage": (15, 1), "max_torque": (16, 2),
    "torque_enable": (40, 1), "goal_position": (42, 2), "goal_time": (44, 2), "goal_speed": (46, 2),
    "present_position": (56, 2), "present_speed": (58, 2), "present_load": (60, 2),
    "present_voltage": (62, 1), "present_temperature": (63, 1), "moving": (66, 1), "present_current": (69, 2),
}
STS3215 = Model("sts3215", True, 4096, 2 * pi, 2048,
                {**_COMMON, "p_gain": (21, 1), "d_gain": (22, 1), "i_gain": (23, 1),
                 "acceleration": (41, 1), "torque_limit": (48, 2), "lock": (55, 1)}, True)
SCS0009 = Model("scs0009", False, 1024, radians(300), 511,
                {**_COMMON, "p_gain": (21, 1), "d_gain": (22, 1), "i_gain": (23, 1), "lock": (48, 1)}, False)
MODELS = {m.name: m for m in (STS3215, SCS0009)}


class BusError(IOError):
    pass


def checksum(body: bytes) -> int:
    return (~sum(body)) & 0xFF


def packet(sid: int, instr: int, params: bytes = b"") -> bytes:
    body = bytes([sid, len(params) + 2, instr]) + bytes(params)
    return b"\xff\xff" + body + bytes([checksum(body)])


class FeetechBus:
    """`model` is one Model for the whole bus, or {servo_id: Model} for a mixed bus
    (e.g. HD-1910 thighs on the STS protocol + SCS0009 everywhere else)."""

    def __init__(self, port, model, retries: int = 2):
        self.port, self.retries = port, retries
        self.models = model if isinstance(model, dict) else None
        self.m = model if not isinstance(model, dict) else next(iter(model.values()))

    def model_of(self, sid) -> Model:
        return self.models.get(sid, self.m) if self.models else self.m

    # -- encoding ----------------------------------------------------------
    def _enc(self, value: int, size: int, m: Model = None) -> bytes:
        m = m or self.m
        value &= (1 << (8 * size)) - 1
        return value.to_bytes(size, "little" if m.little_endian else "big")

    def _dec(self, data: bytes, m: Model = None) -> int:
        m = m or self.m
        return int.from_bytes(data, "little" if m.little_endian else "big")

    @staticmethod
    def _signed(raw: int, sign_bit: int) -> int:
        """Feetech sign-magnitude: bit `sign_bit` set = negative."""
        mag = raw & ((1 << sign_bit) - 1)
        return -mag if raw & (1 << sign_bit) else mag

    # -- transport ---------------------------------------------------------
    def _status(self, expect_id: int) -> bytes:
        head = self.port.read(4)
        if len(head) < 4 or head[:2] != b"\xff\xff":
            raise BusError(f"no reply from servo {expect_id}")
        sid, length = head[2], head[3]
        rest = self.port.read(length)
        if len(rest) < length:
            raise BusError(f"short reply from servo {sid}")
        err, params, chk = rest[0], rest[1:-1], rest[-1]
        if checksum(bytes([sid, length, err]) + params) != chk:
            raise BusError(f"bad checksum from servo {sid}")
        if sid != expect_id:
            raise BusError(f"reply from {sid}, expected {expect_id}")
        if err:
            raise BusError(f"servo {sid} error flags 0x{err:02x}")
        return params

    def _txrx(self, sid, instr, params=b"", reply=True):
        last = None
        for _ in range(self.retries + 1):
            self.port.write(packet(sid, instr, params))
            if not reply or sid == BROADCAST:
                return b""
            try:
                return self._status(sid)
            except BusError as e:
                last = e
        raise last

    # -- primitives --------------------------------------------------------
    def ping(self, sid) -> bool:
        try:
            self._txrx(sid, PING)
            return True
        except BusError:
            return False

    def read(self, sid, name) -> int:
        m = self.model_of(sid)
        addr, size = m.reg[name]
        return self._dec(self._txrx(sid, READ, bytes([addr, size])), m)

    def write(self, sid, name, value: int):
        m = self.model_of(sid)
        addr, size = m.reg[name]
        self._txrx(sid, WRITE, bytes([addr]) + self._enc(value, size, m))

    def _groups(self, ids):
        """Split servo IDs by protocol model (each SYNC packet needs one encoding)."""
        g = {}
        for sid in ids:
            g.setdefault(self.model_of(sid).name, (self.model_of(sid), []))[1].append(sid)
        return list(g.values())

    def sync_write(self, name, values: dict):
        """One packet per protocol group sets the same register on many servos (no replies)."""
        for m, ids in self._groups(values):
            addr, size = m.reg[name]
            params = bytes([addr, size]) + b"".join(bytes([sid]) + self._enc(values[sid], size, m) for sid in ids)
            self._txrx(BROADCAST, SYNC_WRITE, params, reply=False)

    def sync_read(self, name, ids) -> dict:
        out = {}
        for m, group in self._groups(ids):
            addr, size = m.reg[name]
            if not m.sync_read:                        # SCS: read one by one
                out.update({sid: self.read(sid, name) for sid in group})
                continue
            self.port.write(packet(BROADCAST, SYNC_READ, bytes([addr, size]) + bytes(group)))
            out.update({sid: self._dec(self._status(sid), m) for sid in group})
        return out

    def _decode_block(self, m, data, a0, end):
        out = {}
        for name, (a, s) in m.reg.items():
            if a0 <= a and a + s <= end:
                out[name] = self._dec(data[a - a0:a - a0 + s], m)
        return out

    def read_block(self, sid, first, last) -> dict:
        """Read consecutive registers first..last in one transaction."""
        m = self.model_of(sid)
        a0, _ = m.reg[first]
        a1, s1 = m.reg[last]
        data = self._txrx(sid, READ, bytes([a0, a1 + s1 - a0]))
        return self._decode_block(m, data, a0, a1 + s1)

    def sync_read_block(self, ids, first, last) -> dict:
        """Registers first..last from many servos: one SYNC_READ packet per protocol group (STS), so 12
        servos cost one request and 12 back-to-back replies instead of 12 round trips. SCS servos can't
        sync-read and are read one by one. If a reply is missing, the rest of the packet is discarded and
        BusError raised (the caller can fall back to single reads to find out which servo is silent)."""
        out = {}
        for m, group in self._groups(ids):
            a0, _ = m.reg[first]
            a1, s1 = m.reg[last]
            if not m.sync_read:
                out.update({sid: self.read_block(sid, first, last) for sid in group})
                continue
            self.port.write(packet(BROADCAST, SYNC_READ, bytes([a0, a1 + s1 - a0]) + bytes(group)))
            try:
                for sid in group:
                    out[sid] = self._decode_block(m, self._status(sid), a0, a1 + s1)
            except BusError:
                if hasattr(self.port, "reset_input_buffer"):
                    self.port.reset_input_buffer()
                raise
        return out

    def _state_from(self, m, r) -> dict:
        return {"position": m.raw_to_rad(r["present_position"]),
                "speed": self._signed(r["present_speed"], 15) * m.span / m.steps,
                "load": self._signed(r["present_load"], 10) / 1000.0,
                "voltage": r["present_voltage"] / 10.0,
                "temperature": float(r["present_temperature"])}

    def states(self, ids) -> dict:
        """state() for many servos in one SYNC_READ: {sid: {position, speed, load, voltage, temperature}}."""
        raw = self.sync_read_block(ids, "present_position", "present_temperature")
        return {sid: self._state_from(self.model_of(sid), r) for sid, r in raw.items()}

    # -- robot-level helpers --------------------------------------------------
    def set_id(self, old, new):
        """The reply to an ID write may come back under either ID, so don't wait for it:
        send, drop whatever answers, then confirm by pinging the new ID."""
        self.write(old, "lock", 0)
        m = self.model_of(old)
        if self.models is not None:
            self.models[new] = m
        addr, size = m.reg["id"]
        self._txrx(old, WRITE, bytes([addr]) + self._enc(new, size, m), reply=False)
        time.sleep(0.005)
        if hasattr(self.port, "reset_input_buffer"):
            self.port.reset_input_buffer()
        if not self.ping(new):
            raise BusError(f"servo did not come back as ID {new}")
        self.write(new, "lock", 1)

    def torque(self, ids, enable=True):
        self.sync_write("torque_enable", {i: int(bool(enable)) for i in ids})

    def goal_positions(self, rad_by_id: dict):
        raw = {}
        for i, r in rad_by_id.items():
            m = self.model_of(i)
            raw[i] = min(m.steps - 1, max(0, m.rad_to_raw(r)))
        self.sync_write("goal_position", raw)

    def state(self, sid) -> dict:
        m = self.model_of(sid)
        return self._state_from(m, self.read_block(sid, "present_position", "present_temperature"))

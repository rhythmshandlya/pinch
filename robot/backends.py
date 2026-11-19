"""Hardware backends with one interface, so the robot code runs unchanged on
the real crab or in the simulator.

    write_positions({id: servo_rad})      # goal positions
    read_state() -> {id: ServoState}      # position (rad), speed (rad/s), load (-1..1), volts, temp C
    read_imu() -> (gyro[3] rad/s, gravity[3] unit vector, body frame)
    wait(dt)                              # real: sleep to keep the loop period; sim: advance physics
    torque(enable: bool)

RealBackend talks to the 12 leg servos with our own Feetech driver (robot/feetech_bus.py)
over the FE-URT-1 USB adapter; the 6 tiny arm servos hang straight off the Radxa ZERO 3W's own
hardware PWM outputs (BoardPWM); the BNO055 IMU and the INA219 battery monitor are on I2C.
(A PCA9685 PWM board is supported too, if the PWM pins are ever needed for something else.)
SimBackend runs the MuJoCo model; its servos can be "mounted" with random horn
offsets so calibration can be tested before the hardware exists.
"""

import logging
import os
import time
from dataclasses import dataclass
from math import pi

import numpy as np

from crab.params import JOINT_NAMES
from robot.servos import BUS_JOINTS, PWM_CHANNEL, PWM_JOINTS, SERVO_IDS, default_sign

log = logging.getLogger("crab")

# Hobby-servo pulse: 1500 us = centre. The GS-4308MG listing says "180 deg" and "pulse travel
# 1000 to 2000 us", which could mean either 1000..2000 us or 500..2500 us for 180 deg. We assume
# 1000..2000 us = 180 deg: if that's wrong the arms move *less* than asked, never more. The arm
# trim (robot/calibrate.py --trim-arms) measures the real travel and the centre error.
PWM_CENTER_US, PWM_US_PER_RAD = 1500.0, 1000.0 / pi
PWM_MIN_US, PWM_MAX_US = 500.0, 2500.0


def rad_to_us(rad):
    return float(np.clip(PWM_CENTER_US + rad * PWM_US_PER_RAD, PWM_MIN_US, PWM_MAX_US))


def us_to_rad(us):
    return (us - PWM_CENTER_US) / PWM_US_PER_RAD


@dataclass
class ServoState:
    position: float      # rad, servo frame (None: a PWM servo that hasn't been sent anywhere yet)
    speed: float         # rad/s
    load: float          # -1..1 of max torque
    voltage: float       # V
    temperature: float   # C
    feedback: bool = True   # False: PWM servo - position is just the last command, the rest unknown


class SimBackend:
    """MuJoCo stands in for the robot. `mount_offsets` = physical horn offsets (rad) per joint."""

    def __init__(self, mount_offsets=None, render=False, voltage=None, pwm_travel=1.0, model=None):
        from crab.params import PARAMS, JOINT_SPECS
        voltage = PARAMS.bus_voltage if voltage is None else voltage
        self._flim = {j: s.force_limit for j, s in zip(JOINT_NAMES, JOINT_SPECS)}
        import mujoco
        from pathlib import Path
        self.mj = mujoco
        root = Path(__file__).resolve().parents[1]
        self.m = model if model is not None else mujoco.MjModel.from_xml_path(str(root / "sim" / "models" / "crab.xml"))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("stand").id)
        mujoco.mj_forward(self.m, self.d)          # body poses valid before the first step (IMU reads)
        self.mount ={j: 0.0 for j in JOINT_NAMES} if mount_offsets is None else dict(mount_offsets)
        self.sign = {j: default_sign(j) for j in JOINT_NAMES}
        self.qadr = {j: self.m.jnt_qposadr[self.m.joint(j).id] for j in JOINT_NAMES}
        self.vadr = {j: self.m.jnt_dofadr[self.m.joint(j).id] for j in JOINT_NAMES}
        # gear-play joints (if the model has them): the encoder sits on the output shaft, after the play
        names = {self.m.joint(i).name for i in range(self.m.njnt)}
        self.bl_qadr = {j: self.m.jnt_qposadr[self.m.joint(f"{j}_backlash").id]
                        for j in JOINT_NAMES if f"{j}_backlash" in names}
        self.bl_vadr = {j: self.m.jnt_dofadr[self.m.joint(f"{j}_backlash").id] for j in self.bl_qadr}
        self.act = {j: self.m.actuator(j).id for j in JOINT_NAMES}
        self.by_id = {i: j for j, i in SERVO_IDS.items()}
        self.no_feedback = set(PWM_JOINTS)           # like the real arm servos: they report nothing
        self.travel = {j: (pwm_travel if j in self.no_feedback else 1.0) for j in JOINT_NAMES}  # real vs assumed
        self.cmd = {}                                # last servo-frame command per joint
        self.torque_on = True
        self.voltage = voltage                       # servo bus (regulated 5 V)
        self.battery = PARAMS.battery_nominal       # 2S pack voltage, as the INA219 would read it
        self.held = None          # joint vector: "a person holds the robot in this pose" (calibration rehearsal)
        self.viewer = None
        if render:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)

    def link_angle(self, j):          # true joint angle of the link (servo + gear play)
        q = self.d.qpos[self.qadr[j]]
        return q + (self.d.qpos[self.bl_qadr[j]] if j in self.bl_qadr else 0.0)

    def link_speed(self, j):
        w = self.d.qvel[self.vadr[j]]
        return w + (self.d.qvel[self.bl_vadr[j]] if j in self.bl_vadr else 0.0)

    def _servo_angle(self, j):        # what the physical servo would report
        return self.sign[j] * self.link_angle(j) + self.mount[j]

    def write_positions(self, goals):
        for sid, s_rad in goals.items():
            j = self.by_id[sid]
            self.cmd[j] = s_rad
            self.d.ctrl[self.act[j]] = (s_rad - self.mount[j]) * self.sign[j] * self.travel[j]

    def read_state(self):
        out = {}
        for sid, j in self.by_id.items():
            if j in self.no_feedback:
                out[sid] = ServoState(self.cmd.get(j), 0.0, 0.0, self.voltage, 35.0, feedback=False)
                continue
            f = self.d.actuator_force[self.act[j]]
            out[sid] = ServoState(position=float(self._servo_angle(j)),
                                  speed=float(self.sign[j] * self.link_speed(j)),
                                  load=float(np.clip(self.sign[j] * f / self._flim[j], -1, 1)),
                                  voltage=self.voltage, temperature=35.0)
        return out

    def read_imu(self):
        R = self.d.xmat[1].reshape(3, 3)
        return self.d.qvel[3:6].copy(), R.T @ np.array([0, 0, -1.0])

    def torque(self, enable):
        self.torque_on = enable
        if not enable:      # limp: let joints follow gravity (zero actuator stiffness)
            self.m.actuator_gainprm[:, 0] = 0
            self.m.actuator_biasprm[:, 1] = 0

    def wait(self, dt):
        if self.held is not None:                       # held by hand: joints don't move
            for i, j in enumerate(JOINT_NAMES):
                self.d.qpos[self.qadr[j]] = self.held[i]
                if j in self.bl_qadr:
                    self.d.qpos[self.bl_qadr[j]] = 0.0
            self.d.qvel[:] = 0
            self.mj.mj_forward(self.m, self.d)
            return
        for _ in range(max(1, int(round(dt / self.m.opt.timestep)))):
            self.mj.mj_step(self.m, self.d)
        if self.viewer is not None:
            self.viewer.sync()
            time.sleep(dt)

    def battery_voltage(self):
        return self.battery

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


class BusBackend:
    """Leg servos through our own Feetech driver (robot/feetech_bus.py) over any serial-like port
    (pyserial + FE-URT-1 on the robot, robot.bus_emulator.EmulatedPort in tests), arm servos
    through the PCA9685 PWM board.

    pwm:     PCA9685 (arm servos), or None (arms not connected: they are skipped)
    imu:     object with read() -> (gyro, gravity_unit), or None
    battery: object with read() -> pack volts (INA219), or None
    clock:   wait(dt) function; default sleeps in real time (tests pass the simulator's)
    """

    def __init__(self, port, model, imu=None, clock=None, battery=None, pwm=None):
        from robot.feetech_bus import FeetechBus
        self.bus = FeetechBus(port, model)
        self.ids = [SERVO_IDS[j] for j in BUS_JOINTS]
        self.pwm = pwm
        self.pwm_ch = {SERVO_IDS[j]: PWM_CHANNEL[j] for j in PWM_JOINTS}    # servo ID -> PWM output
        self.pwm_cmd = {}                                                 # servo ID -> last command (rad)
        self.imu = imu
        self.battery = battery
        self._clock = clock
        self._t = time.monotonic()

    def battery_voltage(self):
        return None if self.battery is None else self.battery.read()

    def write_positions(self, goals):
        self.bus.goal_positions({i: v for i, v in goals.items() if i not in self.pwm_ch})
        for i, v in goals.items():
            if i in self.pwm_ch and self.pwm is not None:
                self.pwm.set_pulse(self.pwm_ch[i], rad_to_us(v))
                self.pwm_cmd[i] = v

    def read_state(self):
        out = {i: ServoState(**self.bus.state(i)) for i in self.ids}
        nan = float("nan")
        for i in self.pwm_ch:
            out[i] = ServoState(self.pwm_cmd.get(i), 0.0, 0.0, nan, nan, feedback=False)
        return out

    def read_imu(self):
        if self.imu is None:
            return np.zeros(3), np.array([0.0, 0.0, -1.0])
        return self.imu.read()

    def torque(self, enable):
        self.bus.torque(self.ids, enable)
        if self.pwm is None:
            return
        if enable:                                    # arm servos hold again where they were sent
            for i, v in self.pwm_cmd.items():
                self.pwm.set_pulse(self.pwm_ch[i], rad_to_us(v))
        else:                                         # no pulses = hobby servos go limp
            self.pwm.all_off()

    def wait(self, dt):
        if self._clock is not None:
            return self._clock(dt)
        self._t += dt
        delay = self._t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self._t = time.monotonic()                     # fell behind: resync

    def close(self):
        try:
            self.torque(False)
        except Exception:
            pass


def find_i2c_bus(address=0x28):
    """The Linux I2C bus a device answers on: /dev/i2c-1 on a Raspberry Pi, /dev/i2c-3 on the
    Radxa ZERO 3W's header (pins 3/5). Env var CRAB_I2C_BUS overrides it."""
    import glob
    import os
    if os.environ.get("CRAB_I2C_BUS"):
        return int(os.environ["CRAB_I2C_BUS"])
    from smbus2 import SMBus
    for dev in sorted(glob.glob("/dev/i2c-*"), key=lambda d: int(d.rsplit("-", 1)[1])):
        n = int(dev.rsplit("-", 1)[1])
        try:
            with SMBus(n) as bus:
                bus.read_byte_data(address, 0)
            return n
        except OSError:
            continue
    raise RuntimeError(f"no I2C device at 0x{address:02x} - check the wiring and that I2C is enabled")


class BNO055:
    """IMU over plain I2C registers (pip install smbus2), no board-specific library, so it runs on
    any Linux board. IMU fusion mode (gyro + accelerometer, no magnetometer: the shell magnets
    would upset it). Mount it flat, chip up, x arrow pointing forward."""

    ADDR, OPR_MODE, UNIT_SEL, GYR, GRV = 0x28, 0x3D, 0x3B, 0x14, 0x2E

    def __init__(self, bus=None, smbus=None):
        import time
        if smbus is None:
            from smbus2 import SMBus
            smbus = SMBus(find_i2c_bus(self.ADDR) if bus is None else bus)
        self.bus = smbus
        self.bus.write_byte_data(self.ADDR, self.OPR_MODE, 0x00)        # config mode
        time.sleep(0.03)
        self.bus.write_byte_data(self.ADDR, self.UNIT_SEL, 0x02)        # gyro in rad/s, accel in m/s^2
        self.bus.write_byte_data(self.ADDR, self.OPR_MODE, 0x08)        # IMU fusion mode
        time.sleep(0.03)

    def _vec(self, reg, scale):
        raw = self.bus.read_i2c_block_data(self.ADDR, reg, 6)
        return np.array([int.from_bytes(bytes(raw[i:i + 2]), "little", signed=True) for i in (0, 2, 4)],
                        float) / scale

    def read(self):
        g = self._vec(self.GYR, 900.0)                                  # rad/s (900 LSB per rad/s)
        v = self._vec(self.GRV, 100.0)                                  # m/s^2; +9.8 on z when level:
        n = np.linalg.norm(v)                                           # the chip reports "up", we want
        return g, (-v / n if n > 0 else np.array([0.0, 0.0, -1.0]))     # the unit vector toward the ground


class INA219:
    """Battery monitor over plain I2C registers (pip install smbus2), in the battery's + lead.
    The servos sit behind the 5 V regulator, so they can't see the pack voltage - this can."""

    def __init__(self, address=0x40, bus=None, smbus=None):
        if smbus is None:
            from smbus2 import SMBus
            smbus = SMBus(find_i2c_bus(address) if bus is None else bus)
        self.bus, self.addr = smbus, address

    def _reg(self, reg):
        hi, lo = self.bus.read_i2c_block_data(self.addr, reg, 2)
        return (hi << 8) | lo

    def read(self):
        bus_v = (self._reg(0x02) >> 3) * 0.004                          # 4 mV per bit
        shunt = self._reg(0x01)
        shunt = (shunt - 65536 if shunt & 0x8000 else shunt) * 1e-5     # 10 uV per bit
        return float(bus_v + shunt)                                     # pack voltage (before the shunt)


class BoardPWM:
    """The arm servos on the Radxa ZERO 3W's own hardware PWM outputs, through Linux sysfs.
    Enable these six overlays in `sudo rsetup` -> Overlays: PWM8-M0, PWM9-M0, PWM12-M1, PWM13-M1,
    PWM14-M0, PWM15-M1. Hardware PWM: exact 50 Hz frames, no jitter, no extra board.
    Output k drives arm servo k (robot/servos.py PWM_CHANNEL); the header pin is in PINS."""

    # output -> PWM controller (its address in the RK3566 device tree) and 40-pin header pin
    CONTROLLERS = ["fe6f0000", "fe6f0010", "fe700000", "fe700010", "fe700020", "fe700030"]   # PWM8 9 12 13 14 15
    PINS = [16, 18, 21, 24, 7, 19]
    PERIOD_NS = 20_000_000

    def __init__(self, root="/sys/class/pwm", wait=1.0):
        from pathlib import Path
        self.root = Path(root)
        found = {}
        for chip in self.root.glob("pwmchip*"):
            found[self._address(chip)] = chip
        self.out = {}                                                   # output -> .../pwmchipN/pwm0
        for k, addr in enumerate(self.CONTROLLERS):
            chip = found.get(addr)
            if chip is None:
                continue
            pwm = chip / "pwm0"
            if not pwm.exists():
                (chip / "export").write_text("0")
                t0 = time.monotonic()
                while not (pwm / "duty_cycle").exists() and time.monotonic() - t0 < wait:
                    time.sleep(0.02)                                    # udev sets the permissions
            self._w(pwm / "enable", 0)
            self._w(pwm / "period", self.PERIOD_NS)
            self.out[k] = pwm
        self.on = set()
        if not self.out:
            raise OSError(f"no PWM outputs under {root} - enable the six PWM overlays in rsetup")

    @staticmethod
    def _address(chip):
        """'fe6f0000' for the controller behind /sys/class/pwm/pwmchipN."""
        try:
            for line in (chip / "device" / "uevent").read_text().splitlines():
                if line.startswith("OF_FULLNAME="):
                    return line.rsplit("@", 1)[-1].strip()
        except OSError:
            pass
        return os.path.basename(os.path.realpath(chip / "device")).split(".")[0]

    @staticmethod
    def _w(path, value):
        path.write_text(str(int(value)))

    def ping(self):
        return len(self.out) == len(self.CONTROLLERS)

    def missing_pins(self):
        return [self.PINS[k] for k in range(len(self.CONTROLLERS)) if k not in self.out]

    def set_pulse(self, channel, us):
        pwm = self.out.get(channel)
        if pwm is None:
            return
        self._w(pwm / "duty_cycle", round(us * 1000))
        if channel not in self.on:
            self._w(pwm / "enable", 1)
            self.on.add(channel)

    def all_off(self):
        for k, pwm in self.out.items():                                # output low = servo goes limp
            self._w(pwm / "enable", 0)
        self.on.clear()


class PCA9685:
    """16-channel PWM board over plain I2C registers (pip install smbus2) - the alternative way to
    drive the arm servos (not in the BOM: the Radxa's own PWM outputs do the job).
    Address 0x41 (bridge the A0 solder jumper on the board): 0x40 is the INA219's."""

    MODE1, MODE2, LED0, ALL_LED, PRESCALE = 0x00, 0x01, 0x06, 0xFA, 0xFE
    SLEEP, AI, RESTART, OUTDRV, FULL_OFF = 0x10, 0x20, 0x80, 0x04, 0x10

    def __init__(self, address=0x41, bus=None, smbus=None, freq=50.0, osc_hz=25e6):
        if smbus is None:
            from smbus2 import SMBus
            smbus = SMBus(find_i2c_bus(address) if bus is None else bus)
        self.bus, self.addr = smbus, address
        prescale = int(round(osc_hz / (4096 * freq))) - 1              # 121 -> 50 Hz frames
        self.us_per_count = (prescale + 1) / osc_hz * 1e6               # ~4.9 us per step
        self.bus.write_byte_data(address, self.MODE1, self.SLEEP)       # prescale only sets while asleep
        self.bus.write_byte_data(address, self.PRESCALE, prescale)
        self.bus.write_byte_data(address, self.MODE2, self.OUTDRV)      # push-pull outputs
        self.bus.write_byte_data(address, self.MODE1, self.AI)          # wake, register auto-increment
        time.sleep(0.001)                                               # oscillator start-up (500 us)
        self.bus.write_byte_data(address, self.MODE1, self.AI | self.RESTART)
        self.all_off()                                                  # arms limp until first command

    def ping(self):
        try:
            return bool(self.bus.read_byte_data(self.addr, self.MODE1) & self.AI)
        except OSError:
            return False

    def set_pulse(self, channel, us):
        off = int(round(us / self.us_per_count))                        # pulse starts at count 0
        self.bus.write_i2c_block_data(self.addr, self.LED0 + 4 * channel, [0, 0, off & 0xFF, off >> 8])

    def all_off(self):
        self.bus.write_i2c_block_data(self.addr, self.ALL_LED, [0, 0, 0, self.FULL_OFF])


def bus_models():
    """Bus servo ID -> protocol model (the legs; the PWM arm servos aren't on the bus)."""
    from robot.feetech_bus import MODELS
    from crab.params import servo_for
    return {SERVO_IDS[j]: MODELS[servo_for(j).bus_model] for j in BUS_JOINTS}


def RealBackend(port="/dev/ttyUSB0", baudrate=1_000_000, imu=True, battery=True, arms="board"):
    """The physical crab: leg servos via FE-URT-1 (pip install pyserial); arm servos on the
    Radxa's PWM pins (arms="board") or a PCA9685 board (arms="pca9685"); IMU and battery monitor
    via I2C. Missing arm outputs are reported; the legs still work."""
    import serial
    ser = serial.Serial(port, baudrate=baudrate, timeout=0.02)
    pwm = None
    if arms and PWM_JOINTS:
        try:
            pwm = PCA9685() if arms == "pca9685" else BoardPWM()
        except (OSError, RuntimeError) as e:
            log.warning("arm servo outputs not found: %s - arms stay limp", e)
    return BusBackend(ser, bus_models(), imu=BNO055() if imu else None,
                      battery=INA219() if battery else None, pwm=pwm)


class SimPhysics:
    """Lets the emulated servo bus drive the MuJoCo simulator (used by tests)."""

    def __init__(self, sim: "SimBackend"):
        self.sim = sim

    def set_goal(self, sid, rad):
        self.sim.write_positions({sid: rad})

    def set_torque(self, sid, on):
        pass                                               # whole-robot torque handled by the test

    def get_state(self, sid):
        s = self.sim.read_state()[sid]
        return dict(position=s.position, speed=s.speed, load=s.load, voltage=s.voltage, temperature=s.temperature)


class SimPWM:
    """Lets the emulated PCA9685 drive the arm servos in the MuJoCo simulator (used by tests)."""

    def __init__(self, sim: "SimBackend"):
        self.sim = sim
        self.sid = {PWM_CHANNEL[j]: SERVO_IDS[j] for j in PWM_JOINTS}

    def set_pulse(self, channel, us):
        if us is not None and channel in self.sid:                      # None = output off (limp)
            self.sim.write_positions({self.sid[channel]: us_to_rad(us)})


def emulated_bus_backend(mount_offsets=None, pwm_travel=1.0):
    """Full software stack down to the bytes: driver -> emulated servos and emulated PWM chip
    -> MuJoCo physics."""
    from robot.bus_emulator import EmulatedPCA9685, EmulatedPort
    model = bus_models()
    sim = SimBackend(mount_offsets=mount_offsets, pwm_travel=pwm_travel)
    port = EmulatedPort(model, list(model), physics=SimPhysics(sim))
    chip = EmulatedPCA9685(physics=SimPWM(sim))
    imu = type("SimIMU", (), {"read": lambda self: sim.read_imu()})()
    batt = type("SimBattery", (), {"read": lambda self: sim.battery})()
    be = BusBackend(port, model, imu=imu, clock=sim.wait, battery=batt,
                    pwm=PCA9685(smbus=chip) if PWM_JOINTS else None)
    be.sim, be.chip = sim, chip
    return be


def random_horn_offsets(seed=0):
    """Physical mounting error: the 25T spline puts a horn within +-7.2 deg of intent, plus assembly slop."""
    rng = np.random.default_rng(seed)
    return {j: float(rng.uniform(-7.2, 7.2) * pi / 180 + rng.normal(0, 1.0) * pi / 180) for j in JOINT_NAMES}

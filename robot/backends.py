"""Hardware backends with one interface, so the robot code runs unchanged on
the real crab or in the simulator.

    write_positions({id: servo_rad})      # goal positions
    read_state() -> {id: ServoState}      # position (rad), speed (rad/s), load (-1..1), volts, temp C
    read_imu() -> (gyro[3] rad/s, gravity[3] unit vector, body frame)
    wait(dt)                              # real: sleep to keep the loop period; sim: advance physics
    torque(enable: bool)

RealBackend talks to all 18 bus servos (12 STS3215 legs, 6 STS3032 arms) with our own Feetech driver
(robot/feetech_bus.py) over the FE-URT-1 USB adapter; the BNO055 IMU and the INA219 battery monitor are on I2C.
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
from robot.servos import SERVO_IDS, default_sign

log = logging.getLogger("crab")

@dataclass
class ServoState:
    position: float      # rad, servo frame
    speed: float         # rad/s
    load: float          # -1..1 of max torque
    voltage: float       # V, at the servo's own terminals
    temperature: float   # C


class SimBackend:
    """MuJoCo stands in for the robot. `mount_offsets` = physical horn offsets (rad) per joint."""

    def __init__(self, mount_offsets=None, render=False, voltage=None, model=None,
                 realism="ideal", seed=0, soc=1.0, servo_model="first_order", power_model="circuit",
                 enforce_module_min=None, ambient_c=None):
        """realism: "ideal" (exact sensors, no delay, constant power: the default) or "datasheet" (the electronics
        of configs/electronics.toml: command delay, encoder steps, IMU noise and lag, distance-beam noise and
        dropouts, a battery that drains and sags; crab/electronics.py). Seeded: same seed, same numbers.
        soc: the pack's starting state of charge (datasheet realism only).
        servo_model: the leg servo, measured Rhoban BAM m6 numbers from
        configs/electronics.toml: "first_order" (the position actuator in crab_v2.xml plus the firmware's target
        slew limit) or "bam" (the full model, crab/servo_model.py; needs the optional `bam` package).
        power_model (datasheet realism): "circuit" (the default: the netlist of configs/wiring_v2.toml solved every
        control tick, crab/wiring.py: wire drops, both regulators, the fuse, brown-out, the INA219 where it is wired, the
        IMU at its mount) or "lumped" (the earlier battery + one regulator model, kept for comparison).
        enforce_module_min: the regulator modules stop below their listed 7 V input (None: the data file's choice).
        ambient_c: the room's temperature for the servos' heat model (None: configs/electronics.toml servo_heat.ambient_c).
        Every bus servo's present temperature comes from crab.servo_model.ServoHeat, driven by its motor's torque."""
        from crab.params import JOINT_SPECS, PARAMS
        if realism not in ("ideal", "datasheet"):
            raise ValueError(f"realism must be 'ideal' or 'datasheet', not {realism!r}")
        if power_model not in ("circuit", "lumped"):
            raise ValueError(f"power_model must be 'circuit' or 'lumped', not {power_model!r}")
        if servo_model not in ("first_order", "bam"):
            raise ValueError(f"servo_model must be 'first_order' or 'bam', not {servo_model!r}")
        voltage = PARAMS.bus_voltage if voltage is None else voltage
        self._flim = {j: s.force_limit for j, s in zip(JOINT_NAMES, JOINT_SPECS)}
        from pathlib import Path

        import mujoco
        self.mj = mujoco
        root = Path(__file__).resolve().parents[1]
        self.m = model if model is not None else mujoco.MjModel.from_xml_path(str(root / "sim" / "models" / PARAMS.model_file))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("stand").id)
        mujoco.mj_forward(self.m, self.d)          # body poses valid before the first step (IMU reads)
        from crab.servo_model import make_drive  # the measured leg servo (crab/servo_model.py)
        self.drive = make_drive(servo_model, self.m, self.d, JOINT_NAMES)
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
        self.cmd = {}                                # last servo-frame command per joint
        self.torque_on = True
        self.voltage = voltage                       # leg servo rail (ideal realism: every servo reports it)
        self.battery = PARAMS.battery_nominal       # 2S pack voltage, as the INA219 would read it
        self.held = None          # joint vector: "a person holds the robot in this pose" (calibration rehearsal)
        self.realism, self.elec, self.circuit, self._imu_mount = realism, None, None, None
        self._setup_heat(ambient_c, bam=servo_model == "bam" and self.drive is not None)
        if realism == "datasheet" or self.drive is not None:
            self._act_idx = np.array([self.act[j] for j in JOINT_NAMES])
            self._target = self.d.ctrl[self._act_idx].copy()          # what the servos have been told (undelayed)
        if realism == "datasheet":
            from crab.electronics import Data, Realism
            data = Data()                              # configs/electronics.toml
            self.elec = Realism(data, seed=seed, soc=soc,
                                kt=[s.kt_current for s in JOINT_SPECS], dt=float(self.m.opt.timestep),
                                initial=self._target)
            self._setup_mounts(data)
            if power_model == "circuit":
                self._setup_circuit(data, seed, soc, enforce_module_min)
            self._sync_power()
        self.viewer = None
        if render:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)

    def _setup_heat(self, ambient_c, bam):
        """The bus servos' temperatures (crab/servo_model.py ServoHeat). The copper current makes the motor's own torque:
        the position actuator's force less the back-EMF the joint damping stands in for, or the full BAM model's torque."""
        from crab.params import servo_for
        from crab.servo_model import LEG_KINDS, ServoHeat
        fb = list(JOINT_NAMES)
        self.heat_joints = fb
        self._heat_act = np.array([self.act[j] for j in fb])
        self._heat_vadr = np.array([self.vadr[j] for j in fb])
        self._heat_bemf = np.array([0.0 if bam and j.split("_", 1)[1] in LEG_KINDS else servo_for(j).back_emf
                                    for j in fb])
        self._heat_fric = np.array([servo_for(j).frictionloss for j in fb])
        self._heat_visc = np.array([servo_for(j).viscous for j in fb])
        self.heat = ServoHeat([servo_for(j) for j in fb], ambient=ambient_c)
        self._heat_index = {j: i for i, j in enumerate(fb)}

    def _heat_acc(self, acc):
        """Add this physics step's squared motor torque (acc[0]) and gear friction loss (acc[1], W) per bus servo."""
        w = self.d.qvel[self._heat_vadr]
        tau = self.d.actuator_force[self._heat_act] - self._heat_bemf * w
        aw = np.abs(w)
        acc[0] += tau * tau
        acc[1] += aw * (self._heat_fric + self._heat_visc * aw)

    # -- the circuit and the sensors at their mounts ---------------------------------------------------------
    def _setup_mounts(self, data):
        """The IMU where configs/layout_v2.toml mounts it: gyro and specific force are read from the MuJoCo sensors on its
        site (lever arm included), fused by crab.electronics.MountedImu, turned into body axes."""
        from crab.electronics import Layout, MountedImu
        R = Layout()["imu"].R
        self._imu_mount = R
        self._gyro_adr = int(self.m.sensor_adr[self.m.sensor("imu_gyro").id])
        self._acc_adr = int(self.m.sensor_adr[self.m.sensor("imu_accel").id])
        self.elec.imu = MountedImu(data, self.elec.imu.rng, R)

    def _setup_circuit(self, data, seed, soc, enforce):
        from crab.wiring import Circuit
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC1C]))
        self.circuit = Circuit(data=data, rng=rng, soc=soc, enforce_module_min=enforce)
        self.elec.circuit = self.circuit
        legs = [ld["joint"] for ld in self.circuit.legs]
        arms = [ld["joint"] for ld in self.circuit.arms]
        self._leg_act = np.array([self.act[j] for j in legs])
        self._leg_vadr = np.array([self.vadr[j] for j in legs])
        self._arm_act = np.array([self.act[j] for j in arms])
        from crab.params import servo_for
        self._kt_arm = np.array([servo_for(j).kt_current for j in arms])
        self._kt, self._R = float(data("leg_servo.kt_nm_per_a")), float(data("leg_servo.resistance_ohm"))
        self._v_leg = self.circuit.v_servo.copy()
        self._dt = float(self.m.opt.timestep)
        self._supply_to_drive()

    def _supply_to_drive(self):
        if self.drive is not None:
            self.drive.set_supply(self._v_leg)

    def set_activity(self, name):
        """What the computer is doing, for its power draw: "idle", "camera" (the default) or "npu"."""
        if self.circuit is not None:
            if name not in self.circuit.pc:
                raise ValueError(f"activity must be one of {sorted(self.circuit.pc)}")
            self.circuit.activity = name

    def link_angle(self, j):          # true joint angle of the link (servo + gear play)
        q = self.d.qpos[self.qadr[j]]
        return q + (self.d.qpos[self.bl_qadr[j]] if j in self.bl_qadr else 0.0)

    def link_speed(self, j):
        w = self.d.qvel[self.vadr[j]]
        return w + (self.d.qvel[self.bl_vadr[j]] if j in self.bl_vadr else 0.0)

    def _servo_angle(self, j):        # what the physical servo would report
        return self.sign[j] * self.link_angle(j) + self.mount[j]

    def write_positions(self, goals):
        if self.circuit is not None and not self.circuit.computer_up:
            return                                          # the computer is down (brown-out): nobody is talking to the servos
        for sid, s_rad in goals.items():
            j = self.by_id[sid]
            self.cmd[j] = s_rad
            v = (s_rad - self.mount[j]) * self.sign[j]
            if self.elec is not None or self.drive is not None:
                self._target[JOINT_NAMES.index(j)] = v          # reaches the joint after the delay (wait)
            else:
                self.d.ctrl[self.act[j]] = v

    def _sync_power(self):
        e = self.elec
        if self.circuit is not None:
            self.voltage, self.battery = self.circuit.servo_voltage(), self.circuit.v_pack
        else:
            self.voltage, self.battery = e.power.servo_voltage(), e.power.v_pack

    def read_state(self):
        out = {}
        for sid, j in self.by_id.items():
            f = self.d.actuator_force[self.act[j]]
            pos = self._servo_angle(j)
            out[sid] = ServoState(position=float(pos if self.elec is None else self.elec.encoder(pos)),
                                  speed=float(self.sign[j] * self.link_speed(j)),
                                  load=float(np.clip(self.sign[j] * f / self._flim[j], -1, 1)),
                                  voltage=self._servo_report(j),
                                  temperature=float(np.round(self.heat.T[self._heat_index[j]])))
        return out

    def _servo_report(self, j):
        """The voltage a servo reports: its own terminals, in 0.1 V steps (circuit), or the shared bus voltage."""
        if self.circuit is None:
            return self.voltage
        joints = [ld["joint"] for ld in self.circuit.legs]
        if j not in joints:                                 # an arm servo (STS3032) sits on the 5 V rail B
            arms = [ld["joint"] for ld in self.circuit.arms]
            return float(np.round(self.circuit.v_arm[arms.index(j)] / 0.1) * 0.1) if j in arms else float(self.voltage)
        return float(np.round(self.circuit.v_servo[joints.index(j)] / 0.1) * 0.1)

    def read_imu(self):
        if self._imu_mount is not None:                     # datasheet realism: the chip at its mount
            sd = self.d.sensordata
            return self.elec.imu.read(self.d.time, sd[self._gyro_adr:self._gyro_adr + 3],
                                      sd[self._acc_adr:self._acc_adr + 3])
        R = self.d.xmat[1].reshape(3, 3)
        gyro, grav = self.d.qvel[3:6].copy(), R.T @ np.array([0, 0, -1.0])
        return (gyro, grav) if self.elec is None else self.elec.imu.read(self.d.time, gyro, grav)

    def torque(self, enable):
        self.torque_on = enable
        if self.drive is not None:
            self.drive.torque_on = enable
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
        n = max(1, int(round(dt / self.m.opt.timestep)))
        heat = np.zeros((2, len(self._heat_act)))
        cir = self.circuit
        if cir is not None:
            acc_dd, acc_dw, acc_arm = np.zeros(len(self._leg_act)), np.zeros(len(self._leg_act)), np.zeros(len(self._arm_act))
            bam = getattr(self.drive, "kind", "") == "bam"
            stall = np.maximum(self._v_leg, 1e-3) * self._kt / self._R
        if self.elec is None and self.drive is None:
            for _ in range(n):
                self.mj.mj_step(self.m, self.d)
                self._heat_acc(heat)
        else:
            for _ in range(n):
                tgt = self._target if self.elec is None else self.elec.delay.push(self._target)
                if self.drive is None:
                    self.d.ctrl[self._act_idx] = tgt
                else:
                    self.drive.command(tgt)
                    self.drive.step()
                self.mj.mj_step(self.m, self.d)
                self._heat_acc(heat)
                if cir is not None:                  # what each servo's motor model drew this step
                    duty = self.drive.duty if bam else np.clip(self.d.actuator_force[self._leg_act] / stall, -1.0, 1.0)
                    acc_dd += duty * duty
                    acc_dw += duty * self.d.qvel[self._leg_vadr]
                    acc_arm += np.abs(self.d.actuator_force[self._arm_act])
            if cir is not None:
                cir.step(n * self._dt, acc_dd / n / self._R, self._kt * acc_dw / n / self._R, acc_arm / n / self._kt_arm,
                         self.torque_on)
                self._v_leg = cir.v_servo.copy()
                self._supply_to_drive()                 # each servo's torque follows the voltage at its own terminals
                self._sync_power()
            elif self.elec is not None:
                amps = self.elec.servo_amps(self.d.actuator_force[self._act_idx], self.torque_on)
                self.elec.power.step(n * self.m.opt.timestep, amps)
                self._sync_power()
                if self.drive is not None:
                    self.drive.set_supply(self.voltage)         # the full servo model's torque follows the supply
        self.heat.step(n * self.m.opt.timestep, heat[0] / n, self.torque_on, heat[1] / n)
        if self.viewer is not None:
            self.viewer.sync()
            time.sleep(dt)

    def battery_voltage(self):
        """Pack volts as the INA219 reads them (datasheet realism: in its 4 mV steps, with its gain error)."""
        if self.circuit is not None:
            return self.circuit.measure()[0]
        return self.battery if self.elec is None else self.elec.power.measure()[0]

    def battery_current(self):
        """Amps drawn from the pack, as the INA219 reads them (None in ideal realism: nothing models it)."""
        if self.circuit is not None:
            return self.circuit.measure()[1]
        return None if self.elec is None else self.elec.power.measure()[1]

    def battery_fraction(self):
        """0..1 from the pack voltage, the way robot/hw_real.py does it (6.4 V empty, 8.4 V full)."""
        v = self.battery_voltage()
        return float(np.clip((v - 6.4) / (8.4 - 6.4), 0, 1))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


class BusBackend:
    """All 18 servos through our own Feetech driver (robot/feetech_bus.py) over any serial-like port
    (pyserial + FE-URT-1 on the robot, robot.bus_emulator.EmulatedPort in tests).

    imu:     object with read() -> (gyro, gravity_unit), or None
    battery: object with read() -> pack volts (INA219), or None
    clock:   wait(dt) function; default sleeps in real time (tests pass the simulator's)
    """

    # Transient errors (a garbled servo reply, an I2C NACK) are counted and ridden out with the last good
    # value; only a fault that persists this many reads in a row is treated as real.
    MAX_STALE = 10            # servo-state reads (~1 s at the firmware's 10 Hz health + feet checks)
    MAX_IO_FAILS = 25         # consecutive failed writes / IMU / battery reads

    def __init__(self, port, model, imu=None, clock=None, battery=None):
        from collections import Counter

        from robot.feetech_bus import FeetechBus
        self.bus = FeetechBus(port, model)
        self.ids = [SERVO_IDS[j] for j in JOINT_NAMES]
        self.imu = imu
        self.battery = battery
        self._clock = clock
        self._t = time.monotonic()
        self.errors = Counter()                   # source -> transient errors ridden out
        self._fails = Counter()                   # source -> consecutive failures right now
        self._last = {}                           # source -> last good value
        self._stale_reads = 0

    def _guard(self, source, fn, fallback):
        """Run fn(); on an I/O error count it and return the last good value (or `fallback`)."""
        try:
            v = fn()
        except OSError as e:                      # BusError, SerialException and smbus2's errors are OSErrors
            self.errors[source] += 1
            self._fails[source] += 1
            if self._fails[source] == self.MAX_IO_FAILS:
                log.error("%s: %d failures in a row (%s)", source, self._fails[source], e)
            if self._fails[source] > self.MAX_IO_FAILS and source == "servo_write":
                raise
            return self._last.get(source, fallback)
        self._fails[source] = 0
        self._last[source] = v
        return v

    def battery_voltage(self):
        return None if self.battery is None else self._guard("battery", self.battery.read, None)

    def write_positions(self, goals):
        self._guard("servo_write", lambda: self.bus.goal_positions(dict(goals)), None)

    def _read_bus(self):
        from robot.feetech_bus import BusError
        try:
            st = self.bus.states(self.ids)                   # one SYNC_READ for all 18 servos
        except OSError:
            self.errors["servo_read"] += 1
            st = {}
            for i in self.ids:                               # find out who is silent, keep the rest
                try:
                    st[i] = self.bus.state(i)
                except OSError:
                    pass
        missing = [i for i in self.ids if i not in st]
        self._stale_reads = self._stale_reads + 1 if missing else 0
        for i in missing:
            if self._stale_reads > self.MAX_STALE or i not in self._last.get("servos", {}):
                raise BusError(f"servo(s) {missing} not answering ({self._stale_reads} reads in a row)")
            st[i] = self._last["servos"][i]
        self._last["servos"] = st
        return st

    def read_state(self):
        return {i: ServoState(**s) for i, s in self._read_bus().items()}

    def read_imu(self):
        if self.imu is None:
            return np.zeros(3), np.array([0.0, 0.0, -1.0])
        return self._guard("imu", self.imu.read, (np.zeros(3), np.array([0.0, 0.0, -1.0])))

    def torque(self, enable):
        self.bus.torque(self.ids, enable)

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
    The leg servos report their own terminals (after the wires) and the arm servos the 5 V rail B, not the pack - this does.
    The pack voltage is bus voltage + the drop across the shunt, both measured directly, so it doesn't
    depend on the shunt's value; the current does (current() = drop / SHUNT_OHM)."""

    SHUNT_OHM = 0.01          # the board's shunt resistor (R010: 10 mOhm; hardware/BOM.md - was 0.1 Ohm)

    def __init__(self, address=0x40, bus=None, smbus=None):
        if smbus is None:
            from smbus2 import SMBus
            smbus = SMBus(find_i2c_bus(address) if bus is None else bus)
        self.bus, self.addr = smbus, address

    def _reg(self, reg):
        hi, lo = self.bus.read_i2c_block_data(self.addr, reg, 2)
        return (hi << 8) | lo

    def shunt_v(self):
        raw = self._reg(0x01)
        return (raw - 65536 if raw & 0x8000 else raw) * 1e-5             # 10 uV per bit, +-320 mV

    def read(self):
        bus_v = (self._reg(0x02) >> 3) * 0.004                          # 4 mV per bit
        return float(bus_v + self.shunt_v())                            # pack voltage (before the shunt)

    def current(self):
        """Amps drawn from the pack (+ = discharging). +-32 A range with the 10 mOhm shunt."""
        return float(self.shunt_v() / self.SHUNT_OHM)


def bus_models():
    """Bus servo ID -> protocol model, for all 18 servos."""
    from crab.params import servo_for
    from robot.feetech_bus import MODELS
    return {SERVO_IDS[j]: MODELS[servo_for(j).bus_model] for j in JOINT_NAMES}


def servo_port(by_id="/dev/serial/by-id"):
    """The FE-URT-1's serial device. /dev/ttyUSB0 can change number when another USB-serial thing is
    plugged in, so prefer the stable /dev/serial/by-id name. PINCH_SERVO_PORT overrides."""
    import glob
    if os.environ.get("PINCH_SERVO_PORT"):
        return os.environ["PINCH_SERVO_PORT"]
    found = sorted(glob.glob(os.path.join(by_id, "*")))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        log.warning("several USB serial devices %s - using the first; set PINCH_SERVO_PORT to choose", found)
        return found[0]
    return "/dev/ttyUSB0"


def RealBackend(port=None, baudrate=1_000_000, imu=True, battery=True):
    """The physical crab: all 18 servos via the FE-URT-1 (pip install pyserial); IMU and battery monitor
    via I2C."""
    import serial
    ser = serial.Serial(port or servo_port(), baudrate=baudrate, timeout=0.02)
    return BusBackend(ser, bus_models(), imu=BNO055() if imu else None,
                      battery=INA219() if battery else None)


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


def emulated_bus_backend(mount_offsets=None, realism="ideal", seed=0):
    """Full software stack down to the bytes: driver -> emulated servos -> MuJoCo physics."""
    from robot.bus_emulator import EmulatedPort
    model = bus_models()
    sim = SimBackend(mount_offsets=mount_offsets, realism=realism, seed=seed)
    port = EmulatedPort(model, list(model), physics=SimPhysics(sim))
    imu = type("SimIMU", (), {"read": lambda self: sim.read_imu()})()
    batt = type("SimBattery", (), {"read": lambda self: sim.battery_voltage()})()
    be = BusBackend(port, model, imu=imu, clock=sim.wait, battery=batt)
    be.sim = sim
    return be


def random_horn_offsets(seed=0):
    """Physical mounting error: the 25T spline puts a horn within +-7.2 deg of intent, plus assembly slop."""
    rng = np.random.default_rng(seed)
    return {j: float(rng.uniform(-7.2, 7.2) * pi / 180 + rng.normal(0, 1.0) * pi / 180) for j in JOINT_NAMES}

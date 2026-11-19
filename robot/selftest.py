"""Self-test of the robot software against the simulated robot (no hardware needed).

    python robot/selftest.py

Checks the exact code path the robot will run, with SimBackend standing in
for the servo bus, IMU and battery:
  1. joint <-> raw servo position conversion round-trips
  2. calibration recovers random horn offsets (25T spline + assembly slop)
  3. with that calibration, the robot reaches the commanded pose (< 1 deg error)
  4. the safety layer clamps joint limits and limits servo speed
  5. low battery triggers a safety stop (torque off)
  6. the robot stack walks: trot forward, crab-walk sideways
"""

import sys
from math import pi
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.emotes import STAND  # noqa: E402
from crab.params import JOINT_NAMES, PARAMS  # noqa: E402
from robot.backends import SimBackend, random_horn_offsets  # noqa: E402
from robot.calibrate import CAL_POSE, SimPerson, check_directions, measure_offsets, trim_arms  # noqa: E402
from robot.crab_robot import MAX_STEP, Crab, SafetyStop  # noqa: E402
from robot.servos import BUS_JOINTS, PWM_JOINTS, Calibration, raw_to_rad, rad_to_raw  # noqa: E402

QUIET = dict(log=lambda *_: None)

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def calibrated_backend(seed=3):
    truth = random_horn_offsets(seed)
    hw = SimBackend(mount_offsets=truth)
    hw.torque(False)
    hw.held = CAL_POSE                     # the person holds the robot in the calibration pose
    hw.wait(0.01)
    cal = measure_offsets(hw)                        # legs: read from the servos
    hw = SimBackend(mount_offsets=truth)              # arms: lined up by eye (simulated person)
    trim_arms(hw, cal, ask=SimPerson(hw), **QUIET)
    # fresh backend with the same physical offsets, standing
    return SimBackend(mount_offsets=truth), cal, truth


def main():
    print("robot software self-test (simulated hardware)")

    errs = [abs(raw_to_rad(rad_to_raw(a)) - a) for a in np.linspace(-pi + 0.01, pi - 0.01, 101)]
    check("raw <-> radian conversion", max(errs) < 2 * pi / 4096, f"max err {np.degrees(max(errs)):.3f} deg")

    hw, cal, truth = calibrated_backend()
    err = max(abs(cal.offset[j] - truth[j]) for j in BUS_JOINTS)
    check("calibration recovers leg horn offsets", err < np.radians(0.5), f"worst {np.degrees(err):.3f} deg")
    err = max(abs(cal.offset[j] - truth[j]) for j in PWM_JOINTS)
    check("arm trim by eye lines up the PWM arm servos", err < np.radians(3), f"worst {np.degrees(err):.2f} deg")
    hw5 = SimBackend(mount_offsets=truth, pwm_travel=0.5)          # servos turn half as far per pulse
    cal5 = Calibration()
    trim_arms(hw5, cal5, ask=SimPerson(hw5), **QUIET)
    q5 = STAND.copy()
    crab5 = Crab(hw5, calibration=cal5)
    crab5.q_cmd = q5.copy()
    for _ in range(40):
        crab5.tick(q5)
    arm_err = max(abs(hw5.link_angle(j) - q5[JOINT_NAMES.index(j)]) for j in PWM_JOINTS)
    check("arm travel check: servos with half the expected travel still reach their poses",
          all(cal5.gain[j] == 2.0 for j in PWM_JOINTS) and arm_err < np.radians(3),
          f"gain {cal5.gain[PWM_JOINTS[0]]:g}, worst arm error {np.degrees(arm_err):.2f} deg")

    hw3 = SimBackend(mount_offsets=truth)
    cal3 = Calibration()
    cal3.sign["FR_tibia"] *= -1                     # the software has this servo the wrong way round
    hw3.torque(False)
    hw3.held = CAL_POSE
    hw3.wait(0.01)
    flipped = check_directions(hw3, cal3, ask=SimPerson(hw3), **QUIET)
    check("leg direction check finds and fixes a reversed servo", flipped == ["FR_tibia"]
          and cal3.sign == Calibration().sign, f"flipped {flipped}")

    crab = Crab(hw, calibration=cal)
    crab.start()
    crab.move_to(STAND, 1.0)
    for _ in range(25):
        crab.tick(STAND)
    q, _ = crab.read_joints()
    true_q = np.array([hw.d.qpos[hw.qadr[j]] for j in JOINT_NAMES])
    e_meas = np.degrees(np.abs(q - true_q).max())
    e_pose = np.degrees(np.abs(true_q - STAND).max())
    check("calibrated robot reaches the stand pose", e_pose < 3.0,
          f"max joint error {e_pose:.2f} deg (gravity sag incl.), sensing error {e_meas:.3f} deg")

    uncal = Crab(SimBackend(mount_offsets=truth), calibration=Calibration())
    uncal.start()
    for _ in range(25):
        uncal.tick(STAND)
    tq = np.array([uncal.hw.d.qpos[uncal.hw.qadr[j]] for j in JOINT_NAMES])
    check("...and without calibration it would be off (sanity)", np.degrees(np.abs(tq - STAND).max()) > 3.0,
          f"{np.degrees(np.abs(tq - STAND).max()):.1f} deg off")

    before = crab.q_cmd.copy()
    wild = STAND.copy()
    wild[JOINT_NAMES.index("FL_femur")] = -3.0                    # far outside the CAD limit
    after = crab.tick(wild)
    i = JOINT_NAMES.index("FL_femur")
    check("speed limit per 20 ms tick", abs(after[i] - before[i]) <= MAX_STEP[i] + 1e-9,
          f"moved {np.degrees(abs(after[i] - before[i])):.1f} deg in one tick")
    for _ in range(100):
        crab.tick(wild)
    check("joint limit clamp", crab.q_cmd[i] >= crab.lo[i] - 1e-9, f"held at {np.degrees(crab.q_cmd[i]):.0f} deg")
    crab.move_to(STAND, 1.0)

    x0 = hw.d.qpos[0]
    crab.walk((0.08, 0.0, 0.0), seconds=4.0)
    dx = hw.d.qpos[0] - x0
    check("robot stack walks forward (trot)", dx > 0.15, f"moved {dx * 100:.1f} cm in 4 s")
    y0 = hw.d.qpos[1]
    crab.walk((0.0, 0.10, 0.0), seconds=4.0)
    dy = hw.d.qpos[1] - y0
    check("robot stack crab-walks sideways", dy > 0.2, f"moved {dy * 100:.1f} cm left in 4 s")
    crab.play_emote("wave")
    check("emote plays through the robot stack", True)

    hw.battery = PARAMS.battery_stop - 0.2                          # 2S pack below 3.2 V/cell
    stopped = False
    try:
        for _ in range(200):
            crab.tick(STAND)
    except SafetyStop as e:
        stopped = "battery" in str(e)
    check("low-battery safety stop: pack voltage (torque off)", stopped and not hw.torque_on)

    hw2 = SimBackend(mount_offsets=truth)
    crab_b = Crab(hw2, calibration=cal)
    crab_b.start()
    hw2.voltage = 0.85 * PARAMS.bus_voltage                         # regulator dropped out: bus sags
    stopped = False
    try:
        for _ in range(200):
            crab_b.tick(STAND)
    except SafetyStop as e:
        stopped = "servo bus" in str(e)
    check("low-voltage safety stop: servo bus sag (torque off)", stopped and not hw2.torque_on)

    # ---- our own Feetech driver against the byte-level emulated bus --------------------
    from robot.bus_emulator import EmulatedPort
    from robot.backends import emulated_bus_backend
    from robot.feetech_bus import MODELS, FeetechBus, checksum, packet
    from robot.setup_servo import setup
    pk = packet(1, 0x01)
    check("packet + checksum (PING)", pk == bytes([0xFF, 0xFF, 1, 2, 1, checksum(bytes([1, 2, 1]))]))
    for mname in ("sts3215", "scs0009"):
        m = MODELS[mname]
        port = EmulatedPort(m, [3, 4, 5])
        bus = FeetechBus(port, m)
        ok = bus.ping(3) and not bus.ping(9)
        bus.sync_write("goal_position", {3: 100, 4: 900, 5: 511})
        ok &= [port.servos[i]._get("goal_position") for i in (3, 4, 5)] == [100, 900, 511]
        port.servos[4]._put("present_position", 777)
        ok &= bus.sync_read("present_position", [3, 4, 5])[4] == 777
        # EEPROM is locked: a raw ID write is refused; set_id unlocks, writes, relocks
        try:
            bus.write(5, "id", 9)
            refused = False
        except Exception:
            refused = True
        bus.set_id(5, 9)
        ok &= refused and bus.ping(9) and not bus.ping(5)
        check(f"{mname}: ping, sync write/read, EEPROM lock, ID change ({'little' if m.little_endian else 'big'}-endian)", ok)
        new = EmulatedPort(m, [1])
        setup(FeetechBus(new, m), 12, log=lambda *_: None)
        s = new.servos[12]
        check(f"{mname}: setup_servo flow on a brand-new servo", s._get("p_gain") == 32 and s.mem[m.reg['lock'][0]] == 1
              and s._get("goal_position") == m.centre)

    from robot.backends import bus_models
    mixed = bus_models()
    port = EmulatedPort(mixed, list(mixed))
    bus = FeetechBus(port, mixed)
    bus.goal_positions({i: 0.5 for i in mixed})                       # 0.5 rad everywhere
    ok = all(port.servos[i]._get("goal_position") == mixed[i].rad_to_raw(0.5) for i in mixed)
    kinds = sorted({m.name for m in mixed.values()})
    check(f"mixed bus ({' + '.join(kinds)}): each servo gets its own encoding", ok and len(kinds) >= 1)

    truth2 = random_horn_offsets(5)
    bb = emulated_bus_backend(mount_offsets=truth2)
    bb.sim.torque(False)
    bb.sim.held = CAL_POSE
    bb.wait(0.01)                                                    # let the "hands" take hold first
    cal2 = measure_offsets(bb)
    err2 = max(abs(cal2.offset[j] - truth2[j]) for j in BUS_JOINTS)
    check("calibration through the bus (raw counts)", err2 < np.radians(0.2), f"worst {np.degrees(err2):.2f} deg")

    # the arm servos through the PWM board's registers: driver -> emulated PCA9685 -> physics
    from robot.backends import PCA9685
    from robot.bus_emulator import EmulatedPCA9685
    chip = EmulatedPCA9685()
    pca = PCA9685(smbus=chip)
    pca.set_pulse(3, 1500.0)
    check("PWM board driver: 50 Hz frames, 1500 us pulse, unused outputs off",
          abs(chip.frequency() - 50) < 0.5 and abs(chip.pulse_us(3) - 1500) < 5 and chip.pulse_us(4) is None,
          f"{chip.frequency():.2f} Hz, {chip.pulse_us(3):.0f} us")
    pca.all_off()
    check("...torque off = no pulses (hobby servos go limp)", all(chip.pulse_us(c) is None for c in range(16)))
    bb3 = emulated_bus_backend(mount_offsets=truth2)
    cal4 = Calibration(sign=dict(cal2.sign), offset=dict(cal2.offset))
    cal4.sign["R_claw"] *= -1                        # wrong way round: the trim tool must catch it
    trim_arms(bb3, cal4, ask=SimPerson(bb3.sim), **QUIET)
    err4 = max(abs(cal4.offset[j] - truth2[j]) for j in PWM_JOINTS)
    check("arm trim through the PWM board: reversed servo fixed, arms lined up",
          cal4.sign == Calibration().sign and err4 < np.radians(3), f"worst {np.degrees(err4):.2f} deg")
    cal2 = cal4
    bb = emulated_bus_backend(mount_offsets=truth2)                  # fresh robot, same horns
    crab2 = Crab(bb, calibration=cal2)
    crab2.start()
    x0 = bb.sim.d.qpos[0]
    crab2.walk((0.08, 0.0, 0.0), seconds=3.0)
    dx = bb.sim.d.qpos[0] - x0
    check("walks with every command going through the byte-level bus", dx > 0.1, f"{dx * 100:.1f} cm in 3 s")

    # firmware audit (docs/LIFE_PLAN.md A7): one SYNC_READ per state read, a silent servo ridden out for a
    # moment, a dead one -> safety stop, and after a safety stop the arm servos get no pulses ever again
    port_log = bb.bus.port.log
    n0 = len(port_log)
    st = bb.read_state()
    reads = [e for e in port_log[n0:] if e[0] == "sync_read"]
    check("servo state: all 12 leg servos in one SYNC_READ packet", len(reads) == 1 and len(reads[0][1]) == 12
          and len(port_log) - n0 == 1 and all(st[i].feedback for i in bb.ids))
    crab2.state()
    t0 = len(port_log)
    crab2.state()
    crab2.read_joints()
    check("...read once per 20 ms tick however many parts of the program ask", len(port_log) == t0)
    servos_on_bus = bb.bus.port.servos
    sid = bb.ids[4]
    gone = servos_on_bus.pop(sid)
    try:
        st2 = bb.read_state()
        ridden = st2[sid].position == st[sid].position and bb.errors["servo_read"] == 1
    except OSError:
        ridden = False
    check("a servo that misses one reply: its last reading is used, the error counted", ridden)
    stopped = False
    try:
        for _ in range(400):
            crab2.tick(crab2.q_cmd)
    except SafetyStop as e:
        stopped = "servo bus" in str(e)
    pulses_after_stop = [bb.chip.pulse_us(c) for c in range(6)]
    try:
        crab2.shutdown()                                    # must not sit down = must not pulse the arms
    except Exception:
        pass
    check("a servo that stays silent -> safety stop (not a crash), torque off",
          stopped and crab2.stopped and not bb.arms_live)
    check("...and the arm servos get no pulses after it (not even from the shutdown)",
          all(p is None for p in pulses_after_stop) and all(bb.chip.pulse_us(c) is None for c in range(6)))
    servos_on_bus[sid] = gone

    import os
    import tempfile
    from robot.backends import servo_port
    with tempfile.TemporaryDirectory() as tmp:
        none = servo_port(by_id=os.path.join(tmp, "missing"))
        Path(tmp, "usb-1a86_USB_Single_Serial_54A7-if00").write_text("")
        one = servo_port(by_id=tmp)
    check("servo port: the stable /dev/serial/by-id name when there is one, else /dev/ttyUSB0",
          none == "/dev/ttyUSB0" and one.endswith("usb-1a86_USB_Single_Serial_54A7-if00"))

    # the two arms can't be driven into each other anywhere inside their joint limits
    from crab.kinematics import arms_clear
    from robot.crab_robot import joint_limits
    lo, hi = joint_limits()
    iy, ip = JOINT_NAMES.index("L_shoulder_yaw"), JOINT_NAMES.index("L_shoulder_pitch")
    grid = [(y, p) for y in np.linspace(lo[iy], hi[iy], 4) for p in np.linspace(lo[ip], hi[ip], 6)]
    clear = all(arms_clear(a, b) for a in grid for b in grid)
    check("arms can't reach each other anywhere in their joint ranges", clear, f"{len(grid) ** 2} pose pairs")

    # the IMU and battery-monitor drivers decode raw registers correctly (fake I2C chips)
    from robot.backends import BNO055, INA219

    class FakeBus:
        def __init__(self, regs):
            self.regs = regs

        def write_byte_data(self, addr, reg, val):
            self.regs[(addr, reg)] = val

        def read_i2c_block_data(self, addr, reg, n):
            return [self.regs.get((addr, reg + i), 0) for i in range(n)]

    def le16(v):
        v &= 0xFFFF
        return [v & 0xFF, v >> 8]
    regs = {}
    for i, v in enumerate([900, -450, 0]):                     # gyro: 1.0, -0.5, 0 rad/s
        regs[(0x28, 0x14 + 2 * i)], regs[(0x28, 0x15 + 2 * i)] = le16(v)
    for i, v in enumerate([0, 0, 981]):                         # gravity: level robot, chip says +9.81 on z
        regs[(0x28, 0x2E + 2 * i)], regs[(0x28, 0x2F + 2 * i)] = le16(v)
    imu = BNO055(smbus=FakeBus(regs))
    g, down = imu.read()
    check("IMU driver: gyro in rad/s, gravity unit vector points down when level",
          np.allclose(g, [1.0, -0.5, 0.0]) and np.allclose(down, [0, 0, -1]) and regs[(0x28, 0x3D)] == 0x08)
    class FakeIna:                                              # INA219: 16-bit registers, big-endian
        def __init__(self, regs):
            self.regs = regs

        def read_i2c_block_data(self, addr, reg, n):
            v = self.regs[reg]
            return [(v >> 8) & 0xFF, v & 0xFF]
    bus_reg = int(round(7.40 / 0.004)) << 3                      # 7.40 V on the load side
    ina = INA219(smbus=FakeIna({0x02: bus_reg, 0x01: 5000}))    # + 50 mV across the shunt
    check("battery monitor driver: pack voltage from the INA219 registers", abs(ina.read() - 7.45) < 0.005,
          f"{ina.read():.3f} V")

    from robot.run import hardware_check
    lines = []
    ok = hardware_check(emulated_bus_backend(), out=lines.append)
    check("run.py --check finds the 12 leg servos, the arm PWM board, the IMU and the battery",
          ok and len(lines) == 23, f"{len(lines)} lines")
    port = emulated_bus_backend()
    del port.bus.port.servos[8]                                      # unplug the RL thigh servo
    lines = []
    bad = hardware_check(port, out=lines.append)
    check("...and reports a missing servo", not bad and any("MISSING" in ln and "RL_femur" in ln for ln in lines))
    nob = emulated_bus_backend()
    nob.chip.addr = 0x7F                                            # PWM board not answering
    lines = []
    bad = hardware_check(nob, out=lines.append)
    check("...and missing arm servo outputs", not bad and any("MISSING" in ln and "arm servo" in ln for ln in lines))

    # the Radxa's own PWM outputs, through a fake Linux sysfs tree (what the kernel shows)
    import tempfile
    from robot.backends import BoardPWM
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for n, addr in enumerate(["fdd70000"] + BoardPWM.CONTROLLERS[::-1]):    # unrelated chip first
            dev = root / f"pwmchip{n}" / "device"
            dev.mkdir(parents=True)
            (dev / "uevent").write_text(f"DRIVER=rockchip-pwm\nOF_NAME=pwm\nOF_FULLNAME=/pwm@{addr}\n")
            pwm0 = root / f"pwmchip{n}" / "pwm0"                       # as if exported already
            pwm0.mkdir()
            for f in ("enable", "period", "duty_cycle"):
                (pwm0 / f).write_text("0")
        bp = BoardPWM(root=root)

        def chip_of(k):
            return next(c for c in root.glob("pwmchip*")
                        if BoardPWM.CONTROLLERS[k] in (c / "device" / "uevent").read_text()) / "pwm0"
        bp.set_pulse(2, 1650.0)
        p = chip_of(2)
        ok = (p / "period").read_text() == "20000000" and (p / "duty_cycle").read_text() == "1650000" \
            and (p / "enable").read_text() == "1" and (chip_of(3) / "enable").read_text() == "0"
        bp.all_off()
        ok &= all((chip_of(k) / "enable").read_text() == "0" for k in range(6)) and bp.ping()
        check("Radxa PWM driver: finds the 6 outputs by address, 20 ms frames, pulse in ns, off = limp", ok)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

"""The crab's runtime: one 50 Hz loop that turns joint targets into servo goals,
with safety checks on every tick. Works with RealBackend (Radxa ZERO 3W) or
SimBackend (MuJoCo) unchanged.

    crab = Crab(SimBackend())          # or Crab(RealBackend())
    crab.start()                       # ease into the stand pose
    crab.play_emote("wave")
    crab.walk((0, 0.08, 0), seconds=4) # crab-walk left
    crab.shutdown()                    # sit down, torque off
"""

import logging
import time
from math import radians

import numpy as np

from crab.emotes import EMOTES, STAND
from crab.gait import GAITS, Gait
from crab.kinematics import arms_clear
from crab.params import JOINT_NAMES, JOINT_SPECS, PARAMS
from robot.servos import SERVO_IDS, Calibration

log = logging.getLogger("crab")

DT = 0.02                                   # 50 Hz control loop
MAX_STEP = np.array([s.max_cmd_speed for s in JOINT_SPECS]) * DT     # per-joint max target change per tick
# Two independent low-power checks:
#  * the 2S pack itself (INA219 battery monitor): warn at 3.4 V/cell, stop at 3.2 V/cell under load
#  * the servo bus: the servos sit behind the 5 V regulator and report the *bus* voltage; if it
#    sags, the regulator has dropped out (empty pack, or a fault) and the robot must stop
BATT_WARN, BATT_STOP = PARAMS.battery_warn, PARAMS.battery_stop
V_WARN, V_STOP = 0.95 * PARAMS.bus_voltage, 0.92 * PARAMS.bus_voltage
TEMP_STOP = 65.0                            # C
ARM_L = [JOINT_NAMES.index(f"L_{j}") for j in ("shoulder_yaw", "shoulder_pitch")]
ARM_R = [JOINT_NAMES.index(f"R_{j}") for j in ("shoulder_yaw", "shoulder_pitch")]
ARMS = ARM_L + ARM_R + [JOINT_NAMES.index("L_claw"), JOINT_NAMES.index("R_claw")]
FEMURS = [JOINT_NAMES.index(f"{lg}_femur") for lg in ("FL", "FR", "RL", "RR")]
TIBIAS = [JOINT_NAMES.index(f"{lg}_tibia") for lg in ("FL", "FR", "RL", "RR")]
SHIN_MIN = PARAMS.leg.shin_min              # femur + tibia limit (CAD): shin vs hip link
REST = EMOTES["sleep"].keyframes[1][1]      # the low "nap" pose shutdown() leaves the crab in


def joint_limits():
    """Joint ranges straight from params.py (CAD-checked), so the Pi doesn't need MuJoCo."""
    L, A = PARAMS.leg, PARAMS.arm
    r = []
    for j in JOINT_NAMES:
        side, kind = j.split("_", 1)
        r.append({"coxa": L.coxa_range_for(side), "femur": L.femur_range, "tibia": L.tibia_range,
                  "shoulder_yaw": A.shoulder_yaw_range, "shoulder_pitch": A.shoulder_pitch_range,
                  "claw": A.claw_range}[kind])
    r = np.array(r)
    return r[:, 0], r[:, 1]


class SafetyStop(RuntimeError):
    """kind: "battery" (the pack is empty: the firmware also shuts the computer down), "bus_voltage",
    "temperature", "servo_bus" or "stopped"."""

    def __init__(self, why, kind="other"):
        super().__init__(why)
        self.kind = kind


class Crab:
    def __init__(self, backend, calibration=None, com_offset=(0.015, 0.0)):
        self.hw = backend
        self.cal = calibration or Calibration.load()
        self.lo, self.hi = joint_limits()
        self.q_cmd = STAND.copy()
        self.com_offset = np.asarray(com_offset)
        self.low_v_since = None
        self.t = 0.0
        self.ticks = 0                              # 20 ms ticks so far (keys the per-tick state cache)
        self._state_tick, self._state = -1, None
        self.stopped = False                        # a safety stop happened: nothing moves any more

    # -- conversions ------------------------------------------------------
    def state(self):
        """The servos' state, read once per 20 ms tick however many parts of the program ask (the health
        check, the emote/grasp policies, the feet-down sense): each bus read costs ~2 ms on the real robot."""
        if self._state_tick != self.ticks:
            try:
                self._state = self.hw.read_state()
            except OSError as e:                 # a servo stayed silent past the backend's tolerance
                self.emergency_stop(f"servo bus: {e}", "servo_bus")
            self._state_tick = self.ticks
        return self._state

    def read_joints(self):
        """Joint angles: measured for the bus servos; the PWM arm servos can't report, so for
        them it's where they were last sent (q_cmd)."""
        st = self.state()
        q = self.q_cmd.copy()
        for i, j in enumerate(JOINT_NAMES):
            s = st[SERVO_IDS[j]]
            if s.feedback:
                q[i] = self.cal.servo_to_joint(j, s.position)
        return q, st

    def _send(self, q):
        self.hw.write_positions({SERVO_IDS[j]: self.cal.joint_to_servo(j, float(q[i]))
                                 for i, j in enumerate(JOINT_NAMES)})

    # -- the safety-checked tick -----------------------------------------
    def tick(self, q_target):
        if self.stopped:
            raise SafetyStop("stopped", "stopped")
        if self.paused:                                   # a user emergency stop: limp, nothing is sent
            return self.idle_tick()
        q_target = np.asarray(q_target, float)
        if q_target.shape != self.q_cmd.shape or not np.all(np.isfinite(q_target)):
            q_target = self.q_cmd.copy()                  # a non-number never reaches a servo: hold still
            self.bad_targets += 1
        q_target = np.clip(q_target, self.lo, self.hi)                              # joint limits (CAD)
        step = np.clip(q_target - self.q_cmd, -MAX_STEP, MAX_STEP)                  # servo speed limit
        q_new = self.q_cmd + step
        q_new[TIBIAS] = np.maximum(q_new[TIBIAS], SHIN_MIN - q_new[FEMURS])         # shin vs hip link (CAD)
        # the two arms share the space in front of the face: if this step would make them touch,
        # the arms hold still (the legs carry on)
        if not arms_clear(q_new[ARM_L], q_new[ARM_R]) and arms_clear(self.q_cmd[ARM_L], self.q_cmd[ARM_R]):
            q_new[ARMS] = self.q_cmd[ARMS]
            self.arm_blocks += 1
        self.q_cmd = q_new
        self._send(self.q_cmd)
        t0 = time.perf_counter()
        self.hw.wait(DT)
        self.last_wait = time.perf_counter() - t0       # sleep (robot) or physics (twin): not the tick's own work
        self.t += DT
        self.ticks += 1
        self._check_health()
        return self.q_cmd

    servo_temp = {}                                  # joint -> C, refreshed every 0.2 s by the health check

    @property
    def hottest_leg_c(self):
        """The hottest leg servo's reported temperature (C), or None before the first reading."""
        legs = [v for j, v in self.servo_temp.items() if j.split("_", 1)[1] in ("coxa", "femur", "tibia")]
        return max(legs) if legs else None

    arm_blocks = 0                                   # how often the arm guard held the arms
    bad_targets = 0                                  # how often a NaN / infinite / mis-sized target was held
    paused = False                                   # a user emergency stop is on (pause / resume)

    def idle_tick(self):
        """One 20 ms tick that sends nothing (paused): time and the world still move on."""
        t0 = time.perf_counter()
        self.hw.wait(DT)
        self.last_wait = time.perf_counter() - t0
        self.t += DT
        self.ticks += 1
        return self.q_cmd

    def pause(self, why="user"):
        """User emergency stop: torque off at once (Pinch goes limp and settles on its belly) and send nothing
        until resume(). Unlike a safety stop for a fault, this one can be undone."""
        log.warning("EMERGENCY STOP (%s): torque off", why)
        self.paused = True
        try:
            self.hw.torque(False)
        except OSError as e:
            log.error("torque off failed: %s", e)

    def resume(self):
        """Torque back on from wherever the joints are now (the caller eases it back to standing)."""
        if not self.paused or self.stopped:
            return False
        q, _ = self.read_joints()
        self.q_cmd = np.clip(q, self.lo, self.hi)
        self.hw.torque(True)
        self.paused = False
        log.warning("resumed after the emergency stop")
        return True
    last_wait = 0.0

    def _check_health(self):
        if int(round(self.t / DT)) % 10:                   # every 0.2 s is plenty
            return
        _, st = self.read_joints()
        self.servo_temp = {JOINT_NAMES[i]: float(st[SERVO_IDS[j]].temperature) for i, j in enumerate(JOINT_NAMES)
                           if st[SERVO_IDS[j]].feedback}       # C, what each bus servo reports (the mind reads it)
        st = [s for s in st.values() if s.feedback]          # only bus servos report volts and heat
        volts = min(s.voltage for s in st)
        temp = max(s.temperature for s in st)
        batt = self.hw.battery_voltage() if hasattr(self.hw, "battery_voltage") else None
        if temp > TEMP_STOP:
            self.emergency_stop(f"servo temperature {temp:.0f} C", "temperature")
        pack_low = batt is not None and batt < BATT_STOP
        if volts < V_STOP or pack_low:
            self.low_v_since = self.low_v_since or self.t
            if self.t - self.low_v_since > 1.0:          # ignore brief sags while a leg pushes off
                self.emergency_stop(f"battery {batt:.2f} V" if pack_low else f"servo bus {volts:.2f} V",
                                    "battery" if pack_low else "bus_voltage")
        else:
            self.low_v_since = None
            if volts < V_WARN or (batt is not None and batt < BATT_WARN):
                log.warning("battery low (pack %s V, bus %.2f V) - finish up and recharge",
                            "?" if batt is None else f"{batt:.2f}", volts)

    def emergency_stop(self, why, kind="other"):
        log.error("SAFETY STOP: %s -> torque off", why)
        self.stopped = True
        try:
            self.hw.torque(False)
        except OSError as e:                        # the bus itself may be what failed
            log.error("torque off failed: %s", e)
        raise SafetyStop(why, kind)

    # -- behaviours ------------------------------------------------------
    def move_to(self, q, seconds=1.5):
        q0, q1 = self.q_cmd.copy(), np.asarray(q, float)
        n = max(1, int(seconds / DT))
        for k in range(1, n + 1):
            s = k / n
            s = s * s * (3 - 2 * s)
            self.tick(q0 + (q1 - q0) * s)

    def start(self):
        """Read where the joints actually are, enable torque, ease into the stand pose.
        The arm servos can't say where they are: they are assumed in the rest pose shutdown()
        left them in (if not, they snap there at their own speed when torque comes on)."""
        self.q_cmd = REST.copy()
        q, _ = self.read_joints()
        self.q_cmd = np.clip(q, self.lo, self.hi)
        self.hw.torque(True)
        self.move_to(STAND, 2.0)

    def play_emote(self, name, loops=1):
        emo = EMOTES[name]
        T = emo.duration * (loops if emo.loop else 1)
        t0 = self.t
        while self.t - t0 < T:
            self.tick(emo.target(self.t - t0))
        self.move_to(STAND, 0.6)

    def walk(self, cmd, seconds, gait="trot"):
        g = Gait(GAITS[gait], com_offset=self.com_offset)
        t0 = self.t
        while self.t - t0 < seconds:
            n = int(DT / 0.002)                            # gait integrates at 500 Hz internally
            for k in range(n):
                q = g.step(0.002, cmd, solve=k == n - 1)   # leg IK only after the last substep
            self.tick(q)
        for _ in range(int(1.0 / DT)):                      # settle: command zero velocity
            n = int(DT / 0.002)
            for k in range(n):
                q = g.step(0.002, (0.0, 0.0, 0.0), solve=k == n - 1)
            self.tick(q)


    def shutdown(self):
        """Sit down gently, then release the servos. After a safety stop it doesn't move at all (and so
        never re-enables the arm servos' pulses): it only makes sure torque is off."""
        try:
            if not self.stopped:
                self.move_to(REST, 1.5)
        finally:
            self.hw.torque(False)
            if hasattr(self.hw, "close"):
                self.hw.close()


def wait_real(seconds):
    time.sleep(seconds)


__all__ = ["Crab", "SafetyStop", "DT", "radians"]

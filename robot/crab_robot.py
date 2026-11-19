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
    pass


class Crab:
    def __init__(self, backend, calibration=None, com_offset=(0.015, 0.0)):
        self.hw = backend
        self.cal = calibration or Calibration.load()
        self.lo, self.hi = joint_limits()
        self.q_cmd = STAND.copy()
        self.com_offset = np.asarray(com_offset)
        self.low_v_since = None
        self.t = 0.0

    # -- conversions ------------------------------------------------------
    def read_joints(self):
        """Joint angles: measured for the bus servos; the PWM arm servos can't report, so for
        them it's where they were last sent (q_cmd)."""
        st = self.hw.read_state()
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
        q_target = np.clip(np.asarray(q_target, float), self.lo, self.hi)          # joint limits (CAD)
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
        self.hw.wait(DT)
        self.t += DT
        self._check_health()
        return self.q_cmd

    arm_blocks = 0                                   # how often the arm guard held the arms

    def _check_health(self):
        if int(round(self.t / DT)) % 10:                   # every 0.2 s is plenty
            return
        _, st = self.read_joints()
        st = [s for s in st.values() if s.feedback]          # only bus servos report volts and heat
        volts = min(s.voltage for s in st)
        temp = max(s.temperature for s in st)
        batt = self.hw.battery_voltage() if hasattr(self.hw, "battery_voltage") else None
        if temp > TEMP_STOP:
            self.emergency_stop(f"servo temperature {temp:.0f} C")
        pack_low = batt is not None and batt < BATT_STOP
        if volts < V_STOP or pack_low:
            self.low_v_since = self.low_v_since or self.t
            if self.t - self.low_v_since > 1.0:          # ignore brief sags while a leg pushes off
                self.emergency_stop(f"battery {batt:.2f} V" if pack_low else f"servo bus {volts:.2f} V")
        else:
            self.low_v_since = None
            if volts < V_WARN or (batt is not None and batt < BATT_WARN):
                log.warning("battery low (pack %s V, bus %.2f V) - finish up and recharge",
                            "?" if batt is None else f"{batt:.2f}", volts)

    def emergency_stop(self, why):
        log.error("SAFETY STOP: %s -> torque off", why)
        self.hw.torque(False)
        raise SafetyStop(why)

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
            q = STAND
            for _ in range(int(DT / 0.002)):               # gait integrates at 500 Hz internally
                q = g.step(0.002, cmd)
            self.tick(q)
        for _ in range(int(1.0 / DT)):                      # settle: command zero velocity
            q = STAND
            for _ in range(int(DT / 0.002)):
                q = g.step(0.002, (0.0, 0.0, 0.0))
            self.tick(q)

    def run_policy(self, run_dir, cmd_fn, seconds):
        """Learned walking: cmd_fn(t) -> (vx, vy, wz). Same observations as training (crab/policy_io.py)."""
        import json
        import pickle
        from pathlib import Path
        import jax
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
        from crab import policy_io as pio
        from training.train import network_factory
        run_dir = Path(run_dir)
        cfg = json.loads((run_dir / "config.json").read_text())
        params = pickle.loads((run_dir / "params.pkl").read_bytes())
        nets = network_factory(cfg)(pio.OBS_SIZE, pio.N_ACT, preprocess_observations_fn=running_statistics.normalize)
        policy = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
        key, last, t0 = jax.random.PRNGKey(0), np.zeros(pio.N_ACT), self.t
        while self.t - t0 < seconds:
            q, st = self.read_joints()
            qd = np.array([self.cal.sign[j] * st[SERVO_IDS[j]].speed for j in JOINT_NAMES])
            gyro, grav = self.hw.read_imu()
            obs = pio.observation(np, gyro, grav, np.asarray(cmd_fn(self.t - t0), float),
                                  q[pio.LEG_ACT_IDX], qd[pio.LEG_ACT_IDX], last, self.t - t0)
            act, _ = policy(obs.astype(np.float32), key)
            last = np.clip(np.asarray(act), -1, 1)
            self.tick(pio.action_to_targets(np, last))

    def shutdown(self):
        """Sit down gently, then release the servos."""
        try:
            self.move_to(REST, 1.5)
        finally:
            self.hw.torque(False)
            if hasattr(self.hw, "close"):
                self.hw.close()


def wait_real(seconds):
    time.sleep(seconds)


__all__ = ["Crab", "SafetyStop", "DT", "radians"]

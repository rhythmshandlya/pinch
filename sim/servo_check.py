"""Can the real servos deliver what the simulator asked of them? Checked per joint, each
against its own servo (crab/params.py JOINT_SERVO).

* Drive saturation: the actuator force is the voltage-driven torque; hitting the model's
  force limit means the servo is maxed out -> PASS/FAIL criterion.
* Datasheet torque-speed line: stall torque at standstill falling linearly to zero at the
  no-load speed; we compute the motor's electromagnetic torque (actuator force - back-EMF
  slope * speed) and report how often it's beyond that line -> warning (ds%).
* Heating: RMS motor torque vs the servo's continuous *rated* torque -> thermal%.
"""

import numpy as np

from crab.params import JOINT_SPECS

KE = np.array([s.back_emf for s in JOINT_SPECS])
DS_STALL = np.array([s.datasheet_stall for s in JOINT_SPECS])
DS_NOLOAD = np.array([s.datasheet_noload for s in JOINT_SPECS])
FORCE_LIMIT = np.array([s.force_limit for s in JOINT_SPECS])
RATED = np.array([s.rated_torque for s in JOINT_SPECS])
MAX_CMD = np.array([s.max_cmd_speed for s in JOINT_SPECS])
KT = np.array([s.kt_current for s in JOINT_SPECS])
VIN = np.array([s.vin for s in JOINT_SPECS])


class ServoMonitor:
    def __init__(self, nu):
        self.n = 0
        self.volt_sat = np.zeros(nu)
        self.ds_over = np.zeros(nu)
        self.peak_motor = np.zeros(nu)
        self.peak_vel = np.zeros(nu)
        self.sq = np.zeros(nu)                    # sum of squared motor torque (RMS / heating)

    def update(self, actuator_force, actuator_velocity):
        f, w = np.asarray(actuator_force), np.asarray(actuator_velocity)
        tau_m = f - KE * w
        motoring = np.sign(tau_m) == np.sign(w)
        limit = DS_STALL * np.clip(1 - np.where(motoring, np.abs(w), 0.0) / DS_NOLOAD, 0, 1)
        self.n += 1
        self.volt_sat += np.abs(f) >= 0.98 * FORCE_LIMIT
        self.ds_over += np.abs(tau_m) > limit
        self.peak_motor = np.maximum(self.peak_motor, np.abs(tau_m))
        self.peak_vel = np.maximum(self.peak_vel, np.abs(w))
        self.sq += tau_m ** 2

    @property
    def volt_sat_pct(self):
        return 100 * self.volt_sat.max() / max(self.n, 1)

    @property
    def ds_over_pct(self):
        return 100 * self.ds_over.max() / max(self.n, 1)

    @property
    def rms(self):
        return np.sqrt(self.sq / max(self.n, 1))

    @property
    def thermal_pct(self):
        """Per joint: RMS motor torque as % of the servo's continuous rated torque."""
        return 100 * self.rms / RATED

    def worst_joint(self):
        return int(np.argmax(self.peak_motor / DS_STALL))

    def hottest_joint(self):
        return int(np.argmax(self.thermal_pct))

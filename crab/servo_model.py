"""The v2 leg servo (Feetech STS3215, 7.4 V) in the CPU twin, at two levels of detail. Both read their numbers from
configs/electronics.toml [leg_servo] (Rhoban BAM m6, measured; docs/dev/V2_MODEL.md).

    drive = make_drive("bam", model, data, JOINT_NAMES)      # or "first_order"
    drive.command(q18)          # joint targets (rad), JOINT_NAMES order: what the firmware was told
    drive.step()                # once per physics step, just before mujoco.mj_step

  FirstOrderServos  the MuJoCo position actuator already in sim/models/crab_v2.xml (stiffness, back-EMF damping,
                    Coulomb friction, rotor inertia, torque limit: BAM's own bam/to_mujoco.py first-order form),
                    plus the firmware's target slew limit (max_velocity). No load-dependent friction. This is
                    the form MJX / the training environments can use. Torque limit and stiffness do not follow the
                    supply voltage.
  BamServos         BAM's full model through bam.mujoco.MujocoController: the actuator turns into a torque motor
                    and, every step, the voltage-controlled DC motor with back-EMF, the 0.97 duty limit and the slew
                    limit set the torque, and the load-, direction- and speed-dependent (Stribeck) friction of the
                    fitted m6 model is written into the joint's friction and damping. The supply voltage can change
                    while it runs (set_supply). Needs the optional `bam` package (requirements.txt).
The arms (hobby servos) keep their position actuators in both; their targets pass straight to d.ctrl.
Command delay is NOT applied here: SimBackend's realism="datasheet" (crab/electronics.py) delays every target by the
measured 5.0 ms plus bus time before it reaches `command`; standalone loops use sim/latency.py's DelayLine.
"""

import numpy as np

from .electronics import Data

LEG_KINDS = ("coxa", "femur", "tibia")


def measured_parameters(data=None) -> dict:
    """The keys of Rhoban's m6.json, rebuilt from configs/electronics.toml [leg_servo] (every number there has
    source = "measured" and the JSON file as its ref)."""
    d = data if isinstance(data, Data) else Data(data)

    def g(k):
        return float(d("leg_servo." + k))
    return {
        "kt": g("kt_nm_per_a"), "R": g("resistance_ohm"), "armature": g("armature_kgm2"),
        "error_gain_ratio": g("error_gain_ratio"), "q_offset": 0.0, "max_velocity": g("max_velocity_rad_s"),
        "command_delay": g("command_delay_s"), "friction_base": g("friction_base"),
        "friction_stribeck": g("friction_stribeck"), "friction_viscous": g("friction_viscous"),
        "load_friction_motor": g("load_friction_motor"), "load_friction_external": g("load_friction_external"),
        "load_friction_motor_stribeck": g("load_friction_motor_stribeck"),
        "load_friction_external_stribeck": g("load_friction_external_stribeck"),
        "load_friction_motor_quad": g("load_friction_motor_quad"),
        "load_friction_external_quad": g("load_friction_external_quad"),
        "dtheta_stribeck": g("dtheta_stribeck"), "alpha": g("alpha"),
        "model": str(d("leg_servo.model_name")), "actuator": "sts3215",
    }


class _Drive:
    kind = "none"

    def __init__(self, m, d, joints, data=None):
        self.m, self.d, self.joints = m, d, list(joints)
        self.data = data if isinstance(data, Data) else Data(data)
        self.legs = [j for j in self.joints if j.split("_", 1)[1] in LEG_KINDS]
        self.leg_pos = np.array([self.joints.index(j) for j in self.legs])
        self.act = np.array([m.actuator(j).id for j in self.joints])
        self.leg_act = self.act[self.leg_pos]
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in self.legs])
        self.vadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in self.legs])
        self.target = d.ctrl[self.act].copy()
        self.torque_on = True

    def command(self, q):
        self.target[:] = q

    def reset(self):
        """After the simulation state was reset (a keyframe, a teleport): forget the firmware's smoothed target."""
        self.target[:] = self.d.ctrl[self.act]

    def set_supply(self, volts):
        pass

    def step(self):
        raise NotImplementedError


class FirstOrderServos(_Drive):
    kind = "first_order"

    def __init__(self, m, d, joints, data=None):
        super().__init__(m, d, joints, data)
        self.vmax = float(self.data("leg_servo.max_velocity_rad_s"))
        self.smooth = d.qpos[self.qadr].copy()                 # the firmware's internal, rate-limited target
        self.dt = float(m.opt.timestep)

    def reset(self):
        super().reset()
        self.smooth = self.d.qpos[self.qadr].copy()

    def set_supply(self, volts):
        """The voltage at each leg servo's terminals (one number or one per leg joint, `self.legs` order). The servo's
        stiffness (duty per error) and torque limit are proportional to it (BAM's own to_mujoco.py form, written for the
        fitted 7.4 V); the back-EMF damping kt^2 / R does not depend on it. 0 V: no torque."""
        if not hasattr(self, "_nominal"):
            m, a = self.m, self.leg_act
            self._nominal = (m.actuator_gainprm[a, 0].copy(), m.actuator_biasprm[a, 1].copy(),
                             m.actuator_forcerange[a].copy(), float(self.data("leg_servo.supply_v")))
        gain, bias, frc, v0 = self._nominal
        k = np.broadcast_to(np.asarray(volts, float) / v0, gain.shape)
        self.m.actuator_gainprm[self.leg_act, 0] = gain * k
        self.m.actuator_biasprm[self.leg_act, 1] = bias * k
        self.m.actuator_forcerange[self.leg_act] = frc * k[:, None]

    def step(self):
        step = self.vmax * self.dt
        tgt = self.target[self.leg_pos]
        self.smooth = np.clip(tgt, self.smooth - step, self.smooth + step)
        out = self.target.copy()
        out[self.leg_pos] = self.smooth
        self.d.ctrl[self.act] = out


class BamServos(_Drive):
    kind = "bam"

    def __init__(self, m, d, joints, data=None):
        super().__init__(m, d, joints, data)
        try:
            from bam.model import load_model_from_dict
            from bam.mujoco import MujocoController
        except ImportError as e:                                   # pragma: no cover
            raise ImportError("the full BAM servo model needs the optional `bam` package: "
                              "pip install --no-deps --ignore-requires-python git+https://github.com/Rhoban/bam "
                              "(see requirements.txt)") from e
        g = lambda k: float(self.data("leg_servo." + k))           # noqa: E731
        model = load_model_from_dict(measured_parameters(self.data))
        act = model.actuator
        act.vin, act.kp, act.error_gain, act.max_pwm = g("supply_v"), g("firmware_kp"), g("error_gain"), g("max_pwm")
        self.model, self.vin0 = model, act.vin
        # the servos' firmware angle limits (min/max angle registers, set at calibration): a goal outside the joint's
        # range is clamped, exactly as the position actuators' ctrlrange does in the first-order model
        self.goal_lo = np.array([m.actuator_ctrlrange[a, 0] for a in self.leg_act])
        self.goal_hi = np.array([m.actuator_ctrlrange[a, 1] for a in self.leg_act])
        for a in self.leg_act:             # the position actuator becomes a plain torque motor (ctrl = N*m)
            m.actuator_gaintype[a] = 0
            m.actuator_biastype[a] = 0
            m.actuator_gainprm[a, :] = 0
            m.actuator_gainprm[a, 0] = 1.0
            m.actuator_biasprm[a, :] = 0
            m.actuator_ctrllimited[a] = 0
            m.actuator_forcelimited[a] = 0
        self.ctl = MujocoController(model, self.legs, m, d)
        self.ctl.q_target[:] = d.qpos[self.qadr]
        self.torque = np.zeros(len(self.legs))

    def reset(self):
        super().reset()
        self.ctl.reset()
        self.ctl.q_target[:] = self.d.qpos[self.qadr]

    def set_supply(self, volts):
        v = np.asarray(volts, float)                 # one number, or one per leg joint (BAM's maths is element-wise)
        self.model.actuator.vin = float(v) if v.ndim == 0 else v.copy()

    @property
    def duty(self):
        return np.asarray(self.model.actuator.duty_cycle)

    def step(self):
        d = self.d
        d.ctrl[self.act] = self.target                             # arms: straight through
        self.ctl.q_target[:] = np.clip(self.target[self.leg_pos], self.goal_lo, self.goal_hi)
        self.ctl.update()                                          # leg torques into ctrl, friction into the joints
        if not self.torque_on:
            d.ctrl[self.leg_act] = 0.0


def make_drive(kind, m, d, joints, data=None):
    """kind: "first_order" or "bam"."""
    if kind == "first_order":
        return FirstOrderServos(m, d, joints, data)
    if kind == "bam":
        return BamServos(m, d, joints, data)
    raise ValueError(f"servo model must be 'first_order' or 'bam', not {kind!r}")


# ---------------------------------------------------------------------------------------------------------------------
# Heat. A first-order (one-node) thermal model of each bus servo, driven by its motor's copper loss, the form of
# Qian et al. (arXiv:2603.01631, per motor of a Unitree A1) and of Ben Katz's Mini Cheetah temperature observer:
#
#     C dT/dt = P - (T - T_amb) / R_th,    tau = R_th * C
#     P = (tau_motor / kt)^2 * R_w * (1 + a_cu (T - 25 C)) + P_gear + P_idle     (P_idle only while torque is on)
#
# P_gear is the gearbox's friction loss, which heats the same case: |w| * (Coulomb + viscous |w|) with the joint's
# friction numbers (for the STS3215, BAM's measured friction_base and friction_viscous).
# tau_motor is the torque the motor's current makes, at the output shaft; kt (output torque per amp) and the winding
# resistance R_w = V kt / stall come from the servo's spec (crab/params.py; the STS3215's are Rhoban BAM's measured m6
# model). One node models what the servo's present_temperature register reads (the sensor sits on the board, not in
# the winding, so the winding runs hotter than the reading: the mind's limits leave room for that).
# Feetech prints no thermal numbers. configs/electronics.toml [servo_heat] holds the STS3215's thermal resistance and
# time constant, estimated from SO-ARM100 users' hold tests (an STS3215 holding about 15 kg*cm settles near 48 C within
# 10 min); another servo scales by its case's surface area (natural convection: R_th ~ 1 / (h A); its heat capacity
# scales with mass, so tau stays about the same). The phase-3 bench step `heat` (a servo holding a known load while it
# logs present_temperature, then cooling) fits them with fit_heat() and replaces the estimates.

def _area_mm2(size):
    a, b, c = size
    return 2.0 * (a * b + a * c + b * c)


def heat_parameters(spec, data=None):
    """{R_w, kt, R_th, tau, P_idle, T_amb, T_protect, a_cu} for one servo spec (crab.params.ServoSpec)."""
    from crab.params import SERVOS
    d = data if isinstance(data, Data) else Data(data)

    def g(k):
        return float(d("servo_heat." + k))
    base = SERVOS.get(spec.name[:-1], spec) if spec.name.endswith("T") else spec    # a torque-limited variant: same motor
    R_w = base.vin * base.kt_current / base.model_stall                             # stall = V kt / R (BAM's own form)
    big = base.name.startswith("STS3215") or base.name == "ST3215"
    idle_a = float(d("leg_servo.idle_current_a" if big else "arm_servo.idle_current_a"))
    ref_area = _area_mm2(SERVOS["ST3215"].size_mm)                                  # the STS3215's case
    return dict(R_w=R_w, kt=base.kt_current, R_th=g("sts3215_thermal_resistance_k_per_w") * ref_area / _area_mm2(base.size_mm),
                tau=g("time_constant_s"), P_idle=spec.vin * idle_a, T_amb=g("ambient_c"), T_protect=g("protection_c"),
                a_cu=g("copper_tempco_per_k"))


class ServoHeat:
    """The temperature of N servos (C), stepped with each one's mean squared motor torque over the step.

        heat = ServoHeat([servo_for(j) for j in joints])
        heat.step(dt, mean_tau_sq, torque_on=True)
        heat.T                 # what each servo's present_temperature register would read (C, unrounded)
    """

    KEYS = ("R_w", "kt", "R_th", "tau", "P_idle", "T_amb", "T_protect", "a_cu")

    def __init__(self, specs=None, data=None, ambient=None, params=None):
        ps = params if params is not None else [heat_parameters(s, data) for s in specs]
        for k in self.KEYS:
            setattr(self, k, np.array([p[k] for p in ps], float))
        if ambient is not None:
            self.T_amb[:] = ambient
        self.T = self.T_amb.copy()
        self.peak = self.T.copy()

    def power(self, tau_sq, torque_on=True, T=None):
        """Watts dissipated in each servo: copper loss at temperature T (default: now) plus the electronics' idle draw."""
        T = self.T if T is None else T
        cu = np.asarray(tau_sq, float) / self.kt ** 2 * self.R_w * (1 + self.a_cu * (T - 25.0))
        return cu + (self.P_idle if torque_on else 0.0)

    def step(self, dt, tau_sq, torque_on=True, p_gear=0.0):
        """Advance dt seconds with each servo's mean squared motor torque (N*m)^2 and mean gear friction loss (W) over
        that time. Exact for constant power over the step (the copper's resistance is taken at the start temperature)."""
        T_inf = self.T_amb + self.R_th * (self.power(tau_sq, torque_on) + p_gear)
        self.T = T_inf + (self.T - T_inf) * np.exp(-dt / self.tau)
        np.maximum(self.peak, self.T, out=self.peak)
        return self.T

    def steady(self, tau_rms, torque_on=True):
        """Where each servo settles holding a torque of this RMS for good (the copper tempco's fixed point)."""
        T = self.T_amb.copy()
        for _ in range(40):
            T = self.T_amb + self.R_th * self.power(np.asarray(tau_rms, float) ** 2, torque_on, T)
        return T

    def predict(self, tau_rms, seconds, torque_on=True):
        """Each servo's temperature after `seconds` more at this RMS torque (the model itself is not stepped)."""
        T_inf = self.steady(tau_rms, torque_on)
        return T_inf + (self.T - T_inf) * np.exp(-float(seconds) / self.tau)

    def reading(self):
        """The present_temperature register: whole degrees C."""
        return np.round(self.T)


def fit_heat(t, temp, tau_sq, params, torque_on=True):
    """Fit one servo's thermal resistance R_th (K/W) and time constant tau (s) to a logged run that starts cold (at the
    room's temperature) and heats, and may cool again: t (s), temp (C, its register), tau_sq (mean squared motor torque
    over each interval ending at t[k], (N*m)^2). params: its heat_parameters() (R_w, kt, P_idle and a_cu are kept).
    For a fixed tau the rise is linear in R_th (the copper's resistance taken at the measured temperature), so each tau
    on a log grid gets its least-squares R_th; the best pair wins. The hook for robot/bench.py `heat`.
    Returns (R_th, tau, rms error C)."""
    t, temp, tau_sq = (np.asarray(x, float) for x in (t, temp, tau_sq))
    h = ServoHeat(params=[params])
    P = np.array([float(h.power(tau_sq[k], torque_on, np.array([temp[k - 1]]))[0]) if k else 0.0 for k in range(len(t))])
    y = temp - temp[0]
    best = None
    for tau in np.geomspace(10.0, 5000.0, 200):
        g, x = np.zeros(len(t)), 0.0                    # the rise with R_th = 1 K/W
        for k in range(1, len(t)):
            x = P[k] + (x - P[k]) * np.exp(-(t[k] - t[k - 1]) / tau)
            g[k] = x
        R = float(np.dot(g, y) / max(np.dot(g, g), 1e-12))
        err = float(np.sqrt(np.mean((R * g - y) ** 2)))
        if best is None or err < best[2]:
            best = (R, float(tau), err)
    return best

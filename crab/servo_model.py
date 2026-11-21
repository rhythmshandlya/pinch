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
        self.kind = "none"

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
        self.model.actuator.vin = float(volts)

    @property
    def duty(self):
        return np.asarray(self.model.actuator.duty_cycle)

    def step(self):
        d = self.d
        d.ctrl[self.act] = self.target                             # arms: straight through
        self.ctl.q_target[:] = self.target[self.leg_pos]
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

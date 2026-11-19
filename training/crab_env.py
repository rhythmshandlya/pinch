"""MJX (MuJoCo-on-JAX) walking environment for the crab, trained with Brax PPO.

Runs thousands of crabs in parallel on a GPU (Colab) or a handful on CPU for a
smoke test. Uses the same model file as the CPU simulator (sim/models/crab.xml),
adapted at load time:
  * foot-only collisions (the body never needs to touch the floor to learn;
    falls are detected from tilt / height),
  * sensors MJX does not need are removed (we compute IMU signals directly),
  * a larger physics step for speed.

Observations / actions come from crab/policy_io.py, the same code the real
robot uses.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab import policy_io as pio  # noqa: E402
from crab.params import LEG_IDS  # noqa: E402

MODEL = ROOT / "sim" / "models" / "crab.xml"


def _has_joint(m, name):
    return any(m.joint(i).name == name for i in range(m.njnt))


def load_training_model(sim_dt: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for s in list(spec.sensors):                 # touch sensors etc. not needed
        spec.delete(s)
    for g in list(spec.geoms):                   # CAD visual meshes: looks only, slow for MJX
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for msh in list(spec.meshes):
        spec.delete(msh)
    m = spec.compile()
    m.opt.timestep = sim_dt
    m.opt.iterations = 4
    m.opt.ls_iterations = 8
    feet = {m.geom(f"{lg}_foot").id for lg in LEG_IDS}
    floor = m.geom("floor").id
    for g in range(m.ngeom):
        if g == floor:
            continue
        if g in feet:
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 0
        else:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    m.geom_conaffinity[floor], m.geom_contype[floor] = 1, 0
    return m


# v2 (2026-11-19): v1 learned to stand still - Pinch's commands are only a few cm/s, and with a loose
# speed tolerance standing earned ~60 % of the tracking reward with none of walking's costs. Now the
# tolerance is tight (standing earns ~1 %), tracking counts double and effort penalties are lighter.
REWARD_SCALES = dict(
    tracking_lin_vel=3.0,
    tracking_ang_vel=0.8,
    lin_vel_z=-2.0,
    ang_vel_xy=-0.05,
    orientation=-2.0,
    torques=-0.002,
    torque_limit=-1.0,
    action_rate=-0.02,
    energy=-0.0005,
    feet_air_time=2.0,
    foot_slip=-0.5,
    stand_still=-0.5,
    pose=-0.05,
    termination=-5.0,
)


class CrabWalkEnv(Env):
    """Velocity-command walking. Command = (vx, vy, wz) in the body frame."""

    def __init__(self, sim_dt=0.004, episode_seconds=20.0, cmd_resample_s=5.0,
                 cmd_ranges=None, obs_noise=0.02, push_interval_s=4.0, push_vel=None,
                 reward_scales=None):
        from crab.params import PARAMS
        v, w = PARAMS.scale / PARAMS.scale ** 0.5, 1 / PARAMS.scale ** 0.5      # speeds scale with size
        if cmd_ranges is None:
            cmd_ranges = ((-0.10 * v, 0.12 * v), (-0.12 * v, 0.12 * v), (-0.7 * w, 0.7 * w))
        push_vel = 0.15 * v if push_vel is None else push_vel
        self.mj_model = load_training_model(sim_dt)
        self.sys = mjx.put_model(self.mj_model)   # attribute name expected by brax DR wrapper
        self.sim_dt = sim_dt
        self.n_substeps = int(round(pio.CTRL_DT / sim_dt))
        self.dt = pio.CTRL_DT
        self.episode_steps = int(episode_seconds / self.dt)
        self.cmd_resample_steps = int(cmd_resample_s / self.dt)
        self.cmd_lo = jp.array([r[0] for r in cmd_ranges])
        self.cmd_hi = jp.array([r[1] for r in cmd_ranges])
        self.obs_noise = obs_noise
        self.push_steps = int(push_interval_s / self.dt)
        self.push_vel = push_vel
        self.scales = dict(REWARD_SCALES, **(reward_scales or {}))

        m = self.mj_model
        self.init_qpos = jp.array(m.key("stand").qpos)
        # qpos / qvel addresses of the 12 leg joints, actuator ids of legs
        names = pio.LEG_JOINT_NAMES
        self.leg_qadr = jp.array([m.jnt_qposadr[m.joint(n).id] for n in names])
        self.leg_vadr = jp.array([m.jnt_dofadr[m.joint(n).id] for n in names])
        # servo encoders sit after the gear play: observe servo joint + backlash joint
        has_bl = _has_joint(m, f"{names[0]}_backlash")
        self.bl_qadr = jp.array([m.jnt_qposadr[m.joint(f"{n}_backlash").id] for n in names]) if has_bl else None
        self.bl_vadr = jp.array([m.jnt_dofadr[m.joint(f"{n}_backlash").id] for n in names]) if has_bl else None
        self.leg_act = jp.array(pio.LEG_ACT_IDX)
        self.foot_geoms = jp.array([m.geom(f"{lg}_foot").id for lg in LEG_IDS])
        self.foot_radius = float(m.geom_size[m.geom("FL_foot").id][0])
        self.body_id = m.body("body").id
        # per leg joint, from each joint's own servo: penalise torque above its continuous
        # rating (heating), and subtract back-EMF to get the motor's electromagnetic torque
        from crab.params import JOINT_SPECS, PARAMS
        leg_specs = [JOINT_SPECS[i] for i in pio.LEG_ACT_IDX]
        self.torque_soft_limit = jp.array([s.rated_torque for s in leg_specs])
        self.ke = jp.array([s.back_emf for s in leg_specs])
        self.fall_height = 0.37 * float(m.key("stand").qpos[2])
        self.vel_sigma = 0.015                      # m/s on the smoothed velocity (v1: 0.036 instantaneous)
        self.yaw_sigma = 0.15                       # rad/s (v1: 0.3)

    # brax Env interface ---------------------------------------------------
    @property
    def observation_size(self):
        return pio.OBS_SIZE

    @property
    def action_size(self):
        return pio.N_ACT

    @property
    def backend(self):
        return "mjx"

    def _sample_command(self, rng):
        k1, k2, k3 = jax.random.split(rng, 3)
        cmd = jax.random.uniform(k1, (3,), minval=self.cmd_lo, maxval=self.cmd_hi)
        # each component either clearly on or off, so "walk slowly" never looks like "stand"
        on = jax.random.uniform(k3, (3,)) < 0.6
        cmd = jp.where(on & (jp.abs(cmd) < 0.35 * self.cmd_hi), jp.sign(cmd) * 0.35 * self.cmd_hi, cmd) * on
        stand = jax.random.uniform(k2) < 0.08                        # 8 %: stand still
        return jp.where(stand, jp.zeros(3), cmd)

    def reset(self, rng):
        rng, k_q, k_yaw, k_cmd, k_lat = jax.random.split(rng, 5)
        qpos = self.init_qpos
        qpos = qpos.at[self.leg_qadr].add(jax.random.uniform(k_q, (pio.N_ACT,), minval=-0.05, maxval=0.05))
        yaw = jax.random.uniform(k_yaw, minval=-jp.pi, maxval=jp.pi)
        qpos = qpos.at[3:7].set(jp.array([jp.cos(yaw / 2), 0, 0, jp.sin(yaw / 2)]))
        data = mjx.make_data(self.sys).replace(qpos=qpos, ctrl=jp.array(pio.FULL_DEFAULT))
        data = mjx.forward(self.sys, data)

        info = dict(
            rng=rng,
            command=self._sample_command(k_cmd),
            step=jp.array(0),
            last_act=jp.zeros(pio.N_ACT),
            prev_act=jp.zeros(pio.N_ACT),              # for 1-step latency
            latency=jax.random.bernoulli(k_lat, 0.5),  # some episodes act one step late
            feet_air_time=jp.zeros(4),
            last_contact=jp.zeros(4, dtype=bool),
            last_foot_pos=data.geom_xpos[self.foot_geoms],
            v_filt=jp.zeros(3),                        # body velocity smoothed over ~a gait cycle
        )
        metrics = {k: jp.array(0.0) for k in self.scales}
        obs = self._obs(data, info, jax.random.PRNGKey(0))
        return State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        rng, k_noise, k_cmd, k_push = jax.random.split(info["rng"], 4)
        action = jp.clip(action, -1.0, 1.0)

        applied = jp.where(info["latency"], info["prev_act"], action)
        ctrl = pio.action_to_targets(jp, applied)

        data = state.pipeline_state
        # occasional shove so the policy learns to recover
        push = (info["step"] % self.push_steps == 0) & (info["step"] > 0)
        kick = jax.random.uniform(k_push, (2,), minval=-self.push_vel, maxval=self.push_vel)
        qvel = data.qvel.at[0:2].add(jp.where(push, kick, 0.0))
        data = data.replace(qvel=qvel, ctrl=ctrl)

        def substep(_, d):
            return mjx.step(self.sys, d)
        data = jax.lax.fori_loop(0, self.n_substeps, substep, data)

        # --- quantities for rewards -------------------------------------
        R = data.xmat[self.body_id]
        v_body = R.T @ data.qvel[0:3]
        w_body = data.qvel[3:6]
        gravity = R.T @ jp.array([0.0, 0.0, -1.0])
        tau = data.actuator_force[self.leg_act]
        qd = data.qvel[self.leg_vadr]
        tau_motor = tau - self.ke * qd          # electromagnetic torque (what the datasheet rates)
        q = data.qpos[self.leg_qadr]
        foot_pos = data.geom_xpos[self.foot_geoms]
        contact = foot_pos[:, 2] < self.foot_radius * 1.5
        first_contact = contact & ~info["last_contact"]
        air = info["feet_air_time"] + self.dt
        foot_vel = (foot_pos - info["last_foot_pos"]) / self.dt
        cmd = info["command"]
        moving = jp.linalg.norm(cmd) > 0.01
        # a walking body surges and slows within every step: score the speed averaged over ~a cycle
        v_filt = 0.85 * info["v_filt"] + 0.15 * jp.array([v_body[0], v_body[1], w_body[2]])

        r = dict(
            tracking_lin_vel=jp.exp(-jp.sum((cmd[:2] - v_filt[:2]) ** 2) / self.vel_sigma ** 2 / 2),
            tracking_ang_vel=jp.exp(-((cmd[2] - v_filt[2]) ** 2) / self.yaw_sigma ** 2 / 2),
            lin_vel_z=v_body[2] ** 2,
            ang_vel_xy=jp.sum(w_body[:2] ** 2),
            orientation=jp.sum(gravity[:2] ** 2),
            torques=jp.sum(tau ** 2),
            torque_limit=jp.sum(jp.maximum(jp.abs(tau_motor) - self.torque_soft_limit, 0.0) ** 2),
            action_rate=jp.sum((action - info["last_act"]) ** 2),
            energy=jp.sum(jp.abs(tau * qd)),
            feet_air_time=jp.sum((air - 0.15) * first_contact) * moving,
            foot_slip=jp.sum(contact * jp.sum(foot_vel[:, :2] ** 2, axis=1)),
            stand_still=jp.sum(jp.abs(q - pio.DEFAULT_LEG_POSE)) * (~moving),
            pose=jp.sum((q - pio.DEFAULT_LEG_POSE) ** 2),
        )
        height = data.qpos[2]
        fell = (gravity[2] > -0.7) | (height < self.fall_height)     # > ~45 deg tilt or belly down
        r["termination"] = fell.astype(jp.float32)
        reward = sum(self.scales[k] * v for k, v in r.items()) * self.dt
        reward = jp.clip(reward, -1.0, 10.0)

        # --- bookkeeping -------------------------------------------------
        step = info["step"] + 1
        resample = step % self.cmd_resample_steps == 0
        info.update(
            rng=rng,
            command=jp.where(resample, self._sample_command(k_cmd), cmd),
            step=step,
            prev_act=action,
            last_act=action,
            feet_air_time=air * ~contact,
            last_contact=contact,
            last_foot_pos=foot_pos,
            v_filt=jp.where(resample, jp.zeros(3), v_filt),
        )
        done = fell.astype(jp.float32)
        metrics = dict(state.metrics)        # keep keys wrappers add (e.g. "reward")
        metrics.update(r)
        obs = self._obs(data, info, k_noise)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done,
                             metrics=metrics, info=info)

    def _obs(self, data, info, rng):
        R = data.xmat[self.body_id]
        gyro = data.qvel[3:6]
        gravity = R.T @ jp.array([0.0, 0.0, -1.0])
        q, qd = data.qpos[self.leg_qadr], data.qvel[self.leg_vadr]
        if self.bl_qadr is not None:                       # encoder reads the output shaft (after gear play)
            q, qd = q + data.qpos[self.bl_qadr], qd + data.qvel[self.bl_vadr]
        obs = pio.observation(jp, gyro, gravity, info["command"], q, qd,
                              info["last_act"], info["step"] * self.dt)
        return obs + self.obs_noise * jax.random.normal(rng, obs.shape)


def _actuated_dofs():
    """(dof indices driven by a servo, matching actuator ids). Backlash dofs have no actuator."""
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    act_of_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
    dofs, acts = [], []
    for d in range(6, m.nv):
        j = int(m.dof_jntid[d])
        if j in act_of_joint:
            dofs.append(d)
            acts.append(act_of_joint[j])
    return np.array(dofs), np.array(acts)


_ACT_DOFS, _ACT_IDS = _actuated_dofs()


def domain_randomize(sys: mjx.Model, rng: jax.Array):
    """Per-environment physics variations so the policy can't overfit one exact robot.

    rng has shape (num_envs, 2). Returns a batched model and its vmap in_axes.
    """
    @jax.vmap
    def rand(key):
        k = jax.random.split(key, 6)
        friction = sys.geom_friction.at[:, 0].set(jax.random.uniform(k[0], minval=0.5, maxval=1.25))
        body_mass = sys.body_mass * jax.random.uniform(k[1], sys.body_mass.shape, minval=0.85, maxval=1.15)
        # servo stiffness: Open Duck runs ~0.74x of the BAM-derived kp, so cover that and a bit above
        kp_scale = jax.random.uniform(k[2], (sys.nu,), minval=0.75, maxval=1.25)   # unknown firmware P gains
        gain = sys.actuator_gainprm.at[:, 0].multiply(kp_scale)
        bias = sys.actuator_biasprm.at[:, 1].multiply(kp_scale)
        # Motor strength: the micro servos have no published lab measurements, so their
        # datasheet models are randomised widely (-25% .. +15%); force limit and back-EMF
        # damping move together (a weaker motor is also slower).
        s = jax.random.uniform(k[3], (sys.nu,), minval=0.75, maxval=1.15)
        frange = sys.actuator_forcerange * s[:, None]
        damp_scale = s
        noise = jax.random.uniform(k[5], (sys.nu,), minval=0.85, maxval=1.15)
        dof_scale = jp.ones(sys.dof_damping.shape).at[jp.asarray(_ACT_DOFS)].set(
            (damp_scale * noise)[jp.asarray(_ACT_IDS)])
        damping = sys.dof_damping * dof_scale
        armature = sys.dof_armature * jax.random.uniform(k[4], sys.dof_armature.shape, minval=0.8, maxval=1.2)
        return friction, body_mass, gain, bias, frange, armature, damping

    friction, body_mass, gain, bias, frange, armature, damping = rand(rng)
    fields = dict(geom_friction=friction, body_mass=body_mass, actuator_gainprm=gain,
                  actuator_biasprm=bias, actuator_forcerange=frange,
                  dof_armature=armature, dof_damping=damping)
    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace({k: 0 for k in fields})
    return sys.tree_replace(fields), in_axes

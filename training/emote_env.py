"""MJX environment for *expressive motion*: follow any keyframed emote while staying balanced.

The emotes in crab/emotes.py are keyframes (what Pinch should look like). Played open-loop they are
fine on a flat desk, but a shove, a slippery spot or a heavier battery can make Pinch stagger. This
environment trains one policy for all emotes:
  * the arms follow the keyframes exactly (they are PWM servos without feedback: nothing to correct)
  * the policy adds small corrections to the leg targets (residuals) to track the leg keyframes
    and the body attitude the emote has when played on a perfect robot, under random physics and
    pushes.

Observation (crab/policy_io.emote_observation) = IMU + leg encoders + last action + the reference
now and 0.1 s ahead + which emote + its progress. Same function on the robot.
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
from training.crab_env import MODEL, load_training_model  # noqa: E402

REWARD_SCALES = dict(
    track_legs=2.0,
    track_attitude=1.5,
    track_height=0.5,
    ang_vel_xy=-0.02,
    foot_slip=-1.0,
    action_rate=-0.1,
    torque_limit=-1.0,
    residual=-0.2,
    termination=-10.0,
)


def nominal_references(hz=50.0):
    """Play every emote open-loop on the nominal CPU model and record what it should look like.
    Returns dict of arrays: q (E,T,18) joint targets, grav (E,T,3) body-frame gravity, h (E,T) body
    height, n (E,) number of steps per emote."""
    from crab.emotes import STAND
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    names = pio.EMOTE_NAMES
    lens = [int(np.ceil(pio.EMOTE_LIB[n].duration * hz)) + 1 for n in names]
    T = max(lens)
    q = np.zeros((len(names), T, len(STAND)))
    grav = np.zeros((len(names), T, 3))
    h = np.zeros((len(names), T))
    act = np.array([m.actuator(j).id for j in pio.JOINT_NAMES_ALL])
    sub = int(round(1 / hz / m.opt.timestep))
    body = m.body("body").id
    for e, name in enumerate(names):
        emo = pio.EMOTE_LIB[name]
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        for _ in range(200):                                   # settle in the stand pose
            d.ctrl[act] = STAND
            mujoco.mj_step(m, d)
        for k in range(T):
            target = emo.target(min(k / hz, emo.duration))
            d.ctrl[act] = target
            for _ in range(sub):
                mujoco.mj_step(m, d)
            R = d.xmat[body].reshape(3, 3)
            q[e, k], grav[e, k], h[e, k] = target, R.T @ np.array([0, 0, -1.0]), d.qpos[2]
    return dict(q=q, grav=grav, h=h, n=np.array(lens))


class CrabEmoteEnv(Env):
    def __init__(self, sim_dt=0.004, obs_noise=0.02, push_interval_s=3.0, push_vel=0.08,
                 reward_scales=None, refs=None):
        self.mj_model = load_training_model(sim_dt)
        self.sys = mjx.put_model(self.mj_model)
        self.sim_dt = sim_dt
        self.n_substeps = int(round(pio.CTRL_DT / sim_dt))
        self.dt = pio.CTRL_DT
        self.obs_noise = obs_noise
        self.push_steps = int(push_interval_s / self.dt)
        self.push_vel = push_vel
        self.scales = dict(REWARD_SCALES, **(reward_scales or {}))
        refs = refs or nominal_references(1 / self.dt)
        self.ref_q = jp.array(refs["q"])
        self.ref_grav = jp.array(refs["grav"])
        self.ref_h = jp.array(refs["h"])
        self.ref_n = jp.array(refs["n"])
        self.n_emotes = len(refs["n"])
        self.lookahead = int(round(0.1 / self.dt))

        m = self.mj_model
        self.init_qpos = jp.array(m.key("stand").qpos)
        names = pio.LEG_JOINT_NAMES
        self.leg_qadr = jp.array([m.jnt_qposadr[m.joint(n).id] for n in names])
        self.leg_vadr = jp.array([m.jnt_dofadr[m.joint(n).id] for n in names])
        try:
            self.bl_qadr = jp.array([m.jnt_qposadr[m.joint(f"{n}_backlash").id] for n in names])
            self.bl_vadr = jp.array([m.jnt_dofadr[m.joint(f"{n}_backlash").id] for n in names])
        except KeyError:
            self.bl_qadr = self.bl_vadr = None
        self.leg_act = jp.array(pio.LEG_ACT_IDX)
        self.foot_geoms = jp.array([m.geom(f"{lg}_foot").id for lg in LEG_IDS])
        self.foot_radius = float(m.geom_size[m.geom("FL_foot").id][0])
        self.body_id = m.body("body").id
        from crab.params import JOINT_SPECS
        leg_specs = [JOINT_SPECS[i] for i in pio.LEG_ACT_IDX]
        self.torque_soft_limit = jp.array([s.rated_torque for s in leg_specs])
        self.ke = jp.array([s.back_emf for s in leg_specs])
        self.fall_height = 0.37 * float(m.key("stand").qpos[2])

    @property
    def observation_size(self):
        return pio.EMOTE_OBS_SIZE

    @property
    def action_size(self):
        return pio.N_ACT

    @property
    def backend(self):
        return "mjx"

    def _ref(self, e, k):
        k = jp.minimum(k, self.ref_n[e] - 1)
        return self.ref_q[e, k], self.ref_grav[e, k], self.ref_h[e, k]

    def reset(self, rng):
        rng, k_e, k_q = jax.random.split(rng, 3)
        e = jax.random.randint(k_e, (), 0, self.n_emotes)
        qpos = self.init_qpos.at[self.leg_qadr].add(jax.random.uniform(k_q, (pio.N_ACT,), minval=-0.03, maxval=0.03))
        data = mjx.make_data(self.sys).replace(qpos=qpos, ctrl=self.ref_q[e, 0])
        data = mjx.forward(self.sys, data)
        info = dict(rng=rng, emote=e, k=jp.array(0), step=jp.array(0), last_act=jp.zeros(pio.N_ACT),
                    last_foot_pos=data.geom_xpos[self.foot_geoms])
        metrics = {k: jp.array(0.0) for k in self.scales}
        obs = self._obs(data, info, jax.random.PRNGKey(0))
        return State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        rng, k_noise, k_push, k_e = jax.random.split(info["rng"], 4)
        action = jp.clip(action, -1.0, 1.0)
        e, k = info["emote"], info["k"]
        ref_q, ref_g, ref_h = self._ref(e, k + 1)
        ctrl = pio.emote_targets(jp, ref_q, action)

        data = state.pipeline_state
        push = (info["step"] % self.push_steps == 0) & (info["step"] > 0)
        kick = jax.random.uniform(k_push, (2,), minval=-self.push_vel, maxval=self.push_vel)
        data = data.replace(qvel=data.qvel.at[0:2].add(jp.where(push, kick, 0.0)), ctrl=ctrl)
        data = jax.lax.fori_loop(0, self.n_substeps, lambda _, d: mjx.step(self.sys, d), data)

        R = data.xmat[self.body_id]
        gravity = R.T @ jp.array([0.0, 0.0, -1.0])
        w_body = data.qvel[3:6]
        q = data.qpos[self.leg_qadr]
        qd = data.qvel[self.leg_vadr]
        tau = data.actuator_force[self.leg_act]
        tau_motor = tau - self.ke * qd
        foot_pos = data.geom_xpos[self.foot_geoms]
        contact = foot_pos[:, 2] < self.foot_radius * 1.5
        foot_vel = (foot_pos - info["last_foot_pos"]) / self.dt
        legs_ref = ref_q[self.leg_act]
        r = dict(
            track_legs=jp.exp(-jp.sum((q - legs_ref) ** 2) / (2 * 0.15 ** 2)),
            track_attitude=jp.exp(-jp.sum((gravity[:2] - ref_g[:2]) ** 2) / (2 * 0.05 ** 2)),
            track_height=jp.exp(-((data.qpos[2] - ref_h) ** 2) / (2 * 0.006 ** 2)),
            ang_vel_xy=jp.sum(w_body[:2] ** 2),
            foot_slip=jp.sum(contact * jp.sum(foot_vel[:, :2] ** 2, axis=1)) * 100.0,
            action_rate=jp.sum((action - info["last_act"]) ** 2),
            torque_limit=jp.sum(jp.maximum(jp.abs(tau_motor) - self.torque_soft_limit, 0.0) ** 2),
            residual=jp.sum(action ** 2),
        )
        fell = (gravity[2] > -0.7) | (data.qpos[2] < self.fall_height)
        r["termination"] = fell.astype(jp.float32)
        reward = jp.clip(sum(self.scales[n] * v for n, v in r.items()) * self.dt, -1.0, 10.0)

        # next emote when this one is over (Pinch strings emotes together all day)
        k_next = k + 1
        over = k_next >= self.ref_n[e] - 1
        e_new = jax.random.randint(k_e, (), 0, self.n_emotes)
        info.update(rng=rng, emote=jp.where(over, e_new, e), k=jp.where(over, 0, k_next),
                    step=info["step"] + 1, last_act=action, last_foot_pos=foot_pos)
        metrics = dict(state.metrics)
        metrics.update(r)
        obs = self._obs(data, info, k_noise)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=fell.astype(jp.float32),
                             metrics=metrics, info=info)

    def _obs(self, data, info, rng):
        R = data.xmat[self.body_id]
        gyro = data.qvel[3:6]
        gravity = R.T @ jp.array([0.0, 0.0, -1.0])
        q, qd = data.qpos[self.leg_qadr], data.qvel[self.leg_vadr]
        if self.bl_qadr is not None:
            q, qd = q + data.qpos[self.bl_qadr], qd + data.qvel[self.bl_vadr]
        e, k = info["emote"], info["k"]
        now, _, _ = self._ref(e, k + 1)
        ahead, _, _ = self._ref(e, k + 1 + self.lookahead)
        progress = k / jp.maximum(self.ref_n[e] - 1, 1)
        obs = pio.emote_observation(jp, gyro, gravity, q, qd, info["last_act"], now, ahead,
                                    jax.nn.one_hot(e, self.n_emotes), progress)
        return obs + self.obs_noise * jax.random.normal(rng, obs.shape)

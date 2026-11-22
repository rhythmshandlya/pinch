"""MJX environment: get up off its back (decision 76).

A searched two-pose swing (crab/righting.py) already rolls Pinch over in ~0.6 s from 8 test starts. The
research plan expects a learned policy to be faster and more reliable across more starts (OP3: get-up 63 %
faster; ANYmal: >97 % recoveries). This env trains one to compare:

  start:  on its back at a random orientation (rolled 180 +- 30 deg, pitched +-25 deg, any heading), joints
          scattered around the stand pose; robot physics randomised as usual
  policy: 50 Hz targets for all 18 joints, each mapped over its full range (no reference motion)
  contact: the shell (chassis box, dome, visor), legs and feet touch the floor - they are what it rolls on
  reward: getting upright (body z axis up), then standing at the stand height; penalties for jerky and
          wasteful motion. 4 s episodes; success = upright and at height over the last second.
Observation and action mapping: crab/policy_io.py (righting_*), shared with the robot and the bench.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jp
from brax.envs.base import Env, State
from mujoco import mjx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab import policy_io as pio  # noqa: E402
from crab.params import JOINT_NAMES  # noqa: E402
from training.robot_model import load_training_model  # noqa: E402

REWARD_SCALES = dict(upright=1.0, height=1.0, stood=2.0, action_rate=-0.01, energy=-0.0005, termination=0.0)


def load_righting_model(sim_dt=0.002):
    """The training model with the whole robot touching the floor (it rolls on its shell, legs and claws),
    never itself, and a few more solver iterations for the many contacts."""
    from training.robot_model import COLLIDERS
    m = load_training_model(sim_dt)
    m.opt.iterations = 6
    floor = m.geom("floor").id
    for g in range(m.ngeom):
        if g == floor:
            continue
        # the model's own contact shapes (shell, legs, claw jaws); not the electronics boxes inside the shell or the
        # claw's small pads (dozens of extra floor contacts per robot: the get-up run crawled on msi-1)
        on = (m.geom(g).name or "") in COLLIDERS if COLLIDERS else True
        m.geom_contype[g], m.geom_conaffinity[g] = (1, 0) if on else (0, 0)
    return m


class CrabRightingEnv(Env):
    def __init__(self, sim_dt=0.002, episode_seconds=4.0, obs_noise=0.02, reward_scales=None):
        self.mj_model = m = load_righting_model(sim_dt)
        self.sys = mjx.put_model(m)
        self.n_phys = int(round(pio.CTRL_DT / sim_dt))
        self.dt = pio.CTRL_DT
        self.T = int(episode_seconds / self.dt)
        self.obs_noise = obs_noise
        self.scales = dict(REWARD_SCALES, **(reward_scales or {}))
        self.lo = jp.array(m.actuator_ctrlrange[:, 0])
        self.hi = jp.array(m.actuator_ctrlrange[:, 1])
        self.qadr = jp.array([m.jnt_qposadr[m.joint(j).id] for j in JOINT_NAMES])
        self.vadr = jp.array([m.jnt_dofadr[m.joint(j).id] for j in JOINT_NAMES])
        self.body_id = m.body("body").id
        self.init_qpos = jp.array(m.key("stand").qpos)
        self.stand_h = float(m.key("stand").qpos[2])

    @property
    def observation_size(self):
        return pio.RIGHTING_OBS_SIZE

    @property
    def action_size(self):
        return len(JOINT_NAMES)

    @property
    def backend(self):
        return "mjx"

    def reset(self, rng):
        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
        roll = jp.pi + jax.random.uniform(k1, minval=-0.52, maxval=0.52)
        pitch = jax.random.uniform(k2, minval=-0.44, maxval=0.44)
        yaw = jax.random.uniform(k3, minval=-jp.pi, maxval=jp.pi)
        cr, sr, cp, sp, cy, sy = jp.cos(roll / 2), jp.sin(roll / 2), jp.cos(pitch / 2), jp.sin(pitch / 2), \
            jp.cos(yaw / 2), jp.sin(yaw / 2)
        quat = jp.array([cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr, cy * sp * cr + sy * cp * sr,
                         sy * cp * cr - cy * sp * sr])
        q = self.init_qpos.at[2].set(0.08).at[3:7].set(quat)
        q = q.at[self.qadr].add(jax.random.uniform(k4, (len(JOINT_NAMES),), minval=-0.2, maxval=0.2))
        data = mjx.make_data(self.sys).replace(qpos=q, ctrl=jp.clip(q[self.qadr], self.lo, self.hi))
        data = jax.lax.fori_loop(0, 150, lambda _, d: mjx.step(self.sys, d), mjx.forward(self.sys, data))  # land
        info = dict(rng=rng, step=jp.array(0), last_act=jp.zeros(len(JOINT_NAMES)), up_steps=jp.array(0))
        metrics = {k: jp.array(0.0) for k in self.scales}
        metrics["success_rate"] = jp.array(0.0)
        return State(data, self._obs(data, info, jax.random.PRNGKey(0)), jp.array(0.0), jp.array(0.0), metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        if "steps" in info:
            new = info["steps"] == 0
            info.update(step=jp.where(new, 0, info["step"]), last_act=jp.where(new, 0.0, info["last_act"]),
                        up_steps=jp.where(new, 0, info["up_steps"]))
        rng, k_noise = jax.random.split(info["rng"])
        action = jp.clip(action, -1, 1)
        target = pio.righting_targets(jp, action, self.lo, self.hi)
        data = state.pipeline_state.replace(ctrl=target)
        data = jax.lax.fori_loop(0, self.n_phys, lambda _, d: mjx.step(self.sys, d), data)
        R = data.xmat[self.body_id]
        z_up = R[2, 2]                                        # body z axis . world z (1 = upright)
        h = data.qpos[2]
        upright = (z_up + 1.0) / 2.0
        at_height = jp.exp(-((h - self.stand_h) / 0.012) ** 2) * (z_up > 0.85)
        up = (z_up > 0.85) & (h > 0.7 * self.stand_h)
        up_steps = jp.where(up, info["up_steps"] + 1, 0)
        qd = data.qvel[self.vadr]
        r = dict(upright=upright, height=at_height, stood=(up_steps > 25).astype(jp.float32),
                 action_rate=jp.sum((action - info["last_act"]) ** 2), energy=jp.sum(qd ** 2),
                 termination=jp.array(0.0))
        bad = ~(jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel)))
        r = {k: jp.where(bad, 0.0, jp.asarray(v, jp.float32)) for k, v in r.items()}
        reward = jp.where(bad, -1.0, sum(self.scales[k] * v for k, v in r.items()) * self.dt)
        info.update(rng=rng, step=info["step"] + 1, last_act=action, up_steps=up_steps)
        metrics = dict(state.metrics)
        metrics.update(r)
        metrics["success_rate"] = (up_steps > 50).astype(jp.float32)
        obs = self._obs(data, info, k_noise)
        obs = jp.where(jp.isfinite(obs), obs, 0.0)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=bad.astype(jp.float32),
                             metrics=metrics, info=info)

    def _obs(self, data, info, rng):
        R = data.xmat[self.body_id]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        obs = pio.righting_observation(jp, grav, data.qvel[3:6], data.qpos[self.qadr], data.qvel[self.vadr],
                                       info["last_act"])
        return obs + self.obs_noise * jax.random.normal(rng, obs.shape)

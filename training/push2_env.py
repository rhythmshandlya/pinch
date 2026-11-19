"""MJX environment: push a desk thing to a goal, on Pinch's real walking controller (decision 67).

Body's push is: line up behind the thing (scripted), lower the paddle claw, then walk crouched while steering
so the thing slides to the goal. The steering is the brittle part (decision 65: a cap goes in 10 % of the
time). This env learns only that part:

  high level (this policy, 10 Hz): the walking command (vx, vy, turn) - nothing else
  low level: the scripted trot in JAX (crab/gait_jax.py, identical to the robot's), right claw as the
             paddle, crouched 11 mm - exactly what Body.Manipulation._push_to drives

Each episode starts where the scripted line-up ends: the thing at the paddle (with noise), the goal 8-25 cm
ahead within +-35 deg. The thing's size, mass and friction vary; shoves now and then. Reward: the thing getting
closer to the goal, the paddle staying just behind it, a bonus on arrival; penalties for losing it, tilting,
jerky commands and falling. Observation (all available on the robot): gyro, gravity, the last command, the
thing and the goal relative to the body (camera + the desk marker), the thing's velocity, time left.
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
from crab.gait import _S, GAITS  # noqa: E402
from crab.gait_jax import GaitJax  # noqa: E402
from crab.params import JOINT_NAMES, LEG_IDS  # noqa: E402
from crab.arm import PUSH_LOWER, paddle_pose  # noqa: E402

MODEL = ROOT / "sim" / "models" / "crab.xml"
THING_H = 0.006
HL_DT = 0.1                                   # a decision every 0.1 s
SUB = 10                                      # gait substeps per 20 ms control tick (as Body)
CMD_LO = np.array([-0.035, -0.03, -0.6])      # the push envelope (Body pushes at <= 4.2 cm/s, steering gently)
CMD_HI = np.array([0.05, 0.03, 0.6])
REWARD_SCALES = dict(progress=40.0, near=0.05, success=0.5, tilt=-0.2, action_rate=-0.01, lost=-0.3,
                     termination=-1.0)
OBS_SIZE = 3 + 3 + 3 + 2 + 2 + 2 + 1


def load_push2_model(sim_dt=0.002):
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for s in list(spec.sensors):
        spec.delete(s)
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for msh in list(spec.meshes):
        spec.delete(msh)
    b = spec.worldbody.add_body(name="thing", pos=[0.3, 0, THING_H])
    b.add_freejoint(name="thing_free")
    b.add_geom(name="thing_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.012, THING_H], mass=0.005,
               friction=[0.6, 0.005, 0.0001])
    m = spec.compile()
    m.opt.timestep = sim_dt
    m.opt.iterations = 6
    m.opt.ls_iterations = 8
    floor = m.geom("floor").id
    feet = {m.geom(f"{lg}_foot").id for lg in LEG_IDS}
    arm = set()
    for body in ("R_upper", "R_dactyl"):
        bid = m.body(body).id
        arm |= {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid
                and m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE)}
    thing = m.geom("thing_geom").id
    for g in range(m.ngeom):                         # feet: floor + thing; paddle: thing; thing: floor + both
        if g == floor:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 1
        elif g in feet:
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 2
        elif g in arm:
            m.geom_contype[g], m.geom_conaffinity[g] = 2, 0
        elif g == thing:
            m.geom_contype[g], m.geom_conaffinity[g] = 3, 2
        else:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    return m


def push2_randomize(sys_: mjx.Model, rng: jax.Array):
    """The usual physics randomisation, plus the thing's size, mass and friction per robot."""
    from training.robot_model import domain_randomize
    sys_b, in_axes = domain_randomize(sys_, rng)
    gid = _THING_GEOM
    bid = _THING_BODY

    @jax.vmap
    def thing(key, size, mass, fric):
        k1, k2, k3 = jax.random.split(jax.random.fold_in(key, 11), 3)
        half = jax.random.uniform(k1, minval=0.008, maxval=0.016)
        size = size.at[gid, 0].set(half).at[gid, 1].set(half)
        mass = mass.at[bid].set(jax.random.uniform(k2, minval=0.003, maxval=0.010))
        fric = fric.at[gid, 0].set(jax.random.uniform(k3, minval=0.3, maxval=0.9))
        return size, mass, fric

    n = rng.shape[0]
    size, mass, fric = thing(rng, jp.broadcast_to(sys_.geom_size, (n,) + sys_.geom_size.shape),
                             sys_b.body_mass, sys_b.geom_friction)
    in_axes = in_axes.tree_replace({"geom_size": 0})
    return sys_b.tree_replace({"geom_size": size, "body_mass": mass, "geom_friction": fric}), in_axes


_THING_GEOM = _THING_BODY = None


class CrabPush2Env(Env):
    def __init__(self, sim_dt=0.002, episode_seconds=15.0, obs_noise=0.01, push_vel=0.06, push_interval_s=3.0,
                 reward_scales=None):
        global _THING_GEOM, _THING_BODY
        self.mj_model = m = load_push2_model(sim_dt)
        _THING_GEOM, _THING_BODY = m.geom("thing_geom").id, m.body("thing").id
        self.sys = mjx.put_model(m)
        self.n_phys = int(round(pio.CTRL_DT / sim_dt))
        self.n_low = int(round(HL_DT / pio.CTRL_DT))
        self.dt = HL_DT
        self.T = int(episode_seconds / HL_DT)
        self.obs_noise, self.push_vel = obs_noise, push_vel
        self.push_every = int(push_interval_s / HL_DT)
        self.scales = dict(REWARD_SCALES, **(reward_scales or {}))
        self.gait = GaitJax(GAITS["trot"], com_offset=(0.015 * _S, 0.0), arms=paddle_pose("right"))
        self.act_ids = jp.array([m.actuator(j).id for j in JOINT_NAMES])
        self.body_id = m.body("body").id
        self.thing_body = m.body("thing").id
        self.thing_q = m.jnt_qposadr[m.joint("thing_free").id]
        self.site = m.site("R_grip_site").id
        q0 = np.array(m.key("stand").qpos)
        self.init_qpos = jp.array(q0)
        self.fall_height = 0.37 * float(m.key("stand").qpos[2])
        self.cmd_lo, self.cmd_hi = jp.array(CMD_LO), jp.array(CMD_HI)
        self.tip0 = jp.array([0.10, -0.035])           # where the paddle's centre sits in the body frame (approx.)

    @property
    def observation_size(self):
        return OBS_SIZE

    @property
    def action_size(self):
        return 3

    @property
    def backend(self):
        return "mjx"

    def _rel(self, data, p2):
        R = data.xmat[self.body_id]
        d = jp.concatenate([p2 - data.xpos[self.body_id][:2], jp.zeros(1)])
        return (R.T @ d)[:2]

    def reset(self, rng):
        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
        thing = self.tip0 + jax.random.uniform(k1, (2,), minval=jp.array([0.005, -0.012]),
                                               maxval=jp.array([0.03, 0.012]))
        ang = jax.random.uniform(k2, minval=-0.6, maxval=0.6)
        dist = jax.random.uniform(k3, minval=0.08, maxval=0.25)
        goal = thing + dist * jp.array([jp.cos(ang), jp.sin(ang)])
        qpos = self.init_qpos.at[self.thing_q:self.thing_q + 3].set(jp.array([thing[0], thing[1], THING_H]))
        data = mjx.forward(self.sys, mjx.make_data(self.sys).replace(qpos=qpos, ctrl=self.gait.base))
        gs = self.gait.init()
        gs["lower"] = jp.array(PUSH_LOWER)
        info = dict(rng=rng, step=jp.array(0), goal=goal, cmd=jp.zeros(3), last_act=jp.zeros(3), gait=gs,
                    dist=jp.linalg.norm(thing - goal), dist0=jp.linalg.norm(thing - goal), thing0=thing,
                    prev_thing=thing)
        metrics = {k: jp.array(0.0) for k in self.scales}
        metrics["success_rate"] = jp.array(0.0)
        return State(data, self._obs(data, info, jax.random.PRNGKey(0)), jp.array(0.0), jp.array(0.0), metrics, info)

    def step(self, state, action):
        info = dict(state.info)
        if "steps" in info:          # Brax's auto-reset restores only the physics: fresh bookkeeping
            new = info["steps"] == 0
            fresh = self.gait.init()
            fresh["lower"] = jp.array(PUSH_LOWER)
            info["gait"] = jax.tree_util.tree_map(lambda a, b: jp.where(new, b, a), info["gait"], fresh)
            info.update(step=jp.where(new, 0, info["step"]), last_act=jp.where(new, 0.0, info["last_act"]),
                        dist=jp.where(new, info["dist0"], info["dist"]), cmd=jp.where(new, 0.0, info["cmd"]),
                        prev_thing=jp.where(new, info["thing0"], info["prev_thing"]))
        rng, k_noise, k_push = jax.random.split(info["rng"], 3)
        action = jp.clip(action, -1, 1)
        cmd = self.cmd_lo + (self.cmd_hi - self.cmd_lo) * (action + 1) / 2
        data = state.pipeline_state
        shove = (info["step"] % self.push_every == 0) & (info["step"] > 0)
        kick = jax.random.uniform(k_push, (2,), minval=-self.push_vel, maxval=self.push_vel)
        data = data.replace(qvel=data.qvel.at[0:2].add(jp.where(shove, kick, 0.0)))

        def tick(_, carry):
            d, gs = carry
            for _k in range(SUB):                           # (unrolled: SUB gait substeps per control tick)
                gs, q = self.gait.step(gs, pio.CTRL_DT / SUB, cmd)
            d = d.replace(ctrl=q)
            d = jax.lax.fori_loop(0, self.n_phys, lambda __, dd: mjx.step(self.sys, dd), d)
            return d, gs
        data, gs = jax.lax.fori_loop(0, self.n_low, tick, (data, info["gait"]))
        R = data.xmat[self.body_id]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        thing = data.xpos[self.thing_body][:2]
        goal = info["goal"]
        dist = jp.linalg.norm(thing - goal)
        u = (goal - thing) / (dist + 1e-6)
        behind = thing - u * 0.025
        claw = data.site_xpos[self.site][:2]
        r = dict(progress=info["dist"] - dist,
                 near=1.0 - jp.tanh(jp.linalg.norm(claw - behind) / 0.05),
                 success=(dist < 0.025).astype(jp.float32),
                 tilt=jp.sum(grav[:2] ** 2),
                 action_rate=jp.sum((action - info["last_act"]) ** 2),
                 lost=(jp.linalg.norm(thing - claw) > 0.12).astype(jp.float32))
        r = {k: jp.asarray(v, jp.float32) for k, v in r.items()}
        bad = ~(jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel)))
        fell = (grav[2] > -0.7) | (data.qpos[2] < self.fall_height) | bad
        r["termination"] = fell.astype(jp.float32)
        r = {k: jp.where(bad, 0.0, v) for k, v in r.items()}
        reward = jp.clip(sum(self.scales[k] * v for k, v in r.items()), -1.0, 10.0)
        reward = jp.where(bad, -1.0, reward)
        dist = jp.where(bad, info["dist0"], dist)
        info.update(rng=rng, step=info["step"] + 1, cmd=cmd, last_act=action, dist=dist, gait=gs,
                    prev_thing=jp.where(bad, info["prev_thing"], thing))
        metrics = dict(state.metrics)
        metrics.update(r)
        metrics["success_rate"] = (dist < 0.025).astype(jp.float32) / self.T * 20.0
        obs = self._obs(data, info, k_noise)
        obs = jp.where(jp.isfinite(obs), obs, 0.0)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=fell.astype(jp.float32),
                             metrics=metrics, info=info)

    def _obs(self, data, info, rng):
        R = data.xmat[self.body_id]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        thing = data.xpos[self.thing_body][:2]
        vel = (R.T @ jp.concatenate([(thing - info["prev_thing"]) / HL_DT, jp.zeros(1)]))[:2]
        obs = jp.concatenate([data.qvel[3:6] * 0.25, grav, (info["cmd"] - self.cmd_lo) / (self.cmd_hi - self.cmd_lo) * 2 - 1,
                              self._rel(data, thing) * 5.0, self._rel(data, info["goal"]) * 5.0, vel * 5.0,
                              jp.array([info["step"] / self.T])])
        return obs + self.obs_noise * jax.random.normal(rng, obs.shape)

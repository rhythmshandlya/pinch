"""MJX environment: push a desk thing to a goal (hierarchical reinforcement learning).

Why: the scripted pusher (pinch/body.py) works but is brittle - each object needed its own hand-tuned
thresholds. Legged loco-manipulation work learns pushing hierarchically: a locomotion policy underneath,
a manipulation policy on top that decides where to walk and what the limb does. So:

  high level (this policy, 10 Hz): walking command (vx, vy, turn) + right arm (swing, lift, claw)
  low level (the trained walking policy, policies/walk.npz, 50 Hz): turns the command into leg targets

Observation: IMU gravity + gyro, the current command, the arm's commands, and - from the camera on the
robot - the thing and the goal relative to Pinch's body. Reward: the thing getting closer to the goal
(dense progress), a small pull toward having the claw just behind it, a bonus when it arrives;
penalties for falling and for knocking it far off course. The thing's size, mass and friction, the
goal and the robot's physics are randomised.
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
from crab.params import JOINT_NAMES, LEG_IDS  # noqa: E402

MODEL = ROOT / "sim" / "models" / "crab.xml"
PUCK_R, PUCK_H = 0.014, 0.006
HL_DT = 0.1                                   # high-level decision every 0.1 s
REWARD_SCALES = dict(progress=40.0, near=0.05, success=0.5, tilt=-0.2, action_rate=-0.005, lost=-0.2,
                     termination=-1.0)          # per 0.1 s decision
_R_ARM = [JOINT_NAMES.index(f"R_{j}") for j in ("shoulder_yaw", "shoulder_pitch", "claw")]


def load_push_model(sim_dt=0.004):
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for s in list(spec.sensors):
        spec.delete(s)
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for msh in list(spec.meshes):
        spec.delete(msh)
    b = spec.worldbody.add_body(name="puck", pos=[0.3, 0, PUCK_H])
    b.add_freejoint(name="puck_free")
    # a small block (MJX has no cylinder-box collisions; an eraser/cap-sized box stands in for both)
    b.add_geom(name="puck_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.012, PUCK_H], mass=0.006,
               friction=[0.6, 0.005, 0.0001])
    m = spec.compile()
    m.opt.timestep = sim_dt
    m.opt.iterations = 6
    m.opt.ls_iterations = 8
    floor = m.geom("floor").id
    feet = {m.geom(f"{lg}_foot").id for lg in LEG_IDS}
    arm_geoms = set()
    for side in ("R", "L"):
        for body in (f"{side}_upper", f"{side}_dactyl"):
            bid = m.body(body).id
            arm_geoms |= {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid
                          and m.geom_type[g] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CAPSULE)}
    puck = m.geom("puck_geom").id
    for g in range(m.ngeom):
        if g == floor:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 1
        elif g in feet:
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 2          # feet also bump the puck
        elif g in arm_geoms:
            m.geom_contype[g], m.geom_conaffinity[g] = 2, 0
        elif g == puck:
            m.geom_contype[g], m.geom_conaffinity[g] = 3, 2
        else:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    return m


def _walk_policy():
    z = np.load(ROOT / "policies" / "walk.npz")
    n = len([k for k in z.files if k.startswith("W")])
    return dict(mean=jp.array(z["obs_mean"]), std=jp.array(z["obs_std"]),
                layers=[(jp.array(z[f"W{i}"]), jp.array(z[f"b{i}"])) for i in range(n)])


def _mlp(p, obs):
    x = jp.clip((obs - p["mean"]) / p["std"], -5, 5)
    for i, (W, b) in enumerate(p["layers"]):
        x = x @ W + b
        if i < len(p["layers"]) - 1:
            x = x * jax.nn.sigmoid(x)
    return jp.tanh(x[: x.shape[-1] // 2])


class CrabPushEnv(Env):
    def __init__(self, sim_dt=0.004, episode_seconds=15.0, obs_noise=0.01, reward_scales=None):
        self.mj_model = load_push_model(sim_dt)
        self.sys = mjx.put_model(self.mj_model)
        m = self.mj_model
        self.n_sub = int(round(pio.CTRL_DT / sim_dt))                # physics steps per low-level step
        self.n_low = int(round(HL_DT / pio.CTRL_DT))                 # low-level steps per decision
        self.dt = HL_DT
        self.T = int(episode_seconds / HL_DT)
        self.obs_noise = obs_noise
        self.scales = dict(REWARD_SCALES, **(reward_scales or {}))
        self.walk = _walk_policy()
        self.body_id = m.body("body").id
        self.puck_body = m.body("puck").id
        self.puck_q = m.jnt_qposadr[m.joint("puck_free").id]
        self.site = m.site("R_grip_site").id
        self.leg_qadr = jp.array([m.jnt_qposadr[m.joint(n).id] for n in pio.LEG_JOINT_NAMES])
        self.leg_vadr = jp.array([m.jnt_dofadr[m.joint(n).id] for n in pio.LEG_JOINT_NAMES])
        self.bl_q = jp.array([m.jnt_qposadr[m.joint(f"{n}_backlash").id] for n in pio.LEG_JOINT_NAMES])
        self.bl_v = jp.array([m.jnt_dofadr[m.joint(f"{n}_backlash").id] for n in pio.LEG_JOINT_NAMES])
        q0 = np.array(m.key("stand").qpos)
        q0[self.puck_q:self.puck_q + 7] = [0.3, 0, PUCK_H, 1, 0, 0, 0]
        self.init_qpos = jp.array(q0)
        self.fall_height = 0.37 * float(m.key("stand").qpos[2])
        from crab.params import PARAMS
        v, w = PARAMS.scale ** 0.5, 1 / PARAMS.scale ** 0.5
        self.cmd_max = jp.array([0.09 * v, 0.10 * v, 0.7 * w])       # inside the walking policy's envelope
        self.arm_lo = jp.array([r[0] for r in pio.ARM_RANGES])
        self.arm_hi = jp.array([r[1] for r in pio.ARM_RANGES])

    @property
    def observation_size(self):
        return 3 + 3 + 3 + 3 + 3 + 3 + 1

    @property
    def action_size(self):
        return 6

    @property
    def backend(self):
        return "mjx"

    def _rel(self, data, p):
        R = data.xmat[self.body_id]
        return (R.T @ (p - data.xpos[self.body_id]))[:2]

    def reset(self, rng):
        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
        r = jax.random.uniform(k1, minval=0.10, maxval=0.20)
        a = jax.random.uniform(k2, minval=-1.0, maxval=1.0)
        puck = jp.array([r * jp.cos(a), r * jp.sin(a), PUCK_H])
        g_ang = jax.random.uniform(k3, minval=-jp.pi, maxval=jp.pi)
        g_r = jax.random.uniform(k4, minval=0.10, maxval=0.22)
        goal = puck[:2] + g_r * jp.array([jp.cos(g_ang), jp.sin(g_ang)])
        qpos = self.init_qpos.at[self.puck_q:self.puck_q + 3].set(puck)
        data = mjx.forward(self.sys, mjx.make_data(self.sys).replace(qpos=qpos, ctrl=jp.array(pio.FULL_DEFAULT)))
        info = dict(rng=rng, step=jp.array(0), goal=goal, cmd=jp.zeros(3), arm=jp.array(pio.FULL_DEFAULT)[jp.array(_R_ARM)],
                    walk_last=jp.zeros(12), t_walk=jp.array(0.0), last_act=jp.zeros(6),
                    dist=jp.linalg.norm(puck[:2] - goal), dist0=jp.linalg.norm(puck[:2] - goal), puck0=puck[:2])
        metrics = {k: jp.array(0.0) for k in self.scales}
        metrics["success_rate"] = jp.array(0.0)
        return State(data, self._obs(data, info, jax.random.PRNGKey(0)), jp.array(0.0), jp.array(0.0), metrics, info)

    def _walk_obs(self, data, cmd, last, t):
        R = data.xmat[self.body_id]
        q = data.qpos[self.leg_qadr] + data.qpos[self.bl_q]
        qd = data.qvel[self.leg_vadr] + data.qvel[self.bl_v]
        return pio.observation(jp, data.qvel[3:6], R.T @ jp.array([0.0, 0.0, -1.0]), cmd, q, qd, last, t)

    def step(self, state, action):
        info = dict(state.info)
        if "steps" in info:          # Brax's auto-reset restores only the physics: a new episode -> fresh bookkeeping
            new = info["steps"] == 0
            info.update(step=jp.where(new, 0, info["step"]), last_act=jp.where(new, 0.0, info["last_act"]),
                        walk_last=jp.where(new, 0.0, info["walk_last"]), t_walk=jp.where(new, 0.0, info["t_walk"]),
                        dist=jp.where(new, info["dist0"], info["dist"]), cmd=jp.where(new, 0.0, info["cmd"]))
        rng, k_noise = jax.random.split(info["rng"])
        action = jp.clip(action, -1, 1)
        cmd = action[:3] * self.cmd_max
        arm = self.arm_lo + (self.arm_hi - self.arm_lo) * (action[3:] + 1) / 2

        def low(_, carry):
            d, last, t = carry
            a = _mlp(self.walk, self._walk_obs(d, cmd, last, t))
            ctrl = pio.action_to_targets(jp, a).at[jp.array(_R_ARM)].set(arm)
            d = d.replace(ctrl=ctrl)
            d = jax.lax.fori_loop(0, self.n_sub, lambda _, dd: mjx.step(self.sys, dd), d)
            return d, a, t + pio.CTRL_DT
        data, walk_last, t_walk = jax.lax.fori_loop(0, self.n_low, low, (state.pipeline_state, info["walk_last"],
                                                                           info["t_walk"]))
        R = data.xmat[self.body_id]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        puck = data.xpos[self.puck_body][:2]
        goal = info["goal"]
        dist = jp.linalg.norm(puck - goal)
        u = (goal - puck) / (dist + 1e-6)
        behind = puck - u * (PUCK_R + 0.012)
        claw = data.site_xpos[self.site][:2]
        r = dict(
            progress=(info["dist"] - dist),                           # metres closer this decision
            near=1.0 - jp.tanh(jp.linalg.norm(claw - behind) / 0.05),
            success=(dist < 0.025).astype(jp.float32),
            tilt=jp.sum(grav[:2] ** 2),
            action_rate=jp.sum((action - info["last_act"]) ** 2),
            lost=(jp.linalg.norm(puck - info["puck0"]) > 0.45).astype(jp.float32),
        )
        r = {k: jp.asarray(v, jp.float32) for k, v in r.items()}
        bad = ~(jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel)))   # a physics blow-up: end it
        fell = (grav[2] > -0.7) | (data.qpos[2] < self.fall_height) | bad
        r["termination"] = fell.astype(jp.float32)
        r = {k: jp.where(bad, 0.0, v) for k, v in r.items()}
        reward = jp.clip(sum(self.scales[k] * v for k, v in r.items()), -1.0, 10.0)
        reward = jp.where(bad, -1.0, reward)
        dist = jp.where(bad, info["dist0"], dist)
        walk_last = jp.where(jp.all(jp.isfinite(walk_last)), walk_last, 0.0)
        t_walk = jp.where(jp.isfinite(t_walk), t_walk, 0.0)
        info.update(rng=rng, step=info["step"] + 1, cmd=cmd, arm=arm, walk_last=walk_last, t_walk=t_walk,
                    last_act=action, dist=dist)
        metrics = dict(state.metrics)
        metrics.update(r)
        metrics["success_rate"] = (dist < 0.025).astype(jp.float32) / self.T * 20.0       # ~1 if done by the end
        done = fell.astype(jp.float32)
        obs = self._obs(data, info, k_noise)
        obs = jp.where(jp.isfinite(obs), obs, 0.0)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    def _obs(self, data, info, rng):
        R = data.xmat[self.body_id]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        puck = self._rel(data, data.xpos[self.puck_body]) * 5.0
        goal = self._rel(data, jp.array([info["goal"][0], info["goal"][1], 0.0])) * 5.0
        obs = jp.concatenate([data.qvel[3:6] * 0.25, grav, info["cmd"] / self.cmd_max, info["arm"],
                              jp.concatenate([puck, jp.zeros(1)]),
                              jp.concatenate([goal, jp.zeros(1)]), jp.array([info["step"] / self.T])])
        return obs + self.obs_noise * jax.random.normal(rng, obs.shape)

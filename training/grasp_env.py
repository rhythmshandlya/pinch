"""MJX environment: pick up a pen with the right claw (reinforcement learning).

Why RL: a scripted grasp failed in the twin. The open jaw leaves only +-1-2.5 mm to spare around a pen;
a round pen pinched above its middle wedges out of the jaws as they rise; walking alone can't place
the claw that precisely. Legged loco-manipulation work learns such contact-rich skills with RL and
domain randomisation, usually hierarchically - so here Pinch's approach (walking, scripted lining-up)
gets the claw roughly over the pen, and this policy does the last few centimetres: it moves the arm
(swing, lift, claw) and shifts/leans the body with the legs to get the pen between the jaws and lift it.

Reward follows the usual reach -> grasp -> lift recipe: a dense reach term; grasp and lift terms gated
on having reached; a sparse success bonus; penalties for shoving the pen away or tipping over. A
separate `success` metric (pen held up between both jaws) is logged, since reward alone can hide a
policy that never lifts anything.

Observation (crab/policy_io.grasp_observation): IMU, leg encoders, the arm's last commands, the pen's
position and direction relative to the open claw's gap (from the wide-angle camera on the robot), time.
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
PEN_R, PEN_HALF = 0.0045, 0.065

REWARD_SCALES = dict(reach=5.0, align=1.0, grasp=4.0, lift=10.0, success=20.0, shove=-4.0, tilt=-2.0,
                     action_rate=-0.01, termination=-10.0)


def load_grasp_model(sim_dt=0.004):
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for s in list(spec.sensors):
        spec.delete(s)
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for msh in list(spec.meshes):
        spec.delete(msh)
    pen = spec.worldbody.add_body(name="pen", pos=[0.3, 0, PEN_R])
    pen.add_freejoint(name="pen_free")
    pen.add_geom(name="pen_geom", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[PEN_R, PEN_HALF, 0],
                 euler=[0, np.pi / 2, 0], mass=0.009, friction=[0.6, 0.005, 0.0002])
    m = spec.compile()
    m.opt.timestep = sim_dt
    m.opt.iterations = 6
    m.opt.ls_iterations = 8
    floor = m.geom("floor").id
    feet = {m.geom(f"{lg}_foot").id for lg in LEG_IDS}
    fingers = {m.geom(f"{s}_pollex{lip}").id for s in ("R", "L") for lip in ("", "_lip")}
    for side in ("R", "L"):
        b = m.body(f"{side}_dactyl").id
        fingers |= {g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX}
    pen_g = m.geom("pen_geom").id
    for g in range(m.ngeom):
        if g == floor:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 1
        elif g in feet:
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 0
        elif g in fingers:                    # fingers touch the desk and the pen; TPU pads: grippy
            m.geom_contype[g], m.geom_conaffinity[g] = 3, 0
            m.geom_friction[g] = [1.2, 0.02, 0.001]
            m.geom_condim[g] = 6
        elif g == pen_g:
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 2
        else:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    return m, sorted(fingers)


class CrabGraspEnv(Env):
    def __init__(self, sim_dt=0.004, episode_seconds=4.0, obs_noise=0.01, reward_scales=None, pos_noise=0.008,
                 yaw_noise=0.25):
        self.mj_model, fingers = load_grasp_model(sim_dt)
        self.sys = mjx.put_model(self.mj_model)
        m = self.mj_model
        self.dt = pio.CTRL_DT
        self.n_sub = int(round(self.dt / sim_dt))
        self.T = int(episode_seconds / self.dt)
        self.obs_noise = obs_noise
        self.pos_noise, self.yaw_noise = pos_noise, yaw_noise
        self.scales = dict(REWARD_SCALES, **(reward_scales or {}))
        self.body_id = m.body("body").id
        self.pen_body = m.body("pen").id
        self.pen_geom = m.geom("pen_geom").id
        self.pen_q = m.jnt_qposadr[m.joint("pen_free").id]
        self.pen_v = m.jnt_dofadr[m.joint("pen_free").id]
        self.pollex = jp.array([m.geom("R_pollex").id, m.geom("R_pollex_lip").id])
        db = m.body("R_dactyl").id
        self.dactyl = jp.array([g for g in fingers if m.geom_bodyid[g] == db])
        self.site = m.site("R_grip_site").id
        self.pitch_q = m.jnt_qposadr[m.joint("R_shoulder_pitch").id]
        self.claw_q = m.jnt_qposadr[m.joint("R_claw").id]
        self.act = jp.array([m.actuator(j).id for j in JOINT_NAMES])
        self.leg_qadr = jp.array([m.jnt_qposadr[m.joint(n).id] for n in pio.LEG_JOINT_NAMES])
        self.leg_vadr = jp.array([m.jnt_dofadr[m.joint(n).id] for n in pio.LEG_JOINT_NAMES])
        self.bl_q = jp.array([m.jnt_qposadr[m.joint(f"{n}_backlash").id] for n in pio.LEG_JOINT_NAMES])
        self.bl_v = jp.array([m.jnt_dofadr[m.joint(f"{n}_backlash").id] for n in pio.LEG_JOINT_NAMES])
        self.nominal = jp.array(pio.GRASP_NOMINAL)
        q0 = np.array(m.key("stand").qpos)                    # includes the pen (parked out of reach)
        q0[self.pen_q:self.pen_q + 7] = [0.3, 0, PEN_R, 1, 0, 0, 0]
        self.init_qpos = jp.array(q0)
        self.fall_height = 0.3 * float(m.key("stand").qpos[2])

    @property
    def observation_size(self):
        return pio.GRASP_OBS_SIZE

    @property
    def action_size(self):
        return pio.GRASP_N_ACT

    @property
    def backend(self):
        return "mjx"

    def _gap(self, data):
        """Centre of the open claw's gap (4 mm back along the finger, 8 mm off the fixed finger) and the
        finger direction, in the world frame."""
        R = data.site_xmat[self.site]
        p = data.site_xpos[self.site] + R @ jp.array([-0.004, -0.008, 0.0])
        return p, R[:, 0]

    def reset(self, rng):
        rng, k1, k2, k3, k4 = jax.random.split(rng, 5)
        data = mjx.make_data(self.sys).replace(qpos=self.init_qpos)
        ready = pio.grasp_targets(jp, self.nominal, jp.zeros(pio.GRASP_N_ACT).at[13].set(0.3).at[14].set(1.0))
        data = data.replace(ctrl=ready)
        data = jax.lax.fori_loop(0, 120, lambda _, d: mjx.step(self.sys, d.replace(ctrl=ready)), data)  # settle 0.5 s
        # where the open claw's gap will be once the arm is lowered to the desk (the arm sweeps forward
        # and down as it lowers, so this is the spot the approach lines up with)
        low = data.qpos.at[self.pitch_q].set(pio.ARM_RANGES[1][0]).at[self.claw_q].set(pio.ARM_RANGES[2][1])
        gap, fdir = self._gap(mjx.kinematics(self.sys, data.replace(qpos=low)))
        # the pen lies roughly along the finger, roughly under the gap (what the approach achieves)
        yaw0 = jp.arctan2(fdir[1], fdir[0])
        yaw = yaw0 + jax.random.uniform(k1, minval=-self.yaw_noise, maxval=self.yaw_noise + 1e-6)
        ax = jp.array([jp.cos(yaw), jp.sin(yaw), 0.0])
        n = jp.array([-jp.sin(yaw0), jp.cos(yaw0), 0.0])
        off = jax.random.uniform(k2, (2,), minval=-self.pos_noise, maxval=self.pos_noise + 1e-6)
        grab = jp.array([gap[0], gap[1], PEN_R]) + off[0] * jp.array([jp.cos(yaw0), jp.sin(yaw0), 0.0]) + off[1] * n
        flip = jax.random.bernoulli(k3)                           # which end is near
        sgn = jp.where(flip, 1.0, -1.0)
        centre = grab + sgn * ax * (PEN_HALF - 0.018)
        quat = jp.array([jp.cos(yaw / 2), 0, 0, jp.sin(yaw / 2)])
        qpos = data.qpos.at[self.pen_q:self.pen_q + 3].set(centre).at[self.pen_q + 3:self.pen_q + 7].set(quat)
        qvel = data.qvel.at[self.pen_v:self.pen_v + 6].set(0.0)
        data = mjx.forward(self.sys, data.replace(qpos=qpos, qvel=qvel))
        info = dict(rng=rng, step=jp.array(0), last_act=jp.zeros(pio.GRASP_N_ACT), grab=grab, pen0=centre[:2],
                    held=jp.array(0.0))
        metrics = {k: jp.array(0.0) for k in self.scales}
        metrics["success_rate"] = jp.array(0.0)
        return State(data, self._obs(data, info, k4), jp.array(0.0), jp.array(0.0), metrics, info)

    def _touch(self, data, gs):
        """Does the pen touch any of the geoms gs (a jaw and its lip)?"""
        c = data._impl.contact
        other = jp.where(c.geom[:, 0] == self.pen_geom, c.geom[:, 1], jp.where(c.geom[:, 1] == self.pen_geom, c.geom[:, 0], -1))
        return jp.any(jp.isin(other, gs) & (c.dist < 1e-3))

    def step(self, state, action):
        info = dict(state.info)
        rng, k_noise = jax.random.split(info["rng"])
        action = jp.clip(action, -1, 1)
        ctrl = pio.grasp_targets(jp, self.nominal, action)
        data = state.pipeline_state.replace(ctrl=ctrl)
        data = jax.lax.fori_loop(0, self.n_sub, lambda _, d: mjx.step(self.sys, d), data)

        R = data.xmat[self.body_id]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        gap, fdir = self._gap(data)
        pen_c = data.xpos[self.pen_body]
        pen_ax = data.xmat[self.pen_body][:, 0]
        # the point on the pen's axis nearest the gap: where the jaws should close
        s = jp.clip(jp.dot(gap - pen_c, pen_ax), -PEN_HALF, PEN_HALF)
        nearest = pen_c + s * pen_ax
        dist = jp.linalg.norm(gap - nearest)
        reached = dist < 0.012
        both = self._touch(data, self.pollex) & self._touch(data, self.dactyl)
        lift_h = nearest[2] - PEN_R                                  # how high the gripped part of the pen is
        lifted = both & (lift_h > 0.012)
        held = jp.where(lifted, info["held"] + self.dt, 0.0).astype(jp.float32)
        shove = jp.maximum(jp.linalg.norm(pen_c[:2] - info["pen0"]) - 0.01, 0.0)
        r = dict(
            reach=1.0 - jp.tanh(dist / 0.02),
            align=jp.abs(jp.dot(pen_ax[:2], fdir[:2]) / (jp.linalg.norm(fdir[:2]) + 1e-6)) * reached,
            grasp=both * reached,
            lift=jp.clip(lift_h / 0.02, 0.0, 1.0) * both,
            success=(held > 0.4).astype(jp.float32),
            shove=shove * (~both) * 20.0,
            tilt=jp.sum(grav[:2] ** 2),
            action_rate=jp.sum((action - info["last_act"]) ** 2),
        )
        r = {k: jp.asarray(v, jp.float32) for k, v in r.items()}     # (some terms are booleans)
        fell = (grav[2] > -0.6) | (data.qpos[2] < self.fall_height)
        r["termination"] = fell.astype(jp.float32)
        reward = jp.clip(sum(self.scales[k] * v for k, v in r.items()) * self.dt, -1.0, 10.0)
        info.update(rng=rng, step=info["step"] + 1, last_act=action, held=held)
        metrics = dict(state.metrics)
        metrics.update(r)
        metrics["success_rate"] = (held > 0.4).astype(jp.float32) / self.T * 50     # ~1 if held for the last second
        obs = self._obs(data, info, k_noise)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=fell.astype(jp.float32),
                             metrics=metrics, info=info)

    def _obs(self, data, info, rng):
        R = data.xmat[self.body_id]
        gyro = data.qvel[3:6]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        q = data.qpos[self.leg_qadr] + data.qpos[self.bl_q]
        qd = data.qvel[self.leg_vadr] + data.qvel[self.bl_v]
        gap, fdir = self._gap(data)
        pen_c = data.xpos[self.pen_body]
        pen_ax = data.xmat[self.pen_body][:, 0]
        k1, k2 = jax.random.split(rng)
        # the camera's view of the pen, relative to the claw gap, in the body frame (a few mm of noise)
        rel = R.T @ (pen_c - gap) + 0.003 * jax.random.normal(k1, (3,))
        ax_b = R.T @ pen_ax
        ang = jp.arctan2(ax_b[1], ax_b[0]) + 0.05 * jax.random.normal(k2)
        obs = pio.grasp_observation(jp, gyro, grav, q, qd, info["last_act"], rel, ang, info["step"] * self.dt / 4.0)
        return obs + self.obs_noise * jax.random.normal(rng, obs.shape)

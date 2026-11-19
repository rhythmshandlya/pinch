"""MJX environment: pick up a small thing with the right claw (reinforcement learning) - any stick or block that
fits between the jaws (up to ~12 mm wide): pens, pencils and thin markers ("sticks"), erasers, sharpeners,
small boxes ("blocks"). Balls can't be held by flat jaws (two point contacts: they roll out - a scripted
grasp failed at every size), so Pinch pushes and bats those; the kind is kept so the model stays general.

The claw (decision 24) has jaw blades that hang below the palm, lie flat on the desk when the arm points
20 deg down, and are parallel when 9 mm apart - a scripted grasp with a pen placed between them lifts it
(13/13 held in the twin). What a script can't do is get the thing there and keep it there: walking alone
places the claw only to a centimetre, and things shift and roll as the jaws close. Legged
loco-manipulation work learns such contact-rich skills with RL and domain randomisation, usually
hierarchically - so Pinch's approach (walking, lining up) gets the claw roughly over the thing, and this
policy does the last few centimetres: it moves the arm (swing, lift, claw) and shifts/leans the body with
the legs to get the thing between the jaws and lift it.

One policy for all small things: every training environment gets its own sizes (grasp_randomize), each
episode one of three kinds, and the policy is told the kind and size - what the camera measures on the
robot. (Generalising over procedurally varied objects is how grasping policies are made to carry over
to things they never saw, e.g. UniDexGrasp, arXiv 2303.00938.)

Reward follows the usual reach -> grasp -> lift recipe: a dense reach term; grasp and lift terms gated
on having reached; a sparse success bonus. Shoving the thing more than 4 cm away (without holding it)
ends the episode - losing the rest of the episode's reward is the penalty, so it can't pile up and teach
the policy to stay away from the thing altogether (which happened with a per-step shove penalty).
A separate `success` metric (held up between both jaws) is logged, since reward alone can hide a policy
that never lifts anything.

Observation (crab/policy_io.grasp_observation): IMU, leg encoders, the last actions and the arm's
targets, the thing's grab point and direction relative to the open claw's gap, its kind and size, time.
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
P_LOW = -0.11          # arm pitch at which the jaws' lower edges reach the desk in the grasp crouch (measured)
GAP_Y = 0.006          # aim the grab point this far off the fixed jaw, into the open gap (toward the moving jaw)
# rigid things in hand: MuJoCo's default soft contacts let a servo squeeze a 9 g pen 3 mm deep
# rigid things in hand, as in the twin (sim/desk.py): 4 ms contacts on 2 ms physics. grasp-v3 trained at 4 ms with
# 10 ms contacts and fell apart in the twin (decision 47)
STIFF = dict(solref=(0.004, 1.0), solimp=(0.95, 0.99, 0.001, 0.5, 2.0))
KINDS = pio.GRASP_KINDS
KIND_P = (0.6, 0.4, 0.0)                       # how often each kind comes up (no balls: see above)
# default sizes (the pen, the eraser, a marble); each training environment draws its own (grasp_randomize)
DEFAULT = {"stick": (0.0045, 0.065, 0.0), "block": (0.010, 0.006, 0.0045), "ball": (0.006, 0.0, 0.0)}
PARK = [(2.0, 2.0), (2.0, 2.3), (2.0, 2.6)]    # where the two unused things wait, far away on the floor
ARM0 = (0.17, 0.08, 0.96)                      # the arm when the policy takes over: raised a little, claw open

REWARD_SCALES = dict(reach=5.0, align=1.0, grasp=4.0, lift=10.0, success=20.0, push=-2.0, tilt=-2.0,
                     action_rate=-0.01, termination=-10.0)


def load_grasp_model(sim_dt=0.002):
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for s in list(spec.sensors):
        spec.delete(s)
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for msh in list(spec.meshes):
        spec.delete(msh)
    G = mujoco.mjtGeom
    for i, kind in enumerate(KINDS):
        sz = DEFAULT[kind]
        b = spec.worldbody.add_body(name=f"obj_{kind}", pos=[PARK[i][0], PARK[i][1], 0.01])
        b.add_freejoint(name=f"obj_{kind}_free")
        kw = dict(name=f"obj_{kind}_geom", mass=0.008, friction=[0.6, 0.005, 0.0002])
        if kind == "stick":
            b.add_geom(type=G.mjGEOM_CAPSULE, size=[sz[0], sz[1], 0], euler=[0, np.pi / 2, 0], **kw)
        elif kind == "block":
            b.add_geom(type=G.mjGEOM_BOX, size=list(sz), **kw)
        else:
            b.add_geom(type=G.mjGEOM_SPHERE, size=[sz[0], 0, 0], **kw)
    m = spec.compile()
    m.opt.timestep = sim_dt
    m.opt.iterations = 6
    m.opt.ls_iterations = 8
    floor = m.geom("floor").id
    feet = {m.geom(f"{lg}_foot").id for lg in LEG_IDS}
    right = {m.geom(n).id for n in ("R_pollex", "R_pollex_blade")}
    right |= {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("R_dactyl").id and m.geom_type[g] == G.mjGEOM_BOX}
    left = {m.geom(n).id for n in ("L_pollex", "L_pollex_blade")}
    left |= {g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("L_dactyl").id and m.geom_type[g] == G.mjGEOM_BOX}
    objs = {m.geom(f"obj_{k}_geom").id for k in KINDS}
    for g in range(m.ngeom):
        if g == floor:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 1
        elif g in feet or g in left:                  # feet and the left claw: only the desk
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 0
        elif g in right:                              # the right claw: the desk and the thing; TPU pads: grippy
            m.geom_contype[g], m.geom_conaffinity[g] = 3, 0
            m.geom_friction[g] = [1.2, 0.02, 0.001]
            m.geom_condim[g] = 6
            m.geom_solref[g], m.geom_solimp[g] = STIFF["solref"], STIFF["solimp"]
        elif g in objs:                               # things: the desk and the right claw, not each other
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 2
            m.geom_solref[g], m.geom_solimp[g] = STIFF["solref"], STIFF["solimp"]
        else:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    return m, sorted(right)


def _object_inertia(kind, size, mass):
    """Principal inertia of a thing (its geom frame), for grasp_randomize."""
    if kind == "stick":                               # capsule along its geom z: ~ a rod plus two half balls
        r, h = size[0], size[1]
        axial = 0.5 * mass * r ** 2
        perp = mass * ((2 * h) ** 2 / 12 + r ** 2 / 4)
        return jp.stack([perp, perp, axial])
    if kind == "block":
        a, b, c = size[0], size[1], size[2]
        return mass / 3 * jp.stack([b ** 2 + c ** 2, a ** 2 + c ** 2, a ** 2 + b ** 2])
    return jp.full(3, 0.4 * mass * size[0] ** 2)


def grasp_randomize(sys: mjx.Model, rng: jax.Array):
    """Per-environment variations: the things' sizes and masses (one policy for all small things), friction
    (scaled, so the grippy pads stay grippier than plastic), masses, servo strength and stiffness, a tilted
    desk. rng has shape (num_envs, 2); returns a batched model and its vmap in_axes."""
    m, _ = load_grasp_model()
    geoms = [m.geom(f"obj_{k}_geom").id for k in KINDS]
    bodies = [m.body(f"obj_{k}").id for k in KINDS]

    @jax.vmap
    def rand(key):
        k = jax.random.split(key, 16)
        friction = sys.geom_friction.at[:, 0].multiply(jax.random.uniform(k[0], minval=0.75, maxval=1.15))
        body_mass = sys.body_mass * jax.random.uniform(k[1], sys.body_mass.shape, minval=0.85, maxval=1.15)
        kp_scale = jax.random.uniform(k[2], (sys.nu,), minval=0.75, maxval=1.25)
        gain = sys.actuator_gainprm.at[:, 0].multiply(kp_scale)
        bias = sys.actuator_biasprm.at[:, 1].multiply(kp_scale)
        frange = sys.actuator_forcerange * jax.random.uniform(k[3], (sys.nu,), minval=0.7, maxval=1.15)[:, None]
        tilt = jax.random.uniform(k[4], minval=0.0, maxval=jp.radians(3.0))
        az = jax.random.uniform(k[5], minval=0.0, maxval=2 * jp.pi)
        gravity = 9.81 * jp.array([jp.sin(tilt) * jp.cos(az), jp.sin(tilt) * jp.sin(az), -jp.cos(tilt)])

        def u(i, lo, hi):
            return jax.random.uniform(k[6 + i], minval=lo, maxval=hi)

        # the things the claw can hold (a scripted grasp lifted these sizes, not 13 mm+):
        # a stick 6-11 mm thick and 5-15 cm long; a block 12-24 x 7-12 x 6-11 mm; a ball 9-14 mm (unused)
        stick = jp.stack([u(0, 0.003, 0.0055), u(1, 0.025, 0.075), 0.0])
        bx = u(2, 0.006, 0.012)
        block = jp.stack([bx, jp.minimum(u(3, 0.0035, 0.006), bx), u(4, 0.003, 0.0055)])
        ball = jp.stack([u(5, 0.0045, 0.007), 0.0, 0.0])
        masses = jp.stack([u(6, 0.003, 0.012), u(7, 0.003, 0.010), u(8, 0.002, 0.008)])
        size, rbound, inertia = sys.geom_size, sys.geom_rbound, sys.body_inertia
        for i, (kind, s) in enumerate(zip(KINDS, (stick, block, ball))):
            size = size.at[geoms[i]].set(s)
            rbound = rbound.at[geoms[i]].set(s[0] + s[1] if kind == "stick" else jp.linalg.norm(s))
            body_mass = body_mass.at[bodies[i]].set(masses[i])
            inertia = inertia.at[bodies[i]].set(_object_inertia(kind, s, masses[i]))
        return friction, body_mass, gain, bias, frange, gravity, size, rbound, inertia

    friction, body_mass, gain, bias, frange, gravity, size, rbound, inertia = rand(rng)
    fields = dict(geom_friction=friction, body_mass=body_mass, actuator_gainprm=gain, actuator_biasprm=bias,
                  actuator_forcerange=frange, geom_size=size, geom_rbound=rbound, body_inertia=inertia)
    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace({**{k: 0 for k in fields}, "opt.gravity": 0})
    return sys.tree_replace({**fields, "opt.gravity": gravity}), in_axes


class CrabGraspEnv(Env):
    def __init__(self, sim_dt=0.002, episode_seconds=4.0, obs_noise=0.01, reward_scales=None, pos_noise=0.008,
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
        self.obj_body = jp.array([m.body(f"obj_{k}").id for k in KINDS])
        self.obj_geom = jp.array([m.geom(f"obj_{k}_geom").id for k in KINDS])
        self.obj_q = jp.array([m.jnt_qposadr[m.joint(f"obj_{k}_free").id] for k in KINDS])
        self.obj_v = jp.array([m.jnt_dofadr[m.joint(f"obj_{k}_free").id] for k in KINDS])
        self.pollex = jp.array([m.geom("R_pollex").id, m.geom("R_pollex_blade").id])
        db = m.body("R_dactyl").id
        self.dactyl = jp.array([g for g in fingers if m.geom_bodyid[g] == db])
        self.site = m.site("R_grip_site").id
        self.pitch_q = m.jnt_qposadr[m.joint("R_shoulder_pitch").id]
        self.claw_q = m.jnt_qposadr[m.joint("R_claw").id]
        self.leg_qadr = jp.array([m.jnt_qposadr[m.joint(n).id] for n in pio.LEG_JOINT_NAMES])
        self.leg_vadr = jp.array([m.jnt_dofadr[m.joint(n).id] for n in pio.LEG_JOINT_NAMES])
        self.bl_q = jp.array([m.jnt_qposadr[m.joint(f"{n}_backlash").id] for n in pio.LEG_JOINT_NAMES])
        self.bl_v = jp.array([m.jnt_dofadr[m.joint(f"{n}_backlash").id] for n in pio.LEG_JOINT_NAMES])
        self.nominal = jp.array(pio.GRASP_NOMINAL)
        q0 = np.array(m.key("stand").qpos)
        for i, k in enumerate(KINDS):                         # all three things parked out of reach
            a = int(m.jnt_qposadr[m.joint(f"obj_{k}_free").id])
            q0[a:a + 7] = [PARK[i][0], PARK[i][1], 0.01, 1, 0, 0, 0]
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
        """Where a thing's grab point should be (mid-jaw, GAP_Y into the open gap, at about a pen's axis height
        when the jaws touch the desk) and the jaws' direction, in the world frame."""
        R = data.site_xmat[self.site]
        p = data.site_xpos[self.site] + R @ jp.array([0.0, -GAP_Y, 0.0])
        return p, R[:, 0]

    def _shape(self, kind):
        """(half-extents (long, wide, high), resting centre height, half-length of the axis it may be grabbed
        along) of the thing of this kind, from the (per-environment) model."""
        s = self.sys.geom_size[self.obj_geom[kind]]
        ext = jp.stack([jp.stack([s[0] + s[1], s[0], s[0]]), s, jp.stack([s[0], s[0], s[0]])])[kind]
        rest = jp.stack([s[0], s[2], s[0]])[kind]
        half = jp.stack([jp.maximum(s[1] - 0.010, 0.0), jp.maximum(s[0] - 0.004, 0.0), 0.0])[kind]
        return ext, rest, half

    def _grab_point(self, data, kind, gap):
        c = data.xpos[self.obj_body[kind]]
        ax = data.xmat[self.obj_body[kind]][:, 0]
        _, _, half = self._shape(kind)
        s = jp.clip(jp.dot(gap - c, ax), -half, half)
        return c + s * ax, c, ax

    def reset(self, rng):
        rng, k1, k2, k3, k4, k5 = jax.random.split(rng, 6)
        kind = jax.random.choice(k5, 3, p=jp.array(KIND_P))
        data = mjx.make_data(self.sys).replace(qpos=self.init_qpos)
        ready = pio.grasp_targets(jp, self.nominal, jp.zeros(pio.GRASP_N_ACT).at[13].set(0.3).at[14].set(1.0))
        data = data.replace(ctrl=ready)
        n_settle = int(round(0.5 / float(self.mj_model.opt.timestep)))           # settle 0.5 s (in time, not steps)
        data = jax.lax.fori_loop(0, n_settle, lambda _, d: mjx.step(self.sys, d.replace(ctrl=ready)), data)
        # where the open claw's gap will be once the arm is lowered to the desk (the arm sweeps forward
        # and down as it lowers, so this is the spot the approach lines up with)
        low = data.qpos.at[self.pitch_q].set(P_LOW).at[self.claw_q].set(pio.ARM_RANGES[2][1])
        gap, fdir = self._gap(mjx.kinematics(self.sys, data.replace(qpos=low)))
        # the thing lies roughly along the jaws, roughly under the gap (what the approach achieves)
        yaw0 = jp.arctan2(fdir[1], fdir[0])
        yaw = yaw0 + jax.random.uniform(k1, minval=-self.yaw_noise, maxval=self.yaw_noise + 1e-6)
        ax = jp.array([jp.cos(yaw), jp.sin(yaw), 0.0])
        n = jp.array([-jp.sin(yaw0), jp.cos(yaw0), 0.0])
        off = jax.random.uniform(k2, (2,), minval=-self.pos_noise, maxval=self.pos_noise + 1e-6)
        _, rest, half = self._shape(kind)
        grab = jp.array([gap[0], gap[1], rest]) + off[0] * jp.array([jp.cos(yaw0), jp.sin(yaw0), 0.0]) + off[1] * n
        along = jax.random.uniform(k3, minval=-half, maxval=half + 1e-6)   # its middle, or near either end
        centre = grab + along * ax
        quat = jp.array([jp.cos(yaw / 2), 0, 0, jp.sin(yaw / 2)])
        qpos = jax.lax.dynamic_update_slice(data.qpos, jp.concatenate([centre, quat]), (self.obj_q[kind],))
        qvel = jax.lax.dynamic_update_slice(data.qvel, jp.zeros(6), (self.obj_v[kind],))
        data = mjx.forward(self.sys, data.replace(qpos=qpos, qvel=qvel))
        info = dict(rng=rng, step=jp.array(0), last_act=jp.zeros(pio.GRASP_N_ACT), arm=jp.array(ARM0), kind=kind,
                    obj0=centre[:2], held=jp.array(0.0))
        metrics = {k: jp.array(0.0) for k in self.scales}
        metrics["success_rate"] = jp.array(0.0)
        return State(data, self._obs(data, info, k4), jp.array(0.0), jp.array(0.0), metrics, info)

    def _touch(self, data, gs, geom):
        """Does the thing's geom touch any of the geoms gs (one jaw's parts)?"""
        c = data._impl.contact
        other = jp.where(c.geom[:, 0] == geom, c.geom[:, 1], jp.where(c.geom[:, 1] == geom, c.geom[:, 0], -1))
        return jp.any(jp.isin(other, gs) & (c.dist < 1e-3))

    def step(self, state, action):
        info = dict(state.info)
        if "steps" in info:          # Brax's auto-reset restores only the physics: a new episode -> fresh bookkeeping
            new = info["steps"] == 0
            info.update(step=jp.where(new, 0, info["step"]), last_act=jp.where(new, 0.0, info["last_act"]),
                        held=jp.where(new, 0.0, info["held"]), arm=jp.where(new, jp.array(ARM0), info["arm"]))
        rng, k_noise = jax.random.split(info["rng"])
        action = jp.clip(action, -1, 1)
        ctrl = pio.grasp_targets(jp, self.nominal, action, arm_prev=info["arm"])
        arm = ctrl[jp.array(pio.GRASP_ARM_IDX)]
        data = state.pipeline_state.replace(ctrl=ctrl)
        data = jax.lax.fori_loop(0, self.n_sub, lambda _, d: mjx.step(self.sys, d), data)

        kind = info["kind"]
        geom = self.obj_geom[kind]
        R = data.xmat[self.body_id]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        gap, fdir = self._gap(data)
        grab, centre, ax = self._grab_point(data, kind, gap)
        _, rest, _ = self._shape(kind)
        dist = jp.linalg.norm(gap - grab)
        reached = dist < 0.012
        both = self._touch(data, self.pollex, geom) & self._touch(data, self.dactyl, geom)
        lift_h = grab[2] - rest                                     # how high the gripped part of the thing is
        lifted = both & (lift_h > 0.012)
        held = jp.where(lifted, info["held"] + self.dt, 0.0).astype(jp.float32)
        v_xy = jax.lax.dynamic_slice(data.qvel, (self.obj_v[kind],), (2,))
        pushed = (jp.linalg.norm(centre[:2] - info["obj0"]) > 0.04) & ~lifted
        r = dict(
            reach=1.0 - jp.tanh(dist / 0.02),
            align=jp.where(kind == 2, 1.0,
                           jp.abs(jp.dot(ax[:2], fdir[:2]) / (jp.linalg.norm(fdir[:2]) + 1e-6))) * reached,
            grasp=both * reached,
            lift=jp.clip(lift_h / 0.02, 0.0, 1.0) * both,
            success=(held > 0.4).astype(jp.float32),
            push=jp.minimum(jp.linalg.norm(v_xy), 0.5) * (~both),
            tilt=jp.sum(grav[:2] ** 2),
            action_rate=jp.sum((action - info["last_act"]) ** 2),
        )
        r = {k: jp.asarray(v, jp.float32) for k, v in r.items()}     # (some terms are booleans)
        bad = ~(jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel)))   # a physics blow-up: end it
        fell = (grav[2] > -0.6) | (data.qpos[2] < self.fall_height)
        done = fell | pushed | bad
        r["termination"] = (fell | bad).astype(jp.float32)
        r = {k: jp.where(bad, 0.0, v) for k, v in r.items()}
        reward = jp.clip(sum(self.scales[k] * v for k, v in r.items()) * self.dt, -1.0, 10.0)
        reward = jp.where(bad, -0.2, reward)
        held = jp.where(bad, 0.0, held)
        info.update(rng=rng, step=info["step"] + 1, last_act=action, held=held, arm=jp.where(bad, info["arm"], arm))
        metrics = dict(state.metrics)
        metrics.update(r)
        metrics["success_rate"] = (held > 0.4).astype(jp.float32) / self.T * 50     # ~1 if held for the last second
        obs = self._obs(data, info, k_noise)
        obs = jp.where(jp.isfinite(obs), obs, 0.0)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done.astype(jp.float32),
                             metrics=metrics, info=info)

    def _obs(self, data, info, rng):
        R = data.xmat[self.body_id]
        gyro = data.qvel[3:6]
        grav = R.T @ jp.array([0.0, 0.0, -1.0])
        q = data.qpos[self.leg_qadr] + data.qpos[self.bl_q]
        qd = data.qvel[self.leg_vadr] + data.qvel[self.bl_v]
        kind = info["kind"]
        gap, _ = self._gap(data)
        grab, _, ax = self._grab_point(data, kind, gap)
        ext, _, _ = self._shape(kind)
        k1, k2, k3 = jax.random.split(rng, 3)
        # the camera's view of the thing, relative to the claw gap, in the body frame (a few mm of noise)
        rel = R.T @ (grab - gap) + 0.003 * jax.random.normal(k1, (3,))
        ax_b = R.T @ ax
        ang = jp.arctan2(ax_b[1], ax_b[0]) + 0.05 * jax.random.normal(k2)
        size = ext * (1.0 + 0.1 * jax.random.normal(k3, (3,)))         # sizes from the camera: +-10 %
        obs = pio.grasp_observation(jp, gyro, grav, q, qd, info["last_act"], info["arm"], rel, ang, kind, size,
                                    info["step"] * self.dt / 4.0)
        return obs + self.obs_noise * jax.random.normal(rng, obs.shape)

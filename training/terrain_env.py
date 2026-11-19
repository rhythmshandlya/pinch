"""MJX environment: walking over hurdles, ramps and steps (decision 58).

The residual gait env (training/gait_env.py: scripted trot + learned corrections) on rough ground. Every
training robot gets its own course in a ring around where it starts (13-20 cm out, so every walking
direction meets something):
  * two hurdles (a cable, a pencil, a ruler's edge): bars 3-10 mm high, 12 mm deep, 30 cm long
  * a ramp tilted 0-14 deg (a book propped up, a laptop stand) with a plateau at its top (up to 24 mm)
The obstacles' heights vary by how deep their boxes sit in the floor, so only positions and orientations are
randomised per robot (MJX-safe). Feet and shins collide with them (a shin catching a bar is the real failure).

Because the legs must leave the flat-ground reference to clear things, following the reference counts less,
the corrections may be larger (0.25 rad, stored with the exported policy as action_scale), and the body
height / attitude terms are softer (on a ramp the body tilts with the ground). Walking at the commanded
speed still counts most.

terrain_layout(xp, u) is shared by training (jax) and the bench (sim/robustness.py --world terrain, numpy).
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.crab_env import LEG_IDS, MODEL, domain_randomize  # noqa: E402
from training.gait_env import REWARD_SCALES as _GAIT_SCALES, CrabGaitEnv  # noqa: E402

OBSTACLES = [  # name, box half sizes (x radial, y tangent, z)
    ("hurdle_a", (0.006, 0.15, 0.006)),
    ("hurdle_b", (0.006, 0.15, 0.006)),
    ("ramp", (0.05, 0.12, 0.003)),
    ("ramp_top", (0.06, 0.12, 0.015)),
]
N_U = 10                     # uniforms per layout
ACTION_SCALE = 0.25
REWARD_SCALES = dict(_GAIT_SCALES, track_legs=0.4, track_attitude=0.8, track_height=0.2, residual=-0.05)


def _quat_yaw_pitch(xp, yaw, pitch):
    """q = yaw about z, then pitch about the (new) y axis; (w, x, y, z)."""
    cy, sy, cp, sp = xp.cos(yaw / 2), xp.sin(yaw / 2), xp.cos(pitch / 2), xp.sin(pitch / 2)
    return xp.stack([cy * cp, -sy * sp, cy * sp, sy * cp])


def terrain_layout(xp, u):
    """u: N_U uniforms in [0, 1). Returns pos (4, 3) and quat (4, 4) for OBSTACLES, around the origin."""
    two_pi = 2 * xp.pi
    a1 = u[0] * two_pi
    r1 = 0.13 + 0.06 * u[1]
    h1 = 0.003 + 0.007 * u[2]
    a2 = a1 + xp.pi + (u[3] - 0.5) * 1.2
    r2 = 0.13 + 0.06 * u[4]
    h2 = 0.003 + 0.007 * u[5]
    a3 = a1 + xp.pi / 2 + (u[6] - 0.5) * 1.0
    r3 = 0.12 + 0.04 * u[7]
    th = xp.radians(14.0) * u[8]
    hx, hz = OBSTACLES[2][1][0], OBSTACLES[2][1][2]
    top_h = 2 * hx * xp.sin(th)                                    # the ramp's high end
    ramp_c = r3 + hx * xp.cos(th)
    top_c = r3 + 2 * hx * xp.cos(th) + OBSTACLES[3][1][0]
    pos = xp.stack([
        xp.stack([r1 * xp.cos(a1), r1 * xp.sin(a1), h1 - 0.006]),
        xp.stack([r2 * xp.cos(a2), r2 * xp.sin(a2), h2 - 0.006]),
        xp.stack([ramp_c * xp.cos(a3), ramp_c * xp.sin(a3), hx * xp.sin(th) - hz * xp.cos(th)]),
        xp.stack([top_c * xp.cos(a3), top_c * xp.sin(a3), top_h - OBSTACLES[3][1][2]]),
    ])
    zero = a1 * 0.0
    quat = xp.stack([_quat_yaw_pitch(xp, a1, zero), _quat_yaw_pitch(xp, a2, zero),
                     _quat_yaw_pitch(xp, a3, -th), _quat_yaw_pitch(xp, a3, zero)])
    return pos, quat


def add_obstacles(spec, pos=None, quat=None):
    """Add the obstacle boxes to a MjSpec's world body (at the given layout, or parked out of the way)."""
    for i, (name, size) in enumerate(OBSTACLES):
        p = [5.0 + i, 5.0, -0.5] if pos is None else [float(v) for v in pos[i]]
        q = [1, 0, 0, 0] if quat is None else [float(v) for v in quat[i]]
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=p, quat=q,
                                rgba=[0.55, 0.45, 0.35, 1], friction=[0.8, 0.02, 0.001])


def load_terrain_model(sim_dt: float) -> mujoco.MjModel:
    """The training model (no meshes or sensors; feet touch the floor) plus the obstacles, which the feet and
    shins collide with."""
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for s in list(spec.sensors):
        spec.delete(s)
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(g)
    for msh in list(spec.meshes):
        spec.delete(msh)
    add_obstacles(spec)
    m = spec.compile()
    m.opt.timestep = sim_dt
    m.opt.iterations = 4
    m.opt.ls_iterations = 8
    feet = {m.geom(f"{lg}_foot").id for lg in LEG_IDS}
    shins = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.endswith("_tibia_link")
             and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE}
    world = {m.geom("floor").id} | {m.geom(n).id for n, _ in OBSTACLES}
    for g in range(m.ngeom):
        if g in world:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 1
        elif g in feet or g in shins:
            m.geom_contype[g], m.geom_conaffinity[g] = 1, 0
        else:
            m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
    return m


_OBST_IDS = None


def terrain_randomize(sys: mjx.Model, rng: jax.Array):
    """The usual physics randomisation, plus each robot's own obstacle course."""
    sys_b, in_axes = domain_randomize(sys, rng)
    ids = jp.array(_OBST_IDS)

    @jax.vmap
    def course(key):
        u = jax.random.uniform(jax.random.fold_in(key, 7), (N_U,))
        pos, quat = terrain_layout(jp, u)
        return sys.geom_pos.at[ids].set(pos), sys.geom_quat.at[ids].set(quat)

    gpos, gquat = course(rng)
    in_axes = in_axes.tree_replace({"geom_pos": 0, "geom_quat": 0})
    return sys_b.tree_replace({"geom_pos": gpos, "geom_quat": gquat}), in_axes


class CrabTerrainEnv(CrabGaitEnv):
    def __init__(self, **kw):
        kw.setdefault("reward_scales", {})
        kw["reward_scales"] = dict(REWARD_SCALES, **kw["reward_scales"])
        super().__init__(**kw)
        self.action_scale = ACTION_SCALE

    def _load_model(self, sim_dt):
        global _OBST_IDS
        m = load_terrain_model(sim_dt)
        _OBST_IDS = [m.geom(n).id for n, _ in OBSTACLES]
        return m

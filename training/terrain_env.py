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

from crab.terrain import N_U, OBSTACLES, add_obstacles, terrain_layout  # noqa: E402
from training.crab_env import LEG_IDS, MODEL, domain_randomize  # noqa: E402
from training.gait_env import REWARD_SCALES as _GAIT_SCALES, CrabGaitEnv  # noqa: E402

ACTION_SCALE = 0.25
REWARD_SCALES = dict(_GAIT_SCALES, track_legs=0.4, track_attitude=0.8, track_height=0.2, residual=-0.05)


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

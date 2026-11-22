"""The robot model as every MJX training env sees it, and the physics randomisation they share.

load_training_model() adapts the CPU twin's model (sim/models/crab.xml, or crab_v2.xml with CRAB_DESIGN=v2)
at load time: sensors MJX does not need and the CAD display meshes are removed, only the feet collide (envs that need more turn it on), and
the physics step is the caller's. domain_randomize() draws a different robot per env: friction, masses,
servo stiffness and strength, damping, armature and a tilted desk.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.params import LEG_IDS, PARAMS  # noqa: E402

MODEL = ROOT / "sim" / "models" / PARAMS.model_file     # crab.xml; CRAB_DESIGN=v2: crab_v2.xml


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
        k = jax.random.split(key, 10)
        friction = sys.geom_friction.at[:, 0].set(jax.random.uniform(k[0], minval=0.3, maxval=1.25))   # v3: 0.3
        body_mass = sys.body_mass * jax.random.uniform(k[1], sys.body_mass.shape, minval=0.85, maxval=1.15)
        # servo stiffness: Open Duck runs ~0.74x of the BAM-derived kp, so cover that and a bit above
        kp_scale = jax.random.uniform(k[2], (sys.nu,), minval=0.75, maxval=1.25)   # unknown firmware P gains
        gain = sys.actuator_gainprm.at[:, 0].multiply(kp_scale)
        bias = sys.actuator_biasprm.at[:, 1].multiply(kp_scale)
        # servo to servo and maker to maker (owner, 2026-11-22: Feetech STS3215 vs the Waveshare ST3215 rebrand): a
        # zero offset per servo (BAM's fits of one STS3215 put it at 0.05-0.064 rad; horn spline and calibration add
        # more) as a constant term in the position servo (kp * offset shifts its target), and the gearbox's dry
        # friction (BAM friction_base 0.038-0.056 N*m across its six fits; wider for an unmeasured rebrand)
        offset = jax.random.uniform(k[8], (sys.nu,), minval=-0.06, maxval=0.06)
        bias = bias.at[:, 0].set(gain[:, 0] * offset)
        # Motor strength: the micro servos have no published lab measurements, so their
        # datasheet models are randomised widely (-25% .. +15%); force limit and back-EMF
        # damping move together (a weaker motor is also slower).
        s = jax.random.uniform(k[3], (sys.nu,), minval=0.65, maxval=1.15)                          # v3: 0.65
        frange = sys.actuator_forcerange * s[:, None]
        damp_scale = s
        noise = jax.random.uniform(k[5], (sys.nu,), minval=0.85, maxval=1.15)
        dof_scale = jp.ones(sys.dof_damping.shape).at[jp.asarray(_ACT_DOFS)].set(
            (damp_scale * noise)[jp.asarray(_ACT_IDS)])
        damping = sys.dof_damping * dof_scale
        armature = sys.dof_armature * jax.random.uniform(k[4], sys.dof_armature.shape, minval=0.8, maxval=1.2)
        fl_scale = jp.ones(sys.dof_frictionloss.shape).at[jp.asarray(_ACT_DOFS)].set(
            jax.random.uniform(k[9], (len(_ACT_DOFS),), minval=0.6, maxval=1.6))
        frictionloss = sys.dof_frictionloss * fl_scale
        # v3: the desk is never perfectly level - tilt gravity up to 5 deg in a random direction
        tilt = jax.random.uniform(k[6], minval=0.0, maxval=jp.radians(5.0))
        az = jax.random.uniform(k[7], minval=0.0, maxval=2 * jp.pi)
        gravity = 9.81 * jp.array([jp.sin(tilt) * jp.cos(az), jp.sin(tilt) * jp.sin(az), -jp.cos(tilt)])
        return friction, body_mass, gain, bias, frange, armature, damping, gravity, frictionloss

    friction, body_mass, gain, bias, frange, armature, damping, gravity, frictionloss = rand(rng)
    fields = dict(geom_friction=friction, body_mass=body_mass, actuator_gainprm=gain,
                  actuator_biasprm=bias, actuator_forcerange=frange,
                  dof_armature=armature, dof_damping=damping, dof_frictionloss=frictionloss)
    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace({**{k: 0 for k in fields}, "opt.gravity": 0})
    return sys.tree_replace({**fields, "opt.gravity": gravity}), in_axes

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


RANDOMIZE_FRICTIONLOSS = False   # the servo's dry friction spread is covered by the damping and strength ranges meanwhile
COLLIDERS = set()        # names of the last loaded model's own contact shapes (its non-visual geoms), for righting_env


def _boxes_to_capsules(spec):
    """The CAD's contact boxes on the legs become capsules along their longest side (radius: the mean of the other
    two half sizes): a box on the floor makes up to 4 contacts and 16 friction rows, a capsule 2; with five boxes a leg
    the solver's matrices grew 4x and msi-1 crawled (2048 x 690 x 42)."""
    import numpy as np
    for g in spec.geoms:
        if g.type != mujoco.mjtGeom.mjGEOM_BOX or not g.name:
            continue
        if not any(k in g.name for k in ("_coxa_link_", "_femur_link_", "_tibia_link_")):
            continue
        h = np.array(g.size[:3], float)
        ax = int(np.argmax(h))
        r = float(np.mean(np.delete(h, ax)))
        half = max(float(h[ax]) - r, 1e-4)
        rot = {0: [0.7071068, 0.0, 0.7071068, 0.0], 1: [0.7071068, -0.7071068, 0.0, 0.0], 2: [1.0, 0.0, 0.0, 0.0]}[ax]
        q0 = np.array(g.quat, float)
        w1, x1, y1, z1 = q0
        w2, x2, y2, z2 = rot
        g.quat = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        g.size = [r, half, 0.0]


def load_training_model(sim_dt: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for s in list(spec.sensors):                 # touch sensors etc. not needed
        spec.delete(s)
    shell_mesh = False
    for g in list(spec.geoms):                   # CAD visual meshes: looks only, slow for MJX
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
            shell_mesh |= g.name == "shell"
            spec.delete(g)
    for msh in list(spec.meshes):
        spec.delete(msh)
    if shell_mesh:                               # design v2: the pill shell's contact mesh becomes an ellipsoid of its
        from crab.electronics import Layout  # size (MJX collides it with the floor natively; no mass: the body's
        ch = Layout().chassis                    # inertial carries it)
        spec.body("body").add_geom(name="shell_proxy", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                                   size=[v / 1000.0 for v in ch["shell_half_mm"]],
                                   pos=[v / 1000.0 for v in ch["shell_center_mm"]], mass=0.0, contype=1, conaffinity=0)
    _boxes_to_capsules(spec)
    COLLIDERS.clear()
    COLLIDERS.update(g.name for g in spec.geoms if g.name and (g.contype or g.conaffinity)
                     and not any(k in g.name for k in ("_pad", "_blade", "_drop")))
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


# Command latency. Tan et al. (RSS 2018, arXiv:1804.10332) measured 15-19 ms on Minitaur and found policies trained
# without it failed on the robot; ours is the USB adapter, the 1 Mbps bus (a full read cycle 5.4 ms, crab/wiring.py)
# and the servo's own 5 ms command delay. Each control step's new targets reach the servos a random 0..LAG_MAX_S
# late, the previous targets held until then (the mjlab env randomises the same, training/mjlab_pinch/env_cfg.py).
LAG_MAX_S = 0.02


def lagged_physics(sys: mjx.Model, data: mjx.Data, ctrl, n_phys: int, key, ctrl_dt: float = 0.02):
    """n_phys physics steps of one control period (ctrl_dt): the old data.ctrl for the first k steps, then ctrl.
    (The step count comes from plain numbers: under the domain-randomisation vmap sys's fields are traced.)"""
    k = jax.random.randint(key, (), 0, int(round(LAG_MAX_S * n_phys / ctrl_dt)) + 1)
    prev = data.ctrl
    data = jax.lax.fori_loop(0, n_phys, lambda i, d: mjx.step(sys, d.replace(ctrl=jp.where(i < k, prev, ctrl))), data)
    return data.replace(ctrl=ctrl)        # the next period's "previous" targets are these, even if they never landed


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
                  dof_armature=armature, dof_damping=damping)
    if RANDOMIZE_FRICTIONLOSS:                    # off: a per-robot frictionloss made msi-1's first evaluation crawl
        fields["dof_frictionloss"] = frictionloss
    in_axes = jax.tree_util.tree_map(lambda _: None, sys)
    in_axes = in_axes.tree_replace({**{k: 0 for k in fields}, "opt.gravity": 0})
    return sys.tree_replace({**fields, "opt.gravity": gravity}), in_axes

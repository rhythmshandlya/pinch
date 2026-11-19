"""MJX environment: walking as small learned corrections on top of the scripted trot (decision 47).

Learned-from-scratch walkers (v1-v4) tracked commanded speed worse than the scripted gait in the full
twin (decisions 27, 41). The emote policy, which is residual on scripted keyframes, passed the robustness
bench - so walking gets the same structure: the scripted trot (crab/gait.py, 63-69 % tracking, never
falls) is the reference, and the policy adds corrections (+-0.15 rad) to its leg targets, rewarded for
following the gait, keeping the body level and - new - walking at the *commanded* speed, so it can do
better than the open-loop gait under shoves, slippery spots and weak motors. It can't do much worse:
with zero output it is the scripted gait.

The references are recorded on the CPU twin (2 ms physics, like training) for a grid of commands: joint
targets, the body attitude and height the gait has on a perfect robot. On the robot the same scripted gait
produces the reference live, and the same observation function (policy_io.emote_observation) is used.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab import policy_io as pio  # noqa: E402
from crab.emotes import STAND  # noqa: E402
from crab.gait import _S, GAITS, Gait  # noqa: E402
from training.crab_env import MODEL  # noqa: E402
from training.emote_env import REWARD_SCALES as _EMOTE_SCALES, CrabEmoteEnv  # noqa: E402

V = 0.7211          # the gait's speed scale (crab/gait.py _S / _T), as in sim/robustness.py
GAIT_CMDS = [(0.0, 0.0, 0.0),
             (0.02, 0.0, 0.0), (0.04, 0.0, 0.0), (0.06, 0.0, 0.0), (-0.03, 0.0, 0.0), (-0.06, 0.0, 0.0),
             (0.0, 0.03, 0.0), (0.0, -0.03, 0.0), (0.0, 0.05, 0.0), (0.0, -0.05, 0.0),
             (0.0, 0.0, 0.4), (0.0, 0.0, -0.4), (0.0, 0.0, 0.7), (0.0, 0.0, -0.7),
             (0.03, 0.03, 0.0), (0.03, -0.03, 0.0), (0.04, 0.0, 0.4)]
REWARD_SCALES = dict(_EMOTE_SCALES, track_legs=1.0, track_attitude=1.5, track_height=0.5, track_vel=2.5,
                     residual=-0.1)


def walk_references(cmds=GAIT_CMDS, seconds=6.0, hz=50.0, sim_dt=0.002, sub=10):
    """Run the scripted trot on the nominal CPU twin for each command and record the reference:
    q (E,T,18) joint targets, grav (E,T,3), h (E,T), n (E,), and v (E,2) the commanded body velocity."""
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    m.opt.timestep = sim_dt
    d = mujoco.MjData(m)
    T = int(seconds * hz)
    E = len(cmds)
    q, grav, h = np.zeros((E, T, len(STAND))), np.zeros((E, T, 3)), np.zeros((E, T))
    act = np.array([m.actuator(j).id for j in pio.JOINT_NAMES_ALL])
    n_phys = int(round(1 / hz / sim_dt))
    body = m.body("body").id
    for e, cmd in enumerate(cmds):
        gait = Gait(GAITS["trot"], com_offset=np.array([0.015 * _S, 0.0]))
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        for _ in range(int(0.5 / sim_dt)):
            d.ctrl[act] = STAND
            mujoco.mj_step(m, d)
        target = STAND
        for k in range(T):
            for i in range(sub):
                out = gait.step(1 / hz / sub, cmd, solve=(i == sub - 1))
                target = out if out is not None else target
            d.ctrl[act] = target
            for _ in range(n_phys):
                mujoco.mj_step(m, d)
            R = d.xmat[body].reshape(3, 3)
            q[e, k], grav[e, k], h[e, k] = target, R.T @ np.array([0, 0, -1.0]), d.qpos[2]
    return dict(q=q, grav=grav, h=h, n=np.full(E, T), v=np.array([c[:2] for c in cmds]))


class CrabGaitEnv(CrabEmoteEnv):
    def __init__(self, sim_dt=0.002, obs_noise=0.02, push_interval_s=3.0, push_vel=0.08, reward_scales=None,
                 refs=None, vel_alpha=0.15, vel_sigma=0.015):
        super().__init__(sim_dt=sim_dt, obs_noise=obs_noise, push_interval_s=push_interval_s,
                         push_vel=push_vel, reward_scales=dict(REWARD_SCALES, **(reward_scales or {})),
                         refs=refs or walk_references(sim_dt=sim_dt))
        self.vel_alpha, self.vel_sigma = vel_alpha, vel_sigma

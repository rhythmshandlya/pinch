"""Run a trained walking policy in the regular (CPU) MuJoCo simulator.

Same interface as crab.gait.Gait -- `step(dt, cmd) -> 20 servo targets` --
so sim/walk.py can test, view and GIF a learned policy exactly like the
scripted gait. Observations come from crab/policy_io.py, as on the robot.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab import policy_io as pio  # noqa: E402


class PolicyController:
    ctrl_dt = pio.CTRL_DT          # targets change at 50 Hz (walk.py measures commanded speed per period)

    def __init__(self, run_dir, model, data):
        import jax
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks

        from training.train import network_factory

        run_dir = Path(run_dir)
        cfg = json.loads((run_dir / "config.json").read_text())
        with open(run_dir / "params.pkl", "rb") as fh:
            params = pickle.load(fh)
        nets = network_factory(cfg)(pio.OBS_SIZE, pio.N_ACT,
                                    preprocess_observations_fn=running_statistics.normalize)
        self._policy = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
        self._key = jax.random.PRNGKey(0)
        self.m, self.d = model, data
        self.body = model.body("body").id
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in pio.LEG_JOINT_NAMES])
        self.vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in pio.LEG_JOINT_NAMES])
        names = {model.joint(i).name for i in range(model.njnt)}
        bl = [f"{n}_backlash" for n in pio.LEG_JOINT_NAMES]
        self.bl_qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in bl]) if bl[0] in names else None
        self.bl_vadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in bl]) if bl[0] in names else None
        self.reset()

    def reset(self):
        self.t = 0.0
        self.since_update = 1e9
        self.last_action = np.zeros(pio.N_ACT)
        self.targets = pio.FULL_DEFAULT.copy()
        self.unreachable = 0                       # API parity with Gait

    def feasible_scale(self, cmd):                 # API parity with Gait
        return 1.0

    def step(self, dt, cmd=(0.0, 0.0, 0.0)):
        self.since_update += dt
        if self.since_update >= pio.CTRL_DT - 1e-9:
            self.since_update = 0.0
            d = self.d
            R = d.xmat[self.body].reshape(3, 3)
            q, qd = d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy()
            if self.bl_qadr is not None:                    # encoder after the gear play
                q += d.qpos[self.bl_qadr]
                qd += d.qvel[self.bl_vadr]
            obs = pio.observation(np, d.qvel[3:6], R.T @ np.array([0, 0, -1.0]),
                                  np.asarray(cmd, float), q, qd, self.last_action, self.t)
            act, _ = self._policy(obs.astype(np.float32), self._key)
            self.last_action = np.clip(np.asarray(act), -1, 1)
            self.targets = pio.action_to_targets(np, self.last_action)
        self.t += dt
        return self.targets

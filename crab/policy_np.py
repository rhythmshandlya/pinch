"""Trained policies without JAX: plain numpy MLP inference, so they run on the Radxa (and anywhere).

training/export_policy.py turns a Brax PPO checkpoint (params.pkl) into policies/<name>.npz:
  obs_mean, obs_std            running observation statistics
  W0, b0, W1, b1, ...          the policy network (swish hidden layers, tanh on the mean)

    pol = NumpyPolicy.load("policies/walk.npz")
    action = pol(obs)            # deterministic action in -1..1
"""

from pathlib import Path

import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1] / "policies"


class NumpyPolicy:
    def __init__(self, layers, mean, std):
        self.layers, self.mean, self.std = layers, mean, std

    @classmethod
    def load(cls, path):
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
        return cls(layers, z["obs_mean"], z["obs_std"])

    def __call__(self, obs):
        x = (np.asarray(obs, np.float32) - self.mean) / self.std
        x = np.clip(x, -5, 5)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = x / (1 + np.exp(-x))                       # swish (Brax default activation)
        a = x[: x.shape[-1] // 2]                              # mean half of (mean, log-std)
        return np.tanh(a)


def load_policies():
    """(emote_policy, walk_policy) if exported, else None for each (scripted fallbacks are used). Walking is
    policies/gait.npz: learned corrections on the scripted trot (training/gait_env.py, decision 50)."""
    out = []
    for name in ("emote", "gait"):
        p = POLICY_DIR / f"{name}.npz"
        out.append(NumpyPolicy.load(p) if p.exists() else None)
    return tuple(out)

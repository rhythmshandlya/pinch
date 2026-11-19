"""Trained policies without JAX: plain numpy MLP inference, so they run on the Radxa (and anywhere).

training/export_policy.py turns a Brax PPO checkpoint (params.pkl) into policies/<name>.npz:
  obs_mean, obs_std            running observation statistics
  W0, b0, W1, b1, ...          the policy network (swish hidden layers, tanh on the mean)

    pol = NumpyPolicy.load("policies/gait.npz")
    action = pol(obs)            # deterministic action in -1..1
"""

import os
from pathlib import Path

import numpy as np

POLICY_DIR = Path(__file__).resolve().parents[1] / "policies"


class NumpyPolicy:
    def __init__(self, layers, mean, std, meta=None):
        self.layers, self.mean, self.std = layers, mean, std
        self.meta = meta or {}                                 # policies/<name>.json: how it was trained

    @classmethod
    def load(cls, path):
        z = np.load(path)
        n = len([k for k in z.files if k.startswith("W")])
        layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
        meta_path = Path(path).with_suffix(".json")
        meta = __import__("json").loads(meta_path.read_text()) if meta_path.exists() else {}
        return cls(layers, z["obs_mean"], z["obs_std"], meta)

    def __call__(self, obs):
        x = (np.asarray(obs, np.float32) - self.mean) / self.std
        x = np.clip(x, -5, 5)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = x / (1 + np.exp(-x))                       # swish (Brax default activation)
        a = x[: x.shape[-1] // 2]                              # mean half of (mean, log-std)
        return np.tanh(a)


def load_policy(name):
    """policies/<name>.npz if exported, else None (the scripted fallback is used)."""
    p = POLICY_DIR / f"{name}.npz"
    return NumpyPolicy.load(p) if p.exists() else None


def load_policies():
    """(emote_policy, walk_policy) if exported, else None for each (scripted fallbacks are used). Walking is
    policies/gait.npz: learned corrections on the scripted trot (training/gait_env.py, decision 50).
    PINCH_GAIT_POLICY / PINCH_EMOTE_POLICY name a candidate instead, for benching it in the whole pet."""
    return (load_policy(os.environ.get("PINCH_EMOTE_POLICY", "emote")),
            load_policy(os.environ.get("PINCH_GAIT_POLICY", "gait")))

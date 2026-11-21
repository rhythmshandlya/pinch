"""An RSL-RL MLP actor as plain numpy (no torch, no onnx): what the sim-to-sim replay and, later, the robot run.

crab/policy_np.NumpyPolicy cannot load these: it expects Brax's swish hidden layers and a tanh on half of a
(mean, log-std) output, while an RSL-RL actor is an ELU MLP with a linear output (the deterministic action is the
network output itself) and its observation normaliser is (x - mean) / (std + eps), no clipping. So this module has
its own small forward pass and file format:

    obs_mean, obs_std (N,)        the baked normaliser
    obs_eps ()                    its eps (rsl_rl EmpiricalNormalization: 1e-2)
    W0, b0, W1, b1, ...           Linear layers, W stored (in, out)
    activation                    "elu" (the only one exported so far)
    meta_json                     task, checkpoint, layout sizes, action scale (a JSON string)

export.py writes it from a checkpoint and checks it against torch and the ONNX file (1e-4).
"""

import json
from pathlib import Path

import numpy as np


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class NpPolicy:
    def __init__(self, mean, std, layers, eps=1e-2, activation="elu", meta=None):
        if activation != "elu":
            raise ValueError(f"activation {activation!r} not supported")
        self.mean, self.std, self.eps = np.asarray(mean, np.float32), np.asarray(std, np.float32), float(eps)
        self.layers = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in layers]
        self.meta = meta or {}

    @property
    def obs_size(self):
        return self.layers[0][0].shape[0]

    @property
    def action_size(self):
        return self.layers[-1][0].shape[1]

    def __call__(self, obs):
        x = (np.asarray(obs, np.float32) - self.mean) / (self.std + self.eps)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = _elu(x)
        return x

    @classmethod
    def from_state_dict(cls, sd, meta=None, eps=1e-2):
        """sd: the actor's state dict as numpy arrays, keys as rsl_rl names them
        (obs_normalizer._mean / ._std, mlp.<i>.weight / .bias with torch's (out, in) weights)."""
        idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
        layers = [(np.asarray(sd[f"mlp.{i}.weight"]).T, np.asarray(sd[f"mlp.{i}.bias"])) for i in idx]
        return cls(np.asarray(sd["obs_normalizer._mean"]).reshape(-1), np.asarray(sd["obs_normalizer._std"]).reshape(-1),
                   layers, eps=eps, meta=meta)

    def save(self, path):
        out = dict(obs_mean=self.mean, obs_std=self.std, obs_eps=np.float32(self.eps), activation=np.array("elu"),
                   meta_json=np.array(json.dumps(self.meta)))
        for i, (w, b) in enumerate(self.layers):
            out[f"W{i}"], out[f"b{i}"] = w, b
        np.savez_compressed(path, **out)

    @classmethod
    def load(cls, path):
        z = np.load(Path(path), allow_pickle=False)
        n = len([k for k in z.files if k.startswith("W")])
        return cls(z["obs_mean"], z["obs_std"], [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)], float(z["obs_eps"]),
                   str(z["activation"]), json.loads(str(z["meta_json"])))

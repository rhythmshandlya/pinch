"""Export a trained Brax PPO policy to plain numpy (policies/<name>.npz) and check it matches Brax.

    .venv-rl\\Scripts\\python training/export_policy.py emote-v1 emote
    .venv-rl\\Scripts\\python training/export_policy.py walk-v1 walk

The robot and the laptop app run crab/policy_np.NumpyPolicy: no JAX needed (the Radxa can't run it well).
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export(run, name):
    src = ROOT / "training" / "checkpoints" / run
    norm, policy, _ = pickle.load(open(src / "params.pkl", "rb"))
    layers = policy["params"]
    out = dict(obs_mean=np.asarray(norm.mean, np.float32), obs_std=np.asarray(norm.std, np.float32))
    for i in range(len(layers)):
        out[f"W{i}"] = np.asarray(layers[f"hidden_{i}"]["kernel"], np.float32)
        out[f"b{i}"] = np.asarray(layers[f"hidden_{i}"]["bias"], np.float32)
    dst = ROOT / "policies"
    dst.mkdir(exist_ok=True)
    np.savez_compressed(dst / f"{name}.npz", **out)
    cfg = json.loads((src / "config.json").read_text())
    cfg.setdefault("action_scale", 0.15)
    (dst / f"{name}.json").write_text(json.dumps(dict(run=run, steps=int((src / "last_step.txt").read_text())
                                                      if (src / "last_step.txt").exists() else None, **cfg), indent=1))
    return dst / f"{name}.npz", cfg


def check(run, path, cfg):
    """The numpy policy must give the same deterministic action as Brax's own inference function."""
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from crab.policy_np import NumpyPolicy
    from training.train import network_factory
    params = pickle.load(open(ROOT / "training" / "checkpoints" / run / "params.pkl", "rb"))
    n_obs = params[0].mean.shape[0]
    n_act = params[1]["params"][f"hidden_{len(params[1]['params']) - 1}"]["bias"].shape[0] // 2
    nets = network_factory(cfg)(n_obs, n_act, preprocess_observations_fn=running_statistics.normalize)
    brax_pol = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    np_pol = NumpyPolicy.load(path)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(20):
        obs = (np.asarray(params[0].mean) + rng.normal(0, 1, n_obs) * np.asarray(params[0].std)).astype(np.float32)
        a_brax, _ = brax_pol(obs, jax.random.PRNGKey(0))
        errs.append(np.abs(np.asarray(a_brax) - np_pol(obs)).max())
    return float(max(errs))


if __name__ == "__main__":
    run, name = sys.argv[1], sys.argv[2]
    path, cfg = export(run, name)
    err = check(run, path, cfg)
    print(f"exported {run} -> {path.relative_to(ROOT)}; max difference vs Brax {err:.2e}")
    sys.exit(0 if err < 1e-4 else 1)

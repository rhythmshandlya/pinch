"""Train the crab walking policy with Brax PPO on the MJX environment.

    python training/train.py --preset smoke     # minutes on a laptop CPU: proves the pipeline
    python training/train.py --preset full      # GPU (Colab): the real training run

Writes training/checkpoints/<run>/{params.pkl, config.json, progress.jsonl}.
"""

import argparse
import functools
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRESETS = {
    # tiny: just checks that env + PPO + saving all work end to end
    "smoke": dict(num_timesteps=12_800, num_envs=16, batch_size=16, num_minibatches=4,
                  unroll_length=10, num_updates_per_batch=2, episode_length=200,
                  num_evals=2, num_eval_envs=16, learning_rate=3e-4, entropy_cost=1e-2,
                  discounting=0.97, policy_layers=(64, 64), value_layers=(64, 64)),
    # the real thing: ~100M steps, 4096 parallel crabs on one GPU
    "full": dict(num_timesteps=100_000_000, num_envs=4096, batch_size=256, num_minibatches=32,
                 unroll_length=20, num_updates_per_batch=4, episode_length=1000,
                 num_evals=20, num_eval_envs=256, learning_rate=3e-4, entropy_cost=1e-2,
                 discounting=0.97, policy_layers=(128, 128, 128), value_layers=(256, 256, 256)),
}


def network_factory(cfg):
    from brax.training.agents.ppo import networks as ppo_networks
    return functools.partial(ppo_networks.make_ppo_networks,
                             policy_hidden_layer_sizes=tuple(cfg["policy_layers"]),
                             value_hidden_layer_sizes=tuple(cfg["value_layers"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="smoke")
    ap.add_argument("--timesteps", type=int, help="override num_timesteps")
    ap.add_argument("--no-randomize", action="store_true", help="disable domain randomisation")
    ap.add_argument("--run", default=None, help="run name (default: timestamp)")
    a = ap.parse_args()

    import jax
    from brax.training.agents.ppo import train as ppo

    from training.crab_env import CrabWalkEnv, domain_randomize

    cfg = dict(PRESETS[a.preset])
    if a.timesteps:
        cfg["num_timesteps"] = a.timesteps
    run = a.run or f"{a.preset}-{datetime.now():%Y%m%d-%H%M%S}"
    out = ROOT / "training" / "checkpoints" / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(cfg, preset=a.preset,
                                                     randomize=not a.no_randomize), indent=2))

    print(f"devices: {jax.devices()}  |  run -> {out.relative_to(ROOT)}")
    env = CrabWalkEnv()
    eval_env = CrabWalkEnv()
    t0 = time.time()
    log = open(out / "progress.jsonl", "w")

    def progress(step, metrics):
        rec = {"step": int(step), "time_s": round(time.time() - t0, 1),
               **{k: float(v) for k, v in metrics.items()
                  if k.startswith("eval/episode_reward") or k in ("eval/avg_episode_length",)}}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"  step {rec['step']:>11,}  t={rec['time_s']:>7.1f}s  "
              f"reward={rec.get('eval/episode_reward', float('nan')):8.3f}  "
              f"ep_len={rec.get('eval/avg_episode_length', float('nan')):6.1f}")

    keys = ("num_timesteps", "num_envs", "batch_size", "num_minibatches", "unroll_length",
            "num_updates_per_batch", "episode_length", "num_evals", "num_eval_envs",
            "learning_rate", "entropy_cost", "discounting")
    make_policy, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        randomization_fn=None if a.no_randomize else domain_randomize,
        network_factory=network_factory(cfg),
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        seed=0,
        progress_fn=progress,
        **{k: cfg[k] for k in keys},
    )
    with open(out / "params.pkl", "wb") as fh:
        pickle.dump(jax.device_get(params), fh)
    print(f"done in {time.time() - t0:.0f}s -> {(out / 'params.pkl').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Train Pinch's policies with Brax PPO on MJX environments.

    python training/train.py --preset smoke                 # minutes on a laptop CPU: proves the pipeline
    python training/train.py --preset full                  # GPU: the real walking run
    python training/train.py --env emote --preset emote     # GPU: emote tracking (balance while emoting)
    python -m modal run training/modal_train.py ...         # the same on a Modal GPU (see that file)

Writes <out_root>/<run>/{params.pkl, config.json, progress.jsonl}; params.pkl is rewritten at every
evaluation, so a run that is stopped early still leaves its latest policy.
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
    # walking: ~100M steps, 4096 parallel crabs on one GPU
    "full": dict(num_timesteps=100_000_000, num_envs=4096, batch_size=256, num_minibatches=32,
                 unroll_length=20, num_updates_per_batch=4, episode_length=1000,
                 num_evals=20, num_eval_envs=256, learning_rate=3e-4, entropy_cost=1e-2,
                 discounting=0.97, policy_layers=(128, 128, 128), value_layers=(256, 256, 256)),
    # grasp skill: 4 s episodes, contact-rich (slower per step)
    "grasp": dict(num_timesteps=40_000_000, num_envs=4096, batch_size=256, num_minibatches=32,
                  unroll_length=20, num_updates_per_batch=4, episode_length=200,
                  num_evals=16, num_eval_envs=256, learning_rate=3e-4, entropy_cost=1e-2,
                  discounting=0.97, policy_layers=(256, 256, 128), value_layers=(256, 256, 256)),
    # push skill (hierarchical: 10 Hz decisions over the walking policy), 15 s episodes
    "push": dict(num_timesteps=15_000_000, num_envs=2048, batch_size=256, num_minibatches=16,
                 unroll_length=20, num_updates_per_batch=4, episode_length=150,
                 num_evals=10, num_eval_envs=256, learning_rate=3e-4, entropy_cost=1e-2,
                 discounting=0.99, policy_layers=(128, 128), value_layers=(256, 256)),
    # emote tracking: shorter episodes (one or two emotes), fewer steps
    "emote": dict(num_timesteps=40_000_000, num_envs=4096, batch_size=256, num_minibatches=32,
                  unroll_length=20, num_updates_per_batch=4, episode_length=500,
                  num_evals=16, num_eval_envs=256, learning_rate=3e-4, entropy_cost=5e-3,
                  discounting=0.97, policy_layers=(128, 128, 128), value_layers=(256, 256, 256)),
}
PPO_KEYS = ("num_timesteps", "num_envs", "batch_size", "num_minibatches", "unroll_length",
            "num_updates_per_batch", "episode_length", "num_evals", "num_eval_envs",
            "learning_rate", "entropy_cost", "discounting")


def network_factory(cfg):
    from brax.training.agents.ppo import networks as ppo_networks
    return functools.partial(ppo_networks.make_ppo_networks,
                             policy_hidden_layer_sizes=tuple(cfg["policy_layers"]),
                             value_hidden_layer_sizes=tuple(cfg["value_layers"]))


def make_env(name):
    if name == "emote":
        from training.emote_env import CrabEmoteEnv
        return CrabEmoteEnv()
    if name == "grasp":
        from training.grasp_env import CrabGraspEnv
        return CrabGraspEnv()
    if name == "push":
        from training.push_env import CrabPushEnv
        return CrabPushEnv()
    from training.crab_env import CrabWalkEnv
    return CrabWalkEnv()


def run_training(preset="smoke", env="walk", timesteps=None, run=None, randomize=True,
                 out_root=None, on_save=None):
    """Train one policy. on_save() is called after every checkpoint (Modal commits its volume)."""
    import jax
    from brax.training.agents.ppo import train as ppo

    from training.crab_env import domain_randomize

    cfg = dict(PRESETS[preset])
    if timesteps:
        cfg["num_timesteps"] = int(timesteps)
    run = run or f"{env}-{preset}-{datetime.now():%Y%m%d-%H%M%S}"
    out = Path(out_root or ROOT / "training" / "checkpoints") / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(cfg, preset=preset, env=env, randomize=randomize),
                                                indent=2))

    print(f"devices: {jax.devices()}  |  env {env}  |  run -> {out}", flush=True)
    t0 = time.time()
    log = open(out / "progress.jsonl", "a")

    def progress(step, metrics):
        rec = {"step": int(step), "time_s": round(time.time() - t0, 1),
               **{k: float(v) for k, v in metrics.items()
                  if k.startswith("eval/episode_") or k in ("eval/avg_episode_length",)}}
        if "eval/episode_success_rate" in metrics:
            rec["success"] = float(metrics["eval/episode_success_rate"])
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"  step {rec['step']:>11,}  t={rec['time_s']:>7.1f}s  "
              f"reward={rec.get('eval/episode_reward', float('nan')):8.3f}  "
              f"ep_len={rec.get('eval/avg_episode_length', float('nan')):6.1f}"
              + (f"  success={rec['success']:.2f}" if "success" in rec else ""), flush=True)

    def save(step, make_policy, params):
        with open(out / "params.pkl", "wb") as fh:
            pickle.dump(jax.device_get(params), fh)
        (out / "last_step.txt").write_text(str(int(step)))
        if on_save:
            on_save()

    make_policy, params, _ = ppo.train(
        environment=make_env(env),
        eval_env=make_env(env),
        randomization_fn=domain_randomize if randomize else None,
        network_factory=network_factory(cfg),
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        seed=0,
        progress_fn=progress,
        policy_params_fn=save,
        **{k: cfg[k] for k in PPO_KEYS},
    )
    save(cfg["num_timesteps"], make_policy, params)
    print(f"done in {time.time() - t0:.0f}s -> {out / 'params.pkl'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="smoke")
    ap.add_argument("--env", choices=["walk", "emote", "grasp", "push"], default="walk")
    ap.add_argument("--timesteps", type=int, help="override num_timesteps")
    ap.add_argument("--no-randomize", action="store_true", help="disable domain randomisation")
    ap.add_argument("--run", default=None, help="run name (default: env-preset-timestamp)")
    a = ap.parse_args()
    run_training(a.preset, a.env, a.timesteps, a.run, not a.no_randomize)


if __name__ == "__main__":
    main()

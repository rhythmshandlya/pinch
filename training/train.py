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

import numpy as np
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
    # grasp skill: 4 s episodes, contact-rich (slower per step)
    "grasp": dict(num_timesteps=30_000_000, num_envs=4096, batch_size=256, num_minibatches=32,
                  unroll_length=20, num_updates_per_batch=4, episode_length=200,
                  num_evals=16, num_eval_envs=256, learning_rate=3e-4, entropy_cost=1e-2,
                  discounting=0.97, policy_layers=(256, 256, 128), value_layers=(256, 256, 256)),
    "push2": dict(num_timesteps=30_000_000, num_envs=2048, batch_size=256, num_minibatches=16,
                 unroll_length=20, num_updates_per_batch=4, episode_length=150,
                 num_evals=10, num_eval_envs=256, learning_rate=3e-4, entropy_cost=1e-2,
                 discounting=0.99, policy_layers=(128, 128), value_layers=(256, 256)),
    # emote tracking: shorter episodes (one or two emotes), fewer steps
    # residual walking on the scripted trot (decision 50): like emotes, longer episodes of walking clips
    "gait": dict(num_timesteps=60_000_000, num_envs=2048, batch_size=256, num_minibatches=32,
                 unroll_length=20, num_updates_per_batch=4, episode_length=600,
                 num_evals=15, num_eval_envs=128, learning_rate=3e-4, entropy_cost=5e-3,
                 discounting=0.97, policy_layers=(128, 128, 128), value_layers=(256, 256, 256)),
    "gait_long": dict(num_timesteps=100_000_000, num_envs=2048, batch_size=256, num_minibatches=32,
                 unroll_length=20, num_updates_per_batch=4, episode_length=600,
                 num_evals=20, num_eval_envs=128, learning_rate=3e-4, entropy_cost=5e-3,
                 discounting=0.97, policy_layers=(128, 128, 128), value_layers=(256, 256, 256)),
    "gait_v2": dict(num_timesteps=150_000_000, num_envs=2048, batch_size=256, num_minibatches=32,
                 unroll_length=20, num_updates_per_batch=4, episode_length=600,
                 num_evals=20, num_eval_envs=128, learning_rate=3e-4, entropy_cost=5e-3,
                 discounting=0.97, policy_layers=(128, 128, 128), value_layers=(256, 256, 256)),
    "terrain": dict(num_timesteps=100_000_000, num_envs=2048, batch_size=256, num_minibatches=32,
                 unroll_length=20, num_updates_per_batch=4, episode_length=600,
                 num_evals=20, num_eval_envs=128, learning_rate=3e-4, entropy_cost=5e-3,
                 discounting=0.97, policy_layers=(128, 128, 128), value_layers=(256, 256, 256)),
    # get up off its back (decision 76): 4 s episodes, all 18 joints, no reference motion
    "righting": dict(num_timesteps=60_000_000, num_envs=4096, batch_size=256, num_minibatches=32,
                  unroll_length=20, num_updates_per_batch=4, episode_length=200,
                  num_evals=15, num_eval_envs=256, learning_rate=3e-4, entropy_cost=1e-2,
                  discounting=0.97, policy_layers=(128, 128, 128), value_layers=(256, 256, 256)),
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
    if name == "gait":
        from training.gait_env import CrabGaitEnv
        return CrabGaitEnv()
    if name == "gait2":          # walking-research values: a filter over ~a step cycle, a kinder tolerance
        from training.gait_env import CrabGaitEnv
        return CrabGaitEnv(vel_alpha=0.06, vel_sigma=0.03)
    if name == "terrain":         # hurdles, ramps and steps (training/terrain_env.py, decision 58)
        from training.terrain_env import CrabTerrainEnv
        return CrabTerrainEnv()
    if name == "terrain_hard":    # the same, half the robots on bars to 15 mm and ramps to 21 deg
        import training.terrain_env as te
        te.HARD = 1.5
        return te.CrabTerrainEnv()
    if name == "gait_v2":        # walking v2: 48 commands + pushing (paddle, crouched), bench-strength shoves
        from training.gait_env import (GAIT_CMDS_V2, CrabGaitEnv, concat_references, push_references,
                                       walk_references)
        env = CrabGaitEnv(refs=concat_references([walk_references(GAIT_CMDS_V2), push_references()]),
                          push_vel=0.12, push_interval_s=2.5)
        env.covers = ["walk", "push"]
        return env
    if name == "terrain_push":   # one walker for both: hurdles and ramps, 48 commands, crouched pushing (decision 84)
        from training.gait_env import GAIT_CMDS_V2, concat_references, push_references, walk_references
        from training.terrain_env import CrabTerrainEnv
        env = CrabTerrainEnv(refs=concat_references([walk_references(GAIT_CMDS_V2), push_references()]),
                             push_vel=0.12, push_interval_s=2.5)
        env.covers = ["walk", "push"]
        return env
    if name == "push2":          # learned push steering on the real walking controller (decision 67)
        from training.push2_env import CrabPush2Env
        return CrabPush2Env()
    if name == "gait3":          # gait-v1's rewards, shoves as hard as the robustness bench's (0.12 m/s, every 2.5 s)
        from training.gait_env import CrabGaitEnv
        return CrabGaitEnv(push_vel=0.12, push_interval_s=2.5)
    if name == "emote":
        from training.emote_env import CrabEmoteEnv
        return CrabEmoteEnv()
    if name == "emote2":          # the emote balance policy in 2 ms physics, like the twin (emote-v1: 4 ms)
        from training.emote_env import CrabEmoteEnv
        return CrabEmoteEnv(sim_dt=0.002)
    if name == "righting":       # get up off its back with all 18 joints (decision 76)
        from training.righting_env import CrabRightingEnv
        return CrabRightingEnv()
    if name == "righting_smooth":  # gentler on the servos: 5x the action-rate cost (decision 85's current spikes)
        from training.righting_env import CrabRightingEnv
        return CrabRightingEnv(reward_scales=dict(action_rate=-0.05))
    if name == "righting2":      # the same without the energy cost: does the penalty hold back getting up?
        from training.righting_env import CrabRightingEnv
        return CrabRightingEnv(reward_scales=dict(energy=0.0, action_rate=-0.005))
    if name == "grasp":
        from training.grasp_env import CrabGraspEnv
        return CrabGraspEnv()
    raise ValueError(f"unknown env {name!r}")


def run_training(preset="smoke", env="gait", timesteps=None, run=None, randomize=True,
                 out_root=None, on_save=None, num_envs=None):
    """Train one policy. on_save() is called after every checkpoint (Modal commits its volume)."""
    import jax
    from brax.training.agents.ppo import train as ppo

    from training.robot_model import domain_randomize
    from training.grasp_env import grasp_randomize
    from training.terrain_env import terrain_randomize
    from training.push2_env import push2_randomize

    cfg = dict(PRESETS[preset])
    if timesteps:
        cfg["num_timesteps"] = int(timesteps)
    if num_envs:                                  # (a smaller GPU: msi-1's 6 GB RTX 4050)
        cfg["num_envs"] = int(num_envs)
    run = run or f"{env}-{preset}-{datetime.now():%Y%m%d-%H%M%S}"
    out = Path(out_root or ROOT / "training" / "checkpoints") / run
    out.mkdir(parents=True, exist_ok=True)
    environment = make_env(env)
    (out / "config.json").write_text(json.dumps(dict(cfg, preset=preset, env=env, randomize=randomize,
                                                     action_scale=float(getattr(environment, "action_scale", 0.15)),
                                                     covers=list(getattr(environment, "covers", ["walk"]))),
                                                indent=2))

    print(f"devices: {jax.devices()}  |  env {env}  |  run -> {out}", flush=True)
    t0 = time.time()
    log = open(out / "progress.jsonl", "a")

    def progress(step, metrics):
        rec = {"step": int(step), "time_s": round(time.time() - t0, 1),
               **{k: float(v) for k, v in metrics.items()
                  if k.startswith("eval/episode_") or k in ("eval/avg_episode_length",)}}
        if "eval/episode_track" in metrics:
            rec["success"] = float(metrics["eval/episode_track"])          # walking: fraction of commanded speed
        if "eval/episode_success_rate" in metrics:
            rec["success"] = float(metrics["eval/episode_success_rate"])
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"  step {rec['step']:>11,}  t={rec['time_s']:>7.1f}s  "
              f"reward={rec.get('eval/episode_reward', float('nan')):8.3f}  "
              f"ep_len={rec.get('eval/avg_episode_length', float('nan')):6.1f}"
              + (f"  success={rec['success']:.2f}" if "success" in rec else ""), flush=True)
        if step > 0 and not np.isfinite(rec.get("eval/episode_reward", 0.0)):
            raise RuntimeError("training diverged (NaN reward): stopping so the GPU isn't wasted")

    def save(step, make_policy, params):
        with open(out / "params.pkl", "wb") as fh:
            pickle.dump(jax.device_get(params), fh)
        (out / "last_step.txt").write_text(str(int(step)))
        if on_save:
            on_save()

    make_policy, params, _ = ppo.train(
        environment=environment,
        eval_env=make_env(env),
        randomization_fn=({"grasp": grasp_randomize, "terrain": terrain_randomize,
                           "terrain_hard": terrain_randomize, "terrain_push": terrain_randomize,
                           "push2": push2_randomize}.get(env, domain_randomize)
                          if randomize else None),
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
    ap.add_argument("--env", choices=["gait", "gait2", "gait3", "terrain", "terrain_hard", "terrain_push", "gait_v2", "push2", "emote", "emote2", "righting", "righting2", "righting_smooth", "grasp"],
                        default="gait")
    ap.add_argument("--timesteps", type=int, help="override num_timesteps")
    ap.add_argument("--no-randomize", action="store_true", help="disable domain randomisation")
    ap.add_argument("--run", default=None, help="run name (default: env-preset-timestamp)")
    ap.add_argument("--num-envs", type=int, default=None, help="override num_envs (smaller GPUs)")
    a = ap.parse_args()
    run_training(a.preset, a.env, a.timesteps, a.run, not a.no_randomize, num_envs=a.num_envs)


if __name__ == "__main__":
    main()

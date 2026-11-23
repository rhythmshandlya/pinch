"""Train Pinch's policies with Brax PPO on MJX environments.

    python -m training.train --experiment configs/experiments/righting.toml     # what an experiment describes
    python -m training.train --env righting --preset smoke                       # minutes on a CPU: the pipeline
    python -m ops submit configs/experiments/righting.toml --backend modal       # the same on any GPU (ops/)

Environments come from training/registry.py, presets from configs/train/presets.toml, experiments from
configs/experiments/. Writes <out_root>/<run>/: params.pkl (rewritten at every evaluation, so a run stopped early
still leaves its latest policy), progress.jsonl, config.json (read by export_policy) and run.json (the run
record: commit, host, resolved config, state, last metrics).
"""

import argparse
import functools
import json
import pickle
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRESETS_FILE = ROOT / "configs" / "train" / "presets.toml"
PRESETS = {k: {kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}
           for k, v in tomllib.loads(PRESETS_FILE.read_text(encoding="utf-8")).items()}
PPO_KEYS = ("num_timesteps", "num_envs", "batch_size", "num_minibatches", "unroll_length",
            "num_updates_per_batch", "episode_length", "num_evals", "num_eval_envs",
            "learning_rate", "entropy_cost", "discounting")


def network_factory(cfg):
    from brax.training.agents.ppo import networks as ppo_networks
    return functools.partial(ppo_networks.make_ppo_networks,
                             policy_hidden_layer_sizes=tuple(cfg["policy_layers"]),
                             value_hidden_layer_sizes=tuple(cfg["value_layers"]))


def make_env(name):
    from training.registry import make_env as _make
    return _make(name)


def git_state(root=ROOT):
    """(commit, dirty) of the code being trained. A job snapshot has no .git: ops passes PINCH_GIT_SHA."""
    import os
    import subprocess
    if os.environ.get("PINCH_GIT_SHA"):
        return os.environ["PINCH_GIT_SHA"], os.environ.get("PINCH_GIT_DIRTY") == "1"
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def load_experiment(path):
    """configs/experiments/<name>.toml -> dict (format: configs/experiments/README.md)."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    exp = tomllib.loads(p.read_text(encoding="utf-8"))
    e = exp.setdefault("experiment", {})
    e["file"] = p.resolve().relative_to(ROOT).as_posix() if p.resolve().is_relative_to(ROOT) else str(p)
    for key in ("env", "preset"):
        if key not in e:
            raise ValueError(f"{p}: [experiment] needs {key!r}")
    return exp


def resolve_config(preset, overrides=None, timesteps=None, num_envs=None, seed=None):
    """The preset, then the experiment's overrides, then command-line overrides. The seed is part of the
    resolved config, so every run record names it."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; known: {', '.join(PRESETS)}")
    cfg = dict(PRESETS[preset])
    for k, v in (overrides or {}).items():
        if k not in cfg:
            raise ValueError(f"override {k!r} is not a key of preset {preset!r}")
        cfg[k] = tuple(v) if isinstance(v, list) else v
    if timesteps:
        cfg["num_timesteps"] = int(timesteps)
    if num_envs:                                  # (a smaller GPU: msi-1's 6 GB RTX 4050)
        cfg["num_envs"] = int(num_envs)
    cfg["seed"] = int(seed or 0)                  # train a change under several seeds before trusting it
    return cfg


def _write_record(out, record):
    tmp = out / "run.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, default=str))
    tmp.replace(out / "run.json")


def run_training(preset="smoke", env="gait", timesteps=None, run=None, randomize=True,
                 out_root=None, on_save=None, num_envs=None, overrides=None, experiment=None, seed=None):
    """Train one policy. on_save() is called after every checkpoint (Modal commits its volume)."""
    import platform
    import socket

    import jax
    from brax.training.agents.ppo import train as ppo

    from training.registry import randomizer, spec

    skill = spec(env).skill                       # fails fast on an unknown env
    cfg = resolve_config(preset, overrides, timesteps, num_envs, seed)
    run = run or f"{env}-{preset}-{datetime.now():%Y%m%d-%H%M%S}"
    out = Path(out_root or ROOT / "training" / "checkpoints") / run
    out.mkdir(parents=True, exist_ok=True)
    environment = make_env(env)
    (out / "config.json").write_text(json.dumps(dict(cfg, preset=preset, env=env, randomize=randomize,
                                                     action_scale=float(getattr(environment, "action_scale", 0.15)),
                                                     covers=list(getattr(environment, "covers", ["walk"]))),
                                                indent=2))
    sha, dirty = git_state()
    record = dict(run=run, env=env, skill=skill, preset=preset, config=cfg, randomize=randomize,
                  experiment=experiment, git_sha=sha, git_dirty=dirty, host=socket.gethostname(),
                  platform=platform.platform(), devices=[str(d) for d in jax.devices()],
                  started=datetime.now().astimezone().isoformat(timespec="seconds"), state="running", last=None)
    _write_record(out, record)

    restore = warm_start(experiment, environment)
    record["init_from"] = (experiment or {}).get("init_from") if restore is not None else None
    _write_record(out, record)
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
        record["last"] = rec
        _write_record(out, record)
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

    try:
        make_policy, params, _ = ppo.train(
            environment=environment,
            eval_env=make_env(env),
            randomization_fn=randomizer(env) if randomize else None,
            network_factory=network_factory(cfg),
            normalize_observations=True,
            reward_scaling=1.0,
            clipping_epsilon=0.2,
            gae_lambda=0.95,
            max_grad_norm=1.0,
            seed=cfg["seed"],
            progress_fn=progress,
            policy_params_fn=save,
            **({"restore_params": restore} if restore is not None else {}),
            **{k: cfg[k] for k in PPO_KEYS},
        )
        save(cfg["num_timesteps"], make_policy, params)
        record["state"] = "done"
    except BaseException as e:                    # noqa: BLE001 - record why, then let it propagate
        record["state"], record["error"] = "failed", f"{type(e).__name__}: {e}"
        raise
    finally:
        record["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        record["duration_s"] = round(time.time() - t0, 1)
        _write_record(out, record)
        log.close()
        if on_save:
            on_save()
    print(f"done in {time.time() - t0:.0f}s -> {out / 'params.pkl'}", flush=True)
    return out


def warm_start(experiment, environment):
    """An experiment may start from an earlier policy (`init_from = "<params.pkl>"` under [experiment]; ~ expands, a
    relative path is from the repo root), so past training is not thrown away: PPO continues from its normaliser,
    policy and value weights. Used only when its observation and action sizes match the environment; otherwise the
    run starts from scratch and says so."""
    import os
    import pickle
    path = (experiment or {}).get("init_from")
    if not path:
        return None
    p = Path(os.path.expanduser(path))
    p = p if p.is_absolute() else ROOT / p
    if not p.exists():
        print(f"warm start: {p} not found, training from scratch", flush=True)
        return None
    params = pickle.loads(p.read_bytes())
    obs = np.shape(params[0].mean)[-1] if hasattr(params[0], "mean") else None
    want = environment.observation_size
    want = want if isinstance(want, int) else None
    if obs is not None and want is not None and obs != want:
        print(f"warm start: {p.name} has {obs} observations, this env {want}: training from scratch", flush=True)
        return None
    print(f"warm start from {p}", flush=True)
    return params


def run_experiment(path, run=None, out_root=None, on_save=None, num_envs=None, timesteps=None, seed=None):
    """Train what an experiment file describes (its env, preset and overrides)."""
    exp = load_experiment(path)
    e = exp["experiment"]
    return run_training(e["preset"], e["env"], timesteps, run or e.get("name"), e.get("randomize", True),
                        out_root, on_save, num_envs, exp.get("overrides"), experiment=e,
                        seed=e.get("seed") if seed is None else seed)


def main():
    from training.registry import ENVS
    ap = argparse.ArgumentParser(description="Train a policy: an experiment file, or an env and a preset.")
    ap.add_argument("--experiment", help="configs/experiments/<name>.toml (sets env, preset and overrides)")
    ap.add_argument("--preset", choices=list(PRESETS), default="smoke")
    ap.add_argument("--env", choices=sorted(ENVS), default="gait")
    ap.add_argument("--timesteps", type=int, help="override num_timesteps")
    ap.add_argument("--no-randomize", action="store_true", help="disable domain randomisation")
    ap.add_argument("--run", default=None, help="run name (default: the experiment's name, or env-preset-time)")
    ap.add_argument("--num-envs", type=int, default=None, help="override num_envs (smaller GPUs)")
    ap.add_argument("--seed", type=int, default=None, help="training seed (default: the experiment's, else 0)")
    ap.add_argument("--out-root", default=None, help="where run directories go (default training/checkpoints)")
    a = ap.parse_args()
    if a.experiment:
        run_experiment(a.experiment, a.run, a.out_root, num_envs=a.num_envs, timesteps=a.timesteps, seed=a.seed)
    else:
        run_training(a.preset, a.env, a.timesteps, a.run, not a.no_randomize, a.out_root, num_envs=a.num_envs,
                     seed=a.seed)


if __name__ == "__main__":
    main()

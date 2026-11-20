---
name: pinch-new-env
description: Add a new reinforcement-learning environment or task variant for Pinch (MJX + Brax PPO) - reward design, domain randomisation, registry entry, experiment file, observation/action contract for the robot, tests. Use when a new skill needs training (pushing, climbing, a new trick) or an existing env needs a variant (new rewards, terrain, randomisation). Pair with the mujoco-accelerated skill for MJX specifics.
---

# A new training environment

## Before writing code

- **Research first** (project rule): find how recent papers or well-maintained projects (MuJoCo Playground, mjlab,
  Isaac Lab tasks, legged_gym) solve the closest task: observation set, reward terms, curriculum, randomisation.
  Cite them in the decision.
- Decide the policy's slot (`gait`, `emote`, `righting`, `grasp`, `push`, or a new one) and what the robot will
  feed it: the observation must be computable on the robot from its sensors (IMU, joint positions, last action -
  no simulator-only state unless it is for the critic).

## Steps

1. Write `training/<skill>_env.py` from the closest existing env (`righting_env.py` is the smallest full one;
   `gait_env.py` shows residual control over the scripted gait). Shared robot loading and randomisation:
   `training/robot_model.py`. Never import `pinch/` or `robot/` from training (layering test).
2. Put the observation/action contract the robot also uses in `crab/policy_io.py` (numpy, no JAX), so training
   and deployment share one definition.
3. Register it once in `training/registry.py` (`EnvSpec`: name, skill, factory, kwargs, randomiser, description).
4. Add `configs/experiments/<name>.toml` with a last-resort timeout and an `[eval]` that runs a bench.
5. A bench in `sim/benches/` if none measures the skill yet (`common.run_parallel`, `--json`, many seeds).
6. `python tests/run.py ops architecture` then a smoke run: `python -m training.train --env <name> --preset smoke`
   with `.venv-rl` (CPU, minutes).
7. Train with pinch-train; judge with pinch-adopt-policy.

## Reward and stability gotchas

- Scale rewards so the dominant term is the task, not a proximity bonus (push2-v1 learned to hover near the
  object: decision 88).
- Penalise action rate when the policy will drive real servos; check slew on the bench (decision 85).
- Randomise friction, mass, latency and motor strength; train on 2 ms physics when the twin uses 2 ms.
- `num_envs` must fit 6 GB on msi-1: set `small_gpu_num_envs` in the experiment.

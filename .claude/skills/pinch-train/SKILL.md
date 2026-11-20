---
name: pinch-train
description: Train one of Pinch's policies (walking, get-up, emote balance, grasp, push) on any GPU - this laptop, msi-1 or another Tailscale machine, Modal or a SkyPilot cloud - and follow it to a fetched checkpoint. Use when asked to train, retrain, launch, queue, monitor, stop or fetch an RL run, or to keep the GPUs busy. Not for judging or adopting the result (pinch-adopt-policy) or writing a new environment (pinch-new-env).
---

# Train a Pinch policy

Everything goes through `python -m ops` (run from the repo root with `.venv\Scripts\python`). Details:
`docs/dev/TRAINING.md`, `docs/dev/COMPUTE.md`.

## Steps

1. **Pick or write the experiment** in `configs/experiments/` (`python -m ops envs` lists environments). Copy the
   closest file; change `name`, `env`, `preset`, `[overrides]`. Keep `[compute] timeout_min` - paid backends
   refuse to run without a hard stop. Run `python tests/run.py ops` after editing.
2. **Commit and push.** Jobs train a pushed commit (`ops submit` refuses unpushed or dirty trees for remote
   backends). Name the run after the task: `--run <task>-<skill>-v<n>`.
3. **Choose compute:**
   - free first: `--host msi-1` (6 GB: small_gpu_num_envs is applied) or `--host any` for the first free host;
   - Modal only when the hosts are busy or the run is large: `--backend modal` (state the GPU and timeout to the
     user before starting: it costs money), or `--host any --cloud-fallback`;
   - `local` for smoke tests (`configs/experiments/smoke.toml`, ~8 min on CPU).
4. **Submit:** `python -m ops submit configs/experiments/<exp>.toml --host any --cloud-fallback --run <name>`.
5. **Monitor:** `python -m ops jobs -r` (state, step, reward, success). `python -m ops logs <run>` for the
   trainer's output. Don't poll in a tight loop: check every 10-30 minutes, or let `ops dispatch --watch` run.
6. **When done:** `python -m ops fetch <run>`, then hand over to pinch-adopt-policy. A failed run: read
   `ops logs`, record why in the decision log if it teaches something.

## Gotchas

- msi-1 is a Windows laptop; jobs run in WSL under a Scheduled Task. If `ops hosts check` says unreachable, the
  laptop is off or Tailscale is down - ask the user, don't retry in a loop.
- `success` means different things per skill: fraction of commanded speed (walking), success rate (get-up, grasp).
- A NaN reward stops the run on purpose (the trainer raises to save GPU time).
- Never leave paid jobs running without a reason: `python -m ops stop <run>`.

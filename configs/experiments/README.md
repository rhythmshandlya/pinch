# Experiments

One file per experiment: what to train, what compute it needs, and how the result is judged. `ops submit` takes
the file; the trainer reads `[experiment]` and `[overrides]`; `ops eval` runs `[eval]`.

```toml
[experiment]
name = "terrain-push"          # default run name (ops adds -<n> when a run of that name exists)
env = "terrain_push"           # training/registry.py
preset = "gait_v2"             # configs/train/presets.toml
description = "one walker for hurdles, ramps and pushing"
seed = 0                       # optional; `ops submit --seed N` overrides it. The run's run.json records it

[overrides]                    # any preset key
num_timesteps = 150_000_000

[compute]
min_vram_gb = 6                # hosts with less are skipped
small_gpu_num_envs = 2048      # num_envs on GPUs under 12 GB
timeout_min = 1440             # last-resort stop (Modal's 24 h maximum); runs end when their steps are done
gpu = "A10G"                   # Modal / cloud GPU type
cloud_ok = true                # may it run on a paid backend (Modal, SkyPilot)?

[eval]                         # run on the exported candidate; {candidate} is its policies/ name
skill = "gait"                 # sets PINCH_GAIT_POLICY={candidate} for benches that load the whole pet
commands = [
  "sim/benches/hurdles.py --policy {candidate}",
  "sim/benches/robustness.py --what walk --world terrain --episodes 40",
]
```

# Training on mjlab (MuJoCo Warp + PyTorch)

A second trainer next to the Brax/MJX one, started because of the audit (finding 5): Brax now maintains only its
training code, the MuJoCo Playground moved to PyTorch + MuJoCo Warp + RSL-RL, and the measured servo models
published by Rhoban (BAM) plug into mjlab and CPU MuJoCo, not into MJX. The Brax environments are untouched; this
is an addition, and the walking task is ported first as the template for the rest.

Template followed: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) (a small servo
robot on mjlab with BAM actuators: one shared observation layout, delay / bias / misalignment randomisation, ONNX
export with the normaliser baked in, written reward-design rules) on top of
[mujocolab/mjlab](https://github.com/mujocolab/mjlab)'s own velocity task.

## Layout (`training/mjlab_pinch/`)

| File | What | Needs |
|---|---|---|
| `layout.py` | the one observation / action / command layout, history, command buckets | numpy |
| `robot_spec.py` | the robot from `sim/models/crab.xml` for mjlab; servo numbers from `crab/params.py` | mujoco |
| `mdp.py` | leg action (arms hold the stand pose), IMU with mounting error, bucketed velocity command | torch, mjlab |
| `env_cfg.py` | the task: sensors, observations, events (randomisation), rewards, terminations, PPO settings | torch, mjlab |
| `tasks.py`, `train.py` | register `Pinch-Velocity-Flat` (position actuators) and `Pinch-Velocity-Flat-Bam`; run mjlab's trainer | torch, mjlab |
| `check_env.py` | build the task and step it: sizes, joint order, NaNs, stand holds | GPU |
| `export.py`, `np_policy.py` | checkpoint to ONNX (normaliser baked in) and to plain numpy, three-way check to 1e-4 | torch / numpy |
| `sim_replay.py` | the sim-to-sim check on the CPU twin | numpy, mujoco |
| `curves.py` | learning curves from the tensorboard files, penalty sign check | tensorboard |

`ops/mjlab_run.py` starts, follows, stops and fetches a job on an SSH host. It sits in `ops/` because `training/`
may not import `ops/` (`tests/test_architecture.py`).

**The robot model and the servo numbers are swappable.** `PINCH_MJLAB_MODEL` (or `robot_spec.MODEL`) names the MJCF;
a resized robot only has to keep the joint, foot, site and body names (`tests/test_mjlab_pinch.py` checks them).
Servo numbers come from `crab/params.py` for the plain position actuators, or from a BAM file for the BAM
variant (`env_cfg.BAM_MOTOR`).

## The task

- **Control**: 50 Hz (5 ms physics x 4). Action = 12 leg-joint position offsets (rad, x0.3) around the stand
  pose. The six arm servos hold the stand pose and are never observed (they report nothing).
- **Observation (129)**: gyro, gravity direction, leg position (with encoder bias) and velocity, last action, each
  over a 3-frame history (60 ms), then the command once. Built identically by `layout.frame` and `layout.History`
  in the trainer's layout and in the sim-to-sim replay (term-major, oldest first, first frame backfilled: mjlab's
  own history behaviour). The critic additionally sees true joint positions, body velocity and foot contact.
- **Commands**: vx, vy up to 6 cm/s, yaw up to 0.6 rad/s, from explicit buckets: 10 % exactly zero, 15 % turn in
  place, 15 % forward only, 15 % sideways only, 45 % mixed. vy is drawn exactly as densely as vx (a test pins it).
- **Rewards** (microduck_rl's velocity recipe, re-sized for a 6 cm/s crab: its Gaussians are 0.3 to 0.5 m/s wide,
  which would be flat here): velocity tracking (std 4 cm/s, 0.3 rad/s), upright, pose (tight when standing, wide
  when walking), air time, foot clearance / swing height / slip, body angular velocity, action rate, joint limits,
  a fall penalty. The sign rule from its AGENTS.md is checked by `curves.py`: every penalty reads <= 0.
- **Terminations**: tilt over 60 degrees, body lower than 28 mm (only the feet collide, so a collapsed body
  would sink through the desk), NaN state, 20 s timeout.
- **Randomisation**: action delay 0-3 control steps (as physics-step lag, resampled about every half second),
  gyro / gravity delay 0-1 step, encoder bias +-0.025 rad, IMU mounting error up to 5 degrees (per episode),
  foot friction 0.3-1.25, mass with inertia +-15 %, body centre of mass +-5 mm, pushes of 4 cm/s every 1.5-4 s,
  servo stiffness 0.75-1.25x, motor strength 0.65-1.15x, armature and damping +-20 / 15 %. The BAM variant has
  per-env supply voltage (6.6-8.2 V) and friction scale (0.8-1.2) instead of the last four.

## Run it

On the trainer (msi-1: WSL Ubuntu, `~/pinch-mjlab/.venv`; versions in `requirements-mjlab.txt`):

```bash
python -m training.mjlab_pinch.check_env Pinch-Velocity-Flat 64          # sizes, order, NaNs, stand holds
python -m training.mjlab_pinch.train Pinch-Velocity-Flat --env.scene.num-envs 64 --agent.max-iterations 5 \
    --agent.logger tensorboard                                           # the smoke test: always first
python -m training.mjlab_pinch.export logs/rsl_rl/pinch_velocity/<run>/model_N.pt out_dir
python -m training.mjlab_pinch.curves logs/rsl_rl
```

From this laptop (commit and push first; the job trains the pushed commit):

```powershell
.venv\Scripts\python -m ops.mjlab_run start --name walk1 --envs 16384 --iterations 300 --agent.num-steps-per-env 12
.venv\Scripts\python -m ops.mjlab_run status walk1      # also: logs, stop
.venv\Scripts\python -m ops.mjlab_run fetch walk1       # -> training/checkpoints/walk1/{policy/, logs/, train.log, gpu_mem.csv}
.venv\Scripts\python -m training.mjlab_pinch.sim_replay training/checkpoints/walk1/policy/policy.npz
```

Use `--agent.logger tensorboard` (the launcher does): mjlab's default wandb writer crashes on this setup. The job
exports the last checkpoint itself and writes `policy.onnx`, `policy.npz`, `export.json`. A branch that is not
`master` is fetched by name (`ops.mjlab_run` does it; `ops submit`'s `git fetch origin` alone only gets master on
a single-branch clone, which msi-1's is).

## Measured on msi-1 (RTX 4050 Laptop, 6 GB)

| Environments | Steps/s | GPU memory (process peak) |
|---|---|---|
| 64 | about 2,000 | |
| 4,096 | 45,500 | 1.2 GB |
| 8,192 | 53,600 | 2.1 GB |
| 16,384 | 57,600 | 3.9 GB |
| 24,576 | 59,100 | 5.8 GB (of 6.1: too tight) |

Speed saturates around 16k environments; 16,384 is the working size. The Brax/MJX trainer is not measured on the
same robot here, so no comparison is claimed; mjlab's own Go1 probe on the same GPU gave 38,000 steps/s.

PROOF_RUN_PLACEHOLDER

## What differs from the MJX environments

- Direct velocity-command walking from the stand pose, not corrections on the scripted trot (decision 50); the
  scripted gait is not used at all. No reference motion.
- PyTorch + RSL-RL PPO (ELU MLP 256-128-64, running observation normaliser, Gaussian policy), not Brax PPO.
- 50 Hz control at 5 ms physics, not 500 Hz at 2 ms with 10 substeps; contacts are MuJoCo Warp's.
- Observations carry a 3-frame history, the commands are bucketed, delay is randomised 0-3 steps. The MJX
  environments have no history and a fixed action delay.
- A policy is ONNX / `.npz` (ELU, linear output, normaliser `(x - mean) / (std + 0.01)`), not the swish + tanh of
  `crab/policy_np.py`. `np_policy.NpPolicy` loads it; the robot's loader does not exist yet (see below).

## The version gap

The trainer runs mujoco 3.11 and mujoco-warp 3.11 (what mjlab 1.6.1 needs); the twin and the tests run mujoco
4.0.1. They differ in the compiled solver, so a policy trained under one meets slightly different contacts in the
other. This is exactly what the sim-to-sim check measures (it also differs in time step: 5 ms trained, 2 ms
replayed, and in the twin's backlash joints which the trainer also has). BAM's own code declares mjlab < 1.4 and
Python < 3.13; its mjlab actuator imports and runs on mjlab 1.6.1 (64-env smoke run, no NaN), installed with
`--no-deps`. Treat BAM on mjlab 1.6.1 as working but unsupported until a longer run confirms it.

## What is left

- **BAM parameters for the STS3215 and the resized robot.** The `Pinch-Velocity-Flat-Bam` task uses
  `feetech_sts3215_7_4V/m6` (decision 100) on today's small, light model: the file's reflected inertia (0.022 kg m^2)
  is far larger than this robot's links, so that variant only proves the plumbing. Once the resized model exists:
  set `PINCH_MJLAB_MODEL`, keep the joint names, re-check the stand height, joint limits and the action scale,
  and re-run the smoke test and a training run. Also open: BAM's stiff friction-loss constraint and delay terms
  against the real servo (decision 100's risk).
- **Get-up task** (righting from the back) and **motion tracking** (emotes and tricks as reference motions);
  mjlab has a tracking task to start from. Both reuse `layout.py` and the leg action.
- **A robot-side loader** for the ONNX / `.npz` policy and a decision on which of the two the firmware runs.
- **Experiment files**: `trainer = "mjlab"` in `configs/experiments/` and the matching command in the ssh job
  script, so `ops submit --host any` can place these jobs; `ops/mjlab_run.py` is the stopgap.
- **Reward and curriculum tuning.** The first runs prove the pipeline, not a gait: see the proof run above. The
  microduck_rl budgets are 4,000 to 6,000 iterations at 4,096 environments for gaits (about 100 times this proof
  run). The CAD shin-clamp (`crab/policy_io.shin_clamp`) is not applied in the mjlab action.
- **Desk tilt** (gravity direction randomisation of the MJX environments) is not ported; pushes and the IMU
  mounting error cover part of it.

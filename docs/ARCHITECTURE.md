# Architecture

How the Pinch project is laid out, how code is developed and tested, how policies are trained and adopted, and how
work runs in parallel across GPUs and agents. This is the reference; the workflows built on it are in
[docs/dev/](dev/).

## 1. The repository

```
pinch/                 the pet: mind, body, face, senses, memory, routine (runs on the robot and in the twin)
  mind/ body/ activities/ twin/
crab/                  the robot itself: geometry, kinematics, gait, emotes, policy I/O (no dependencies above it)
robot/                 firmware: drivers, servo bus, supervisor, HTTP API, selftest (runs on the Radxa)
sim/                   MuJoCo models, the desk world, benches (sim/benches/), clips, CAD checks
  worlds/ scenarios/   seeded homes, desks, tables and floors with people (1,000 training, 200 held-out test
                       worlds), the fisheye camera, behaviour scenarios as data (section 8)
training/              learning: MJX environments, the environment registry, the trainer, export to numpy
configs/               data, not code: training presets (configs/train/presets.toml) and experiments
  train/ experiments/  (configs/experiments/*.toml): what to train, on what compute, judged by which bench
ops/                   the operations layer: job specs, the compute backends (local, SSH/Tailscale, Modal),
                       the job ledger, host checks, worktrees for parallel work. `python -m ops ...`
policies/              adopted policies as numpy (.npz + .json) and registry.toml: which run each came from
tests/                 the suite (`python tests/run.py` or `pytest`): replay, golden trace, layering, ops...
cad/ hardware/         printable parts and the bill of materials, wiring, assembly
deploy/                the robot's installer and systemd service
docs/                  decisions (DECISIONS.md), reports, this file, workflows (docs/dev/)
```

### Layers (enforced by `tests/test_architecture.py`)

```
            ops  (controller side: may read configs and the registry, launches jobs; nothing imports it)
             |
   training  ->  crab                 training never imports pinch/robot (it ships to GPUs without them)
   pinch     ->  crab, sim(models)    the pet never imports training (the robot has no JAX)
   robot     ->  pinch, crab          firmware wraps the same mind and body
   crab      ->  (nothing of ours)
```

Learned skills cross from `training/` to the robot only as exported numpy files in `policies/`, loaded by
`crab/policy_np.py`. That one boundary is why the robot needs neither JAX nor a GPU.

## 2. Development

- **One project file:** `pyproject.toml` declares the dependency sets (`dev`, `train`, `cad`, `robot`) and the
  tool settings (ruff, pytest). Two environments in practice: `.venv` (twin, tests, CAD) and `.venv-rl` (JAX/MJX
  training, pinned in `requirements-train.txt`).
- **No editable install.** Every entry point puts its own checkout first on `sys.path`, so several git
  worktrees can share one `.venv` without importing each other's code.
- **Checks:** `ruff check .` (lint), `python tests/run.py` (about 100 tests, including the golden trace and the
  layering rules), `python -m ops check` (both, plus the private-file guard and the robot's two self-tests
  against emulated hardware). Pre-commit runs the fast ones; CI runs all of them on every push and pull request,
  on Linux with a virtual display so that rendering and the robot's camera checks run there too.
- **Private files never leave the machine:** a guard (`ops/guard.py`, run by pre-commit and CI) refuses commits
  that add the local-only documents or AI attribution lines.
- **Decisions:** every non-trivial choice gets a numbered entry in `docs/DECISIONS.md` with its measurement.

## 3. Training

```
configs/experiments/<name>.toml  ->  ops submit  ->  backend (local | ssh host | modal)
        |                                                |
   env (training/registry.py)                     training/train.py --experiment ...
   preset (configs/train/presets.toml)                   |
   compute needs, eval bench, target skill        <run>/ params.pkl, progress.jsonl, run.json
                                                         |
                                ops fetch -> training/checkpoints/<run>
                                ops export -> policies/<candidate>.npz
                                ops eval   -> the experiment's bench, --json
                                adopt      -> policies/<skill>.npz + registry.toml + a decision
```

- **Environment registry** (`training/registry.py`): every environment is registered once with its factory,
  its domain-randomisation function, the skill it trains and a one-line description. The trainer, the CLI and
  the tests all read it; there is no second list.
- **Presets are data** (`configs/train/presets.toml`): PPO sizes and schedules. An experiment names a preset and
  may override any key.
- **Experiments are data** (`configs/experiments/*.toml`): env, preset, overrides, compute needs (minimum VRAM,
  number of parallel robots on a small GPU, hard timeout) and how the result is judged (bench and baseline).
- **Every run is reproducible:** jobs run from a pushed git commit, and each run writes `run.json` with the
  commit, whether the tree was dirty, the host, the full resolved config and its final metrics.
- **Adoption gate:** a candidate replaces a policy only after its experiment's bench beats the current one;
  `policies/registry.toml` records which run, which commit and which decision each adopted policy came from.
  Benches print every success count with its 95 % range, and `sim/benches/stats.py` says whether two results
  differ or the comparison is inconclusive. Each run names its seed, so a change can be trained several times.

## 4. Compute

One job spec, three kinds of backend, one ledger:

| Backend | Where | How a job runs |
|---|---|---|
| `local` | this laptop | a background process from a snapshot of the commit |
| `ssh` | any machine on the Tailscale network: msi-1, a friend's rented GPU, a future server | the job script goes over SSH (into WSL on Windows hosts); the worker snapshots the commit with `git archive`, trains in `~/pinch-jobs/<run>/`, and results come back as a tar stream |
| `modal` | Modal cloud GPUs (A10G default) | `training/modal_train.py` with a hard timeout per run; checkpoints on the `pinch-checkpoints` volume |

- **Hosts** are described in `ops/hosts.local.toml` (addresses are private, so the file is gitignored; the
  template is `ops/hosts.example.toml`). Adding a friend's GPU is one entry plus `python -m ops hosts bootstrap
  <name>`, which clones the repo and builds the training environment there.
- **The ledger** (`.ops/jobs.json`, local) lists every job: spec, backend, host, commit, state. `ops jobs`
  refreshes it from every backend; `ops dispatch` starts queued jobs on whichever host is free, Modal only when
  the job allows it (Modal costs money: each job states its timeout).
- **Isolation:** each job trains from its own snapshot of its own commit, so two jobs on one host, or a job and
  ongoing development, never disturb each other.

## 5. Parallel work

- **One task = one git worktree = one branch** (`python -m ops worktree new <task>` →
  `../pinch-wt/<task>` on `feat/<task>`). Several agents, or an agent and a human, work at once without touching
  each other's files.
- **Areas** (`ops/areas.toml`) name who may change what in a task: mind, body, twin, a training environment,
  firmware, CAD. Two tasks that need the same area run one after the other.
- **Training from any branch:** a task pushes its branch and submits an experiment; the job pins that commit.
  The ledger shows every agent what is training where.
- **Merging:** a branch merges into `master` after `ops check` passes. If the golden trace changes, the branch
  says why in its decision entry and re-records it once, after rebasing.

The workflows (start a task, run an experiment, add a GPU, adopt a policy) are in [docs/dev/](dev/).

## 6. Agent skills

`.claude/skills/` holds two kinds:
- **`pinch-*`**: this project's workflows (train, adopt a policy, parallel task, add a GPU host, new environment,
  decision entry), written to Anthropic's skill format.
- **`mujoco-*`**: Google DeepMind's official MuJoCo skills (accelerated MJX/MuJoCo Warp, CPU/GPU parity, Python
  bindings, MjSpec editing, rendering), copied from the installed mujoco package by `python -m ops skills sync`
  so they match the pinned version.

Official Anthropic plugins enabled for the project are listed in `.claude/settings.json` (Python language
server, PR review, CLAUDE.md upkeep, security guidance, skill authoring). NVIDIA's official Isaac Lab skills are
not vendored: they describe Isaac Lab's API, which Pinch doesn't use yet; they come in with that migration.

## 7. Why these tools (evidence)

Every tool is an established, actively maintained project (GitHub stars and last commit checked 2026-11-20).

| Need | Choice | Evidence | Rejected |
|---|---|---|---|
| Lint and format | **ruff** | astral-sh/ruff, 51k stars, active | flake8 + isort + black: three tools for one job |
| Tests | **pytest**-compatible suite, zero-dependency runner | pytest-dev/pytest, 15k stars | - |
| Commit hooks | **pre-commit** | pre-commit/pre-commit, 16k stars | hand-written git hooks |
| Python installs on workers | **uv** | astral-sh/uv, 92k stars | pip alone: slow, no interpreter management |
| Private network to GPUs | **Tailscale** | tailscale/tailscale, 39k stars; node sharing for a friend's machine | opening SSH ports, VPN servers |
| Cloud GPUs | **Modal** (in use), **SkyPilot** for any other cloud | skypilot-org/skypilot, 10.8k stars, UC Berkeley, NSDI '23 paper | dstack (2.3k stars); Ray (44k) needs a Linux head node and open ports |
| Own GPUs (incl. Windows) | small **SSH backend** (`ops/backends/ssh.py`) | SkyPilot's SSH Node Pools install k3s + NVIDIA GPU Operator, and WSL2 is not on the GPU Operator's platform list | SkyPilot SSH pools on WSL2 |
| Configs | **TOML** files read with Python's standard `tomllib` | PEP 680; no dependency on every worker | Hydra (10.8k stars): powerful, but adds a dependency and indirection the current configs don't need |
| Run records | `run.json` per run (commit, host, config, metrics) | the minimum MLflow/W&B track; importable into MLflow (29k stars) later | a tracking server reachable from every worker |
| Training stack | Brax PPO on MJX today; **mjlab** (MuJoCo Warp + RSL-RL) for new environments | mujocolab/mjlab 3.4k stars (arXiv 2601.22074), with two templates on it: pollen-robotics/microduck_rl (2.9k, a small servo robot) and KingKongRobotics/jumper (3.2k, a crab). Brax's README says only its training code is still maintained, and MuJoCo Playground 1.0 (November 2026) moved to PyTorch + MuJoCo Warp + RSL-RL (checked 2026-11-21, [the audit](research/PROJECT_AUDIT.md)) | Isaac Lab 3.0 as the main stack: a heavier install. It now ships Newton with a MuJoCo-Warp solver, so it stays the fallback |
| Training and test worlds | **ProcTHOR-10K's ready-made homes** (allenai/procthor-10k, Apache-2.0, arXiv 2206.06994: 12,000 houses of 1-10 rooms) furnished with **AI2-THOR's own models** (Apache-2.0, read from the Habitat glb mirror haosulab/AI2THOR one range request at a time), converted once to MuJoCo (`sim/worlds/procthor.py`: coloured parts, 5 cm voxel boxes for contacts); our seeded generator (`sim/worlds/generate.py`) picks the house and Pinch's surface and adds everything else, and builds its own simple homes when the download is absent; held-out splits as in Procgen (arXiv 1912.01588), our test worlds drawn only from ProcTHOR's test houses | the only source tried that has whole furnished homes under an open licence and loads without Unity or Blender: ReplicaCAD and HSSD (non-commercial), RoboCasa and NVlabs scene_synthesizer (kitchens only), Infinigen Indoors and BEHAVIOR-1K (Blender / Isaac Sim) |
| Everyday things, people, furniture looks | Google Scanned Objects in MuJoCo (kevinzakka/mujoco_scanned_objects, MIT; scans CC-BY 4.0, arXiv 2204.11918); Microsoft Rocketbox avatars (MIT, 42 rigged photo-textured people, 4 of them children) as MuJoCo skins, MakeHuman bodies (CC0) as the fallback; SFHQ synthetic faces (CC0/MIT) | all fetched on demand into `models/` (not in git) except 29 small face crops; checked by `tests/test_worlds.py` without them | SMPL / SMPL-X (non-commercial licence), Mixamo (no redistribution), Microsoft Face Synthetics (research-only) |
| Training terrains | mjlab's terrain generator in curriculum mode, families at Pinch's scale, Rudin et al.'s promotion rule (arXiv 2109.11978) | arXiv 2610.40364 (14 curricula, 5 quadrupeds): the success-driven terrain curriculum is most often in the top group, uniform sampling never | PAIRED / ACCEL (unsupervised environment design): rarely better in that comparison, and ACCEL's level edits don't fit batched GPU worlds |
| Behaviour scenarios | TOML files with sampled worlds, timelines and assertions (`sim/worlds/scenario.py`), Scenic's semantics (arXiv 1809.09310: require by rejection) and CARLA ScenarioRunner's criteria, temporal checks (eventually / always / never) | BerkeleyLearnVerify/Scenic 389 stars, carla-simulator/scenario_runner 687 stars, both active | rtamt / stlpy (signal temporal logic, < 100 stars): more than these checks need |
| Bench statistics | Wilson range, Fisher's exact and McNemar tests in `sim/benches/stats.py` | pure Python (so it runs in CI and on the robot), pinned to SciPy's values in `tests/test_stats.py`; scipy/scipy 15k stars is the reference | rliable (google-research, archived since 2024): built for many tasks with few seeds, not one yes/no rate |
| Agent instructions | **AGENTS.md** + CLAUDE.md, skills in `.claude/skills/` | the format used by mjlab, MuJoCo Playground and MuJoCo Warp; anthropics/skills (198k stars) | - |

## 8. Worlds, people and scenarios (phase 5)

```
world(split, i) -> World (plain data)          sim/worlds/generate.py: a home of 1-3 rooms with doorways and
   |                                            furniture; Pinch's surface (desk, table or floor) centred at the
   |                                            origin; tray, its things, 0-10 everyday things, rug or desk mat,
   |                                            lights, people with a style. The home is a ready-made ProcTHOR-10K
   |                                            house when fetched (procthor.py), else generate.py's own
   +-> PinchSim(world=w)                        the twin builds it in (sim/worlds/build.py) and the mind gets the
   |     + Life(sim, camera=True)               surface; people move and touch (people.py, humans.py); the eye
   |                                            camera's 160-degree fisheye view is rendered (fisheye.py) and the
   |                                            robot's own face pipeline runs on it (pinch/faces.py)
   +-> scenario files (sim/scenarios/*.toml)    each seed: a held-out world, sampled event times, words and
   |     sim/benches/worlds.py                  places, temporal assertions; pass rates with 95 % ranges
   +-> training/mjlab_pinch/terrains.py         the walking task on indoor terrain families with a curriculum
```

Splits: training worlds 0-999, validation 0-99, held-out test 0-199, each world's random stream seeded by
(split, index) and the test layouts checked disjoint from the training ones. Benches judge on test worlds only;
nothing is tuned on them. The heavy assets (scanned objects, ProcTHOR homes, Rocketbox people) are fetched with
`python -m sim.worlds.assets fetch`, `python -m sim.worlds.procthor fetch` and `python -m sim.worlds.rocketbox fetch`; without them worlds use their own simple homes, boxes and
capsule people, so every test and CI run works from a fresh checkout.

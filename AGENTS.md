# Working on Pinch (for coding agents and people)

Pinch is a hand-sized robot crab pet, built in software first: CAD, a MuJoCo digital twin, learned locomotion
and motion (Brax PPO on MJX), the pet mind, and firmware that runs the same mind and body on the robot.

Read first: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), then the workflow you need in [docs/dev/](docs/dev/).

## Commands

```powershell
.venv\Scripts\python -m ops check                 # guard + ruff + all tests: must pass before any push
.venv\Scripts\python tests\run.py <filter>        # some tests (architecture, golden, ops, states, ...)
.venv\Scripts\python -m ops envs                  # training environments
.venv\Scripts\python -m ops submit <exp> --host any --cloud-fallback   # train on the first free GPU
.venv\Scripts\python -m ops jobs -r               # what is training where
.venv\Scripts\python -m ops worktree new <task>   # isolate a parallel task
```

Two environments: `.venv` (twin, tests, CAD, ops) and `.venv-rl` (JAX training, export). Never install the
project as a package: entry points put their own checkout on `sys.path`, which is what lets worktrees share a venv.

## Rules

- **Evidence first.** Back each technical decision with research papers or well-maintained, widely used projects
  (check stars and recent activity); cite them in `docs/DECISIONS.md`. Prefer an established tool over new code.
- **Layers.** crab ← training; crab ← pinch ← robot; nothing imports ops; pinch/robot never import training
  (`tests/test_architecture.py`). Learned skills reach the robot only as `policies/*.npz`.
- **Behaviour is pinned.** The golden trace must stay identical through refactors; re-record only for intended
  behaviour changes, with the reason in the decision.
- **Measure.** Skill changes are judged on the benches (`sim/benches/`, many seeds/layouts), never on one run.
  Policies are adopted only through docs/dev/TRAINING.md and recorded in `policies/registry.toml`.
- **Parallel work.** One task per worktree and branch; claim areas from `ops/areas.toml`; shared files (golden
  trace, DECISIONS.md, README, policy registry) are edited only by the integrator when merging.
- **Public repository.** Never commit the local-only documents or machine addresses (`ops/guard.py` enforces it),
  never add AI attribution lines to commits, commit with the project's noreply address.
- **Paid compute.** Runs are not cut short: a job ends when its steps are done. `timeout_min` (24 h) is only a
  last-resort net. Check every long run about 4 h in (`ops jobs -r`, `ops logs`): stop it if it's stuck, diverging
  or no longer improving, and stop anything not needed.

## Agent skills in this repo (`.claude/skills/`)

- `pinch-*`: this project's workflows (train, adopt a policy, start a parallel task, add a GPU host, add an env).
- `mujoco-*`: Google DeepMind's official MuJoCo skills, vendored from the installed mujoco package
  (`python -m ops skills sync`; do not edit by hand).

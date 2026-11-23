# Development

## Set up a clone

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt ruff pytest pre-commit
.venv\Scripts\pre-commit install -t pre-commit -t commit-msg
.venv\Scripts\python -m ops skills sync          # official MuJoCo agent skills for the installed mujoco
```

Training needs a second environment, `.venv-rl`, from `requirements-train.txt` (pinned JAX/Brax/MJX; on an NVIDIA
Linux or WSL machine add `"jax[cuda12]==0.9.2"`). Most work never needs it: GPU jobs run elsewhere (TRAINING.md).

## The checks

| Command | What | When |
|---|---|---|
| `python -m ops check` | private-file guard, ruff, the whole test suite, the robot's two self-tests | before every push; CI runs the same |
| `python tests/run.py <filter>` | some test files (`architecture`, `golden`, `ops`, ...) | while working |
| `pytest` | the same tests under pytest | if you prefer it |
| `python robot/firmware_test.py` | firmware on emulated hardware (about 45 checks, a minute) | part of `ops check`; alone when touching robot/ or the mind |
| `python robot/selftest.py` | servo bus, calibration, drivers, safety (40 checks, seconds) | part of `ops check`; alone when touching robot/ |
| `python sim/benches/<bench>.py --json out.json` | behaviour and skill benches | when a change could move a number |

Pre-commit runs ruff, the TOML/YAML checks, the guard and the layering/ops tests on each commit (seconds).

CI (`.github/workflows/tests.yml`) runs the same steps on Linux for every push and pull request. MuJoCo needs
somewhere to draw, so CI starts a virtual display (Xvfb); without one MuJoCo aborts the Python process instead of
raising. A push is finished when its CI run is green: look with `gh run list --limit 3`, and read a failure with
`gh run view <id> --log-failed`. Two checks are platform-bound and say so when they skip: the golden trace is
exact only on the platform that recorded it (Windows), and the face-recognition check needs the 39 MB models
that `deploy/install.sh` fetches.

## Rules that keep the code clean

- **Layers** (`tests/test_architecture.py`): crab → nothing of ours; training → crab only; pinch and robot never
  import training; nothing imports ops.
- **Golden trace** (`tests/test_golden.py`): a refactor must leave a seeded session identical. If behaviour is
  meant to change, re-record it once (`python tests/test_golden.py --record`) and say why in the decision.
- **Decisions**: every non-trivial change gets a numbered entry in `docs/DECISIONS.md` with what was measured,
  and the sources (papers, well-maintained projects) behind the approach.
- **Private files stay local**: the guard refuses the local-only documents, `ops/hosts.local.toml`, tailnet
  addresses and AI attribution lines. Commits use the project's noreply address.
- **Style**: ruff (line length 120). Comments say why, not what.

## Layout

See [ARCHITECTURE.md](../ARCHITECTURE.md). In short: `pinch/` the pet, `crab/` the robot, `robot/` the firmware,
`sim/` the world and benches, `training/` learning, `configs/` data, `ops/` operations, `policies/` adopted skills.

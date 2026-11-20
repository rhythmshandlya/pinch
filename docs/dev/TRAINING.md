# Training and adopting a policy

The loop: **experiment file → job → checkpoint → numpy policy → bench → adopt (or not) → decision.**

## 1. Describe the experiment

Copy the closest file in `configs/experiments/` (format in its README). An experiment names an environment from
`training/registry.py` (`python -m ops envs`), a preset from `configs/train/presets.toml`, any overrides, its
compute needs (with a last-resort timeout) and the benches that judge it.

A new environment: add its module under `training/`, register it once in `training/registry.py`, add an
experiment and run `python tests/run.py ops` (it checks every experiment resolves).

## 2. Run it

Commit and push first: every job trains a pushed commit, and its `run.json` records which.

```powershell
.venv\Scripts\python -m ops submit configs/experiments/smoke.toml                       # this laptop (CPU, minutes)
.venv\Scripts\python -m ops submit configs/experiments/righting-smooth.toml --host msi-1   # a Tailscale GPU
.venv\Scripts\python -m ops submit configs/experiments/terrain-push.toml --backend modal   # Modal (paid)
.venv\Scripts\python -m ops submit configs/experiments/grasp.toml --host any --cloud-fallback   # first free GPU
.venv\Scripts\python -m ops jobs -r            # every job, live state, latest reward/success
.venv\Scripts\python -m ops logs <run>
.venv\Scripts\python -m ops stop <run>
```

On a GPU under 12 GB the experiment's `small_gpu_num_envs` is used automatically.

## 3. Judge it

```powershell
.venv\Scripts\python -m ops fetch <run>                  # -> training/checkpoints/<run>
.venv\Scripts\python -m ops export <run> <slot>_<tag>    # -> policies/<slot>_<tag>.npz (checks it matches Brax)
.venv\Scripts\python -m ops eval <experiment> <slot>_<tag>   # the experiment's benches, JSON in .ops/eval/
```

Compare with the incumbent's numbers in `policies/registry.toml` (re-run the same benches on the incumbent when
in doubt: the twin's contacts are chaotic, so compare pass rates over many layouts, never single runs).

## 4. Adopt it

Only if it beats the incumbent on the experiment's benches without losing elsewhere (the golden trace and the
behaviour scenarios still pass):

1. copy `policies/<slot>_<tag>.*` over `policies/<slot>.*`;
2. update the slot in `policies/registry.toml` (run, steps, env, evidence, decision number);
3. add the decision to `docs/DECISIONS.md` with the numbers, re-record the golden trace if the walk changed;
4. `python -m ops check`, commit, push.

Candidates that lose stay local (gitignored) and get a line in the decision log.

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

One run is one draw of the dice. Before believing that a change to an environment or a reward helped, train it
under at least three seeds: `python -m ops submit <experiment> --seed 1` (then 2, 3; the default is 0, or the
`seed` in the file's `[experiment]` section). The seed is in each run's `run.json`.

## 3. Judge it

```powershell
.venv\Scripts\python -m ops fetch <run>                  # -> training/checkpoints/<run>
.venv\Scripts\python -m ops export <run> <slot>_<tag>    # -> policies/<slot>_<tag>.npz (checks it matches Brax)
.venv\Scripts\python -m ops eval <experiment> <slot>_<tag>   # the experiment's benches, JSON in .ops/eval/
```

Compare with the incumbent's numbers in `policies/registry.toml` (re-run the same benches on the incumbent when
in doubt: the twin's contacts are chaotic, so compare pass rates over many layouts, never single runs).

Every bench prints a success count with its 95 % range, for example `11/24 = 46 % (95 % range 28-65 %)`. To
compare two results:

```powershell
.venv\Scripts\python sim/benches/stats.py 11/24 2/24        # -> p = 0.008, different
.venv\Scripts\python sim/benches/stats.py 19/48 14/48       # -> p = 0.390, inconclusive
.venv\Scripts\python sim/benches/stats.py --need 0.46 0.67  # -> about 87 layouts per policy
```

With 24 layouts only a gap of about 30 points can be told from chance. Use 100 or more layouts for an adoption
decision, the same layouts for both policies (`stats.paired` counts only the layouts where they disagree).
A bench with pass criteria exits non-zero when it fails (the robustness bench does when any verdict is FAIL),
and `ops eval` then fails too.

## 4. Adopt it

Only if it beats the incumbent on the experiment's benches without losing elsewhere (the golden trace and the
behaviour scenarios still pass). "Beats" means the comparison says `different`; `inconclusive` is not a win:

1. copy `policies/<slot>_<tag>.*` over `policies/<slot>.*`;
2. update the slot in `policies/registry.toml` (run, steps, env, evidence, decision number);
3. add the decision to `docs/DECISIONS.md` with the numbers, re-record the golden trace if the walk changed;
4. `python -m ops check`, commit, push.

Candidates that lose stay local (gitignored) and get a line in the decision log.

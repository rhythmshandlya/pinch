# Parallel work: several agents (and people) at once

## The unit of work

**One task = one git worktree = one branch = one PR.**

```powershell
.venv\Scripts\python -m ops worktree new tidy-ball      # ../pinch-wt/tidy-ball on feat/tidy-ball
cd ..\pinch-wt\tidy-ball
..\..\robotic-crab\.venv\Scripts\python tests\run.py    # this checkout's venv works in every worktree
```

In the Claude desktop app, a task started as its own session with a worktree gets the same isolation
automatically; the rules below still apply.

## Rules

1. **Claim areas.** A task names the areas it changes (`ops/areas.toml`: mind, body, twin, training, firmware,
   cad, ...). Two tasks that claim the same area run one after the other, not at once.
2. **Shared files are the integrator's.** The golden trace, `docs/DECISIONS.md`, the README and
   `policies/registry.toml` are edited only when merging, after rebasing on `master`: a task writes its decision
   text in its PR description and the integrator numbers and appends it.
3. **Train from the branch.** Push the branch, then `python -m ops submit <exp> --host any`. The job pins the
   commit; the ledger (`python -m ops jobs -r`) shows every task what is training where. Name runs after the task
   (`--run tidy-ball-push-v1`).
4. **Merge only green.** `python -m ops check` passes on the rebased branch. A changed golden trace is re-recorded
   once, by the integrator, with the reason in the decision.
5. **Benches decide.** A task that claims a skill got better shows the bench numbers before and after (same
   layouts, same seeds) in its PR.

## A typical parallel day

| Track | Area | Compute |
|---|---|---|
| Feature: clear the work area before tidying | mind, body | CPU benches on the laptop |
| Training: terrain-push walker | training | Modal A10G |
| Training: gentler get-up | training (another env) | msi-1 |
| Framework: unknown-object curiosity | mind, senses, twin | laptop |
| Hardware: OLED mount tweak | cad | none |

Two training tracks may share the `training` area when they only add their own environment module and
experiment file; the registry line is the one shared edit (trivial to merge).

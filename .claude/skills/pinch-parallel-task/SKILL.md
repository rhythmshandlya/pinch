---
name: pinch-parallel-task
description: Start, work in, and merge an isolated parallel task on Pinch so several agents can build features at once without colliding - git worktree and branch per task, claimed code areas, shared-file rules, checks before merge. Use when splitting work across agents or sessions, starting a feature branch, or merging a finished task back to master.
---

# A parallel task

Reference: `docs/dev/PARALLEL.md`; areas in `ops/areas.toml`.

## Start

1. Name it (lowercase-with-dashes) and list the areas it changes (`ops/areas.toml`). If another running task
   claims one of them, wait or re-scope.
2. `python -m ops worktree new <task>` -> `../pinch-wt/<task>` on `feat/<task>` (or a worktree session in the
   desktop app). Use the main checkout's `.venv` from inside it.

## Work

- Stay inside the claimed areas. Shared files - `tests/golden/`, `docs/DECISIONS.md`, `README.md`,
  `policies/registry.toml` - are not edited in task branches; put the proposed decision text in the PR body.
- Training from the branch: push it, then pinch-train with `--run <task>-...`; the ops ledger shows it to others.
- Commit small, with the project's noreply address and no AI attribution lines.

## Finish

1. Rebase on `master`; `python -m ops check` must pass.
2. If the golden trace fails because behaviour changed on purpose, say so in the PR; the integrator re-records it.
3. Open the PR with: what changed, areas touched, bench numbers before/after, the proposed decision text.
4. After merge: `python -m ops worktree remove <task>`.

---
name: pinch-adopt-policy
description: Judge a trained Pinch policy against the one in use and adopt it only if it wins - export a checkpoint to numpy, run the experiment's benches, compare with policies/registry.toml, update the slot and the decision log. Use after a training run finishes, or when asked whether a candidate (gait, righting, emote, grasp) is better. Not for launching training (pinch-train).
---

# Adopt (or reject) a policy

Workflow reference: `docs/dev/TRAINING.md` sections 3-4.

## Steps

1. `python -m ops fetch <run>` if not already in `training/checkpoints/<run>`.
2. `python -m ops export <run> <slot>_<tag>` (uses `.venv-rl`; it checks the numpy policy matches Brax).
3. `python -m ops eval configs/experiments/<exp>.toml <slot>_<tag>` - runs the experiment's `[eval]` benches with
   `PINCH_<SKILL>_POLICY=<slot>_<tag>`; JSON lands in `.ops/eval/<slot>_<tag>/`.
4. Compare with the incumbent (`policies/registry.toml` evidence). If the incumbent's numbers came from another
   bench version or seed set, re-run the same command with the incumbent name. Compare pass rates over the whole
   layout/seed set; one run decides nothing.
5. **Adopt only if** it wins on its benches and loses nothing elsewhere: `python tests/run.py` (scenarios, golden)
   and, for walking, robustness on desk/rough/stuck/terrain.
6. Adopting: copy `policies/<slot>_<tag>.{npz,json}` over `policies/<slot>.*`; update the slot in
   `policies/registry.toml`; write the decision (pinch-decision skill) with before/after numbers; re-record the
   golden trace if the walk changed (`python tests/test_golden.py --record`); `python -m ops check`; commit.
7. Rejecting: keep the candidate local (gitignored), add one line to the decision log with its numbers.

## Gotchas

- The twin's contacts are chaotic: tidy needs 24-48 layouts, righting 200 starts, robustness 40 episodes/world.
- The golden trace is recorded on Windows; it skips on Linux. Re-record on the same platform it came from.
- Smoother is not automatically better: check the slew/current numbers the bench prints when a policy is
  meant to be gentler on servos.

@AGENTS.md

## Claude Code specifics

- Skills: use the `pinch-*` skills for project workflows and the vendored `mujoco-*` skills for MuJoCo, MJX and
  MuJoCo Warp questions. Official Anthropic plugins enabled for this project are listed in `.claude/settings.json`.
- Parallel sessions: start each independent task in its own worktree session; keep training jobs in the `ops`
  ledger so every session can see them.
- Machine-specific and private notes live in `CLAUDE.local.md` (gitignored).

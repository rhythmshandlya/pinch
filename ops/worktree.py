"""One task = one git worktree = one branch, so parallel agents never edit the same checkout.

    python -m ops worktree new tidy-ball        # ../pinch-wt/tidy-ball on branch feat/tidy-ball
    python -m ops worktree list
    python -m ops worktree remove tidy-ball     # after the branch is merged

Worktrees share this checkout's .venv and .venv-rl (every entry point puts its own checkout first on sys.path,
so code is never imported from the wrong tree). Local-only files (policies benched here, the private docs) are
not copied; a task that needs them links them explicitly.
"""

import re

from ops import ROOT, gitutil

WT_ROOT = ROOT.parent / "pinch-wt"


def _check(task):
    if not task or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,40}", task):
        raise SystemExit("task names are lowercase words joined by '-', e.g. tidy-ball")


def new(task, base="master"):
    _check(task)
    path = WT_ROOT / task
    if path.exists():
        raise SystemExit(f"{path} exists")
    WT_ROOT.mkdir(exist_ok=True)
    gitutil.git("worktree", "add", "-b", f"feat/{task}", str(path), base)
    return (f"worktree {path} on feat/{task} (from {base})\n"
            f"  run things from there with this checkout's venv: {ROOT / '.venv'}\n"
            f"  when done: commit, push, `python -m ops check`, open a PR, then `python -m ops worktree remove {task}`")


def remove(task):
    _check(task)
    gitutil.git("worktree", "remove", str(WT_ROOT / task))
    print(f"removed {WT_ROOT / task} (branch feat/{task} kept)")


def list_():
    return gitutil.git("worktree", "list")

"""The bash job script every SSH host (and the local backend under WSL/Linux) runs. One place, so all workers
behave the same: snapshot the commit with `git archive` (no worktree, no shared state between jobs), train with a
hard timeout, write the exit code."""

JOB_ROOT = "$HOME/pinch-jobs"

PREPARE = r"""
set -euo pipefail
JOB="{job_root}/{run}"; REPO={repo}
mkdir -p "$JOB/out"
if [ ! -d "$REPO/.git" ]; then git clone -q {remote} "$REPO"; fi
git -C "$REPO" fetch -q origin
git -C "$REPO" cat-file -e {sha}^{{commit}}
rm -rf "$JOB/src" && mkdir -p "$JOB/src"
git -C "$REPO" archive {sha} | tar -x -C "$JOB/src"
cat > "$JOB/job.sh" <<'JOBEOF'
#!/usr/bin/env bash
JOB="{job_root}/{run}"; REPO={repo}
cd "$JOB/src"
export PINCH_GIT_SHA={sha} PINCH_GIT_DIRTY=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 PYTHONUNBUFFERED=1
echo $$ > "$JOB/pid"
timeout {timeout_min}m "$REPO/{python}" -m training.train --experiment {experiment} --run {run} --out-root "$JOB/out" {extra} > "$JOB/train.log" 2>&1
echo $? > "$JOB/exit_code"
JOBEOF
chmod +x "$JOB/job.sh"
echo prepared "$JOB"
"""

# Linux: detach from the SSH session
LAUNCH_LINUX = r"""
JOB="{job_root}/{run}"
setsid nohup bash "$JOB/job.sh" > /dev/null 2>&1 < /dev/null &
sleep 1; echo launched
"""

STATUS = r"""
JOB="{job_root}/{run}"
if [ ! -d "$JOB" ]; then echo "state=lost"; exit 0; fi
if [ -f "$JOB/exit_code" ]; then echo "exit_code=$(cat $JOB/exit_code)";
elif [ -f "$JOB/pid" ] && kill -0 "$(cat $JOB/pid)" 2>/dev/null; then echo "state=running";
elif [ -f "$JOB/pid" ]; then echo "state=lost"; else echo "state=starting"; fi
P="$JOB/out/{run}/progress.jsonl"
if [ -f "$P" ]; then echo "last=$(tail -1 $P)"; fi
"""

LOGS = r"""tail -n {n} "{job_root}/{run}/train.log" 2>/dev/null || echo "(no log yet)" """

FETCH = r"""tar -C "{job_root}/{run}/out" -czf - {run} | base64 -w0"""

STOP = r"""
JOB="{job_root}/{run}"
if [ -f "$JOB/pid" ]; then pkill -TERM -P "$(cat $JOB/pid)" 2>/dev/null; kill -TERM "$(cat $JOB/pid)" 2>/dev/null; fi
echo 143 > "$JOB/exit_code"; echo stopped
"""


def parse_status(out):
    """STATUS output -> (state, last metrics dict)."""
    import json
    facts = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    if "exit_code" in facts:
        code = facts["exit_code"].strip()
        state = "done" if code == "0" else ("stopped" if code in ("143", "124") else "failed")
        if code == "124":
            state = "failed"               # hit its timeout: the policy so far is still in params.pkl
    else:
        state = facts.get("state", "lost")
        state = "running" if state == "starting" else state
    last = {}
    if facts.get("last"):
        try:
            last = json.loads(facts["last"])
        except ValueError:
            pass
    return state, last

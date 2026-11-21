"""Train the mjlab Pinch task on an SSH host (msi-1 and friends) without touching the Brax job machinery.

    python -m ops.mjlab_run start --host msi-1 --envs 8192 --iterations 300 [--task Pinch-Velocity-Flat] [--name RUN]
    python -m ops.mjlab_run status|logs|stop RUN
    python -m ops.mjlab_run fetch RUN            # -> training/checkpoints/<RUN>/ (checkpoints, policy.onnx, policy.npz)

It is a script next to `ops submit` rather than a trainer key in the experiment files (decision pending, see
docs/dev/MJLAB.md): the mjlab trainer has its own Python environment on the worker (requirements-mjlab.txt,
`~/pinch-mjlab/.venv`), its own flags and its own outputs. Everything else is shared with the SSH backend: the job
is the pushed commit unpacked with `git archive` into `~/pinch-jobs/<run>/src`, on Windows it runs under the same
one-shot Scheduled Task (the process survives the SSH session), and status/logs/fetch/stop are ops.backends.ssh's.

The job: train with `--agent.logger tensorboard`, export the last checkpoint (ONNX with the normaliser baked in,
numpy .npz; it must match torch to 1e-4), and sample GPU memory every 5 s into gpu_mem.csv.
It lives under ops/ because training/ may not import ops (tests/test_architecture.py).
"""

import argparse
import sys
from datetime import datetime

from ops import ROOT, gitutil
from ops.backends import jobscript as js
from ops.backends import ssh
from ops.hosts import load_hosts, run_bash
from ops.jobs import Job, now

MJLAB_PYTHON = "$HOME/pinch-mjlab/.venv/bin/python"

PREPARE = r"""
set -euo pipefail
JOB="{job_root}/{run}"; REPO={repo}
mkdir -p "$JOB/out/{run}"
if [ ! -d "$REPO/.git" ]; then git clone -q {remote} "$REPO"; fi
git -C "$REPO" fetch -q origin "+refs/heads/{branch}:refs/remotes/origin/{branch}"
git -C "$REPO" cat-file -e {sha}^{{commit}}
rm -rf "$JOB/src" && mkdir -p "$JOB/src"
git -C "$REPO" archive {sha} | tar -x -C "$JOB/src"
cat > "$JOB/job.sh" <<'JOBEOF'
#!/usr/bin/env bash
JOB="{job_root}/{run}"; OUT="$JOB/out/{run}"
cd "$JOB/src"
export PINCH_GIT_SHA={sha} PYTHONUNBUFFERED=1
echo $$ > "$JOB/pid"
( while true; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$OUT/gpu_mem.csv"; sleep 5; done ) &
SAMPLER=$!
timeout {timeout_min}m {python} -W ignore -m training.mjlab_pinch.train {task} --env.scene.num-envs {envs} \
  --agent.max-iterations {iterations} --agent.logger tensorboard --agent.run-name {run} --log-root "$OUT/logs" \
  --agent.seed {seed} {extra} > "$JOB/train.log" 2>&1
CODE=$?
if [ $CODE -eq 0 ] || [ $CODE -eq 124 ]; then
  CKPT=$(ls -t "$OUT"/logs/*/*/model_*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then {python} -W ignore -m training.mjlab_pinch.export "$CKPT" "$OUT/policy" --task {task} >> "$JOB/train.log" 2>&1 || CODE=3; fi
fi
kill $SAMPLER 2>/dev/null
cp "$JOB/train.log" "$OUT/train.log"
echo $CODE > "$JOB/exit_code"
JOBEOF
chmod +x "$JOB/job.sh"
echo prepared "$JOB"
"""


def _job(host, run, sha, timeout_min):
    return Job(run=run, experiment="mjlab", backend="ssh", host=host.name, sha=sha, timeout_min=timeout_min,
               state="running", started=now())


def start(a):
    host = load_hosts()[a.host]
    sha, branch = gitutil.head(), gitutil.branch()
    if not gitutil.pushed(sha):
        sys.exit(f"commit {sha[:8]} is not pushed: a job trains a pushed commit (git push origin {branch})")
    run = a.name or f"mjlab-{datetime.now():%Y%m%d-%H%M%S}"
    job = _job(host, run, sha, a.timeout_min)
    script = PREPARE.format(job_root=host.job_root, run=run, repo=host.repo, remote=gitutil.remote_url(), sha=sha,
                            branch=branch, timeout_min=a.timeout_min, python=a.python, task=a.task, envs=a.envs,
                            iterations=a.iterations, seed=a.seed, extra=" ".join(a.extra))
    print(run_bash(host, script, timeout=300).strip())
    if host.os == "windows-wsl":
        print(ssh.launch_windows(job, host).strip())
    else:
        print(run_bash(host, ssh._fmt(js.LAUNCH_LINUX, job, host)).strip())
    print(f"started {run} at {sha[:8]} ({a.task}, {a.envs} envs, {a.iterations} iterations)")


def _existing(a):
    host = load_hosts()[a.host]
    return _job(host, a.run, "", 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--host", default="msi-1")
    s.add_argument("--task", default="Pinch-Velocity-Flat")
    s.add_argument("--envs", type=int, default=8192)
    s.add_argument("--iterations", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--name")
    s.add_argument("--timeout-min", type=int, default=30, help="last-resort stop (this is the free laptop)")
    s.add_argument("--python", default=MJLAB_PYTHON)
    s.add_argument("extra", nargs="*", help="more mjlab train flags")
    for name in ("status", "logs", "stop", "fetch"):
        q = sub.add_parser(name)
        q.add_argument("run")
        q.add_argument("--host", default="msi-1")
    a = p.parse_args(argv)
    if a.cmd == "start":
        return start(a)
    job = _existing(a)
    if a.cmd == "status":
        state, last, raw = ssh.status(job)
        print(state, raw)
    elif a.cmd == "logs":
        print(ssh.logs(job, 60))
    elif a.cmd == "stop":
        print(ssh.stop(job))
    else:
        dest = ROOT / "training" / "checkpoints" / a.run
        ssh.fetch(job, dest)
        print("fetched to", dest)


if __name__ == "__main__":
    sys.exit(main() or 0)

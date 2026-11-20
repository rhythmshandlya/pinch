"""Modal cloud GPUs (paid: every job carries a hard timeout and an explicit GPU type).

Drives training/modal_train.py through the Modal CLI. The image carries this checkout's files, so submit
requires a clean tree at the job's commit (ops.cli enforces it) and passes the commit to the run record.
"""

import json
import os
import re
import subprocess
import sys

from ops import ROOT
from ops.jobs import now

SCRIPT = "training/modal_train.py"


def _modal(*args, env=None, timeout=600):
    py = os.environ.get("PINCH_MODAL_PYTHON", sys.executable)
    r = subprocess.run([py, "-m", "modal", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8", **(env or {})))
    if r.returncode != 0:
        raise RuntimeError(f"modal {' '.join(args[:2])}: {r.stderr.strip()[-800:] or r.stdout.strip()[-800:]}")
    return r.stdout


def submit(job):
    args = ["run", "--detach", f"{SCRIPT}::launch", "--experiment", job.experiment, "--run", job.run,
            "--git-sha", job.sha]
    if job.num_envs:
        args += ["--num-envs", str(job.num_envs)]
    if job.timesteps:
        args += ["--timesteps", str(job.timesteps)]
    if job.dirty:
        args += ["--git-dirty"]
    out = _modal(*args, env=dict(PINCH_GPU=job.gpu or "A10G", PINCH_TIMEOUT_MIN=str(job.timeout_min)))
    app = re.search(r"/apps/[^/\s]+/[^/\s]+/(ap-[A-Za-z0-9]+)", out)
    call = re.search(r'"call_id":\s*"([^"]+)"', out)
    job.state, job.started = "running", now()
    job.handle = dict(app_id=app.group(1) if app else "", call_id=call.group(1) if call else "")
    return job


def status(job):
    out = _modal("run", f"{SCRIPT}::status", "--run", job.run, timeout=300)
    line = next((ln for ln in reversed(out.splitlines()) if ln.startswith("{")), "{}")
    st = json.loads(line)
    rec = st.get("record") or {}
    last = rec.get("last") or {}
    if not last and st.get("progress"):             # runs started before run.json existed
        try:
            last = json.loads(st["progress"][-1])
        except ValueError:
            pass
    state = rec.get("state", "running" if not rec else "lost")
    if state == "running" and job.handle.get("app_id") and not _app_running(job.handle["app_id"]):
        state = "failed"            # the app ended without writing a final record: timeout or crash
    return state, last, json.dumps(st)[:2000]


def _app_running(app_id):
    try:
        out = _modal("app", "list", "--json", timeout=120)
        apps = json.loads(out)
    except (RuntimeError, ValueError):
        return True                 # can't tell: don't declare it dead
    for a in apps:
        if a.get("app_id") == app_id:
            return not a.get("stopped_at") and a.get("state") != "stopped"
    return False


def logs(job, n=40):
    app = job.handle.get("app_id")
    if not app:
        return "(no app id recorded)"
    try:
        return "\n".join(_modal("app", "logs", app, timeout=20).splitlines()[-n:])
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        return f"(logs unavailable: {e})"


def fetch(job, dest):
    _modal("run", f"{SCRIPT}::fetch", "--run", job.run, "--dest", str(dest), timeout=900)
    return dest


def stop(job):
    app = job.handle.get("app_id")
    if not app:
        raise RuntimeError("no Modal app id recorded for this job; stop it from the Modal dashboard")
    return _modal("app", "stop", "--yes", app, timeout=120)

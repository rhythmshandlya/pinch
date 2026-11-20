"""This machine. Trains from a snapshot of the commit (`git archive`) into .ops/local/<run>/, so development in
the working tree can carry on while it runs. Uses .venv-rl (Windows or Linux layout)."""

import io
import os
import shutil
import subprocess
import sys
import tarfile

from ops import ROOT, STATE
from ops.backends.jobscript import parse_status
from ops.jobs import now

BASE = STATE / "local"


def _python():
    for p in (ROOT / ".venv-rl" / "Scripts" / "python.exe", ROOT / ".venv-rl" / "bin" / "python"):
        if p.exists():
            return str(p)
    return sys.executable


def _dir(job):
    return BASE / job.run


def submit(job):
    d = _dir(job)
    if d.exists():
        shutil.rmtree(d)
    (d / "out").mkdir(parents=True)
    tar = subprocess.run(["git", "archive", job.sha], cwd=ROOT, capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
        tf.extractall(d / "src", filter="data")
    cmd = [_python(), "-m", "training.train", "--experiment", job.experiment, "--run", job.run,
           "--out-root", str(d / "out")]
    if job.num_envs:
        cmd += ["--num-envs", str(job.num_envs)]
    if job.timesteps:
        cmd += ["--timesteps", str(job.timesteps)]
    env = dict(os.environ, PINCH_GIT_SHA=job.sha, PINCH_GIT_DIRTY="0", PYTHONUNBUFFERED="1")
    log = open(d / "train.log", "w")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
    p = subprocess.Popen(cmd, cwd=d / "src", stdout=log, stderr=subprocess.STDOUT, env=env,
                         creationflags=flags, start_new_session=os.name != "nt")
    (d / "pid").write_text(str(p.pid))
    job.state, job.started, job.handle = "running", now(), dict(pid=p.pid, dir=str(d))
    return job


def _alive(pid):
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def status(job):
    d = _dir(job)
    rec = d / "out" / job.run / "run.json"
    lines = []
    if rec.exists():
        import json
        r = json.loads(rec.read_text())
        if r.get("state") in ("done", "failed"):
            lines.append(f"exit_code={0 if r['state'] == 'done' else 1}")
    if not lines:
        pid = int((d / "pid").read_text()) if (d / "pid").exists() else None
        lines.append("state=running" if pid and _alive(pid) else "state=lost")
    prog = d / "out" / job.run / "progress.jsonl"
    if prog.exists() and prog.read_text().strip():
        lines.append("last=" + prog.read_text().strip().splitlines()[-1])
    state, last = parse_status("\n".join(lines))
    return state, last, "\n".join(lines)


def logs(job, n=40):
    p = _dir(job) / "train.log"
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:]) if p.exists() else "(no log yet)"


def fetch(job, dest):
    src = _dir(job) / "out" / job.run
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return dest


def stop(job):
    pid = job.handle.get("pid")
    if pid:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(pid, 15)
    return "stopped"

"""SkyPilot (skypilot-org/skypilot, UC Berkeley, NSDI '23): cheapest-available cloud GPUs across RunPod, Lambda,
Vast.ai, AWS, GCP, Azure... and Linux machines registered as SSH Node Pools. Optional: `pip install skypilot`
on a Linux/WSL controller, then `sky check`. Each job becomes a managed job (`sky jobs launch`), which SkyPilot
recovers on preemption and tears down at the end.

Why not for msi-1: SSH Node Pools build a k3s cluster with NVIDIA's GPU Operator on each host, and WSL2 is not
on the GPU Operator's platform list; Windows hosts use the plain ssh backend instead.
"""

import json
import shutil
import subprocess

from ops import STATE, gitutil
from ops.jobs import now


def task_yaml(job):
    """The SkyPilot task for a job: same snapshot-and-train contract as the ssh backend."""
    extra = (f" --num-envs {job.num_envs}" if job.num_envs else "") + (f" --timesteps {job.timesteps}" if job.timesteps else "")
    extra += f" --seed {job.seed}" if job.seed else ""
    return f"""name: pinch-{job.run}
resources:
  accelerators: {job.gpu or "A10G"}:1
  disk_size: 50
envs:
  PINCH_GIT_SHA: {job.sha}
  PINCH_GIT_DIRTY: "0"
setup: |
  git clone -q {gitutil.remote_url()} pinch && cd pinch && git checkout -q {job.sha}
  pip install -q uv && uv pip install -q --system -r requirements-train.txt "jax[cuda12]==0.9.2"
run: |
  cd pinch && timeout {job.timeout_min}m python -m training.train --experiment {job.experiment} --run {job.run} --out-root ~/sky_ckpt{extra}; code=$?
  echo "PINCH_RESULT_B64:$(tar -C ~/sky_ckpt -czf - {job.run} | base64 -w0)"
  exit $code
file_mounts: {{}}
"""


def _sky(*args, timeout=900):
    if not shutil.which("sky"):
        raise RuntimeError("SkyPilot is not installed here (pip install 'skypilot[runpod,lambda]', then `sky check`)")
    r = subprocess.run(["sky", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-800:] or r.stdout.strip()[-800:])
    return r.stdout


def submit(job):
    STATE.mkdir(exist_ok=True)
    path = STATE / f"sky-{job.run}.yaml"
    path.write_text(task_yaml(job))
    _sky("jobs", "launch", "-y", "-n", f"pinch-{job.run}", str(path))
    job.state, job.started, job.handle = "running", now(), dict(name=f"pinch-{job.run}", task=str(path))
    return job


def status(job):
    out = _sky("jobs", "queue", "-a", timeout=120)
    line = next((ln for ln in out.splitlines() if f"pinch-{job.run}" in ln), "")
    up = line.upper()
    state = ("done" if "SUCCEEDED" in up else "failed" if "FAILED" in up else
             "stopped" if "CANCELLED" in up else "running" if line else "lost")
    return state, {}, json.dumps(dict(line=line))


def logs(job, n=40):
    return "\n".join(_sky("jobs", "logs", "-n", f"pinch-{job.run}", "--no-follow", timeout=120).splitlines()[-n:])


def fetch(job, dest):
    """The run step prints the run directory as one base64 line at the end (no cloud bucket needed)."""
    import base64
    import io
    import tarfile
    out = _sky("jobs", "logs", "-n", f"pinch-{job.run}", "--no-follow", timeout=600)
    line = next((ln for ln in reversed(out.splitlines()) if "PINCH_RESULT_B64:" in ln), None)
    if not line:
        raise RuntimeError("no result in the job's log yet (still running, or it failed before the end)")
    data = base64.b64decode(line.split("PINCH_RESULT_B64:", 1)[1].strip())
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.startswith(job.run + "/") and ".." not in m.name:
                m.name = m.name[len(job.run) + 1:]
                tf.extract(m, dest, filter="data")
    return dest


def stop(job):
    return _sky("jobs", "cancel", "-y", "-n", f"pinch-{job.run}", timeout=300)

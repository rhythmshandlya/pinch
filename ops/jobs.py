"""The job ledger: every training job, wherever it runs. Local and gitignored (.ops/jobs.json).

A job is one run of one experiment at one commit on one backend. States:
    queued -> running -> done | failed | stopped        (lost: the backend no longer knows it)
"""

import json
import os
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from ops import ROOT, STATE

LEDGER = STATE / "jobs.json"
ACTIVE = ("queued", "running")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Job:
    run: str
    experiment: str                    # repo-relative path of the experiment file
    backend: str                       # local | ssh | modal | skypilot
    host: str = ""                     # ssh host name, or "any" while queued for whichever is free
    sha: str = ""
    dirty: bool = False
    state: str = "queued"
    num_envs: int = 0                  # 0: the preset's
    timesteps: int = 0                 # 0: the experiment's
    timeout_min: int = 90
    gpu: str = ""
    cloud_ok: bool = False             # may dispatch fall back to a paid backend?
    created: str = field(default_factory=now)
    started: str = ""
    updated: str = ""
    handle: dict = field(default_factory=dict)   # backend-specific: pid, app id, job dir...
    last: dict = field(default_factory=dict)     # latest evaluation from progress.jsonl
    note: str = ""


def load():
    if not LEDGER.exists():
        return []
    return [Job(**j) for j in json.loads(LEDGER.read_text(encoding="utf-8"))]


def save(jobs):
    STATE.mkdir(exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(j) for j in jobs], indent=2), encoding="utf-8")
    os.replace(tmp, LEDGER)


def upsert(job):
    jobs = [j for j in load() if j.run != job.run]
    job.updated = now()
    jobs.append(job)
    save(jobs)
    return job


def get(run):
    for j in load():
        if j.run == run:
            return j
    raise KeyError(f"no job {run!r} in the ledger ({LEDGER})")


def read_experiment(path):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    exp = tomllib.loads(p.read_text(encoding="utf-8"))
    return p.resolve().relative_to(ROOT).as_posix(), exp


def unique_run(base, taken=None):
    """base, or base-2, base-3 ... so a new run never overwrites a ledger entry or a local checkpoint."""
    taken = set(taken if taken is not None else (j.run for j in load()))
    ck = ROOT / "training" / "checkpoints"
    n, name = 1, base
    while name in taken or (ck / name).exists():
        n += 1
        name = f"{base}-{n}"
    return name


def wait_s(seconds):
    time.sleep(seconds)

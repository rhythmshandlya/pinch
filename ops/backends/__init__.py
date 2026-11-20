"""Compute backends. Each module exposes the same five functions:

    submit(job)            start it; fills job.handle and job.started; returns the job
    status(job)            -> (state, last_metrics, detail): state in running | done | failed | stopped | lost
    logs(job, n)           -> the last n lines of its training log
    fetch(job, dest)       copy the run directory (params.pkl, progress.jsonl, config.json, run.json) to dest
    stop(job)              stop it early
"""

import importlib

NAMES = ("local", "ssh", "modal", "skypilot")


def get(name):
    if name not in NAMES:
        raise ValueError(f"unknown backend {name!r}; known: {', '.join(NAMES)}")
    return importlib.import_module(f"ops.backends.{name}")

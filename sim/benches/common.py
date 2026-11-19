"""What every bench shares: the process pool and the JSON output."""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def run_parallel(fn, jobs, workers=None, chunksize=1):
    """fn over jobs in worker processes, results in job order. workers defaults to the CPU count minus one."""
    jobs = list(jobs)
    workers = max(1, min(len(jobs), workers or (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, jobs, chunksize=chunksize))


def save_json(path, result):
    """Write a bench's results (anything JSON can hold) to path, if one was given."""
    if path:
        Path(path).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        print(f"wrote {path}")

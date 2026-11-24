"""What every bench shares: the process pool and the JSON output."""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# one BLAS thread per worker: the pool already uses every core, and each OpenBLAS thread pool reserves memory
# (16 workers x many threads exhausted 16 GB when benches ran alongside other work)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def run_parallel(fn, jobs, workers=None, chunksize=1, fresh_every=None, on_result=None, keep_going=False):
    """fn over jobs in worker processes, results in job order. workers defaults to the CPU count minus one.
    fresh_every: replace each worker after that many jobs (rendering benches: GL drivers keep some memory per
    closed context, which adds up over hundreds of worlds). Done by a new pool for each batch of workers x
    fresh_every jobs, not max_tasks_per_child: with that, CPython 3.13 can fail to start the replacement worker and
    the pool waits forever (python/cpython#115634; a 200-world face check hung after its first 8).
    on_result(job, result): called as each job finishes (to save progress, so a long bench can resume).
    keep_going: a job whose worker failed (out of memory when other work filled the machine, a crashed process)
    gives None instead of ending the bench."""
    jobs = list(jobs)
    workers = max(1, min(len(jobs), workers or (os.cpu_count() or 2) - 1))
    batch = workers * fresh_every if fresh_every else len(jobs)
    if on_result is None and not keep_going:
        out = []
        for k in range(0, len(jobs), batch):
            with ProcessPoolExecutor(workers) as ex:
                out += ex.map(fn, jobs[k:k + batch], chunksize=chunksize)
            if fresh_every:
                print(f"  {len(out)}/{len(jobs)} done", flush=True)
        return out
    out, done = [None] * len(jobs), 0
    for k in range(0, len(jobs), batch):
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(fn, jobs[i]): i for i in range(k, min(k + batch, len(jobs)))}
            for f in as_completed(futs):
                i = futs[f]
                try:
                    out[i] = f.result()
                except Exception as e:                    # noqa: BLE001 - see keep_going
                    if not keep_going:
                        raise
                    print(f"  job {jobs[i]!r} lost: {type(e).__name__}: {e}", flush=True)
                    continue
                done += 1
                if on_result is not None:
                    on_result(jobs[i], out[i])
        print(f"  {done}/{len(jobs)} done", flush=True)
    return out


def save_json(path, result):
    """Write a bench's results (anything JSON can hold) to path, if one was given."""
    if path:
        Path(path).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        print(f"wrote {path}")

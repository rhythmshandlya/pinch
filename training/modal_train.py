"""Train Pinch on a Modal GPU; checkpoints on the `pinch-checkpoints` volume. Normally driven by `ops`:

    python -m ops submit configs/experiments/righting.toml --backend modal     # what you usually run

Directly (what ops runs underneath):

    PINCH_GPU=A10G PINCH_TIMEOUT_MIN=120 python -m modal run --detach training/modal_train.py::launch \
        --experiment configs/experiments/righting.toml --run righting-v2
    python -m modal run training/modal_train.py::status --run righting-v2     # run.json + the last log lines
    python -m modal run training/modal_train.py::fetch --run righting-v2      # -> training/checkpoints/<run>
    python -m modal app stop --yes <app id>                                   # stop it early

Cost guard: every run has a hard timeout (PINCH_TIMEOUT_MIN) and the GPU type is explicit (PINCH_GPU). Nothing
keeps running after a run: the function exits and the app stops. The image carries this checkout's code; ops
records the commit (and refuses a dirty tree unless told otherwise), so a run stays reproducible.
"""

import json
import os
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP, VOLUME = "pinch-train", "pinch-checkpoints"
GPU = os.environ.get("PINCH_GPU", "A10G")           # A10G ~$1.10/h; L4 ~$0.80/h; H100 ~$4/h (faster, not better)
TIMEOUT_S = int(os.environ.get("PINCH_TIMEOUT_MIN", "90")) * 60   # hard stop for every run

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jax[cuda12]==0.9.2", "flax==0.12.6", "optax==0.2.9", "orbax-checkpoint==0.12.11",
                 "brax==0.14.2", "mujoco==4.0.1", "mujoco-mjx==4.0.1", "numpy", "pillow")
    .add_local_dir(ROOT / "crab", "/root/pinch/crab", ignore=["__pycache__"])
    .add_local_dir(ROOT / "training", "/root/pinch/training", ignore=["__pycache__", "checkpoints", "runs"])
    .add_local_dir(ROOT / "configs", "/root/pinch/configs")
    .add_local_dir(ROOT / "sim" / "models", "/root/pinch/sim/models")
    .add_local_dir(ROOT / "policies", "/root/pinch/policies")             # skills that run on top of the walk policy
)
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={"/ckpt": vol})
def train(env: str = "", preset: str = "", run: str = "", timesteps: int = 0, experiment: str = "",
          num_envs: int = 0, git_sha: str = "", git_dirty: bool = False):
    import time
    sys.path.insert(0, "/root/pinch")
    os.chdir("/root/pinch")
    if git_sha:
        os.environ["PINCH_GIT_SHA"], os.environ["PINCH_GIT_DIRTY"] = git_sha, "1" if git_dirty else "0"
    from training.train import run_experiment, run_training
    last = [0.0]

    def commit():                                   # at most once a minute
        if time.time() - last[0] > 60:
            vol.commit()
            last[0] = time.time()
    try:
        if experiment:
            run_experiment(experiment, run or None, "/ckpt", commit, num_envs or None, timesteps or None)
        else:
            run_training(preset, env, timesteps or None, run, True, out_root="/ckpt", on_save=commit,
                         num_envs=num_envs or None)
    finally:
        vol.commit()
    return run


@app.local_entrypoint()
def main(env: str = "gait", preset: str = "gait", run: str = "gait-v1", timesteps: int = 0, experiment: str = ""):
    print(f"training {experiment or env + '/' + preset} as {run} on a Modal {GPU} (timeout {TIMEOUT_S // 60} min)")
    print("finished:", train.remote(env, preset, run, timesteps, experiment))


@app.local_entrypoint()
def launch(env: str = "", preset: str = "", run: str = "", timesteps: int = 0, experiment: str = "",
           num_envs: int = 0, git_sha: str = "", git_dirty: bool = False):
    """Start a run and return at once (use with `modal run --detach`): the run lives on Modal, not on this
    laptop. Its hard timeout still applies; `status` and `fetch` read its results."""
    call = train.spawn(env, preset, run, timesteps, experiment, num_envs, git_sha, git_dirty)
    print(json.dumps(dict(launched=run, gpu=GPU, timeout_min=TIMEOUT_S // 60, call_id=call.object_id)))


@app.function(volumes={"/ckpt": vol}, timeout=300)
def read_run(run: str) -> dict:
    out = {}
    for p in Path(f"/ckpt/{run}").glob("*"):
        if p.is_file():
            out[p.name] = p.read_bytes()
    return out


@app.function(volumes={"/ckpt": vol}, timeout=120)
def read_status(run: str) -> dict:
    vol.reload()
    d = Path(f"/ckpt/{run}")
    rec = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else None
    prog = (d / "progress.jsonl").read_text().splitlines()[-3:] if (d / "progress.jsonl").exists() else []
    return dict(record=rec, progress=prog)


@app.local_entrypoint()
def status(run: str = "gait-v1"):
    print(json.dumps(read_status.remote(run)))


@app.local_entrypoint()
def fetch(run: str = "gait-v1", dest: str = ""):
    files = read_run.remote(run)
    dst = Path(dest) if dest else ROOT / "training" / "checkpoints" / run
    dst.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (dst / name).write_bytes(data)
    print(f"downloaded {sorted(files)} -> {dst}")


# ---- throughput benchmark: which GPU gives the most simulation steps per dollar on OUR workload ----------------

BENCH_GPUS = ("A10G", "L4", "L40S", "A100-40GB", "H100")
BENCH_CASES = (("gait_v2", "gait_v2", 2048), ("gait_v2", "gait_v2", 8192), ("grasp", "grasp", 4096))


@app.function(gpu="A10G", timeout=1800)
def bench_one(env: str, preset: str, num_envs: int, gpu_name: str) -> dict:
    """A short real training run; steps/s between the 2nd and last evaluation (after XLA compilation)."""
    import subprocess
    import time
    sys.path.insert(0, "/root/pinch")
    os.chdir("/root/pinch")
    from training.train import run_training
    t0 = time.time()
    out = run_training(preset, env, 3_000_000, f"bench-{env}-{num_envs}", True, out_root="/tmp/bench",
                       num_envs=num_envs, overrides=dict(num_evals=4, num_eval_envs=128))
    rows = [json.loads(x) for x in (out / "progress.jsonl").read_text().splitlines()]
    a, b = rows[1], rows[-1]
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return dict(gpu=gpu_name, gpu_seen=smi, env=env, num_envs=num_envs,
                steps_per_s=round((b["step"] - a["step"]) / max(1e-6, b["time_s"] - a["time_s"])),
                compile_s=round(rows[1]["time_s"] - rows[0]["time_s"]), wall_s=round(time.time() - t0))


@app.local_entrypoint()
def throughput(gpus: str = ",".join(BENCH_GPUS)):
    """python -m modal run training/modal_train.py::throughput  ->  one JSON line per (GPU, case)."""
    calls = [(g, c, bench_one.with_options(gpu=g).spawn(c[0], c[1], c[2], g))
             for g in gpus.split(",") for c in BENCH_CASES]
    for g, c, call in calls:
        try:
            print(json.dumps(call.get()), flush=True)
        except Exception as e:                       # noqa: BLE001 - one GPU's failure must not hide the rest
            print(json.dumps(dict(gpu=g, env=c[0], num_envs=c[2], error=repr(e)[:500])), flush=True)

"""Train Pinch on a Modal GPU, checkpoints on a Modal volume, results downloaded locally.

    python -m modal run --detach training/modal_train.py --env walk --preset full --run walk-v1
    python -m modal run training/modal_train.py::fetch --run walk-v1      # download into training/checkpoints/
    python -m modal app stop pinch-train                                  # stop anything still running

Cost guard: every run has a hard timeout (TIMEOUT_S) and an L4 GPU (~$0.80/h). Check spend on the Modal
dashboard. Nothing keeps running after a run: the function exits and the
app stops.
"""

import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
import os
APP, VOLUME = "pinch-train", "pinch-checkpoints"
GPU = os.environ.get("PINCH_GPU", "L4")          # L4 ~$0.80/h; A10G ~$1.10/h when L4s are queued
TIMEOUT_S = int(os.environ.get("PINCH_TIMEOUT_MIN", "90")) * 60   # hard stop for every run

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jax[cuda12]==0.9.2", "flax==0.12.6", "optax==0.2.9", "orbax-checkpoint==0.12.11",
                 "brax==0.14.2", "mujoco==4.0.1", "mujoco-mjx==4.0.1", "numpy", "pillow")
    .add_local_dir(ROOT / "crab", "/root/pinch/crab", ignore=["__pycache__"])
    .add_local_dir(ROOT / "training", "/root/pinch/training", ignore=["__pycache__", "checkpoints", "runs"])
    .add_local_dir(ROOT / "sim" / "models", "/root/pinch/sim/models")
    .add_local_dir(ROOT / "policies", "/root/pinch/policies")             # skills that run on top of the walk policy
)
app = modal.App(APP, image=image)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)


@app.function(gpu=GPU, timeout=TIMEOUT_S, volumes={"/ckpt": vol})
def train(env: str = "gait", preset: str = "gait", run: str = "gait-v1", timesteps: int = 0):
    import os
    import time
    sys.path.insert(0, "/root/pinch")
    os.chdir("/root/pinch")
    from training.train import run_training
    last = [0.0]

    def commit():                                   # at most once a minute
        if time.time() - last[0] > 60:
            vol.commit()
            last[0] = time.time()
    run_training(preset, env, timesteps or None, run, True, out_root="/ckpt", on_save=commit)
    vol.commit()
    return run


@app.local_entrypoint()
def main(env: str = "gait", preset: str = "gait", run: str = "gait-v1", timesteps: int = 0):
    print(f"training {env}/{preset} as {run} on a Modal {GPU} (timeout {TIMEOUT_S // 60} min)")
    print("finished:", train.remote(env, preset, run, timesteps))


@app.local_entrypoint()
def launch(env: str = "gait", preset: str = "gait", run: str = "gait-v1", timesteps: int = 0):
    """Start a run and return at once (use with `modal run --detach`): the run lives on Modal, not on this
    laptop, so closing the terminal doesn't stop it. Its hard timeout still applies; `fetch` gets the results."""
    call = train.spawn(env, preset, run, timesteps)
    print(f"launched {env}/{preset} as {run} on a Modal {GPU} (timeout {TIMEOUT_S // 60} min): {call.object_id}")


@app.function(volumes={"/ckpt": vol}, timeout=300)
def read_run(run: str) -> dict:
    out = {}
    for p in Path(f"/ckpt/{run}").glob("*"):
        if p.is_file():
            out[p.name] = p.read_bytes()
    return out


@app.local_entrypoint()
def fetch(run: str = "gait-v1"):
    files = read_run.remote(run)
    dst = ROOT / "training" / "checkpoints" / run
    dst.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (dst / name).write_bytes(data)
    print(f"downloaded {sorted(files)} -> {dst}")

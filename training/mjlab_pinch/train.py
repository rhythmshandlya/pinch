"""Train a Pinch policy on mjlab (run from the repo root, with the trainer's environment, see MJLAB.md):

    python -m training.mjlab_pinch.train Pinch-Velocity-Flat --env.scene.num-envs 64 --agent.max-iterations 5 \
        --agent.logger tensorboard

Takes every mjlab train flag (tyro): --env.*, --agent.*, --agent.logger tensorboard (wandb is not used here).
After training it exports the last checkpoint (export.py): policy.onnx with the observation normaliser baked in,
and policy.npz for plain numpy.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from mjlab.scripts.train import main as mjlab_train

    import training.mjlab_pinch.tasks  # noqa: F401  (registers the Pinch tasks)
    mjlab_train()


if __name__ == "__main__":
    main()

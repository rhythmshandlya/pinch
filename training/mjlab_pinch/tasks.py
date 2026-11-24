"""Registers the Pinch tasks with mjlab (needs mjlab). Import this module, then use mjlab's train/play CLI:
train.py does both in one process, because the project is never installed as a package."""

from mjlab.tasks.registry import register_mjlab_task

from training.mjlab_pinch.env_cfg import make_env_cfg, make_rl_cfg

TASKS = {"Pinch-Velocity-Flat": "position", "Pinch-Velocity-Flat-Bam": "bam"}

for task_id, kind in TASKS.items():
    register_mjlab_task(task_id=task_id, env_cfg=make_env_cfg(actuators=kind),
                        play_env_cfg=make_env_cfg(play=True, actuators=kind), rl_cfg=make_rl_cfg())

# indoor terrain families with a curriculum (training/mjlab_pinch/terrains.py, phase 5)
from training.mjlab_pinch.terrains import make_rough_env_cfg  # noqa: E402

register_mjlab_task(task_id="Pinch-Velocity-Rough", env_cfg=make_rough_env_cfg(),
                    play_env_cfg=make_rough_env_cfg(play=True), rl_cfg=make_rl_cfg())

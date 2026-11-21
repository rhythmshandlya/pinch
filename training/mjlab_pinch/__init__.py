"""Pinch on mjlab (MuJoCo Warp + PyTorch + RSL-RL): the velocity-command walking task.

layout.py, robot_spec.py and np_policy.py import without torch or mjlab (tests, the sim-to-sim replay).
Everything else needs the trainer's environment (requirements-mjlab.txt) and is imported by train.py.
See docs/dev/MJLAB.md.
"""

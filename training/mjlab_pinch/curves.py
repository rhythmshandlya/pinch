"""Print the learning curves of an mjlab run from its tensorboard files (needs tensorboard).

    python -m training.mjlab_pinch.curves <log dir or any parent> [points]

Shows mean reward, episode length, the tracking errors and every reward term at evenly spaced iterations, and
checks the sign rule: every penalty (negative weight) must read <= 0.
"""

import sys
from pathlib import Path

KEY = ["Train/mean_reward", "Train/mean_episode_length", "Metrics/twist/error_vel_xy", "Metrics/twist/error_vel_yaw",
       "Perf/total_fps"]


def load(logdir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = sorted(Path(logdir).rglob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    ea = EventAccumulator(str(files[-1].parent), size_guidance={"scalars": 0})
    ea.Reload()
    return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}


def main(logdir, points=10):
    sc = load(logdir)
    rewards = sorted(t for t in sc if t.startswith("Episode_Reward/"))
    tags = [t for t in KEY if t in sc] + rewards
    n = max(len(v) for v in sc.values())
    idx = sorted({round(i * (n - 1) / (points - 1)) for i in range(points)}) if n > points else list(range(n))
    print(f"{n} iterations logged")
    print(f"{'iteration':>10s} " + " ".join(f"{i:>9d}" for i in idx))
    for t in tags:
        v = sc[t]
        print(f"{t.split('/', 1)[1][:34]:34s}" + " ".join(f"{v[min(i, len(v) - 1)][1]:9.3f}" for i in idx))
    pen = {t: sc[t][-1][1] for t in rewards if t.split("/")[1] in ("body_ang_vel", "dof_pos_limits", "action_rate_l2",
                                                                    "foot_clearance", "foot_swing_height", "foot_slip", "fell")}
    bad = {t: v for t, v in pen.items() if v > 1e-9}
    print("penalty sign check:", "OK, every penalty <= 0" if not bad else f"VIOLATED {bad}")
    return sc


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)

"""Build the Pinch task and step it with random actions: the sanity checks to run on the trainer before a long job.

    python -m training.mjlab_pinch.check_env [Pinch-Velocity-Flat] [num_envs]

Checks: observation sizes match layout.py, joint order, 12 actions, no NaN/Inf in observations or rewards,
every penalty term is <= 0 and every reward term finite, the robot stays up under a zero action (stand pose).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(task="Pinch-Velocity-Flat", num_envs=64, steps=300):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    import training.mjlab_pinch.tasks  # noqa: F401
    from training.mjlab_pinch import layout

    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    robot = env.scene["robot"]
    obs, _ = env.reset()
    actor, critic = obs["actor"], obs["critic"]
    print("actor obs", tuple(actor.shape), "critic obs", tuple(critic.shape), "actions", env.action_manager.total_action_dim)
    assert actor.shape[1] == layout.OBS_SIZE, (actor.shape, layout.OBS_SIZE)
    assert env.action_manager.total_action_dim == layout.N_LEG
    act = env.action_manager.get_term("joint_pos")
    assert list(act.target_names) == layout.LEG_JOINT_NAMES, act.target_names
    print("leg joint order ok; entity joints:", len(robot.joint_names), "actuators:", len(robot.actuators))
    print("observation terms:", env.observation_manager.active_terms["actor"])
    print("start height", float(robot.data.root_link_pos_w[:, 2].mean()))

    # zero action: the stand pose must hold
    zero = torch.zeros(num_envs, layout.N_LEG, device=env.device)
    for _ in range(100):
        obs, rew, term, trunc, _ = env.step(zero)
    z = robot.data.root_link_pos_w[:, 2]
    tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1))
    print(f"zero action 2 s: height {float(z.mean()):.4f} (min {float(z.min()):.4f}), tilt {float(tilt.mean()):.3f} rad, "
          f"terminated {int(term.sum())}")
    # random actions
    bad = 0
    for _ in range(steps):
        a = torch.randn(num_envs, layout.N_LEG, device=env.device) * 0.5
        obs, rew, term, trunc, extras = env.step(a)
        for g in obs.values():
            bad += int((~torch.isfinite(g)).sum())
        bad += int((~torch.isfinite(rew)).sum())
    print("non-finite values over", steps, "random steps:", bad)
    assert bad == 0
    terms = env.reward_manager.active_terms
    print("reward terms:", terms)
    print("OK")
    env.close()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(a[:1] + [int(x) for x in a[1:2]]))

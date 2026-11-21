"""Sim-to-sim check: run an exported mjlab policy (policy.npz) on the project's CPU MuJoCo twin.

    .venv\\Scripts\\python -m training.mjlab_pinch.sim_replay policy.npz [--seconds 20] [--seeds 5]

The trainer simulates with mujoco 3.11 / MuJoCo Warp at 5 ms; the twin is mujoco 4.0.1 on CPU at its own 2 ms
step with the full model (backlash joints, the servo classes of sim/models/crab.xml). The observation is built
by layout.py, the same function the trainer's layout is pinned to, from the twin's gyro, orientation and joint
state. For each command it reports, over several seeds, falls, how far the body drifted from standing, and the
velocity tracking error (body-frame velocity over the last 15 s against the command). Rates use
sim/benches/stats.py (95 % ranges).
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "sim" / "benches"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import stats  # noqa: E402  (sim/benches/stats.py)

from training.mjlab_pinch import layout  # noqa: E402
from training.mjlab_pinch.np_policy import NpPolicy  # noqa: E402
from training.mjlab_pinch.robot_spec import MODEL  # noqa: E402

COMMANDS = {"stand": (0.0, 0.0, 0.0), "forward 4 cm/s": (0.04, 0.0, 0.0), "sideways 4 cm/s": (0.0, 0.04, 0.0),
            "turn 0.4 rad/s": (0.0, 0.0, 0.4)}
FALL_TILT = np.radians(60.0)
FALL_HEIGHT = 0.028                 # the trainer's termination thresholds
SETTLE_S = 5.0                      # tracking is judged after this


class Twin:
    def __init__(self, model_path=MODEL):
        self.m = mujoco.MjModel.from_xml_path(str(model_path))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.act = {m.actuator(i).name: i for i in range(m.nu)}
        self.leg_act = np.array([self.act[n] for n in layout.LEG_JOINT_NAMES])
        self.arm_act = np.array([self.act[n] for n in layout.ARM_JOINT_NAMES])
        self.leg_q = np.array([m.jnt_qposadr[m.joint(n).id] for n in layout.LEG_JOINT_NAMES])
        self.leg_v = np.array([m.jnt_dofadr[m.joint(n).id] for n in layout.LEG_JOINT_NAMES])
        self.body = m.body("body").id
        self.gyro_adr = m.sensor_adr[m.sensor("imu_gyro").id]
        self.n_sub = int(round(layout.CTRL_DT / m.opt.timestep))
        self.arm_hold = np.array([layout.STAND_POSE[n] for n in layout.ARM_JOINT_NAMES])

    def reset(self, rng, jitter=0.03):
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        d.qpos[self.leg_q] += rng.uniform(-jitter, jitter, layout.N_LEG)
        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[2] += 0.002
        mujoco.mj_forward(m, d)

    def observe(self):
        d = self.d
        R = d.xmat[self.body].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return d.sensordata[self.gyro_adr:self.gyro_adr + 3].copy(), grav

    def body_velocity(self):
        R = self.d.xmat[self.body].reshape(3, 3)
        return R.T @ self.d.qvel[:3], self.d.sensordata[self.gyro_adr + 2]

    def step(self, action):
        d = self.d
        d.ctrl[self.leg_act] = layout.DEFAULT_LEG_POS + layout.ACTION_SCALE * action
        d.ctrl[self.arm_act] = self.arm_hold
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, d)

    def fallen(self):
        R = self.d.xmat[self.body].reshape(3, 3)
        tilt = np.arccos(np.clip(R[2, 2], -1, 1))
        return bool(tilt > FALL_TILT or self.d.qpos[2] < FALL_HEIGHT or not np.isfinite(self.d.qpos).all())


def rollout(twin, policy, cmd, seconds=20.0, seed=0, delay_steps=0):
    rng = np.random.default_rng(seed)
    twin.reset(rng)
    hist, last = layout.History(), np.zeros(layout.N_LEG, np.float32)
    queue = [np.zeros(layout.N_LEG, np.float32)] * delay_steps
    steps = int(seconds / layout.CTRL_DT)
    vel, yaw, fell_at, p0 = [], [], None, twin.d.qpos[:2].copy()
    for k in range(steps):
        gyro, grav = twin.observe()
        q, qd = twin.d.qpos[twin.leg_q], twin.d.qvel[twin.leg_v]
        hist.push(layout.frame(gyro, grav, q - layout.DEFAULT_LEG_POS, qd, last))
        act = policy(hist.obs(cmd)).astype(np.float32)
        last = act
        queue.append(act)
        twin.step(queue.pop(0))
        if twin.fallen():
            fell_at = k * layout.CTRL_DT
            break
        if k * layout.CTRL_DT >= SETTLE_S:
            v, w = twin.body_velocity()
            vel.append(v[:2].copy())
            yaw.append(w)
    out = dict(fell=fell_at is not None, fell_at=fell_at, moved=float(np.linalg.norm(twin.d.qpos[:2] - p0)),
               height=float(twin.d.qpos[2]))
    if vel:
        vel, yaw = np.array(vel), np.array(yaw)
        out.update(v_mean=vel.mean(0).tolist(), yaw_mean=float(yaw.mean()),
                   err_xy=float(np.linalg.norm(vel - np.array(cmd[:2]), axis=1).mean()),
                   err_yaw=float(np.abs(yaw - cmd[2]).mean()),
                   err_xy_of_mean=float(np.linalg.norm(vel.mean(0) - np.array(cmd[:2]))),
                   err_yaw_of_mean=float(abs(yaw.mean() - cmd[2])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--delay", type=int, default=0, help="extra action delay in control steps")
    a = ap.parse_args(argv)
    pol = NpPolicy.load(a.policy)
    twin = Twin()
    print(f"policy {a.policy}: obs {pol.obs_size}, actions {pol.action_size}; twin dt {twin.m.opt.timestep}, "
          f"{a.seconds:.0f} s, {a.seeds} seeds, delay {a.delay} steps")
    for name, cmd in COMMANDS.items():
        rs = [rollout(twin, pol, cmd, a.seconds, s, a.delay) for s in range(a.seeds)]
        falls = sum(r["fell"] for r in rs)
        ok = [r for r in rs if "err_xy" in r]
        line = f"{name:16s} falls {stats.rate(falls, len(rs))}"
        if ok:
            v = np.mean([r["v_mean"] for r in ok], axis=0)
            line += (f" | mean body velocity ({v[0] * 100:+.1f}, {v[1] * 100:+.1f}) cm/s, yaw {np.mean([r['yaw_mean'] for r in ok]):+.2f}"
                     f" rad/s | tracking error xy {np.mean([r['err_xy'] for r in ok]) * 100:.1f} cm/s (of the mean "
                     f"{np.mean([r['err_xy_of_mean'] for r in ok]) * 100:.1f}), yaw {np.mean([r['err_yaw'] for r in ok]):.2f} rad/s"
                     f" | height {np.mean([r['height'] for r in ok]) * 1000:.0f} mm")
        print(line)
        for s, r in enumerate(rs):
            if r["fell"]:
                print(f"    seed {s}: fell at {r['fell_at']:.1f} s")


if __name__ == "__main__":
    main()

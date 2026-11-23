"""Self-righting bench: the searched swing (crab/righting.py) against a learned policy (policies/<name>.npz),
from the same random starts on its back, in the CPU twin's physics (sim/models/crab_v2.xml;
2 ms steps).

    python sim/benches/righting.py                      # swing + policies/righting.npz if there is one
    python sim/benches/righting.py --policy <candidate> --starts 200

Each start: rolled 180 +- 30 deg, pitched +- 25 deg, any heading, joints +-0.2 rad around the stand pose,
dropped from 8 cm; floor friction x0.4-1.2, masses +-15 %, servo strength x0.7-1.1. Both controllers are
sensed as on the robot and handled like Body._right_self: once upright for 0.2 s they blend to the stand pose; success is still
upright 2 s later. Reports the success rate, the time to upright (mean, 90th percentile) and how hard it drives
the servos while getting up: the 95th percentile of the commanded joint speed (rad/s) and its peak (decision 85).
--realism datasheet adds the electronics of configs/electronics.toml: servo targets arrive late (bus + servo delay),
joint angles come in encoder steps, the IMU has the BNO055's rate, lag, noise and bias (crab/electronics.py).
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crab import policy_io as pio  # noqa: E402
from crab import righting  # noqa: E402
from crab.emotes import STAND  # noqa: E402
from crab.params import (
    JOINT_NAMES,  # noqa: E402
    PARAMS,  # noqa: E402
)
from crab.policy_np import POLICY_DIR, NumpyPolicy  # noqa: E402
from sim.benches.common import run_parallel, save_json  # noqa: E402
from sim.benches.stats import rate  # noqa: E402

MODEL = ROOT / "sim" / "models" / PARAMS.model_file   # the robot model, generated from the CAD
DT, CTRL_DT, TIMEOUT = 0.002, pio.CTRL_DT, 8.0


def _quat(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2),
                              np.cos(yaw / 2), np.sin(yaw / 2))
    return np.array([cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr, cy * sp * cr + sy * cp * sr,
                     sy * cp * cr - cy * sp * sr])


def episode(args):
    ctrl, seed, policy, *more = args
    realism = more[0] if more else "ideal"
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    m.opt.timestep = DT
    m.geom_friction[:, 0] *= rng.uniform(0.4, 1.2)
    m.body_mass[:] *= rng.uniform(0.85, 1.15, m.nbody)
    m.actuator_forcerange[:] *= rng.uniform(0.7, 1.1, m.nu)[:, None]
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINT_NAMES])
    vadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in JOINT_NAMES])
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    d.qpos[2] = 0.08
    d.qpos[3:7] = _quat(np.pi + rng.uniform(-0.52, 0.52), rng.uniform(-0.44, 0.44), rng.uniform(-np.pi, np.pi))
    d.qpos[qadr] = np.clip(STAND + rng.uniform(-0.2, 0.2, len(JOINT_NAMES)), lo, hi)
    d.ctrl[:] = d.qpos[qadr]
    for _ in range(int(0.3 / DT)):                       # land on its back
        mujoco.mj_step(m, d)
    pol = NumpyPolicy.load(POLICY_DIR / f"{policy}.npz") if ctrl == "policy" else None
    elec = None
    if realism == "datasheet":
        from crab.electronics import Realism
        elec = Realism(seed=seed, dt=DT, initial=d.ctrl)
    body = m.body("body").id
    sub = int(round(CTRL_DT / DT))
    last = np.zeros(len(JOINT_NAMES))
    t, up, up_t = 0.0, 0, None
    prev, rates = d.ctrl.copy(), []
    while t < TIMEOUT:
        if pol is not None:
            R = d.xmat[body].reshape(3, 3)
            q, qd = d.qpos[qadr].copy(), d.qvel[vadr].copy()
            grav, gyro = R.T @ np.array([0.0, 0.0, -1.0]), d.qvel[3:6]
            if elec is not None:
                q = elec.encoder(q)
                gyro, grav = elec.imu.read(d.time, gyro, grav)
            obs = pio.righting_observation(np, grav, gyro, q, qd, last)
            last = np.clip(pol(obs), -1, 1)
            d.ctrl[:] = pio.righting_targets(np, last, lo, hi)
        else:
            d.ctrl[:] = np.clip(righting.target(t), lo, hi)
        rates.append(np.abs(d.ctrl - prev) / CTRL_DT)      # commanded joint speed this tick, every joint
        prev = d.ctrl.copy()
        goal = d.ctrl.copy()
        for _ in range(sub):
            if elec is not None:
                d.ctrl[:] = elec.delay.push(goal)
            mujoco.mj_step(m, d)
        if elec is not None:
            d.ctrl[:] = goal                                     # (the policy and the rate check read the command)
        t += CTRL_DT
        up = up + 1 if d.xmat[body][8] > 0.85 else 0
        if up >= int(0.2 / CTRL_DT):
            up_t = t
            break
    r = np.concatenate(rates) if rates else np.zeros(1)
    slew = (round(float(np.percentile(r, 95)), 2), round(float(r.max()), 2))
    if up_t is None:
        return ctrl, seed, False, None, slew
    q0 = d.ctrl.copy()
    for k in range(int(2.0 / DT)):                       # blend to stand (0.6 s) and stay up
        s = min(1.0, k * DT / 0.6)
        d.ctrl[:] = q0 + (STAND - q0) * s
        mujoco.mj_step(m, d)
    return ctrl, seed, bool(d.xmat[body][8] > 0.85), up_t, slew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="righting", help="policies/<name>.npz")
    ap.add_argument("--starts", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--json", default=None, help="also write the results here")
    ap.add_argument("--realism", default="ideal", choices=["ideal", "datasheet"],
                    help="datasheet: command delay, encoder steps and the IMU of configs/electronics.toml")
    a = ap.parse_args()
    ctrls = ["swing"] + (["policy"] if (POLICY_DIR / f"{a.policy}.npz").exists() else [])
    if "policy" not in ctrls:
        print(f"no policies/{a.policy}.npz - the swing only")
    jobs = [(c, 5000 + s, a.policy, a.realism) for c in ctrls for s in range(a.starts)]
    rows = run_parallel(episode, jobs, a.jobs, chunksize=4)
    save_json(a.json, rows)
    for c in ctrls:
        r = [x for x in rows if x[0] == c]
        ok = [x for x in r if x[2]]
        times = np.array([x[3] for x in ok]) if ok else np.array([np.nan])
        name = "swing (crab/righting.py)" if c == "swing" else f"policy ({a.policy})"
        print(f"{name:28s} upright and standing {rate(len(ok), len(r))}   time to upright mean "
              f"{np.mean(times):.2f} s, p90 {np.percentile(times, 90):.2f} s   commanded joint speed p95 "
              f"{np.median([x[4][0] for x in r]):.1f} rad/s, peak {np.median([x[4][1] for x in r]):.0f} rad/s (medians)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

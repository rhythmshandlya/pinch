"""What can Pinch get over? One hurdle (a bar 4-20 mm high) or one ramp (5-25 deg, 10 cm long, onto a plateau)
straight across its path, walking forward or sideways at its normal speed; 3 robots per case (nominal and two
randomised: friction, masses, servo strength). Crossed = the body ends 12 cm past the near side of the
obstacle within 15 s without falling.

    python sim/benches/hurdles.py                          # scripted trot and the walking policy (policies/gait.npz)
    python sim/benches/hurdles.py --policy terrain_v1      # another walking policy
Writes docs/HURDLES.md when run with the default policy.
"""

import argparse
import copy
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crab import policy_io as pio  # noqa: E402
from crab.emotes import STAND  # noqa: E402
from crab.gait import _S, GAITS, Gait  # noqa: E402
from crab.params import (
    JOINT_NAMES,  # noqa: E402
    PARAMS,  # noqa: E402
)
from crab.policy_np import POLICY_DIR, NumpyPolicy  # noqa: E402
from sim.benches.common import run_parallel, save_json  # noqa: E402

MODEL = ROOT / "sim" / "models" / PARAMS.model_file   # the robot model, generated from the CAD
HURDLES_MM = (4, 6, 8, 10, 12, 15, 18, 20)
RAMPS_DEG = (5, 10, 15, 20, 25)
START_GAP = 0.03          # m from the feet's reach to the obstacle's near side
SPEED = 0.035


def world(kind, size, way, seed):
    spec = mujoco.MjSpec.from_file(str(MODEL))
    reach = 0.052 if way == "forward" else 0.105            # how far the feet reach that way
    near = reach + START_GAP
    along = np.array((1.0, 0.0)) if way == "forward" else np.array((0.0, 1.0))
    yaw = 0.0 if way == "forward" else np.pi / 2
    q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    if kind == "hurdle":
        h = size / 1000
        c = along * (near + 0.006)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.006, 0.3, h / 2], pos=[c[0], c[1], h / 2],
                                quat=q, friction=[0.8, 0.02, 0.001])
    else:
        th = np.radians(size)
        L = 0.05
        top = 2 * L * np.sin(th)
        c = along * (near + L * np.cos(th))
        qp = [np.cos(-th / 2), 0, np.sin(-th / 2), 0]                     # pitch: rises along +x
        qr = _qmul(q, qp)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[L, 0.3, 0.003],
                                pos=[c[0], c[1], L * np.sin(th) - 0.003 * np.cos(th)], quat=qr,
                                friction=[0.8, 0.02, 0.001])
        c2 = along * (near + 2 * L * np.cos(th) + 0.15)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 0.3, max(top, 0.001) / 2 + 0.0005],
                                pos=[c2[0], c2[1], max(top, 0.001) / 2 - 0.0005], quat=q, friction=[0.8, 0.02, 0.001])
    m = spec.compile()
    if seed:
        rng = np.random.default_rng(seed)
        m.geom_friction[:, 0] = rng.uniform(0.5, 1.2)
        m.body_mass[:] *= rng.uniform(0.85, 1.15, m.nbody)
        m.actuator_forcerange[:] *= rng.uniform(0.8, 1.1, m.nu)[:, None]
    return m, along, near


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def case(args, on_tick=None):
    """One crossing attempt. on_tick(m, d, t), if given, is called every control tick (sim/clips.py records it)."""
    kind, size, way, ctrl, seed, policy = args
    m, along, near = world(kind, size, way, seed)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)
    act = np.array([m.actuator(j).id for j in JOINT_NAMES])
    leg_q = np.array([m.jnt_qposadr[m.joint(n).id] for n in pio.LEG_JOINT_NAMES])
    leg_v = np.array([m.jnt_dofadr[m.joint(n).id] for n in pio.LEG_JOINT_NAMES])
    body = m.body("body").id
    pol = NumpyPolicy.load(POLICY_DIR / f"{policy}.npz") if ctrl == "policy" else None
    k_scale = (pol.meta.get("action_scale", pio.EMOTE_ACTION_SCALE) / pio.EMOTE_ACTION_SCALE) if pol else 1.0
    gait = Gait(GAITS["trot"], com_offset=np.array([0.015 * _S, 0.0]))
    cmd = (SPEED, 0.0, 0.0) if way == "forward" else (0.0, SPEED, 0.0)
    last = np.zeros(12)
    t, fell = 0.0, False
    target = STAND.copy()
    for _k in range(int(15.0 / pio.CTRL_DT)):
        for i in range(10):
            out = gait.step(pio.CTRL_DT / 10, cmd, solve=i == 9)
            target = out if out is not None else target
        if pol is not None:
            g = copy.deepcopy(gait)
            for i in range(5):
                ahead = g.step(pio.CTRL_DT, cmd, solve=i == 4)
            R = d.xmat[body].reshape(3, 3)
            grav = R.T @ np.array([0, 0, -1.0])
            obs = pio.emote_observation(np, d.qvel[3:6], grav, d.qpos[leg_q], d.qvel[leg_v], last, target, ahead,
                                        None, (t % 6.0) / 6.0)
            last = np.clip(pol(obs), -1, 1)
            ctrl_q = pio.emote_targets(np, target, last * k_scale)
        else:
            ctrl_q = target
        d.ctrl[act] = ctrl_q
        for _ in range(10):
            mujoco.mj_step(m, d)
        t += pio.CTRL_DT
        if on_tick is not None:
            on_tick(m, d, t)
        R = d.xmat[body].reshape(3, 3)
        if R[2, 2] < 0.5:
            fell = True
            break
        if float(d.qpos[0:2] @ along) > near + 0.12:
            break
    progress = float(d.qpos[0:2] @ along)
    return dict(kind=kind, size=size, way=way, ctrl=ctrl, seed=seed, fell=fell,
                crossed=(not fell) and progress > near + 0.12, t=round(t, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="gait")
    ap.add_argument("--json", default=None, help="also write the results here")
    a = ap.parse_args()
    ctrls = ["scripted"] + (["policy"] if (POLICY_DIR / f"{a.policy}.npz").exists() else [])
    if "policy" not in ctrls:
        print(f"no policies/{a.policy}.npz - the scripted trot only")
    jobs = []
    for kind, sizes in (("hurdle", HURDLES_MM), ("ramp", RAMPS_DEG)):
        for size in sizes:
            for way in ("forward", "sideways"):
                for ctrl in ctrls:
                    for seed in (0, 1, 2):
                        jobs.append((kind, size, way, ctrl, seed, a.policy))
    res = run_parallel(case, jobs, 14)
    save_json(a.json, res)
    lines = ["# What Pinch gets over (simulation)", "",
             f"Generated by `python sim/benches/hurdles.py`: one obstacle straight across the path, 3 robots per case "
             f"(nominal + 2 randomised). Cells: crossed / 3 (falls in brackets). Walking policy: `{a.policy}`.", ""]
    for kind, sizes, unit in (("hurdle", HURDLES_MM, "mm"), ("ramp", RAMPS_DEG, "deg")):
        lines += [f"| {kind} | " + " | ".join(f"{s} {unit}" for s in sizes) + " |",
                  "|---|" + "---|" * len(sizes)]
        for way in ("forward", "sideways"):
            for ctrl in ctrls:
                cells = []
                for s in sizes:
                    r = [x for x in res if x["kind"] == kind and x["size"] == s and x["way"] == way and x["ctrl"] == ctrl]
                    f = sum(x["fell"] for x in r)
                    cells.append(f"{sum(x['crossed'] for x in r)}/3" + (f" ({f})" if f else ""))
                lines.append(f"| {way}, {'trot + ' + a.policy if ctrl == 'policy' else 'scripted trot'} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.policy == "gait":
        (ROOT / "docs" / "HURDLES.md").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

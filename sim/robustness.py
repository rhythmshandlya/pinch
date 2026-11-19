"""Robustness tests: trained policies vs the scripted controllers, under randomised physics.

    python sim/robustness.py                 # both walking and emotes, 40 episodes each, writes docs/ROBUSTNESS.md
    python sim/robustness.py --quick         # 8 episodes each

Each episode draws a different robot and world (independently of training's randomisation):
  floor friction 0.35..1.2, every body's mass +-20%, each servo's strength (force limit) -30..+15%
  and stiffness +-25%, the desk tilted up to 4 deg in a random direction, 20 ms control latency in
  half the episodes, and a shove every 2.5 s (a sideways velocity kick of up to 0.12 m/s, about a
  flick of the finger).
Walking: 12 s of commands (forward, crab-walk left, turn, back, diagonal). Emotes: five random emotes
back to back. A fall = tipped over 45 deg or body down on the desk.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab import policy_io as pio  # noqa: E402
from crab.emotes import EMOTES, STAND  # noqa: E402
from crab.gait import _S, _T, GAITS, Gait  # noqa: E402
from crab.params import JOINT_NAMES  # noqa: E402
from crab.policy_np import POLICY_DIR, NumpyPolicy  # noqa: E402

MODEL = ROOT / "sim" / "models" / "crab.xml"
V, W = _S / _T, 1 / _T
# commands inside the trained policy's envelope (it learned "clearly moving" or "stopped"; slower requests
# are raised to its minimum by the body, and precise slow moves use the scripted gait)
WALK_CMDS = [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.05 * V, 0, 0)), (3.5, (0, 0.06 * V, 0)), (6.0, (0, 0, 0.5 * W)),
             (8.5, (-0.09 * V, 0, 0)), (10.5, (0.045 * V, 0.045 * V, 0))]
SUB = 10


def perturbed(seed, hard=True):
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    k = 1.0 if hard else 0.5
    m.geom_friction[:, 0] = rng.uniform(1.2 - 0.85 * k, 1.2)
    m.body_mass[:] *= rng.uniform(1 - 0.2 * k, 1 + 0.2 * k, m.nbody)
    s = rng.uniform(1 - 0.3 * k, 1 + 0.15 * k, m.nu)
    m.actuator_forcerange[:] *= s[:, None]
    kp = rng.uniform(1 - 0.25 * k, 1 + 0.25 * k, m.nu)
    m.actuator_gainprm[:, 0] *= kp
    m.actuator_biasprm[:, 1] *= kp
    tilt = np.radians(rng.uniform(0, 4 * k))
    az = rng.uniform(0, 2 * np.pi)
    m.opt.gravity[:] = 9.81 * np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)])
    latency = rng.random() < 0.5
    if FEET_ONLY:                                    # diagnostic: the training model's contacts (feet only)
        feet = {m.geom(f"{lg}_foot").id for lg in ("FL", "FR", "RL", "RR")}
        for g in range(m.ngeom):
            if g not in feet and m.geom(g).name != "floor":
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
    return m, rng, latency


FEET_ONLY = False


class Sensors:
    def __init__(self, m):
        self.body = m.body("body").id
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in pio.LEG_JOINT_NAMES])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in pio.LEG_JOINT_NAMES])
        self.bq = np.array([m.jnt_qposadr[m.joint(f"{n}_backlash").id] for n in pio.LEG_JOINT_NAMES])
        self.bv = np.array([m.jnt_dofadr[m.joint(f"{n}_backlash").id] for n in pio.LEG_JOINT_NAMES])

    def read(self, d, rng):
        R = d.xmat[self.body].reshape(3, 3)
        q = d.qpos[self.qadr] + d.qpos[self.bq] + rng.normal(0, 0.003, 12)
        qd = d.qvel[self.vadr] + d.qvel[self.bv]
        return d.qvel[3:6] + rng.normal(0, 0.02, 3), R.T @ np.array([0, 0, -1.0]), q, qd


def episode(args):
    kind, ctrl, seed, hard = args
    m, rng, latency = perturbed(seed, hard)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)                                  # body orientation valid before the first read
    act = np.array([m.actuator(j).id for j in JOINT_NAMES])
    sens = Sensors(m)
    stand_h = float(m.key("stand").qpos[2])
    pol = NumpyPolicy.load(POLICY_DIR / f"{'walk' if kind == 'walk' else 'emote'}.npz") if ctrl == "policy" else None
    gait = Gait(GAITS["trot"], com_offset=np.array([0.015 * _S, 0.0])) if kind == "walk" else None
    seq = rng.choice(pio.EMOTE_NAMES, 5) if kind == "emote" else None
    plan = []
    if kind == "emote":
        t = 0.5
        for name in seq:
            plan.append((t, name))
            t += EMOTES[name].duration + 0.3
        T = t
    else:
        T = 12.5
    last, prev = np.zeros(12), np.zeros(12)
    target = STAND.copy()
    tilts, track, fell = [], [], False
    t = 0.0
    for k in range(int(T / pio.CTRL_DT)):
        if k > 0 and k % int(2.5 / pio.CTRL_DT) == 0:                    # a shove
            d.qvel[0:2] += rng.uniform(-0.12, 0.12, 2) * (1 if hard else 0.5)
        gyro, grav, q, qd = sens.read(d, rng)
        if kind == "walk":
            cmd = next(c for t0, c in reversed(WALK_CMDS) if t >= t0)
            if pol is None:
                for _ in range(SUB):
                    target = gait.step(pio.CTRL_DT / SUB, cmd)
            else:
                obs = pio.observation(np, gyro, grav, np.asarray(cmd, float), q, qd, last, t)
                a = np.clip(pol(obs), -1, 1)
                target = pio.action_to_targets(np, prev if latency else a)
                prev, last = a, a
            v_body = d.xmat[sens.body].reshape(3, 3).T @ d.qvel[0:3]
            cn = np.linalg.norm(cmd[:2])
            if cn > 1e-3 and t > 1.5:
                track.append(np.dot(v_body[:2], cmd[:2]) / cn ** 2)
        else:
            cur = next(((t0, n) for t0, n in reversed(plan) if t >= t0), None)
            if cur is None:
                ref, ahead, prog = STAND, STAND, 0.0
            else:
                emo = EMOTES[cur[1]]
                te = t - cur[0]
                ref, ahead = emo.target(min(te, emo.duration)), emo.target(min(te + 0.1, emo.duration))
                prog = min(te / emo.duration, 1.0)
            if pol is None:
                target = ref
            else:
                obs = pio.emote_observation(np, gyro, grav, q, qd, last, ref, ahead, None, prog)
                a = np.clip(pol(obs), -1, 1)
                target = pio.emote_targets(np, ref, prev if latency else a)
                prev, last = a, a
        d.ctrl[act] = target
        for _ in range(SUB):
            mujoco.mj_step(m, d)
        t += pio.CTRL_DT
        tilt = np.degrees(np.arccos(np.clip(-grav[2], -1, 1)))
        tilts.append(tilt)
        if grav[2] > -0.7 or d.qpos[2] < 0.37 * stand_h:
            fell = True
            break
    return dict(kind=kind, ctrl=ctrl, seed=seed, fell=fell, tilt_rms=float(np.sqrt(np.mean(np.square(tilts)))),
                tilt_max=float(np.max(tilts)), track=float(np.mean(track)) if track else float("nan"))


def summary(rows):
    out = {}
    for kind in ("walk", "emote"):
        for ctrl in ("scripted", "policy"):
            r = [x for x in rows if x["kind"] == kind and x["ctrl"] == ctrl]
            if not r:
                continue
            tr = [x["track"] for x in r if not x["fell"] and not np.isnan(x["track"])]
            out[(kind, ctrl)] = dict(n=len(r), falls=sum(x["fell"] for x in r),
                                     tilt_rms=float(np.mean([x["tilt_rms"] for x in r])),
                                     tilt_max=float(np.max([x["tilt_max"] for x in r])),
                                     track=float(np.mean(tr)) if tr else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--what", default="walk,emote")
    a = ap.parse_args()
    n = 8 if a.quick else a.episodes
    jobs = []
    for kind in a.what.split(","):
        if not (POLICY_DIR / f"{kind}.npz").exists():
            print(f"no policies/{kind}.npz yet - skipping the trained {kind} policy")
        for ctrl in ("scripted", "policy"):
            if ctrl == "policy" and not (POLICY_DIR / f"{kind}.npz").exists():
                continue
            jobs += [(kind, ctrl, 1000 + s, True) for s in range(n)]
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(episode, jobs))
    s = summary(rows)
    lines = ["# Robustness tests (simulation)", "",
             "Generated by `python sim/robustness.py`. Every episode draws a different robot and world, on top of "
             "(and independent of) training's randomisation: friction 0.35-1.2, masses +-20 %, servo strength "
             "-30..+15 % and stiffness +-25 %, desk tilted up to 4 deg, 20 ms extra latency in half the episodes, "
             "a shove (up to 0.12 m/s) every 2.5 s.", "",
             "| What | Controller | Episodes | Falls | Body tilt RMS | Worst tilt | Speed tracking |",
             "|---|---|---|---|---|---|---|"]
    for (kind, ctrl), v in s.items():
        lines.append(f"| {kind} | {ctrl} | {v['n']} | {v['falls']} ({100 * v['falls'] / v['n']:.0f} %) | "
                     f"{v['tilt_rms']:.1f} deg | {v['tilt_max']:.1f} deg | "
                     + ("-" if np.isnan(v['track']) else f"{100 * v['track']:.0f} %") + " |")
    lines += ["", "Pass criteria: the trained policy falls in at most 5 % of these episodes and does no worse than the "
              "scripted controller; walking tracks at least 60 % of the commanded speed.", ""]
    verdicts = []
    for kind in ("walk", "emote"):
        p, sc = s.get((kind, "policy")), s.get((kind, "scripted"))
        if p:
            ok = p["falls"] <= 0.05 * p["n"] and (sc is None or p["falls"] <= sc["falls"]) and \
                 (kind != "walk" or p["track"] >= 0.6)
            verdicts.append(f"- **{kind}: {'PASS' if ok else 'FAIL'}**")
    lines += verdicts
    (ROOT / "docs" / "ROBUSTNESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

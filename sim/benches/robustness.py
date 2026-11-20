"""Robustness tests: trained policies vs the scripted controllers, under randomised physics.

    python sim/benches/robustness.py                 # both walking and emotes, 40 episodes each, writes docs/ROBUSTNESS.md
    python sim/benches/robustness.py --quick         # 8 episodes each

Each episode draws a different robot and world (independently of training's randomisation):
  floor friction 0.35..1.2, every body's mass +-20%, each servo's strength (force limit) -30..+15%
  and stiffness +-25%, the desk tilted up to 4 deg in a random direction, 20 ms control latency in
  half the episodes, and a shove every 2.5 s (a sideways velocity kick of up to 0.12 m/s, about a
  flick of the finger).
Worlds (--world): desk (flat), rough (40 bits of desk clutter 1-4 mm high under the walking path: cables,
paper, an eraser), stuck (once per episode a random foot is caught for 1.5 s - a spring of up to 3 N holds
it where it was - then let go), terrain (two hurdles 3-10 mm, a ramp up to 14 deg onto a plateau up to 24 mm, in
a ring 13-20 cm around the start: crab/terrain.py, the course terrain training uses), terrain_hard (the same,
hurdles up to 15 mm and ramps up to 21 deg: half again beyond anything in training).
Walking: 12 s of commands (forward, crab-walk left, turn, back, diagonal). Emotes: five random emotes
back to back, each a variant as the robot plays them (crab/emotes.py EmoteStyler: tempo, size, asymmetry, arm
lag, mirror, body turn) under a random mood; PINCH_EMOTE_PLAIN=1 plays the plain keyframes instead.
A fall = tipped over 45 deg or body down on the desk.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from crab import policy_io as pio  # noqa: E402
from crab.emotes import EMOTES, STAND, EmoteStyler  # noqa: E402
from crab.gait import _S, _T, GAITS, Gait  # noqa: E402
from crab.params import JOINT_NAMES  # noqa: E402
from crab.policy_np import POLICY_DIR, NumpyPolicy  # noqa: E402
from sim.benches.common import run_parallel, save_json  # noqa: E402

MODEL = ROOT / "sim" / "models" / "crab.xml"
V, W = _S / _T, 1 / _T
# commands inside the trained policy's envelope (it learned "clearly moving" or "stopped"; slower requests
# are raised to its minimum by the body, and precise slow moves use the scripted gait)
WALK_CMDS = [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.05 * V, 0, 0)), (3.5, (0, 0.06 * V, 0)), (6.0, (0, 0, 0.5 * W)),
             (8.5, (-0.09 * V, 0, 0)), (10.5, (0.045 * V, 0.045 * V, 0))]
SUB = 10


def clutter_model(rng, n=40):
    """The crab model plus n small flat boxes on the floor around the walking path (rough world)."""
    spec = mujoco.MjSpec.from_file(str(MODEL))
    for _i in range(n):
        x, y = rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)
        if abs(x) < 0.06 and abs(y) < 0.11:                     # not under the feet at the start
            x += 0.12 * np.sign(x or 1)
        h = rng.uniform(0.0005, 0.002)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[rng.uniform(0.004, 0.015),
                                rng.uniform(0.002, 0.012), h], pos=[x, y, h],
                                euler=[0, 0, rng.uniform(0, np.pi)], rgba=[0.5, 0.5, 0.5, 1])
    return spec.compile()


def perturbed(seed, hard=True, world="desk"):
    rng = np.random.default_rng(seed)
    if world == "rough":
        m = clutter_model(np.random.default_rng(seed + 50_000))
    elif world in ("terrain", "terrain_hard"):    # two hurdles, a ramp and its plateau (crab/terrain.py)
        from crab.terrain import add_obstacles, terrain_layout
        spec = mujoco.MjSpec.from_file(str(MODEL))
        add_obstacles(spec, *terrain_layout(np, np.random.default_rng(seed + 70_000).random(10),
                                            hard=1.5 if world == "terrain_hard" else 1.0))
        m = spec.compile()
    else:
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
    kind, ctrl, seed, hard, world = args
    m, rng, latency = perturbed(seed, hard, world)
    catch = None
    if world == "stuck":                                     # [foot body, foot geom, when, how long, anchor]
        lg = ("FL", "FR", "RL", "RR")[rng.integers(4)]
        g = m.geom(f"{lg}_foot").id
        catch = [m.geom_bodyid[g], g, rng.uniform(2.0, 9.0), 1.5, None]
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    mujoco.mj_forward(m, d)                                  # body orientation valid before the first read
    act = np.array([m.actuator(j).id for j in JOINT_NAMES])
    sens = Sensors(m)
    stand_h = float(m.key("stand").qpos[2])
    pol = None
    if ctrl == "policy":                         # the emote balance policy (or a candidate)
        pol = NumpyPolicy.load(POLICY_DIR / f"{os.environ.get('PINCH_EMOTE_POLICY', 'emote')}.npz")
    if ctrl == "residual":                       # the scripted trot + learned corrections (training/gait_env.py)
        pol = NumpyPolicy.load(POLICY_DIR / f"{os.environ.get('PINCH_GAIT_POLICY', 'gait')}.npz")
    look = int(round(0.1 / pio.CTRL_DT))
    gait = Gait(GAITS["trot"], com_offset=np.array([0.015 * _S, 0.0])) if kind == "walk" else None
    seq = rng.choice(pio.EMOTE_NAMES, 5) if kind == "emote" else None
    plan = []
    if kind == "emote":
        t = 0.5
        styler = None if os.environ.get("PINCH_EMOTE_PLAIN") else EmoteStyler(np.random.default_rng(seed + 7))
        mood = np.random.default_rng(seed + 11)              # (own streams: the world draws stay as they were)
        for name in seq:
            emo = EMOTES[name]
            if styler is not None:
                side, gaze = mood.choice([0, 1, -1]), mood.uniform(-0.15, 0.15)
                emo = styler.vary(name, arousal=mood.uniform(0, 1), valence=mood.uniform(-1, 1),
                                  sleepy=mood.uniform(0, 1), gain=mood.uniform(0.3, 1.0),
                                  side=int(side) or None, gaze=gaze if mood.random() < 0.5 else None)
            plan.append((t, emo))
            t += emo.duration + 0.3
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
            if ctrl == "residual":                 # as pinch/body.py Body._residual does it on the robot
                for _ in range(SUB):
                    ref = gait.step(pio.CTRL_DT / SUB, cmd)
                g = copy.deepcopy(gait)                # the look-ahead: the gait run on 0.1 s, same command
                for i in range(look):
                    ahead = g.step(pio.CTRL_DT, cmd, solve=i == look - 1)
                obs = pio.emote_observation(np, gyro, grav, q, qd, last, ref, ahead, None, (t % 6.0) / 6.0)
                a = np.clip(pol(obs), -1, 1)
                k = getattr(pol, "meta", {}).get("action_scale", pio.EMOTE_ACTION_SCALE) / pio.EMOTE_ACTION_SCALE
                target = pio.emote_targets(np, ref, (prev if latency else a) * k)
                prev, last = a, a
            else:                                      # the scripted trot alone
                for _ in range(SUB):
                    target = gait.step(pio.CTRL_DT / SUB, cmd)
            v_body = d.xmat[sens.body].reshape(3, 3).T @ d.qvel[0:3]
            cn = np.linalg.norm(cmd[:2])
            if cn > 1e-3 and t > 1.5:
                track.append(np.dot(v_body[:2], cmd[:2]) / cn ** 2)
        else:
            cur = next(((t0, n) for t0, n in reversed(plan) if t >= t0), None)
            if cur is None:
                ref, ahead, prog = STAND, STAND, 0.0
            else:
                emo = cur[1]
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
            if catch is not None:
                b, g, t0, dur, anchor = catch
                d.qfrc_applied[:] = 0
                if t0 <= t < t0 + dur:
                    p = d.geom_xpos[g].copy()
                    if anchor is None:
                        catch[4] = anchor = p
                    f = 200.0 * (anchor - p)
                    f[2] = 0.0
                    n = np.linalg.norm(f)
                    if n > 3.0:
                        f *= 3.0 / n
                    mujoco.mj_applyFT(m, d, f, np.zeros(3), p, b, d.qfrc_applied)
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
        for ctrl in ("scripted", "policy", "residual"):
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
    ap.add_argument("--world", default="desk", choices=["desk", "rough", "stuck", "terrain", "terrain_hard"])
    ap.add_argument("--json", default=None, help="also write the results here")
    a = ap.parse_args()
    n = 8 if a.quick else a.episodes
    jobs = []
    for kind in a.what.split(","):
        jobs += [(kind, "scripted", 1000 + s, True, a.world) for s in range(n)]
        ctrl, learned = (("residual", os.environ.get("PINCH_GAIT_POLICY", "gait")) if kind == "walk" else
                         ("policy", os.environ.get("PINCH_EMOTE_POLICY", "emote")))
        if (POLICY_DIR / f"{learned}.npz").exists():
            jobs += [(kind, ctrl, 1000 + s, True, a.world) for s in range(n)]
        else:
            print(f"no policies/{learned}.npz yet - skipping the trained {kind} policy")
    rows = run_parallel(episode, jobs, 12)
    save_json(a.json, rows)
    s = summary(rows)
    title = {"desk": "", "rough": ": rough desk", "stuck": ": a foot gets caught", "terrain": ": hurdles and ramps",
             "terrain_hard": ": higher hurdles, steeper ramps"}[a.world]
    lines = [f"# Robustness tests (simulation){title}", "",
             "Generated by `python sim/benches/robustness.py`. Every episode draws a different robot and world, on top of "
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
    for kind, ctrl in (("walk", "policy"), ("walk", "residual"), ("emote", "policy")):
        p, sc = s.get((kind, ctrl)), s.get((kind, "scripted"))
        if p:
            ok = p["falls"] <= 0.05 * p["n"] and (sc is None or p["falls"] <= sc["falls"])
            ok = ok and (kind != "walk" or p["track"] >= 0.6)
            verdicts.append(f"- **{kind} ({ctrl}): {'PASS' if ok else 'FAIL'}**")
    lines += verdicts
    if a.world != "desk":
        lines.insert(3, f"World: **{a.world}** - see the module docstring.")
    out = "ROBUSTNESS.md" if a.world == "desk" else f"ROBUSTNESS_{a.world}.md"
    if os.environ.get("PINCH_GAIT_POLICY", "gait") == "gait" and "PINCH_EMOTE_POLICY" not in os.environ:
        (ROOT / "docs" / out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

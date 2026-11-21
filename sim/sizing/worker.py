"""Evaluate ONE design in its own process (the existing gait reads the robot size at import time).

    python sim/sizing/worker.py '<json design>' out.json

The existing gait generator (crab/gait.py) is reused unchanged: before it is imported, crab.params' leg
geometry (links, hip mounts, foot radius) is overwritten with the scaled values, so crab.kinematics and crab.gait
see the scaled crab, and the gait's own rules scale with it (lengths x S, periods x sqrt(S), S = tibia / 100 mm).
Everything else (STAND pose, joint ranges, gait shapes) is today's.
"""

import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from dataclasses import replace  # noqa: E402

import mujoco  # noqa: E402
from model import Design, compile_stand, servo_for_joint, span_and_height, stand_height  # noqa: E402
from servos import repo_servo, sts3215  # noqa: E402

from crab.params import JOINT_NAMES  # noqa: E402

DT = 0.002
G = 9.81
SERVO_BY_NAME = {"STS3032": lambda: repo_servo("STS3032"), "STS3215-m1": lambda: sts3215("m1"),
                 "STS3215-m6": lambda: sts3215("m6")}


def patch_geometry(s: float):
    """Scale crab.params' leg geometry in place, BEFORE crab.kinematics / crab.gait are imported."""
    import crab.params as cp
    leg = cp.PARAMS.leg
    for k, v in (("coxa_len", 0.022 * s), ("femur_len", 0.030 * s), ("tibia_len", 0.052 * s),
                 ("foot_radius", 0.005 * s)):
        object.__setattr__(leg, k, v)
    for k, ((x, y, z), yaw) in list(cp.PARAMS.leg_mounts.items()):
        cp.PARAMS.leg_mounts[k] = ((x * s, y * s, z), yaw)


def design_from(cfg: dict) -> Design:
    sv = SERVO_BY_NAME[cfg["servo"]]()
    if "armature_scale" in cfg:                    # diagnostic: how much of the torque is rotor inertia?
        sv = replace(sv, armature=sv.armature * cfg["armature_scale"])
    return Design(cfg["name"], cfg["s"], sv, cfg.get("batt_mah", 850.0),
                  cfg.get("batt_v", 7.4), cfg.get("print_exp", 3.0), cfg.get("extra_electronics", 0.0))


class Meter:
    """Per-step monitor of one scenario: torque, speed, saturation, electrical power."""

    def __init__(self, d: Design):
        sv = [servo_for_joint(d, j) for j in JOINT_NAMES]
        self.ke = np.array([x.ke for x in sv])
        self.kt = np.array([x.kt for x in sv])
        self.R = np.array([x.R for x in sv])
        self.vin = np.array([x.vin for x in sv])
        self.idle = np.array([x.idle_a for x in sv])
        self.frng = np.array([x.forcerange for x in sv])
        self.n = 0
        self.sq = np.zeros(len(sv))
        self.peak_tau = np.zeros(len(sv))
        self.peak_w = np.zeros(len(sv))
        self.peak_cmd = np.zeros(len(sv))
        self.sat = np.zeros(len(sv))
        self.p_phys = []
        self.p_ub = []
        self.leg = np.array([n[0] != "L" and n[0] != "R" or n[1] != "_" for n in JOINT_NAMES])

    def update(self, f, w, cmd_speed):
        tau = f - self.ke * w                           # electromagnetic torque of the motor
        i = tau / self.kt
        p = i * i * self.R + np.maximum(tau * w, 0.0) + self.idle * self.vin   # copper + mechanical + idle
        ub = self.vin * (np.abs(i) + self.idle)        # the repo's way: rail voltage x motor current
        self.n += 1
        self.sq += tau ** 2
        self.peak_tau = np.maximum(self.peak_tau, np.abs(tau))
        self.peak_w = np.maximum(self.peak_w, np.abs(w))
        self.peak_cmd = np.maximum(self.peak_cmd, cmd_speed)
        self.sat += np.abs(f) >= 0.98 * self.frng
        self.p_phys.append((p[self.leg].sum(), p[~self.leg].sum()))
        self.p_ub.append((ub[self.leg].sum(), ub[~self.leg].sum()))


def run(m, data, d: Design, ctrl_fn, seconds, t_measure=0.0):
    """Drive the crab with ctrl_fn(t) -> 20 joint targets through the firmware model (target slew limit at the
    servo's max velocity, command delay), for `seconds`; statistics only after t_measure."""
    sv = [servo_for_joint(d, j) for j in JOINT_NAMES]
    maxv = np.array([x.max_vel for x in sv])
    lines = [deque([np.array(data.ctrl[i])] * max(0, round(x.delay / DT))) for i, x in enumerate(sv)]
    tgt = np.array(data.ctrl, float)
    prev = tgt.copy()
    meter = Meter(d)
    n = int(seconds / DT)
    start_pos = None
    min_z, max_tilt = 1e9, 0.0
    for k in range(n):
        t = k * DT
        q = np.asarray(ctrl_fn(t), float)
        cmd_speed = np.abs(q - prev) / DT
        prev = q
        tgt += np.clip(q - tgt, -maxv * DT, maxv * DT)
        out = np.empty_like(tgt)
        for i, ln in enumerate(lines):
            if ln:
                ln.append(tgt[i])
                out[i] = ln.popleft()
            else:
                out[i] = tgt[i]
        data.ctrl[:] = out
        mujoco.mj_step(m, data)
        if t >= t_measure:
            if start_pos is None:
                start_pos = (data.qpos[:2].copy(), data.xmat[1].reshape(3, 3)[:, 0].copy())
            meter.update(data.actuator_force, data.actuator_velocity, cmd_speed)
            min_z = min(min_z, data.qpos[2])
            max_tilt = max(max_tilt, math.degrees(math.acos(np.clip(data.xmat[1].reshape(3, 3)[2, 2], -1, 1))))
    heading = start_pos[1][:2] / max(np.linalg.norm(start_pos[1][:2]), 1e-9)
    disp = data.qpos[:2] - start_pos[0]
    t_win = seconds - t_measure
    return dict(meter=meter, fwd=float(disp @ heading / t_win), side=float(disp @ np.array([-heading[1], heading[0]]) / t_win),
                min_z=float(min_z), max_tilt=float(max_tilt))


def summarise(res, d: Design) -> dict:
    mt = res["meter"]
    legs = np.where(mt.leg)[0]
    kinds = {"coxa": [], "femur": [], "tibia": []}
    for i in legs:
        kinds[JOINT_NAMES[i].split("_", 1)[1]].append(i)
    sv = d.servo
    out = {"fwd": res["fwd"], "side": res["side"], "min_z": res["min_z"], "tilt": res["max_tilt"],
           "p_phys_w": float(np.mean([a + b for a, b in mt.p_phys])), "p_ub_w": float(np.mean([a + b for a, b in mt.p_ub])),
           "p_phys_peak_w": float(np.max([a + b for a, b in mt.p_phys])),
           "p_phys_legs_w": float(np.mean([a for a, _ in mt.p_phys])), "p_phys_arms_w": float(np.mean([b for _, b in mt.p_phys])),
           "p_ub_legs_w": float(np.mean([a for a, _ in mt.p_ub])), "p_ub_arms_w": float(np.mean([b for _, b in mt.p_ub])),
           "joint": {}}
    for kd, ids in kinds.items():
        rms = np.sqrt(mt.sq[ids] / mt.n)
        out["joint"][kd] = {
            "rms_over_rated": float(rms.max() / sv.rated), "peak_over_stall": float(mt.peak_tau[ids].max() / sv.ds_stall),
            "peak_over_model_stall": float(mt.peak_tau[ids].max() / sv.forcerange),
            "peak_tau": float(mt.peak_tau[ids].max()), "rms_tau": float(rms.max()),
            "peak_speed": float(mt.peak_w[ids].max()), "peak_cmd_speed": float(mt.peak_cmd[ids].max()),
            "sat_pct": float(100 * mt.sat[ids].max() / mt.n)}
    out["peak_speed"] = max(v["peak_speed"] for v in out["joint"].values())
    out["peak_cmd_speed"] = max(v["peak_cmd_speed"] for v in out["joint"].values())
    return out


def main():
    cfg = json.loads(sys.argv[1])
    out_path = Path(sys.argv[2])
    d = design_from(cfg)
    result = {"cfg": cfg, "mass_items_g": {k: v * 1000 for k, v in d.mass_items().items()}, "mass_kg": d.mass,
              "servo": d.servo.name}

    # ---- static: standing, joint friction removed so actuator force = the torque the pose needs
    m0, d0 = compile_stand(d, frictionless=True)
    for _ in range(int(3.0 / DT)):
        mujoco.mj_step(m0, d0)
    fs = []
    for _ in range(int(0.5 / DT)):
        mujoco.mj_step(m0, d0)
        fs.append(d0.actuator_force.copy())
    f_stand = np.mean(fs, axis=0)
    stat = {}
    for kd in ("coxa", "femur", "tibia"):
        ids = [i for i, n in enumerate(JOINT_NAMES) if n.endswith("_" + kd) and n[1] != "_"]
        tmax = float(np.abs(f_stand[ids]).max())
        stat[kd] = {"tau": tmax, "of_stall": tmax / d.servo.ds_stall, "of_model_stall": tmax / d.servo.forcerange,
                    "of_rated": tmax / d.servo.rated}
    result["static"] = stat
    result["geometry"] = {"hip_height": float(stand_height(d)), "tilt_stand": float(d0.xmat[1].reshape(3, 3)[2, 2])}

    # ---- the real model (with friction): stand, then gait scenarios
    m, data = compile_stand(d)
    for _ in range(int(1.0 / DT)):
        data.ctrl[:] = _stand_vec()
        mujoco.mj_step(m, data)
    L_, W_, H_ = span_and_height(m, data)
    result["span_cm"] = [L_ * 100, W_ * 100, H_ * 100]
    result["body_z_cm"] = float(data.qpos[2] * 100)
    com = data.subtree_com[1] - data.xpos[1]
    snap = (data.qpos.copy(), data.qvel.copy())
    result["com_offset_mm"] = [float(com[0] * 1000), float(com[1] * 1000), float(com[2] * 1000)]
    result["foot_height_ok"] = bool(data.qpos[2] > 0.5 * stand_height(d))

    patch_geometry(d.s)
    from crab.emotes import EMOTES, STAND
    from crab.gait import _S, _T, GAITS, Gait

    GAITS["trot"] = replace(GAITS["trot"], period=GAITS["trot"].period * cfg.get("period_mult", 1.0))

    def fresh():
        data.qpos[:], data.qvel[:] = snap
        data.ctrl[:] = STAND
        data.qacc_warmstart[:] = 0
        mujoco.mj_forward(m, data)
        for _ in range(250):               # half a second to settle
            mujoco.mj_step(m, data)

    result["gait"] = {"S": _S, "T": _T, "trot_period_s": GAITS["trot"].period, "trot_max_step_mm": GAITS["trot"].max_step * 1000,
                      "trot_step_height_mm": GAITS["trot"].step_height * 1000}
    comxy = np.array(com[:2])

    def top_speed(gait_name, direction):
        g = Gait(GAITS[gait_name], com_offset=comxy)
        cmd = np.array(direction, float)
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if g.feasible_scale(cmd * mid) >= 1.0 - 1e-9 else (lo, mid)
        return lo

    result["cap_speed"] = {"trot_fwd": top_speed("trot", (1, 0, 0)), "trot_side": top_speed("trot", (0, 1, 0)),
                           "crawl_fwd": top_speed("crawl", (1, 0, 0)), "crawl_side": top_speed("crawl", (0, 1, 0))}

    scen = {}
    fresh()
    scen["stand"] = summarise(run(m, data, d, lambda t: STAND, 4.0, 1.0), d)
    fresh()
    idle = EMOTES["idle"]
    scen["idle"] = summarise(run(m, data, d, idle.target, 6.4, 0.0), d)

    def walk(name, gait_name, cmd, secs=9.0, t_meas=4.0):
        fresh()
        g = Gait(GAITS[gait_name], com_offset=comxy)
        scen[name] = summarise(run(m, data, d, lambda t: g.step(DT, cmd), secs, t_meas), d)
        scen[name]["cmd"] = [float(c) for c in cmd]

    for v in (0.05, 0.10):
        walk(f"trot_fwd_{int(v*100)}", "trot", (v, 0, 0))
        walk(f"trot_side_{int(v*100)}", "trot", (0, v, 0))
    cap = result["cap_speed"]
    walk("trot_fwd_cap", "trot", (cap["trot_fwd"], 0, 0))
    walk("trot_side_cap", "trot", (0, cap["trot_side"], 0))
    walk("crawl_fwd_cap", "crawl", (cap["crawl_fwd"], 0, 0), 12.0, 6.0)
    result["scenarios"] = scen
    out_path.write_text(json.dumps(result, indent=1))
    print("done", cfg["name"])


def _stand_vec():
    from crab.emotes import STAND
    return STAND


if __name__ == "__main__":
    main()

"""Situation bench for design v2 layouts (R-04, R-34, R-05): can this body do what Pinch's days need?

    python sim/situations.py                     # every candidate in CANDIDATES
    python sim/situations.py --only today,tall   # some

Each candidate is a set of overrides (stance angles, shell size, arm mount, arm swing range) applied without editing
any file: the model is built from crab/params.py and configs/layout_v2.toml with them (sim/build_model.py --primitive
--out ...) in a fresh process. Situations (scripted gaits, the model's own position servos):
  clearance   the shell's lowest point above the floor, standing (target >= 25 mm: 15 mm thresholds + tilt and sag)
  threshold   a 15 mm x 30 mm bar across the path, walked over forward and sideways (the climb gait, 25 mm lift)
  cable       a 10 mm round cable across the path, forward and sideways
  carpet      a soft floor (compliant contact, friction 1.2): forward speed reached / commanded
  slope       walking up a 10 deg slope: forward speed reached / commanded
  arms        shoulder height, the highest claw tip (a wave above the face), whether the jaws reach the desk, swing range
A crossing passes when the body ends past the obstacle, upright, within the time.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = {
    "today": {},
    "thigh15": {"thigh": 15},                                        # thighs lower: the body stands higher
    # v2.1 after the research (docs/research/v2_1): the 104 mm shell kept (the upright arm yaw servos fit at mid-height),
    # the body raised by the stance, the arms at mid-shell with a wide swing and a deep lift
    "v21_t10": {"thigh": 10, "arm_yaw": [-60, 80], "arm_pitch": [-70, 72],
                "layout": {"part.shoulder_l.joint_mm": [60.0, 40, 0.0], "part.shoulder_r.joint_mm": [60.0, -40, 0.0]}},
    "v21_t12": {"thigh": 12, "arm_yaw": [-60, 80], "arm_pitch": [-70, 72],
                "layout": {"part.shoulder_l.joint_mm": [60.0, 40, 0.0], "part.shoulder_r.joint_mm": [60.0, -40, 0.0]}},
    "thigh0": {"thigh": 0},
    "shell40": {"layout": {"chassis.shell_half_mm": [84.0, 94.0, 40.0], "chassis.shell_center_mm": [6.9, 0.0, 6.0]}},
    "thigh15_shell44": {"thigh": 15, "layout": {"chassis.shell_half_mm": [84.0, 94.0, 44.0],
                                                 "chassis.shell_center_mm": [6.9, 0.0, 4.0]}},
}


def env_for(c):
    e = dict(os.environ, CRAB_DESIGN="v2")
    if "thigh" in c:
        e["CRAB_THIGH_DEG"] = str(c["thigh"])
    if "shin" in c:
        e["CRAB_SHIN_DEG"] = str(c["shin"])
    if "arm_yaw" in c:
        e["CRAB_ARM_YAW_DEG"] = ",".join(str(v) for v in c["arm_yaw"])
    if "arm_pitch" in c:
        e["CRAB_ARM_PITCH_DEG"] = ",".join(str(v) for v in c["arm_pitch"])
    if "layout" in c:
        e["CRAB_LAYOUT_JSON"] = json.dumps(c["layout"])
    return e


# ---------------------------------------------------------------------------------------------- inside one process
def _run_one(xml):
    sys.path.insert(0, str(ROOT))
    import mujoco
    import numpy as np

    from crab.emotes import STAND
    from crab.gait import GAITS, Gait
    from crab.params import JOINT_NAMES, PARAMS

    base = mujoco.MjSpec.from_file(xml)

    def model(extra=None, carpet=False, slope=0.0):
        spec = base.copy() if hasattr(base, "copy") else mujoco.MjSpec.from_file(xml)
        if extra:
            extra(spec)
        m = spec.compile()
        if carpet:
            fl = m.geom("floor").id
            m.geom_solref[fl] = [0.03, 1.0]
            m.geom_solimp[fl] = [0.5, 0.95, 0.008, 0.5, 2.0]
            m.geom_friction[fl] = [1.2, 0.02, 0.001]
        if slope:
            g = 9.81
            m.opt.gravity[:] = [-g * np.sin(np.radians(slope)), 0.0, -g * np.cos(np.radians(slope))]
        return m

    def start(m):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        act = np.array([m.actuator(j).id for j in JOINT_NAMES])
        for _ in range(500):
            d.ctrl[act] = STAND
            mujoco.mj_step(m, d)
        return d, act

    def upright(m, d):
        return d.xmat[m.body("body").id].reshape(3, 3)[2, 2] > np.cos(np.radians(40))

    peak = [0.0]

    def walk(m, d, act, cmd, secs, gait="trot", lift=None):
        gp = GAITS[gait]
        if lift:
            from dataclasses import replace
            gp = replace(gp, step_height=lift)
        g = Gait(gp, com_offset=np.zeros(2))
        g.reset()
        for _ in range(int(secs / m.opt.timestep)):
            d.ctrl[act] = g.step(m.opt.timestep, cmd)
            mujoco.mj_step(m, d)
            peak[0] = max(peak[0], float(np.abs(d.actuator_force[act[:12]]).max()))
            if not upright(m, d):
                return False
        return True

    out = {}
    m = model()
    d, act = start(m)
    body = m.body("body").id
    shell = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_contype[g]]
    low = 9.0
    for g in shell:
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R = d.geom_xmat[g].reshape(3, 3)
        low = min(low, d.geom_xpos[g][2] + (R @ c)[2] - (np.abs(R) @ h)[2])
    out["clearance_mm"] = round(low * 1000, 1)
    out["stand_h_mm"] = round(d.qpos[2] * 1000, 1)

    # obstacles across the path (x for forward, y for sideways), 0.12 m ahead of the body centre
    def crossing(kind, axis):
        def add(spec):
            if kind == "threshold":
                size, z, typ = [0.015, 0.25, 0.0075], 0.0075, mujoco.mjtGeom.mjGEOM_BOX
            else:
                size, z, typ = [0.005, 0.25, 0], 0.005, mujoco.mjtGeom.mjGEOM_CAPSULE
            g = spec.worldbody.add_geom(name="obstacle", type=typ, size=size, pos=[0.12, 0, z] if axis == 0 else [0, 0.12, z],
                                        rgba=[0.3, 0.3, 0.8, 1])
            if typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
                g.quat = [0.7071068, 0.7071068, 0, 0] if axis == 0 else [0.7071068, 0, 0.7071068, 0]
            elif axis == 1:
                g.size = [0.25, 0.015, 0.0075]
        mm = model(add)
        dd, aa = start(mm)
        cmd = (0.05, 0.0, 0.0) if axis == 0 else (0.0, 0.04, 0.0)
        ok = walk(mm, dd, aa, cmd, 14.0, gait="climb", lift=0.025)
        p = dd.qpos[axis]
        return bool(ok and p > 0.12 + 0.10), round(float(p) * 1000)

    for kind in ("threshold", "cable"):
        for axis, nm in ((0, "fwd"), (1, "side")):
            ok, p = crossing(kind, axis)
            out[f"{kind}_{nm}"] = ok
            out[f"{kind}_{nm}_mm"] = p

    for name, kw, cmd in (("carpet", dict(carpet=True), (0.08, 0, 0)), ("slope", dict(slope=10.0), (0.06, 0, 0)),
                          ("flat", {}, (0.08, 0, 0))):
        mm = model(**kw)
        dd, aa = start(mm)
        x0 = dd.qpos[0]
        peak[0] = 0.0
        ok = walk(mm, dd, aa, cmd, 8.0)
        out[f"{name}_leg_peak_pct"] = round(100 * peak[0] / 1.91)       # % of the STS3215's stall (1.91 N*m)
        out[f"{name}_speed"] = round(float((dd.qpos[0] - x0) / 8.0 / cmd[0]), 2) if ok else 0.0

    # arms: shoulder height; the highest claw tip over the lift range; can the jaws reach the desk
    d, act = start(m)
    sh = m.body("L_shoulder").id
    out["shoulder_mm"] = round(d.xpos[sh][2] * 1000, 1)
    jp = m.joint("L_shoulder_pitch").id
    qa = m.jnt_qposadr[jp]
    tip, lowest = -9.0, 9.0
    site = m.site("L_grip_site").id
    for p in np.linspace(*m.jnt_range[jp], 30):
        d.qpos[qa] = p
        mujoco.mj_kinematics(m, d)
        tip = max(tip, d.site_xpos[site][2])
        for g in range(m.ngeom):
            if (m.geom(g).name or "").startswith("L_pollex"):
                R = d.geom_xmat[g].reshape(3, 3)
                lowest = min(lowest, d.geom_xpos[g][2] - (np.abs(R) @ m.geom_size[g])[2])
    out["claw_top_mm"] = round(tip * 1000, 1)
    out["reaches_desk"] = bool(lowest < 0.002)
    out["arm_swing_deg"] = [round(float(np.degrees(v))) for v in PARAMS.arm.shoulder_yaw_range]
    out["mass_kg"] = round(float(sum(m.body_mass)), 3)
    print("RESULT " + json.dumps(out))


def run(name, c):
    xml = Path(tempfile.mkdtemp()) / "crab_v2.xml"
    e = env_for(c)
    r = subprocess.run([sys.executable, str(ROOT / "sim" / "build_model.py"), "--design", "v2", "--primitive", "--out", str(xml)],
                       env=e, capture_output=True, text=True)
    if r.returncode:
        return {"error": r.stderr[-400:]}
    r = subprocess.run([sys.executable, __file__, "--one", str(xml)], env=e, capture_output=True, text=True)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    return json.loads(line[-1][7:]) if line else {"error": (r.stderr or r.stdout)[-600:]}


TARGETS = dict(clearance_mm=25.0, carpet_speed=0.6, slope_speed=0.5, shoulder_mm=65.0)
LIMITS = dict(flat_leg_peak_pct=80, slope_leg_peak_pct=80)              # R-03: peak <= 80 % of stall


def verdict(r):
    if "error" in r:
        return False, ["error"]
    bad = [k for k in ("threshold_fwd", "threshold_side", "cable_fwd", "cable_side", "reaches_desk") if not r.get(k)]
    bad += [k for k, t in TARGETS.items() if r.get(k, 0) < t]
    bad += [k for k, t in LIMITS.items() if r.get(k, 999) > t]
    return not bad, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.one:
        return _run_one(a.one)
    names = a.only.split(",") if a.only else list(CANDIDATES)
    allr = {}
    for n in names:
        r = run(n, CANDIDATES[n])
        ok, bad = verdict(r)
        allr[n] = r
        print(f"{n:14s} {'PASS' if ok else 'fails: ' + ', '.join(bad)}")
        print("    " + json.dumps(r))
    if a.json:
        Path(a.json).write_text(json.dumps(allr, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

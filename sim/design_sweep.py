"""Design sweep for v2: score a grid of layouts on every requirement at once and rank them by their smallest margin.

    python sim/design_sweep.py                       # the default grid, 10 at a time
    python sim/design_sweep.py --jobs 6 --json .ops/sweep.json

Swept (the numbers the owner's situations depend on most): thigh angle at the stance (body height), shin angle (where
the feet land, how much the thighs carry), shoulder height (the arms' yaw axis, body z). Each candidate is a model
built without editing any file (sim/build_model.py --primitive --out, overrides through CRAB_* variables, as
sim/situations.py does) and measured with the measured servo model (Rhoban BAM, sim/v2_check.py):
  clearance   shell's lowest point above the floor, standing               >= 30 mm  (15 mm threshold + swing and pitch)
  peak        legs' peak torque, forward and sideways at the speed caps    <= 80 % of stall   (R-03)
  rms         hottest leg joint's RMS torque, the same walks and standing  <= 70 % of rated   (R-03)
  carpet      forward speed reached on a soft floor / commanded            >= 0.6
  slope       forward speed reached up 10 deg / commanded                  >= 0.5
  shoulder    the arm axis at least 45 % up the shell (mid-height; arms/notes.md suggests 55-65 %)
  fits        the arm yaw servo's case fits inside the shell, 2 mm wall, upside down above its horn ("down", today) or
              horn up with the case hanging below the joint ("up")
  feet        feet at least 40 mm outboard of the hips                     (Chen 2022: never under the hips)
  desk        the jaws reach the desk;  wave: the claw tip's highest point
Margin of a requirement = how far inside its limit the candidate is, as a fraction of the limit; the score is the
smallest margin (a candidate is only as good as its weakest requirement). Obstacle crossing is not scored here: a blind
scripted gait catches its shins on a bar whatever the layout (measured); the learned terrain gait is checked on the
winner instead (quick training on msi-1).
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = dict(thigh=(0, 5, 10, 15), shin=(85, 90, 95), shoulder_z=(-6.0, 0.0, 6.0, 12.0, 18.0), yaw=("down", "up"))
SHOULDER_X, SHOULDER_Y = 56.0, 40.0


def env_for(c, xml):
    layout = {"part.shoulder_l.joint_mm": [SHOULDER_X, SHOULDER_Y, c["shoulder_z"]],
              "part.shoulder_r.joint_mm": [SHOULDER_X, -SHOULDER_Y, c["shoulder_z"]]}
    return dict(os.environ, CRAB_YAW_SERVO=c["yaw"], CRAB_DESIGN="v2", CRAB_THIGH_DEG=str(c["thigh"]), CRAB_SHIN_DEG=str(c["shin"]),
                CRAB_LAYOUT_JSON=json.dumps(layout), CRAB_MODEL_PATH=str(xml), PYTHONIOENCODING="utf-8")


def measure(xml):
    """Inside one process (the candidate's CRAB_* variables set): every measurement, printed as one JSON line."""
    sys.path.insert(0, str(ROOT))
    import mujoco
    import numpy as np

    from crab.electronics import Layout
    from crab.emotes import STAND
    from crab.kinematics import _mirror, _mount, leg_fk
    from crab.params import JOINT_NAMES, LEG_IDS
    from sim import v2_check as V

    out = {}
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    act = np.array([m.actuator(j).id for j in JOINT_NAMES])
    for _ in range(500):
        d.ctrl[act] = STAND
        mujoco.mj_step(m, d)
    body = m.body("body").id
    low = 9.0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == body and m.geom_contype[g]:
            c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            R = d.geom_xmat[g].reshape(3, 3)
            low = min(low, d.geom_xpos[g][2] + (R @ c)[2] - (np.abs(R) @ h)[2])
    out["clearance"] = low * 1000
    sh = m.body("L_shoulder").id
    shell = Layout().chassis
    bottom = d.xpos[body][2] * 1000 + shell["shell_center_mm"][2] - shell["shell_half_mm"][2]
    out["shoulder_frac"] = (d.xpos[sh][2] * 1000 - bottom) / (2 * shell["shell_half_mm"][2])
    idx = {lg: [JOINT_NAMES.index(f"{lg}_{k}") for k in ("coxa", "femur", "tibia")] for lg in LEG_IDS}
    from crab.shapes import Shell
    sh_shape = Shell.from_chassis(shell)
    jz = float(Layout()["shoulder_l"].d["joint_mm"][2])
    jx = float(Layout()["shoulder_l"].d["joint_mm"][0])
    # the arm yaw servo: "down" = upside down above its horn (today), "up" = horn up, the case hanging below the joint
    if os.environ.get("CRAB_YAW_SERVO", "down") == "down":
        z_case = jz + 9.5 + 19.2 + 13.75                            # its top (cad/v2/arms.py YAW_HORN_Z, horn face)
    else:
        z_case = jz - 9.5 - 19.2 - 13.75                            # its bottom
    corners = [np.array([jx + dx, 40 + dy, z_case]) for dx in (-17.3, 5.7) for dy in (-6.0, 6.0)]
    out["fits"] = bool(all(sh_shape.inside(c, margin=2.0) for c in corners))
    out["feet_out"] = min(abs(leg_fk(lg, STAND[idx[lg]])[1]) - abs(_mirror(_mount(lg)[0], lg)[1]) for lg in LEG_IDS) * 1000
    # arms: the highest claw tip, can the jaws reach the desk
    jp = m.joint("L_shoulder_pitch").id
    site = m.site("L_grip_site").id
    tip, lowest = -9.0, 9.0
    for p in np.linspace(*m.jnt_range[jp], 30):
        d.qpos[m.jnt_qposadr[jp]] = p
        mujoco.mj_kinematics(m, d)
        tip = max(tip, d.site_xpos[site][2])
        for g in range(m.ngeom):
            if (m.geom(g).name or "").startswith("L_pollex"):
                R = d.geom_xmat[g].reshape(3, 3)
                lowest = min(lowest, d.geom_xpos[g][2] - (np.abs(R) @ m.geom_size[g])[2])
    out["wave_top"] = tip * 1000
    out["desk"] = bool(lowest < 0.002)
    # torque with the measured servo model: standing, forward and sideways at the caps
    peaks, rmss = [], []
    st = V.stand("bam", 3.0)
    rmss.append(st["rms_rated"])
    for cmd in ((0.10, 0.0, 0.0), (0.0, 0.10, 0.0)):
        s = V.walk("bam", cmd, seconds=6.0)
        peaks.append(s["peak_ds"])
        rmss.append(s["rms_rated"])
        out["fell"] = out.get("fell", False) or bool(s["fell"])
    out["peak"], out["rms"] = max(peaks), max(rmss)
    # carpet and slope: speed reached / commanded (first-order servos, as sim/situations.py)
    for name, carpet, slope in (("carpet", True, 0.0), ("slope", False, 10.0)):
        mm = mujoco.MjModel.from_xml_path(xml)
        if carpet:
            fl = mm.geom("floor").id
            mm.geom_solref[fl], mm.geom_solimp[fl] = [0.03, 1.0], [0.5, 0.95, 0.008, 0.5, 2.0]
            mm.geom_friction[fl] = [1.2, 0.02, 0.001]
        if slope:
            mm.opt.gravity[:] = [-9.81 * np.sin(np.radians(slope)), 0, -9.81 * np.cos(np.radians(slope))]
        dd = mujoco.MjData(mm)
        mujoco.mj_resetDataKeyframe(mm, dd, mm.key("stand").id)
        for _ in range(500):
            dd.ctrl[act] = STAND
            mujoco.mj_step(mm, dd)
        from crab.gait import GAITS, Gait
        g = Gait(GAITS["trot"], com_offset=np.zeros(2))
        g.reset()
        x0, cmd = dd.qpos[0], 0.08 if carpet else 0.06
        for _ in range(int(8.0 / mm.opt.timestep)):
            dd.ctrl[act] = g.step(mm.opt.timestep, (cmd, 0, 0))
            mujoco.mj_step(mm, dd)
        out[name] = float((dd.qpos[0] - x0) / 8.0 / cmd)
    out["mass"] = float(sum(m.body_mass))
    print("RESULT " + json.dumps(out))


# requirement -> (kind, limit): "min" means value >= limit, "max" value <= limit, "band" lo <= value <= hi
REQ = dict(clearance=("min", 30.0), peak=("max", 80.0), rms=("max", 70.0), carpet=("min", 0.6), slope=("min", 0.5),
           shoulder_frac=("min", 0.45), feet_out=("min", 40.0), desk=("true", None), fits=("true", None))


def margins(r):
    out = {}
    for k, (kind, lim) in REQ.items():
        v = r[k]
        if kind == "min":
            out[k] = (v - lim) / lim
        elif kind == "max":
            out[k] = (lim - v) / lim
        elif kind == "band":
            lo, hi = lim
            out[k] = min(v - lo, hi - v) / ((hi - lo) / 2)
        else:
            out[k] = 1.0 if v else -1.0
    if r.get("fell"):
        out["fell"] = -1.0
    return out


def run(c):
    xml = Path(tempfile.mkdtemp()) / "crab_v2.xml"
    e = env_for(c, xml)
    r = subprocess.run([sys.executable, str(ROOT / "sim" / "build_model.py"), "--design", "v2", "--primitive", "--out", str(xml)],
                       env=e, capture_output=True, text=True)
    if r.returncode:
        return c, {"error": r.stderr[-300:]}
    r = subprocess.run([sys.executable, __file__, "--one", str(xml)], env=e, capture_output=True, text=True)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    return c, (json.loads(line[-1][7:]) if line else {"error": (r.stderr or r.stdout)[-400:]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--json", default=None)
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        return measure(a.one)
    cands = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    rows = []
    with ThreadPoolExecutor(a.jobs) as ex:
        for c, r in ex.map(run, cands):
            if "error" in r:
                print("error", c, r["error"][-200:], flush=True)
                continue
            mg = margins(r)
            rows.append(dict(c, **r, score=min(mg.values()), weakest=min(mg, key=mg.get), passes=all(v >= 0 for v in mg.values())))
            print(f"{c}  score {rows[-1]['score']:+.2f} ({rows[-1]['weakest']})", flush=True)
    rows.sort(key=lambda x: -x["score"])
    print("\nrank thigh shin shoulder_z | clearance peak rms carpet slope shoulder% feet_out wave desk | score weakest")
    for i, x in enumerate(rows[:12]):
        print(f"{i + 1:4d} {x['thigh']:5.0f} {x['shin']:4.0f} {x['shoulder_z']:+6.1f} | {x['clearance']:9.1f} {x['peak']:4.0f} "
              f"{x['rms']:4.0f} {x['carpet']:6.2f} {x['slope']:5.2f} {100 * x['shoulder_frac']:8.0f} {x['feet_out']:8.1f} "
              f"{x['wave_top']:4.0f} {str(x['desk']):5s} {'fits' if x['fits'] else 'NO':4s} | {x['score']:+.2f} {x['weakest']}{'' if x['passes'] else '  (fails)'}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

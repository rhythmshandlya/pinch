"""Walk the crab with the scripted gait.

    python sim/walk.py                     # headless test of all gaits x all commands
    python sim/walk.py --gait trot         # one gait
    python sim/walk.py --view              # live viewer demo (forward, sideways, turn ...)
    python sim/walk.py --gif               # render docs/images/walk_<gait>.gif
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crab.emotes import STAND  # noqa: E402
from crab.gait import _S, _T, GAITS, Gait  # noqa: E402
from crab.params import JOINT_NAMES, JOINT_SPECS, PARAMS  # noqa: E402
from sim.latency import DelayLine  # noqa: E402
from sim.servo_check import MAX_CMD, ServoMonitor  # noqa: E402

MODEL = ROOT / "sim" / "models" / PARAMS.model_file          # crab.xml; CRAB_DESIGN=v2: crab_v2.xml
IMG_DIR = ROOT / "docs" / "images"
DELAY = max(s.command_delay for s in JOINT_SPECS)
V, W = _S / _T, 1 / _T          # speed scale vs the 100 mm-shin original (same strides per body size)

# (label, (vx, vy, wz), seconds) - original speeds scaled to this robot
TESTS = [
    ("forward", (0.08 * V, 0.0, 0.0), 6.0),
    ("backward", (-0.08 * V, 0.0, 0.0), 6.0),
    ("side left", (0.0, 0.10 * V, 0.0), 6.0),
    ("side right", (0.0, -0.10 * V, 0.0), 6.0),
    ("turn left", (0.0, 0.0, 0.6 * W), 6.0),
    ("diag+turn", (0.05 * V, 0.06 * V, 0.3 * W), 6.0),
]

DEMO = [  # for --view / --gif
    ("stand", (0, 0, 0), 1.5),
    ("forward", (0.08 * V, 0, 0), 4.0),
    ("crab walk left", (0, 0.10 * V, 0), 4.0),
    ("crab walk right", (0, -0.10 * V, 0), 4.0),
    ("turn", (0, 0, 0.6 * W), 4.0),
    ("stop", (0, 0, 0), 2.0),
]


def make_controller(gait_name, m, d, com):
    return Gait(GAITS[gait_name], com_offset=com)


def load():
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
    for _ in range(500):                        # settle 1 s
        d.ctrl[:] = STAND
        mujoco.mj_step(m, d)
    com = d.subtree_com[1] - d.xpos[1]
    return m, d, com[:2].copy()


def yaw_of(d):
    R = d.xmat[1].reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def tilt_deg(d):
    return np.degrees(np.arccos(np.clip(d.xmat[1].reshape(3, 3)[2, 2], -1, 1)))


def evaluate(gait_name):
    m, d, com = load()
    gait = make_controller(gait_name, m, d, com)
    dt = m.opt.timestep
    rows, ok_all = [], True
    for label, cmd, secs in TESTS:
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        gait.reset()
        warm = int(2.0 / dt)                            # first 2 s = acceleration, not measured
        period = getattr(getattr(gait, "params", None), "period", None)
        if period:                                      # measure whole gait cycles (slow gaits vary within one)
            k = max(2, int((secs - 2.0) / period))
            n = warm + int(round(k * period / dt))
        else:
            n = int(secs / dt)
        v_acc, wz_acc, cnt = np.zeros(2), 0.0, 0
        mon = ServoMonitor(m.nu)
        max_tilt, min_h = 0.0, 1.0
        prev_q, peak_cmd_v = None, 0.0                  # peak commanded speed as a fraction of each servo's limit
        cmd_dt = getattr(gait, "ctrl_dt", dt)          # policies update targets at 50 Hz
        delay = DelayLine(DELAY, dt, STAND)
        for i in range(n):
            q = gait.step(dt, cmd)
            if prev_q is not None and not np.array_equal(q, prev_q):
                peak_cmd_v = max(peak_cmd_v, float((np.abs(q - prev_q) / cmd_dt / MAX_CMD).max()))
            prev_q = q
            d.ctrl[:] = delay(q)
            mujoco.mj_step(m, d)
            mon.update(d.actuator_force, d.actuator_velocity)
            max_tilt, min_h = max(max_tilt, tilt_deg(d)), min(min_h, d.xpos[1, 2])
            if i >= warm:
                R = d.xmat[1].reshape(3, 3)
                v_acc += (R.T @ d.qvel[0:3])[:2]
                wz_acc += d.qvel[5]
                cnt += 1
        v, wz = v_acc / cnt, wz_acc / cnt
        fell = max_tilt > 40 or min_h < 0.37 * float(m.key("stand").qpos[2])
        worst = mon.worst_joint()
        # The gait caps commands to what its stride can reach; score against that.
        # Yaw error is converted to foot speed (feet sit ~0.2 m * scale from the centre)
        # so translation and rotation errors are in the same units.
        eff = np.asarray(cmd) * gait.feasible_scale(cmd)
        R_FOOT = 0.2 * _S
        err = np.linalg.norm([v[0] - eff[0], v[1] - eff[1], R_FOOT * (wz - eff[2])])
        scale = np.linalg.norm([eff[0], eff[1], R_FOOT * eff[2]])
        track = max(0.0, 1 - err / scale) * 100
        # Servo check (sim/servo_check.py): FAIL if the measured motor model's drive
        # voltage saturates > 3% of the time or the gait commands > VMAX; the datasheet
        # torque-speed margin is reported as a warning column (ds%).
        hard = mon.volt_sat_pct > 3 or peak_cmd_v > 1.0
        heat = float(mon.thermal_pct.max())
        status = "FELL" if fell else ("SERVO LIMIT" if hard else ("weak" if track < 60 else "ok"))
        if status == "ok" and mon.ds_over_pct > 3:
            status = "ok (ds!)"
        if status.startswith("ok") and heat > 100:
            status += " (hot!)"                          # RMS torque above the servo's continuous rating
        ok_all &= status.startswith("ok")
        peak_pct = 100 * mon.peak_motor[worst] / JOINT_SPECS[worst].datasheet_stall
        rows.append((label, eff, v, wz, track, max_tilt, peak_pct, JOINT_NAMES[worst],
                     mon.volt_sat_pct, mon.ds_over_pct, heat, JOINT_NAMES[mon.hottest_joint()],
                     100 * peak_cmd_v, status))
    return rows, gait.unreachable, ok_all


def print_rows(name, rows, unreachable):
    print(f"\n=== {name} ===")
    print("(commands after the gait's stride cap; peak = peak motor torque % of that servo's stall;"
          " vsat% = time at drive limit; ds% = beyond datasheet torque-speed line; heat = RMS % of rated; "
          "cmd = commanded speed % of limit)")
    print(f"{'command':11s} {'vx':>6s} {'vy':>6s} {'wz':>6s} | {'got vx':>6s} {'vy':>6s} {'wz':>6s} | "
          f"{'track':>5s} {'tilt':>5s} {'peak':>5s} {'@joint':>14s} {'vsat%':>5s} {'ds%':>5s} "
          f"{'heat':>5s} {'@joint':>14s} {'cmd':>5s}  status")
    for label, c, v, wz, tr, tilt, pk, jn, vs, ds, ht, hj, cv, st in rows:
        print(f"{label:11s} {c[0]:6.3f} {c[1]:6.3f} {c[2]:6.2f} | {v[0]:6.3f} {v[1]:6.3f} {wz:6.2f} | "
              f"{tr:4.0f}% {tilt:4.1f}° {pk:4.0f}% {jn:>14s} {vs:5.1f} {ds:5.1f} {ht:4.0f}% {hj:>14s} {cv:4.0f}%  {st}")
    if unreachable:
        print(f"  ! IK clipped {unreachable} times")


def demo_controller(gait_name, m, d, com):
    gait = make_controller(gait_name, m, d, com)
    t_marks = np.cumsum([s for _, _, s in DEMO])

    def ctrl(t):
        k = int(np.searchsorted(t_marks, t, side="right"))
        k = min(k, len(DEMO) - 1)
        return DEMO[k], gait
    return ctrl, t_marks[-1]


def view(gait_name):
    import time

    import mujoco.viewer
    m, d, com = load()
    ctrl, total = demo_controller(gait_name, m, d, com)
    last = None
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = m.body("body").id
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.9 * _S, 135, -25    # sized to the crab
        t0 = d.time
        while v.is_running():
            start = time.time()
            for _ in range(int((1 / 60) / m.opt.timestep)):
                (label, cmd, _), gait = ctrl((d.time - t0) % total)
                if label != last:
                    print(f"> {label}  cmd={cmd}")
                    last = label
                d.ctrl[:] = gait.step(m.opt.timestep, cmd)
                mujoco.mj_step(m, d)
            v.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - start)))


def gif(gait_name, fps=12):
    from PIL import Image, ImageDraw
    m, d, com = load()
    ctrl, total = demo_controller(gait_name, m, d, com)
    r = mujoco.Renderer(m, 250, 400)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("body").id
    cam.distance, cam.azimuth, cam.elevation = 0.85 * _S, 135, -25         # sized to the crab
    frames, every, t0 = [], int(1 / fps / m.opt.timestep), d.time
    i = 0
    while d.time - t0 < total:
        (label, cmd, _), gait = ctrl(d.time - t0)
        d.ctrl[:] = gait.step(m.opt.timestep, cmd)
        mujoco.mj_step(m, d)
        if i % every == 0:
            r.update_scene(d, camera=cam)
            im = Image.fromarray(r.render())
            ImageDraw.Draw(im).text((10, 8), f"{gait_name}: {label}", fill=(25, 25, 25))
            frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=64))
        i += 1
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / f"walk_{gait_name}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"wrote {out.relative_to(ROOT)} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gait", choices=list(GAITS))
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()
    gaits = [a.gait] if a.gait else list(GAITS)
    if a.view:
        return view(gaits[-1])
    if a.gif:
        for g in gaits:
            gif(g)
        return
    ok = True
    for g in gaits:
        rows, unr, good = evaluate(g)
        print_rows(g, rows, unr)
        ok &= good
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
